/******************************************************************************
 *             Copyright 2020 DeepFrame AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "Conv2dCuDNN.hpp"
#include "iMatrix_func.hpp"
#include "iMatrix_func_adv.hpp"
#include "cuda_utility.cuh"

#include <cassert>
#include <cstdlib>

template <typename DT>
__global__ void __type2float_1d(float *dst_data, DT *src_data, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        dst_data[i] = src_data[i];
    }
}

template <typename DT>
__global__ void __float2type_1d(DT *dst_data, float *src_data, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        dst_data[i] = src_data[i];
    }
}

namespace julie
{
namespace la
{

template <typename DT>
cudnnHandle_t Conv2dCuDNN<DT>::CUDNN_HANDLE = nullptr;

template <typename DT>
void* Conv2dCuDNN<DT>::CUDNN_WORKSPACE = nullptr;

template <typename DT>
Conv2dCuDNN<DT>::Conv2dCuDNN (lint pad_h, lint pad_w, lint stride_h, lint stride_w)
    :
    m_pad_h {pad_h},
    m_pad_w {pad_w},
    m_stride_h {stride_h},
    m_stride_w {stride_w},
    m_fused_cache {iMatrix<DT>{}, iMatrix<DT>{}}
{
    //cudaSetDevice(0);

    if (!CUDNN_HANDLE)
    {
        cudnnCreate(&CUDNN_HANDLE);
    }
}
template
Conv2dCuDNN<int>::Conv2dCuDNN (lint pad_h, lint pad_w, lint stride_h, lint stride_w);
template
Conv2dCuDNN<float>::Conv2dCuDNN (lint pad_h, lint pad_w, lint stride_h, lint stride_w);


template <typename DT>
void Conv2dCuDNN<DT>::forward (
    iMatrix<DT> &output, const iMatrix<DT> & input, const iMatrix<DT> & weights, const iMatrix<DT> & bias)
{
    if (input.shape().dim() != 4)
    {
        throw std::invalid_argument(std::string("Conv2dCuDNN input should be 4-dimensional."));
    }

    if (weights.shape().dim() != 4)
    {
        throw std::invalid_argument(std::string("Conv2dCuDNN weight should be 4-dimensional."));
    }

    if (input.shape()[1] != weights.shape()[1])
    {
        throw std::invalid_argument(std::string("Number of channel should be consistent between input and conv filters."));
    }

    if (weights.shape()[0] != bias.shape().size())
    {
        throw std::invalid_argument(std::string("Number of filters and numbers of bias should be identical."));
    }

    // Resolve shapes
    lint w_n = weights.shape()[0];
    lint w_ch = weights.shape()[1];
    lint w_h = weights.shape()[2];
    lint w_w = weights.shape()[3];

    lint in_bat = input.shape()[0];
    lint in_ch = input.shape()[1];
    lint in_h = input.shape()[2];
    lint in_w = input.shape()[3];

    lint out_h = (in_h + this->m_pad_h * 2 - w_h) / this->m_stride_h + 1;
    lint out_w = (in_w + this->m_pad_w * 2 - w_w) / this->m_stride_w + 1;

    this->m_input = input;

    output.set_matrix_type(julie::MatrixType::CUDA);
    la::renew_if_shape_not_match(output, la::Shape{in_bat, w_n, out_h, out_w});

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&(this->input_descriptor)));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/in_bat,
                                        /*channels=*/in_ch,
                                        /*image_height=*/in_h,
                                        /*image_width=*/in_w));

    CUDNN_CHECK(cudnnCreateFilterDescriptor(&(this->kernel_descriptor)));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(this->kernel_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/w_n,
                                        /*in_channels=*/in_ch,
                                        /*kernel_height=*/w_h,
                                        /*kernel_width=*/w_w));

    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&(this->convolution_descriptor)));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(this->convolution_descriptor,
                                            /*pad_height=*/this->m_pad_h,
                                            /*pad_width=*/this->m_pad_w,
                                            /*vertical_stride=*/this->m_stride_h,
                                            /*horizontal_stride=*/this->m_stride_w,
                                            /*dilation_height=*/1,
                                            /*dilation_width=*/1,
                                            /*mode=*/CUDNN_CROSS_CORRELATION,
                                            /*computeType=*/CUDNN_DATA_FLOAT));

    int batch_size{0}, channels{0}, height{0}, width{0};
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(this->convolution_descriptor,
                                                    this->input_descriptor,
                                                    this->kernel_descriptor,
                                                    &batch_size,
                                                    &channels,
                                                    &height,
                                                    &width));

    //std::cerr << batch_size << " " << channels << " " << height << " " << width << std::endl;
    //std::cout << "Sheng Chunnan says shape of output should be" << output.shape() << std::endl;
    assert(batch_size == output.shape()[0]);
    assert(channels == output.shape()[1]);
    assert(height == output.shape()[2]);
    assert(width == output.shape()[3]);

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&(this->output_descriptor)));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/in_bat,
                                        /*channels=*/w_n,
                                        /*image_height=*/out_h,
                                        /*image_width=*/out_w));

    int n_elements = 0;
    CUDNN_CHECK(
        cudnnGetConvolutionForwardAlgorithm_v7(CUDNN_HANDLE,
                                          this->input_descriptor,
                                          this->kernel_descriptor,
                                          this->convolution_descriptor,
                                          this->output_descriptor,
                                          1,
                                          &n_elements,
                                          &(this->convolution_fwd_algorithm)));

    //std::cout << "Number of elements: " << n_elements << std::endl;

    size_t workspace_bytes{0};
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(CUDNN_HANDLE,
                                                    this->input_descriptor,
                                                    this->kernel_descriptor,
                                                    this->convolution_descriptor,
                                                    this->output_descriptor,
                                                    this->convolution_fwd_algorithm.algo,
                                                    &workspace_bytes));

    //std::cout << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
    //        << std::endl;
    //assert(workspace_bytes > 0);

    //std::cout << input << std::endl;

    void* workspace{nullptr};
    if (workspace_bytes > 0)
    {
        CUDA_CHECK( cudaMalloc(&workspace, workspace_bytes) );
    }

    DT *d_input = input.get_cuda_instance()->m_data;
    DT *d_weight = weights.get_cuda_instance()->m_data;
    DT *d_output = output.get_cuda_instance()->m_data;

    float *f_input;
    float *f_weight;
    float *f_output;

    if (std::is_same<DT, float>::value)
    {
        f_input = reinterpret_cast<float*>(d_input);
        f_weight = reinterpret_cast<float*>(d_weight);
        f_output = reinterpret_cast<float*>(d_output);
    }
    else
    {
        CUDA_CHECK( cudaMalloc(&f_input, sizeof(float) * input.shape().size()) );
        CUDA_CHECK( cudaMalloc(&f_weight, sizeof(float) * weights.shape().size()) );
        CUDA_CHECK( cudaMalloc(&f_output, sizeof(float) * output.shape().size()) );

        //std::cout << "__type2float_1d for input" << std::endl;
        lint size = input.shape().size();
        lint n_blocks = std::min(size / julie::la::cuda::BLOCK_WIDTH_1D + 1, julie::la::cuda::MAX_N_BLOCK_1D);
        __type2float_1d<<<n_blocks, julie::la::cuda::BLOCK_WIDTH_1D>>>(f_input, d_input, size);

        //std::cout << "__type2float_1d for weight" << std::endl;
        size = weights.shape().size();
        n_blocks = std::min(size / julie::la::cuda::BLOCK_WIDTH_1D + 1, julie::la::cuda::MAX_N_BLOCK_1D);
        __type2float_1d<<<n_blocks, julie::la::cuda::BLOCK_WIDTH_1D>>>(f_weight, d_weight, size);
    }

    //std::cout << "cudnnConvolutionForward" << std::endl;
    const float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(CUDNN_HANDLE,
                                        &alpha,
                                        this->input_descriptor,
                                        f_input,
                                        this->kernel_descriptor,
                                        f_weight,
                                        this->convolution_descriptor,
                                        this->convolution_fwd_algorithm.algo,
                                        workspace,
                                        workspace_bytes,
                                        &beta,
                                        this->output_descriptor,
                                        f_output));

    if (std::is_same<DT, float>::value)
    {
    }
    else
    {
        //std::cout << "__float2type_1d for output" << std::endl;
        lint size = output.shape().size();
        lint n_blocks = std::min(size / julie::la::cuda::BLOCK_WIDTH_1D + 1, julie::la::cuda::MAX_N_BLOCK_1D);
        __float2type_1d<<<n_blocks, julie::la::cuda::BLOCK_WIDTH_1D>>>(d_output, f_output, size);

        CUDA_CHECK( cudaFree(f_input) );
        CUDA_CHECK( cudaFree(f_weight) );
        CUDA_CHECK( cudaFree(f_output) );

    }
    
    CUDA_CHECK( cudaFree(workspace) );

    // [w_n, out_h]
    bias.get_right_extended(this->m_bias_ch_out_h, out_h);
    // [w_n, out_h, out_w]
    this->m_bias_ch_out_h.get_right_extended(this->m_bias_ch_out, out_w);
    this->m_bias_ch_out.reshape(julie::la::Shape {w_n, out_h, out_w});

    output += this->m_bias_ch_out;
}
template
void Conv2dCuDNN<int>::forward (
    iMatrix<int> &output, const iMatrix<int> & input, const iMatrix<int> & weights, const iMatrix<int> & bias);
template
void Conv2dCuDNN<float>::forward (
    iMatrix<float> &output, const iMatrix<float> & input, const iMatrix<float> & weights, const iMatrix<float> & bias);


template <typename DT>
void Conv2dCuDNN<DT>::backward (
        iMatrix<DT> & in_gradient_out,
        iMatrix<DT> & w_gradient_out,
        iMatrix<DT> & b_gradient_out,
        const iMatrix<DT> & gradient_in, const Shape & input_sh, const iMatrix<DT> & weights)
{
    if (input_sh.dim() != 4)
    {
        throw std::invalid_argument(std::string("Conv2dCuDNN input should be 4-dimensional."));
    }

    if (weights.shape().dim() != 4)
    {
        throw std::invalid_argument(std::string("Conv2dCuDNN weight should be 4-dimensional."));
    }

    if (input_sh[1] != weights.shape()[1])
    {
        throw std::invalid_argument(std::string("Number of channel should be consistent between input and conv filters."));
    }

    // Resolve shapes
    lint w_n = weights.shape()[0];
    lint w_ch = weights.shape()[1];
    lint w_h = weights.shape()[2];
    lint w_w = weights.shape()[3];

    lint in_bat = input_sh[0];
    lint in_ch = input_sh[1];
    lint in_h = input_sh[2];
    lint in_w = input_sh[3];

    lint out_h = (in_h + this->m_pad_h * 2 - w_h) / this->m_stride_h + 1;
    lint out_w = (in_w + this->m_pad_w * 2 - w_w) / this->m_stride_w + 1;

    //std::cout << "Initialize backward outputs" << std::endl;
    in_gradient_out.set_matrix_type(julie::MatrixType::CUDA);
    la::renew_if_shape_not_match(in_gradient_out, la::Shape{in_bat, in_ch, in_h, in_w});
    w_gradient_out.set_matrix_type(julie::MatrixType::CUDA);
    la::renew_if_shape_not_match(w_gradient_out, la::Shape{w_n, w_ch, w_h, w_w});
    b_gradient_out.set_matrix_type(julie::MatrixType::CUDA);
    la::renew_if_shape_not_match(b_gradient_out, la::Shape{w_n});

    //std::cout << "input_gradient_descriptor" << std::endl;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&(this->input_gradient_descriptor)));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        this->input_gradient_descriptor,
        /*format=*/CUDNN_TENSOR_NCHW,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*batch_size=*/in_bat,
        /*channels=*/in_ch,
        /*image_height=*/in_h,
        /*image_width=*/in_w));

    //std::cout << "output_gradient_descriptor" << std::endl;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&(this->output_gradient_descriptor)));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        this->output_gradient_descriptor,
        /*format=*/CUDNN_TENSOR_NCHW,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*batch_size=*/in_bat,
        /*channels=*/w_n,
        /*image_height=*/out_h,
        /*image_width=*/out_w));

    CUDNN_CHECK(cudnnCreateFilterDescriptor(&(this->kernel_gradient_descriptor)));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        this->kernel_gradient_descriptor,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*format=*/CUDNN_TENSOR_NCHW,
        /*out_channels=*/w_n,
        /*in_channels=*/in_ch,
        /*kernel_height=*/w_h,
        /*kernel_width=*/w_w));

    //std::cout << "cudnnGetConvolutionBackwardFilterAlgorithm_v7" << std::endl;
    int bwd_kernel_count = 0;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        CUDNN_HANDLE,
        this->input_descriptor,
        this->output_gradient_descriptor,
        this->convolution_descriptor,
        this->kernel_gradient_descriptor,
        1,
        &bwd_kernel_count,
        &(this->convolution_bwd_kernel_algorithm)
    ));

    //std::cout << "cudnnGetConvolutionBackwardFilterWorkspaceSize" << std::endl;
    size_t bwd_kernel_workspace_bytes{0};
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        CUDNN_HANDLE,
        this->input_descriptor,
        this->output_gradient_descriptor,
        this->convolution_descriptor,
        this->kernel_gradient_descriptor,
        this->convolution_bwd_kernel_algorithm.algo,
        &bwd_kernel_workspace_bytes));

    //std::cout << "cudnnGetConvolutionBackwardDataAlgorithm_v7" << std::endl;
    int bwd_data_count = 0;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        CUDNN_HANDLE,
        this->kernel_descriptor,
        this->output_gradient_descriptor,
        this->convolution_descriptor,
        this->input_gradient_descriptor,
        1,
        &bwd_data_count,
        &(this->convolution_bwd_data_algorithm)
    ));

    //std::cout << "cudnnGetConvolutionBackwardDataWorkspaceSize" << std::endl;
    size_t bwd_data_workspace_bytes{0};
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        CUDNN_HANDLE,
        this->kernel_descriptor,
        this->output_gradient_descriptor,
        this->convolution_descriptor,
        this->input_gradient_descriptor,
        this->convolution_bwd_data_algorithm.algo,
        &bwd_data_workspace_bytes
    ));

    void* bwd_kernel_workspace{nullptr};
    void* bwd_data_workspace{nullptr};
    if (bwd_kernel_workspace_bytes > 0)
    {
        CUDA_CHECK( cudaMalloc(&bwd_kernel_workspace, bwd_kernel_workspace_bytes) );
    }
    if (bwd_data_workspace_bytes > 0)
    {
        CUDA_CHECK( cudaMalloc(&bwd_data_workspace, bwd_data_workspace_bytes) );
    }

    DT *d_input = this->m_input.get_cuda_instance()->m_data;
    DT *d_output_gradient = gradient_in.get_cuda_instance()->m_data;
    DT *d_weight = weights.get_cuda_instance()->m_data;
    DT *d_input_gradient = in_gradient_out.get_cuda_instance()->m_data;
    DT *d_weight_gradient = w_gradient_out.get_cuda_instance()->m_data;
    
    float *f_input;
    float *f_output_gradient;
    float *f_weight;
    float *f_input_gradient;
    float *f_weight_gradient;
    
    if (std::is_same<DT, float>::value)
    {
        f_input = reinterpret_cast<float*>(d_input);
        f_output_gradient = reinterpret_cast<float*>(d_output_gradient);
        f_weight = reinterpret_cast<float*>(d_weight);
        f_input_gradient = reinterpret_cast<float*>(d_input_gradient);
        f_weight_gradient = reinterpret_cast<float*>(d_weight_gradient);
    }
    else
    {
        CUDA_CHECK( cudaMalloc(&f_input, sizeof(float) * this->m_input.shape().size()) );
        CUDA_CHECK( cudaMalloc(&f_output_gradient, sizeof(float) * gradient_in.shape().size()) );
        CUDA_CHECK( cudaMalloc(&f_weight, sizeof(float) * weights.shape().size()) );
        CUDA_CHECK( cudaMalloc(&f_input_gradient, sizeof(float) * in_gradient_out.shape().size()) );
        CUDA_CHECK( cudaMalloc(&f_weight_gradient, sizeof(float) * weights.shape().size()) );
        
        lint size = this->m_input.shape().size();
        lint n_blocks = std::min(size / julie::la::cuda::BLOCK_WIDTH_1D + 1, julie::la::cuda::MAX_N_BLOCK_1D);
        __type2float_1d<<<n_blocks, julie::la::cuda::BLOCK_WIDTH_1D>>>(f_input, d_input, size);

        size = gradient_in.shape().size();
        n_blocks = std::min(size / julie::la::cuda::BLOCK_WIDTH_1D + 1, julie::la::cuda::MAX_N_BLOCK_1D);
        __type2float_1d<<<n_blocks, julie::la::cuda::BLOCK_WIDTH_1D>>>(f_output_gradient, d_output_gradient, size);

        size = weights.shape().size();
        n_blocks = std::min(size / julie::la::cuda::BLOCK_WIDTH_1D + 1, julie::la::cuda::MAX_N_BLOCK_1D);
        __type2float_1d<<<n_blocks, julie::la::cuda::BLOCK_WIDTH_1D>>>(f_weight, d_weight, size);
    }

    const float alpha = 1.0f, beta = 0.0f;

    //std::cout << "cudnnConvolutionBackwardFilter" << std::endl;
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(
        CUDNN_HANDLE,
        &alpha,
        this->input_descriptor,
        f_input,
        this->output_gradient_descriptor,
        f_output_gradient,
        this->convolution_descriptor,
        this->convolution_bwd_kernel_algorithm.algo,
        bwd_kernel_workspace,
        bwd_kernel_workspace_bytes,
        &beta,
        this->kernel_gradient_descriptor,
        f_weight_gradient
    ));

    //std::cout << "cudnnConvolutionBackwardData" << std::endl;
    CUDNN_CHECK(cudnnConvolutionBackwardData(
        CUDNN_HANDLE,
        &alpha,
        this->kernel_descriptor,
        f_weight,
        this->output_gradient_descriptor,
        f_output_gradient,
        this->convolution_descriptor,
        this->convolution_bwd_data_algorithm.algo,
        bwd_data_workspace,
        bwd_data_workspace_bytes,
        &beta,
        this->input_gradient_descriptor,
        f_input_gradient
    ));

    if (std::is_same<DT, float>::value)
    {
    }
    else
    {
        lint size = w_gradient_out.shape().size();
        lint n_blocks = std::min(size / julie::la::cuda::BLOCK_WIDTH_1D + 1, julie::la::cuda::MAX_N_BLOCK_1D);
        __float2type_1d<<<n_blocks, julie::la::cuda::BLOCK_WIDTH_1D>>>(d_weight_gradient, f_weight_gradient, size);

        size = in_gradient_out.shape().size();
        n_blocks = std::min(size / julie::la::cuda::BLOCK_WIDTH_1D + 1, julie::la::cuda::MAX_N_BLOCK_1D);
        __float2type_1d<<<n_blocks, julie::la::cuda::BLOCK_WIDTH_1D>>>(d_input_gradient, f_input_gradient, size);

        CUDA_CHECK( cudaFree(f_input) );
        CUDA_CHECK( cudaFree(f_output_gradient) );
        CUDA_CHECK( cudaFree(f_weight) );
        CUDA_CHECK( cudaFree(f_input_gradient) );
        CUDA_CHECK( cudaFree(f_weight_gradient) );
    }

    CUDA_CHECK( cudaFree(bwd_kernel_workspace) );
    CUDA_CHECK( cudaFree(bwd_data_workspace) );
    

    gradient_in.get_reduce_sum(this->m_fused_cache[0], 0);
    this->m_fused_cache[0].get_reduce_sum(this->m_fused_cache[1], 1);
    this->m_fused_cache[1].get_reduce_sum(b_gradient_out, 1);

/*
    std::cout << "gradient_in: " << gradient_in << std::endl;
    std::cout << "in_gradient_out: " << in_gradient_out << std::endl;
    std::cout << "w_gradient_out: " << w_gradient_out << std::endl;
    std::cout << "b_gradient_out: " << b_gradient_out << std::endl;
*/
}
template
void Conv2dCuDNN<int>::backward (
        iMatrix<int> & in_gradient_out,
        iMatrix<int> & w_gradient_out,
        iMatrix<int> & b_gradient_out,
        const iMatrix<int> & gradient_in, const Shape & input_sh, const iMatrix<int> & weights);
template
void Conv2dCuDNN<float>::backward (
        iMatrix<float> & in_gradient_out,
        iMatrix<float> & w_gradient_out,
        iMatrix<float> & b_gradient_out,
        const iMatrix<float> & gradient_in, const Shape & input_sh, const iMatrix<float> & weights);



template <typename DT>
void Conv2dCuDNN<DT>::clear_cache()
{

}
template
void Conv2dCuDNN<int>::clear_cache();
template
void Conv2dCuDNN<float>::clear_cache();


} // la
} // julie
