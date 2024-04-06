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

#include "Matrix_CUDA_func_adv.hpp"
#include "Matrix_CUDA_func.hpp"
#include "nsqrt.hpp"

#include <algorithm>
#include <limits>

template <typename DT>
__global__ void __pad_2d_kernel_3d(
    DT *out_data, DT *in_data,
    int64_t in_bat, int64_t in_ch, int64_t in_h, int64_t in_w,
    int64_t pad_h, int64_t pad_w)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    int64_t offset_h = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t n_threads_h = gridDim.y * blockDim.y;

    int64_t offset_w = blockIdx.z * blockDim.z + threadIdx.z;
    int64_t n_threads_w = gridDim.z * blockDim.z;

    int64_t len = in_bat * in_ch;

    int64_t out_h = in_h + pad_h * 2;
    int64_t out_w = in_w + pad_w * 2;

    int64_t in_h_plus_pad = in_h + pad_h;
    int64_t in_w_plus_pad = in_w + pad_w;

    int64_t block_in = in_h * in_w;
    int64_t block_out = out_h * out_w;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        for (lint out_h_i = offset_h; out_h_i < out_h; out_h_i += n_threads_h)
        {
            if (out_h_i < pad_h || out_h_i >= in_h_plus_pad)
            {
                for (lint out_w_i = offset_w; out_w_i < out_w; out_w_i += n_threads_w)
                {
                    out_data[i * block_out + out_h_i * out_w + out_w_i] = 0;
                }
            }
            else
            {
                for (lint out_w_i = offset_w; out_w_i < out_w; out_w_i += n_threads_w)
                {
                    if (out_w_i >= pad_w && out_w_i < in_w_plus_pad)
                    {
                        out_data[i * block_out + out_h_i * out_w + out_w_i] = 
                            in_data[i * block_in + (out_h_i - pad_h) * in_w + (out_w_i - pad_w)];
                    }
                    else
                    {
                        out_data[i * block_out + out_h_i * out_w + out_w_i] = 0;
                    }
                }
            }
        }
    }
}


template <typename DT>
__global__ void __pad_2d_backward_kernel_3d(
    DT *out_data, DT *in_data,
    int64_t out_bat, int64_t out_ch, int64_t out_h, int64_t out_w,
    int64_t pad_h, int64_t pad_w)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    int64_t offset_h = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t n_threads_h = gridDim.y * blockDim.y;

    int64_t offset_w = blockIdx.z * blockDim.z + threadIdx.z;
    int64_t n_threads_w = gridDim.z * blockDim.z;

    int64_t len = out_bat * out_ch;

    int64_t in_h = out_h - pad_h * 2;
    int64_t in_w = out_w - pad_w * 2;

    int64_t out_h_sub_pad = out_h - pad_h;
    int64_t out_w_sub_pad = out_w - pad_w;

    int64_t block_in = out_h * out_w;
    int64_t block_out = in_h * in_w;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        for (lint out_h_i = pad_h + offset_h; out_h_i < out_h_sub_pad; out_h_i += n_threads_h)
        {
            for (lint out_w_i = pad_w + offset_w; out_w_i < out_w_sub_pad; out_w_i += n_threads_w)
            {
                out_data[i * block_out + (out_h_i - pad_h) * in_w + (out_w_i - pad_w)] =
                    in_data[i * block_in + out_h_i * out_w + out_w_i];
            }
        }
    }
}
    


template <typename DT>
__global__ void __img2row_2d_kernel_3d(
    DT *out_data, DT *in_data,
    int64_t in_bat, int64_t in_ch, int64_t in_h, int64_t in_w,
    int64_t stride_h, int64_t stride_w,
    int64_t w_h, int64_t w_w)
{
    int64_t offset_hw = blockIdx.x * blockDim.x + threadIdx.x; // Parallel for stride_h and stride_w
    int64_t n_threads_hw = gridDim.x * blockDim.x;

    int64_t offset_ch = blockIdx.y * blockDim.y + threadIdx.y; // Parallel for channel
    int64_t n_threads_ch = gridDim.y * blockDim.y;

    int64_t offset_b = blockIdx.z * blockDim.z + threadIdx.z; // Parallel for batch size
    int64_t n_threads_b = gridDim.z * blockDim.z;

    int64_t block_in = in_ch * in_h * in_w;
    int64_t in_ch_size = in_h * in_w;
    int64_t conv_out_h = (in_h - w_h) / stride_h + 1;
    int64_t conv_out_w = (in_w - w_w) / stride_w + 1;

    int64_t out_h = conv_out_h * conv_out_w;
    int64_t out_w = in_ch * w_h * w_w;
    int64_t block_out = out_h * out_w;

    int64_t w_ch_size = w_h * w_w;

    for (int64_t batch_i = offset_b; batch_i < in_bat; batch_i += n_threads_b)
    {
        for (int64_t out_h_i = offset_hw; out_h_i < out_h; out_h_i += n_threads_hw) 
        {
            int64_t conv_out_h_i = out_h_i / conv_out_w;
            int64_t conv_out_w_i = out_h_i % conv_out_w;

            int64_t filter_begin = batch_i * block_in + conv_out_h_i * in_w * stride_h + conv_out_w_i * stride_w;
            int64_t out_h_begin = batch_i * block_out + out_h_i * out_w;

            for (int64_t in_ch_i = offset_ch; in_ch_i < in_ch; in_ch_i += n_threads_ch)
            {
                int64_t filter_h_begin = filter_begin + in_ch_i * in_ch_size;
                int64_t out_pos_a = out_h_begin + in_ch_i * w_ch_size;

                for (int64_t w_h_i = 0; w_h_i < w_h; ++w_h_i)
                {   
                    int64_t filter_pos = filter_h_begin;
                    int64_t out_pos_b = out_pos_a;

                    for (int64_t w_w_i = 0; w_w_i < w_w; ++w_w_i)
                    {
                        out_data[out_pos_b] = in_data[filter_pos];
                        ++filter_pos;
                        ++out_pos_b;
                    }

                    filter_h_begin += in_w;
                    out_pos_a += w_w;
                }
            }
        }
    }
}



template <typename DT>
__global__ void __img2row_2d_backward_kernel_3d(
    DT *in_grad_data, DT *grad_data,
    int64_t out_bat, int64_t out_h, int64_t out_w,
    int64_t in_h, int64_t in_w,
    int64_t stride_h, int64_t stride_w,
    int64_t w_ch, int64_t w_h, int64_t w_w)
{
    int64_t offset_hw = blockIdx.x * blockDim.x + threadIdx.x; // Parallel for stride_h and stride_w
    int64_t n_threads_hw = gridDim.x * blockDim.x;

    int64_t offset_ch = blockIdx.y * blockDim.y + threadIdx.y; // Parallel for channel
    int64_t n_threads_ch = gridDim.y * blockDim.y;

    int64_t offset_b = blockIdx.z * blockDim.z + threadIdx.z; // Parallel for batch size
    int64_t n_threads_b = gridDim.z * blockDim.z;

    int64_t block_out = w_ch * in_h * in_w;
    int64_t out_ch_size = in_h * in_w;

    // int64_t out_w = w_ch * w_h * w_w;
    int64_t block_in = out_h * out_w;

    int64_t w_ch_size = w_h * w_w;

    //int64_t conv_out_h = (in_h - w_h) / stride_h + 1;
    int64_t conv_out_w = (in_w - w_w) / stride_w + 1;


    // out_h == conv_out_h * conv_out_w

    for (int64_t batch_i = offset_b; batch_i < out_bat; batch_i += n_threads_b)
    {
        for (int64_t out_h_i = offset_hw; out_h_i < out_h; out_h_i += n_threads_hw) 
        {
            int64_t conv_out_h_i = out_h_i / conv_out_w;
            int64_t conv_out_w_i = out_h_i % conv_out_w;

            int64_t filter_begin = batch_i * block_out + conv_out_h_i * in_w * stride_h + conv_out_w_i * stride_w;
            int64_t grad_h_begin = batch_i * block_in + out_h_i * out_w;

            for (int64_t w_ch_i = offset_ch; w_ch_i < w_ch; w_ch_i += n_threads_ch)
            {
                int64_t filter_h_begin = filter_begin + w_ch_i * out_ch_size;
                int64_t grad_pos_a = grad_h_begin + w_ch_i * w_ch_size;

                for (int64_t w_h_i = 0; w_h_i < w_h; ++w_h_i)
                {   
                    int64_t filter_pos = filter_h_begin;
                    int64_t grad_pos_b = grad_pos_a;

                    for (int64_t w_w_i = 0; w_w_i < w_w; ++w_w_i)
                    {
                        DT plus = grad_data[grad_pos_b];
                        // Synchronization problem here
                        // Should use atomic operation to add
                        atomicAdd(in_grad_data + filter_pos, plus);
                        //in_grad_data[filter_pos] += plus;

                        ++filter_pos;
                        ++grad_pos_b;
                    }

                    filter_h_begin += in_w;
                    grad_pos_a += w_w;
                }
            }
        }
    }
    
}


template <typename DT>
__global__ void __maxpool_2d_kernel_2d(
    DT *out_data, DT *diff_data, DT *in_data,
    int64_t in_bat, int64_t in_ch,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w,
    int64_t diff_h, int64_t diff_w,
    int64_t stride_h, int64_t stride_w,
    int64_t k_h, int64_t k_w,
    DT max)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x; // Parallel for batch * channel
    int64_t n_threads = gridDim.x * blockDim.x;

    int64_t offset_hw = blockIdx.y * blockDim.y + threadIdx.y; // Parallel for stride_h and stride_w
    int64_t n_threads_hw = gridDim.y * blockDim.y;

    int64_t len = in_bat * in_ch;
    int64_t len_hw = out_h * out_w;

    int64_t in_ch_size = in_h * in_w;
    int64_t out_ch_size = len_hw;
    int64_t diff_ch_size = diff_h * diff_w;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        for (int64_t j = offset_hw; j < len_hw; j += n_threads_hw)
        {
            int64_t out_h_i = j / out_w;
            int64_t out_w_i = j % out_w;

            int64_t in_kernel_pos = i * in_ch_size + out_h_i * in_w * stride_h + out_w_i * stride_w;
            int64_t diff_kernel_pos = i * diff_ch_size + out_h_i * diff_w * k_h + out_w_i * k_w;
            int64_t out_pos = i * out_ch_size + j;

            int64_t max_k_h_i = 0;
            int64_t max_k_w_i = 0;

            DT m = max;

            for (lint k_h_i = 0; k_h_i < k_h; ++k_h_i)
            {
                for (lint k_w_i = 0; k_w_i < k_w; ++k_w_i)
                {
                    DT in_value = in_data[in_kernel_pos + k_h_i * in_w + k_w_i];
                    if (in_value > m)
                    {
                        m = in_value;
                        max_k_h_i = k_h_i;
                        max_k_w_i = k_w_i;
                    }
                }
            }

            diff_data[diff_kernel_pos + max_k_h_i * diff_w + max_k_w_i] = 1;
            out_data[out_pos] = m;
        }
    }
}


template <typename DT>
__global__ void __avgpool_2d_kernel_2d(
    DT *out_data, DT *in_data,
    int64_t in_bat, int64_t in_ch,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w,
    int64_t diff_h, int64_t diff_w,
    int64_t stride_h, int64_t stride_w,
    int64_t k_h, int64_t k_w,
    DT avg_coe)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x; // Parallel for batch * channel
    int64_t n_threads = gridDim.x * blockDim.x;

    int64_t offset_hw = blockIdx.y * blockDim.y + threadIdx.y; // Parallel for stride_h and stride_w
    int64_t n_threads_hw = gridDim.y * blockDim.y;

    int64_t len = in_bat * in_ch;
    int64_t len_hw = out_h * out_w;

    int64_t in_ch_size = in_h * in_w;
    int64_t out_ch_size = len_hw;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        for (int64_t j = offset_hw; j < len_hw; j += n_threads_hw)
        {
            int64_t out_h_i = j / out_w;
            int64_t out_w_i = j % out_w;

            int64_t in_kernel_pos = i * in_ch_size + out_h_i * in_w * stride_h + out_w_i * stride_w;
            int64_t out_pos = i * out_ch_size + j;

            DT sum = 0;

            for (lint k_h_i = 0; k_h_i < k_h; ++k_h_i)
            {
                for (lint k_w_i = 0; k_w_i < k_w; ++k_w_i)
                {
                    sum += in_data[in_kernel_pos + k_h_i * in_w + k_w_i];
                }
            }

            out_data[out_pos] = sum * avg_coe;
        }
    }
}


template <typename DT>
__global__ void __pool_generate_gradient_cache_backward_2d(
    DT *cache_data, DT *grad_data,
    int64_t bat, int64_t ch,
    int64_t diff_h, int64_t diff_w,
    int64_t out_h, int64_t out_w,
    int64_t k_h, int64_t k_w)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x; // Parallel for batch * channel
    int64_t n_threads = gridDim.x * blockDim.x;

    int64_t offset_hw = blockIdx.y * blockDim.y + threadIdx.y; // Parallel for diff_h and diff_w
    int64_t n_threads_hw = gridDim.y * blockDim.y;

    int64_t len = bat * ch;
    int64_t len_hw = diff_h * diff_w;

    int64_t diff_ch_size = len_hw;
    int64_t out_ch_size = out_h * out_w;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        for (int64_t j = offset_hw; j < len_hw; j += n_threads_hw)
        {
            int64_t diff_h_i = j / diff_w;
            int64_t diff_w_i = j % diff_w;
            int64_t out_h_i = diff_h_i / k_h;
            int64_t out_w_i = diff_w_i / k_w;

            cache_data[i * diff_ch_size + diff_h_i * diff_w + diff_w_i] = 
                grad_data[i * out_ch_size + out_h_i * out_w + out_w_i];
        }
    }
}


template <typename DT>
__global__ void __pool_generate_input_gradient_backward_2d(
    DT *in_grad_data, DT *cache_data,
    int64_t bat, int64_t ch,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w,
    int64_t diff_h, int64_t diff_w,
    int64_t stride_h, int64_t stride_w,
    int64_t k_h, int64_t k_w)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x; // Parallel for batch * channel
    int64_t n_threads = gridDim.x * blockDim.x;

    int64_t offset_hw = blockIdx.y * blockDim.y + threadIdx.y; // Parallel for out_h and out_w
    int64_t n_threads_hw = gridDim.y * blockDim.y;

    int64_t len = bat * ch;
    int64_t len_hw = out_h * out_w;

    int64_t in_ch_size = in_h * in_w;
    int64_t diff_ch_size = diff_h * diff_w;
    // int64_t out_ch_size = len_hw;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        for (int64_t j = offset_hw; j < len_hw; j += n_threads_hw)
        {
            int64_t out_h_i = j / out_w;
            int64_t out_w_i = j % out_w;

            int64_t kernel_pos_in_h = out_h_i * stride_h;
            int64_t kernel_pos_in_w = out_w_i * stride_w;

            int64_t kernel_pos_diff_h = out_h_i * k_h;
            int64_t kernel_pos_diff_w = out_w_i * k_w;

            // Calculation for one kernel
            DT *in_grad_pos = in_grad_data + (i * in_ch_size + kernel_pos_in_h * in_w + kernel_pos_in_w);
            DT *cache_pos = cache_data + (i * diff_ch_size + kernel_pos_diff_h * diff_w + kernel_pos_diff_w);
            for (lint k_h_i = 0; k_h_i < k_h; ++k_h_i)
            {      
                for (lint k_w_i = 0; k_w_i < k_w; ++k_w_i)
                {
                    DT plus = cache_pos[k_w_i];
                    // Synchronization problem here
                    // Should use atomic operation to add
                    atomicAdd(in_grad_pos + k_w_i, plus);
                    // in_grad_pos[k_w_i] += cache_pos[k_w_i];
                }

                in_grad_pos += in_w;
                cache_pos += diff_w;
            }
        }
    }
}


namespace julie
{
namespace la
{
namespace cuda
{

/*
Do padding for a 4-dimensional array like this:
from [b, c, h, w] to [b, c, h + pad_h, w + pad_w]
*/
template <typename DT>
void pad_2d(Matrix_CUDA<DT> & output, const Matrix_CUDA<DT> & input, lint pad_h, lint pad_w)
{
    julie::la::Shape output_sh{input.m_shape[0], input.m_shape[1], input.m_shape[2] + pad_h * 2, input.m_shape[3] + pad_w * 2};

    renew_if_shape_not_match(output, output_sh);
    
    lint size_x = input.m_shape[0] * input.m_shape[1];
    lint size_y = output.m_shape[2];
    lint size_z = output.m_shape[3];

    lint size = size_x * size_y * size_z;
    
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        lint coe = static_cast<lint>(pow(size / n_blocks, 1.0f/3.0f));
        lint x = std::max(size_x / coe, 1L);
        lint y = std::max(size_y / coe, 1L);
        lint z = std::max(size_z / coe, 1L);

        // std::cout << "pad_2d 3D grid: " << x << " " << y << " " << z << std::endl;

        dim3 dimGrid(x, y, z);
        dim3 dimBlock(BLOCK_WIDTH_1D / 64, 8, 8);

        __pad_2d_kernel_3d<<<dimGrid, dimBlock>>>(
            output.m_data, input.m_data,
            input.m_shape[0], input.m_shape[1], input.m_shape[2], input.m_shape[3],
            pad_h, pad_w);
    }
}
template
void pad_2d(Matrix_CUDA<int> & output, const Matrix_CUDA<int> & input, lint pad_h, lint pad_w);
template
void pad_2d(Matrix_CUDA<float> & output, const Matrix_CUDA<float> & input, lint pad_h, lint pad_w);


/*
Do back-propagation for a 4-dimensional array like this:
from [b, c, h + pad_h, w + pad_w] to [b, c, h, w]
*/
template <typename DT>
void pad_2d_backward(Matrix_CUDA<DT> & in_gradient, const Matrix_CUDA<DT> & gradient, lint pad_h, lint pad_w)
{
    julie::la::Shape in_gradient_sh {gradient.m_shape[0], gradient.m_shape[1], gradient.m_shape[2] - pad_h * 2, gradient.m_shape[3] - pad_w * 2};

    renew_if_shape_not_match(in_gradient, in_gradient_sh);

    lint size_x = gradient.m_shape[0] * gradient.m_shape[1];
    lint size_y = gradient.m_shape[2];
    lint size_z = gradient.m_shape[3];

    lint size = size_x * size_y * size_z;
    
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        lint coe = static_cast<lint>(pow(size / n_blocks, 1.0f/3.0f));
        lint x = std::max(size_x / coe, 1L);
        lint y = std::max(size_y / coe, 1L);
        lint z = std::max(size_z / coe, 1L);

        // std::cout << "pad_2d_backward 3D grid: " << x << " " << y << " " << z << std::endl;

        dim3 dimGrid(x, y, z);
        dim3 dimBlock(BLOCK_WIDTH_1D / 64, 8, 8);

        __pad_2d_backward_kernel_3d<<<dimGrid, dimBlock>>>(
            in_gradient.m_data, gradient.m_data,
            gradient.m_shape[0], gradient.m_shape[1], gradient.m_shape[2], gradient.m_shape[3],
            pad_h, pad_w);
    }
}
template
void pad_2d_backward(Matrix_CUDA<int> & output, const Matrix_CUDA<int> & gradient, lint pad_h, lint pad_w);
template
void pad_2d_backward(Matrix_CUDA<float> & output, const Matrix_CUDA<float> & gradient, lint pad_h, lint pad_w);


// Convert an array of [b, c, h, w] into an array of [b, n_conv_outputs, c * w_h * w_w]
// where n_conv_outputs == conv_output_h * conv_output_w
template <typename DT>
void img2row_2d(Matrix_CUDA<DT> & output, const Matrix_CUDA<DT> & input, lint stride_h, lint stride_w, lint w_h, lint w_w)
{
    if (input.m_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("img2row_2d input should be 4-dimensional."));
    }

    lint w_ch = input.m_shape[1];

    lint in_bat = input.m_shape[0];
    lint in_ch = input.m_shape[1];
    lint in_h = input.m_shape[2];
    lint in_w = input.m_shape[3];

    lint out_bat = in_bat;

    lint conv_out_h = (in_h - w_h) / stride_h + 1;
    lint conv_out_w = (in_w - w_w) / stride_w + 1;

    lint out_h = conv_out_h * conv_out_w;
    lint out_w = in_ch * w_h * w_w;

    //std::cout << "Mass of output (num of elements): " << out_bat * out_h * out_w << std::endl; 
    renew_if_shape_not_match(output, julie::la::Shape{out_bat, out_h, out_w});

    lint size = out_h * in_ch * in_bat;
    
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        lint coe = static_cast<lint>(pow(size / n_blocks, 1.0f/3.0f));
        lint x = std::max(out_h / coe, 1L);
        lint y = std::max(in_ch / coe, 1L);
        lint z = std::max(in_bat / coe, 1L);

        // std::cout << "img2row_2d 3D grid: " << x << " " << y << " " << z << std::endl;

        dim3 dimGrid(x, y, z);
        dim3 dimBlock(BLOCK_WIDTH_1D / 64, 8, 8);

        __img2row_2d_kernel_3d <<<dimGrid, dimBlock>>> (
            output.m_data, input.m_data,
            in_bat, in_ch, in_h, in_w,
            stride_h, stride_w,
            w_h, w_w);
    }
}
template
void img2row_2d(Matrix_CUDA<int> & output, const Matrix_CUDA<int> & input, lint stride_h, lint stride_w, lint w_h, lint w_w);
template
void img2row_2d(Matrix_CUDA<float> & output, const Matrix_CUDA<float> & input, lint stride_h, lint stride_w, lint w_h, lint w_w);


template <typename DT>
void img2row_2d_backward(Matrix_CUDA<DT> & in_gradient, const Shape & in_shape, const Matrix_CUDA<DT> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w)
{
    if (gradient.m_shape.dim() != 3)
    {
        throw std::invalid_argument(std::string("Input gradient should be 3-dimensional."));
    }

    if (in_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("Input for conv2d should be 4-dimensional."));
    }

    Shape out_shape = gradient.shape();
    lint out_bat = out_shape[0];
    lint out_h = out_shape[1];
    lint out_w = out_shape[2];
    lint in_ch = out_w / (w_h * w_w);

    if (in_shape[0] != out_bat || in_shape[1] != in_ch)
    {
        throw std::invalid_argument(std::string("Batch size for input and ouput for img2row should be she same."));
    }

    
    lint conv_out_h = (in_shape[2] - w_h) / stride_h + 1;
    lint conv_out_w = (in_shape[3] - w_w) / stride_w + 1;

    // std::cout << "stride: " << stride_h << " " << stride_w << std::endl;
    // std::cout << "Shape of input: " << in_shape << std::endl;
    // std::cout << "Shape of img2row output gradient: " << gradient.shape() << std::endl;

    if (conv_out_h * conv_out_w != out_h)
    {
        throw std::invalid_argument(std::string("Shape of conv2d input does not match shape of img2row"));
    }

    renew_if_shape_not_match(in_gradient, in_shape);

    in_gradient = 0;

    lint size = out_h * in_ch * out_bat;
    
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        lint coe = static_cast<lint>(pow(size / n_blocks, 1.0f/3.0f));
        lint x = std::max(out_h / coe, 1L);
        lint y = std::max(in_ch / coe, 1L);
        lint z = std::max(out_bat / coe, 1L);

        // std::cout << "img2row_2d_backward 3D grid: " << x << " " << y << " " << z << std::endl;

        dim3 dimGrid(x, y, z);
        dim3 dimBlock(BLOCK_WIDTH_1D / 64, 8, 8);

        __img2row_2d_backward_kernel_3d <<<dimGrid, dimBlock>>> (
            in_gradient.m_data, gradient.m_data,
            out_bat, out_h, out_w,
            in_shape[2], in_shape[3],
            stride_h, stride_w,
            in_shape[1], w_h, w_w);

    }
}
template
void img2row_2d_backward(Matrix_CUDA<int> & in_gradient, const Shape & in_shape, const Matrix_CUDA<int> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w);
template
void img2row_2d_backward(Matrix_CUDA<float> & in_gradient, const Shape & in_shape, const Matrix_CUDA<float> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w);



// Convert an array of [b, c, h, w] into an array of [b, c * w_h * w_w, n_conv_outputs]
// where n_conv_outputs == conv_output_h * conv_output_w
template <typename DT>
void img2col_2d(Matrix_CUDA<DT> & output, const Matrix_CUDA<DT> & input,
                                lint stride_h, lint stride_w, lint w_h, lint w_w)
{
    if (input.m_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("img2row_2d input should be 4-dimensional."));
    }

    Matrix_CUDA<DT> img2row_output;
    img2row_2d(img2row_output, input, stride_h, stride_w, w_h, w_w);

    transpose_neighboring_dims(output, img2row_output, 1, 1, 2, 2);
}
template
void img2col_2d(Matrix_CUDA<float> & output, const Matrix_CUDA<float> & input,
                                lint stride_h, lint stride_w, lint w_h, lint w_w);
template
void img2col_2d(Matrix_CUDA<int> & output, const Matrix_CUDA<int> & input,
                                lint stride_h, lint stride_w, lint w_h, lint w_w);


template <typename DT>
void img2col_2d_backward(Matrix_CUDA<DT> & in_gradient, const Shape & in_shape, const Matrix_CUDA<DT> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w)
{
    Matrix_CUDA<DT> trans_gradient;
    transpose_neighboring_dims(trans_gradient, gradient, 1, 1, 2, 2);

    img2row_2d_backward(in_gradient, in_shape, trans_gradient, stride_h, stride_w, w_h, w_w);
}
template
void img2col_2d_backward(Matrix_CUDA<float> & in_gradient, const Shape & in_shape, const Matrix_CUDA<float> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w);
template
void img2col_2d_backward(Matrix_CUDA<int> & in_gradient, const Shape & in_shape, const Matrix_CUDA<int> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w);


template <typename DT>
void maxpool_2d(Matrix_CUDA<DT> &output, Matrix_CUDA<DT> &diff, const Matrix_CUDA<DT> &input,
                lint stride_h, lint stride_w, lint k_h, lint k_w)
{
    if (input.m_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("maxpool_2d input should be 4-dimensional."));
    }

    lint in_bat = input.m_shape[0];
    lint in_ch = input.m_shape[1];
    lint in_h = input.m_shape[2];
    lint in_w = input.m_shape[3];

    lint out_h = (in_h - k_h) / stride_h + 1;
    lint out_w = (in_w - k_w) / stride_w + 1;

    int64_t diff_h = out_h * k_h;
    int64_t diff_w = out_w * k_w;

    renew_if_shape_not_match(output, julie::la::Shape{in_bat, in_ch, out_h, out_w});
    renew_if_shape_not_match(diff, julie::la::Shape{in_bat, in_ch, diff_h, diff_w});

    diff = 0;

    lint size_h = in_bat * in_ch;
    lint size_w = out_h * out_w;
    lint n_blocks = std::min(size_h * size_w / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    int64_t grid_x = std::max(nsqrt(n_blocks * size_h / size_w), 1L);
    int64_t grid_y = std::max(n_blocks / grid_x, 1L);

    dim3 dimGrid(grid_x, grid_y, 1);
    dim3 dimBlock(BLOCK_WIDTH_1D / 16, 16, 1);

    DT max = std::numeric_limits<DT>::max() * (-1);

    __maxpool_2d_kernel_2d<<<dimGrid, dimBlock>>>(
        output.m_data, diff.m_data, input.m_data,
        in_bat, in_ch,
        in_h, in_w,
        out_h, out_w,
        diff_h, diff_w,
        stride_h, stride_w,
        k_h, k_w,
        max);
    
}
template
void maxpool_2d(Matrix_CUDA<float> &output, Matrix_CUDA<float> &diff, const Matrix_CUDA<float> &input,
                lint stride_h, lint stride_w, lint k_h, lint k_w);
template
void maxpool_2d(Matrix_CUDA<int> &output, Matrix_CUDA<int> &diff, const Matrix_CUDA<int> &input,
                lint stride_h, lint stride_w, lint k_h, lint k_w);

  
template <typename DT>
void avgpool_2d(Matrix_CUDA<DT> &output, Matrix_CUDA<DT> &diff, const Matrix_CUDA<DT> &input,
                lint stride_h, lint stride_w, lint k_h, lint k_w)
{
    if (input.m_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("maxpool_2d input should be 4-dimensional."));
    }

    lint in_bat = input.m_shape[0];
    lint in_ch = input.m_shape[1];
    lint in_h = input.m_shape[2];
    lint in_w = input.m_shape[3];

    lint out_h = (in_h - k_h) / stride_h + 1;
    lint out_w = (in_w - k_w) / stride_w + 1;

    int64_t diff_h = out_h * k_h;
    int64_t diff_w = out_w * k_w;

    renew_if_shape_not_match(output, julie::la::Shape{in_bat, in_ch, out_h, out_w});
    renew_if_shape_not_match(diff, julie::la::Shape{in_bat, in_ch, diff_h, diff_w});

    DT avg_coe = static_cast<DT>(1) / (k_h * k_w);
    diff = avg_coe;

    lint size_h = in_bat * in_ch;
    lint size_w = out_h * out_w;
    lint n_blocks = std::min(size_h * size_w / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    int64_t grid_x = std::max(nsqrt(n_blocks * size_h / size_w), 1L);
    int64_t grid_y = std::max(n_blocks / grid_x, 1L);

    dim3 dimGrid(grid_x, grid_y, 1);
    dim3 dimBlock(BLOCK_WIDTH_1D / 16, 16, 1);

    __avgpool_2d_kernel_2d<<<dimGrid, dimBlock>>>(
        output.m_data, input.m_data,
        in_bat, in_ch,
        in_h, in_w,
        out_h, out_w,
        diff_h, diff_w,
        stride_h, stride_w,
        k_h, k_w,
        avg_coe);
    
}
template
void avgpool_2d(Matrix_CUDA<float> &output, Matrix_CUDA<float> &diff, const Matrix_CUDA<float> &input,
                lint stride_h, lint stride_w, lint k_h, lint k_w);
template
void avgpool_2d(Matrix_CUDA<int> &output, Matrix_CUDA<int> &diff, const Matrix_CUDA<int> &input,
                lint stride_h, lint stride_w, lint k_h, lint k_w);


template <typename DT>
void pool_2d_backward(Matrix_CUDA<DT> &in_gradient, Matrix_CUDA<DT> &gradient_cache,
                        const Shape &in_shape, const Matrix_CUDA<DT> &diff, const Matrix_CUDA<DT> &gradient,
                        lint stride_h, lint stride_w, lint k_h, lint k_w)
{
    if (gradient.m_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("Gradient of maxpool_2d output should be 4-dimensional."));
    }

    if (diff.m_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("Derivative of maxpool_2d should be 4-dimensional."));
    }

    if (in_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("maxpool_2d input should be 4-dimensional."));
    }

    Shape out_shape = gradient.shape();
    lint out_bat = out_shape[0];
    lint out_ch = out_shape[1];

    if (in_shape[0] != out_bat || in_shape[1] != out_ch)
    {
        throw std::invalid_argument(std::string("Batch size and channel size of input and ouput gradients for max pooling 2d should be she same."));
    }

    lint in_bat = in_shape[0];
    lint in_ch = in_shape[1];
    lint in_h = in_shape[2];
    lint in_w = in_shape[3];

    lint out_h = (in_h - k_h) / stride_h + 1;
    lint out_w = (in_w - k_w) / stride_w + 1;

    if (out_h != out_shape[2] || out_w != out_shape[3])
    {
        throw std::invalid_argument(std::string("Height and width of input and output gradients for max pooling 2d do not match."));
    }

    lint diff_h = out_h * k_h;
    lint diff_w = out_w * k_w;

    Shape diff_sh = julie::la::Shape{in_bat, in_ch, diff_h, diff_w};
    if (diff_sh != diff.m_shape)
    {
        throw std::invalid_argument(std::string("Shape of derivative does not match the shape required for max pooling 2d."));
    }

    renew_if_shape_not_match(in_gradient, in_shape);
    renew_if_shape_not_match(gradient_cache, diff_sh);

    in_gradient = 0;


    lint size_h = in_bat * in_ch;
    lint size_w = diff_h * diff_w;
    lint n_blocks = std::min(size_h * size_w / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    int64_t grid_x = std::max(nsqrt(n_blocks * size_h / size_w), 1L);
    int64_t grid_y = std::max(n_blocks / grid_x, 1L);

    dim3 dimGrid(grid_x, grid_y, 1);
    dim3 dimBlock(BLOCK_WIDTH_1D / 16, 16, 1);

    __pool_generate_gradient_cache_backward_2d<<<dimGrid, dimBlock>>> (
        gradient_cache.m_data, gradient.m_data,
        in_bat, in_ch,
        diff_h, diff_w,
        out_h, out_w,
        k_h, k_w);

    // The chain rule
    gradient_cache *= diff;


    size_h = in_bat * in_ch;
    size_w = out_h * out_w;
    n_blocks = std::min(size_h * size_w / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    grid_x = std::max(nsqrt(n_blocks * size_h / size_w), 1L);
    grid_y = std::max(n_blocks / grid_x, 1L);

    dimGrid = dim3(grid_x, grid_y, 1);
    dimBlock = dim3(BLOCK_WIDTH_1D / 16, 16, 1);

    __pool_generate_input_gradient_backward_2d<<<dimGrid, dimBlock>>> (
        in_gradient.m_data, gradient_cache.m_data,
        in_bat, in_ch,
        in_h, in_w,
        out_h, out_w,
        diff_h, diff_w,
        stride_h, stride_w,
        k_h, k_w);
}
template
void pool_2d_backward(Matrix_CUDA<float> &in_gradient, Matrix_CUDA<float> &gradient_cache,
                        const Shape &in_shape, const Matrix_CUDA<float> &diff, const Matrix_CUDA<float> &gradient,
                        lint stride_h, lint stride_w, lint k_h, lint k_w);
template
void pool_2d_backward(Matrix_CUDA<int> &in_gradient, Matrix_CUDA<int> &gradient_cache,
                        const Shape &in_shape, const Matrix_CUDA<int> &diff, const Matrix_CUDA<int> &gradient,
                        lint stride_h, lint stride_w, lint k_h, lint k_w);

} // namsepace cuda
} // namespace la
} // namespace julie
