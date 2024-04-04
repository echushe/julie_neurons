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

#include "iMatrix_func_adv.hpp"
#include "Matrix_CPU_func_adv.hpp"
#ifdef WITH_CUDA
#include "Matrix_CUDA_func_adv.hpp"
#endif


namespace julie
{
namespace la
{

template <typename DT>
void pad_2d(iMatrix<DT> & output, const iMatrix<DT> & input, lint pad_h, lint pad_w)
{
    output.set_matrix_type(input.get_matrix_type());

    if (input.get_matrix_type() == MatrixType::CPU)
    {
        cpu::pad_2d( *(output.get_cpu_instance()), *(input.get_cpu_instance()), pad_h, pad_w );
    }
    else if (input.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::pad_2d( *(output.get_cuda_instance()), *(input.get_cuda_instance()), pad_h, pad_w );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (input.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void pad_2d(iMatrix<float> & output, const iMatrix<float> & input, lint pad_h, lint pad_w);
template
void pad_2d(iMatrix<int> & output, const iMatrix<int> & input, lint pad_h, lint pad_w);


template <typename DT>
void pad_2d_backward(iMatrix<DT> & in_gradient, const iMatrix<DT> & gradient, lint pad_h, lint pad_w)
{
    in_gradient.set_matrix_type(gradient.get_matrix_type());

    if (gradient.get_matrix_type() == MatrixType::CPU)
    {
        cpu::pad_2d_backward( *(in_gradient.get_cpu_instance()), *(gradient.get_cpu_instance()), pad_h, pad_w );
    }
    else if (gradient.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::pad_2d_backward( *(in_gradient.get_cuda_instance()), *(gradient.get_cuda_instance()), pad_h, pad_w );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (gradient.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void pad_2d_backward(iMatrix<float> & in_gradient, const iMatrix<float> & gradient, lint pad_h, lint pad_w);
template
void pad_2d_backward(iMatrix<int> & in_gradient, const iMatrix<int> & gradient, lint pad_h, lint pad_w);


template <typename DT>
void img2row_2d(iMatrix<DT> & output, const iMatrix<DT> & input, lint stride_h, lint stride_w, lint w_h, lint w_w)
{
    output.set_matrix_type(input.get_matrix_type());

    if (input.get_matrix_type() == MatrixType::CPU)
    {
        cpu::img2row_2d( *(output.get_cpu_instance()), *(input.get_cpu_instance()), stride_h, stride_w, w_h, w_w );
    }
    else if (input.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::img2row_2d( *(output.get_cuda_instance()), *(input.get_cuda_instance()), stride_h, stride_w, w_h, w_w );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (input.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void img2row_2d(iMatrix<float> & output, const iMatrix<float> & input, lint stride_h, lint stride_w, lint w_h, lint w_w);
template
void img2row_2d(iMatrix<int> & output, const iMatrix<int> & input, lint stride_h, lint stride_w, lint w_h, lint w_w);


template <typename DT>
void img2row_2d_backward(iMatrix<DT> & in_gradient, const Shape & in_shape, const iMatrix<DT> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w)
{
    in_gradient.set_matrix_type(gradient.get_matrix_type());

    if (gradient.get_matrix_type() == MatrixType::CPU)
    {
        cpu::img2row_2d_backward( *(in_gradient.get_cpu_instance()), in_shape, *(gradient.get_cpu_instance()), stride_h, stride_w, w_h, w_w );
    }
    else if (gradient.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::img2row_2d_backward( *(in_gradient.get_cuda_instance()), in_shape, *(gradient.get_cuda_instance()), stride_h, stride_w, w_h, w_w );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (gradient.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void img2row_2d_backward(iMatrix<float> & in_gradient, const Shape & in_shape, const iMatrix<float> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w);
template
void img2row_2d_backward(iMatrix<int> & in_gradient, const Shape & in_shape, const iMatrix<int> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w);


template <typename DT>
void img2col_2d(iMatrix<DT> & output, const iMatrix<DT> & input, lint stride_h, lint stride_w, lint w_h, lint w_w)
{
    output.set_matrix_type(input.get_matrix_type());

    if (input.get_matrix_type() == MatrixType::CPU)
    {
        cpu::img2col_2d( *(output.get_cpu_instance()), *(input.get_cpu_instance()), stride_h, stride_w, w_h, w_w );
    }
    else if (input.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::img2col_2d( *(output.get_cuda_instance()), *(input.get_cuda_instance()), stride_h, stride_w, w_h, w_w );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (input.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void img2col_2d(iMatrix<float> & output, const iMatrix<float> & input, lint stride_h, lint stride_w, lint w_h, lint w_w);
template
void img2col_2d(iMatrix<int> & output, const iMatrix<int> & input, lint stride_h, lint stride_w, lint w_h, lint w_w);


template <typename DT>
void img2col_2d_backward(iMatrix<DT> & in_gradient, const Shape & in_shape, const iMatrix<DT> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w)
{
    in_gradient.set_matrix_type(gradient.get_matrix_type());

    if (gradient.get_matrix_type() == MatrixType::CPU)
    {
        cpu::img2col_2d_backward( *(in_gradient.get_cpu_instance()), in_shape, *(gradient.get_cpu_instance()), stride_h, stride_w, w_h, w_w );
    }
    else if (gradient.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::img2col_2d_backward( *(in_gradient.get_cuda_instance()), in_shape, *(gradient.get_cuda_instance()), stride_h, stride_w, w_h, w_w );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (gradient.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void img2col_2d_backward(iMatrix<float> & in_gradient, const Shape & in_shape, const iMatrix<float> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w);
template
void img2col_2d_backward(iMatrix<int> & in_gradient, const Shape & in_shape, const iMatrix<int> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w);


template <typename DT>
void maxpool_2d(iMatrix<DT> &output, iMatrix<DT> &diff, const iMatrix<DT> &input,
                    lint stride_h, lint stride_w, lint k_h, lint k_w)
{
    output.set_matrix_type(input.get_matrix_type());
    diff.set_matrix_type(input.get_matrix_type());

    if (input.get_matrix_type() == MatrixType::CPU)
    {
        cpu::maxpool_2d( *(output.get_cpu_instance()), *(diff.get_cpu_instance()), *(input.get_cpu_instance()),
                        stride_h, stride_w, k_h, k_w);
    }
    else if (input.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::maxpool_2d( *(output.get_cuda_instance()), *(diff.get_cuda_instance()), *(input.get_cuda_instance()),
                stride_h, stride_w, k_h, k_w);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (input.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void maxpool_2d(iMatrix<float> &output, iMatrix<float> &diff, const iMatrix<float> &input,
                    lint stride_h, lint stride_w, lint k_h, lint k_w);
template
void maxpool_2d(iMatrix<int> &output, iMatrix<int> &diff, const iMatrix<int> &input,
                    lint stride_h, lint stride_w, lint k_h, lint k_w);


template <typename DT>
void avgpool_2d(iMatrix<DT> &output, iMatrix<DT> &diff, const iMatrix<DT> &input,
                    lint stride_h, lint stride_w, lint k_h, lint k_w)
{
    output.set_matrix_type(input.get_matrix_type());
    diff.set_matrix_type(input.get_matrix_type());

    if (input.get_matrix_type() == MatrixType::CPU)
    {
        cpu::avgpool_2d( *(output.get_cpu_instance()), *(diff.get_cpu_instance()), *(input.get_cpu_instance()),
                        stride_h, stride_w, k_h, k_w);
    }
    else if (input.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::avgpool_2d( *(output.get_cuda_instance()), *(diff.get_cuda_instance()), *(input.get_cuda_instance()),
                stride_h, stride_w, k_h, k_w);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (input.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void avgpool_2d(iMatrix<float> &output, iMatrix<float> &diff, const iMatrix<float> &input,
                    lint stride_h, lint stride_w, lint k_h, lint k_w);
template
void avgpool_2d(iMatrix<int> &output, iMatrix<int> &diff, const iMatrix<int> &input,
                    lint stride_h, lint stride_w, lint k_h, lint k_w);


template <typename DT>
void pool_2d_backward(iMatrix<DT> &in_gradient, iMatrix<DT> &gradient_cache,
                        const Shape &in_shape, const iMatrix<DT> &diff, const iMatrix<DT> &gradient,
                        lint stride_h, lint stride_w, lint k_h, lint k_w)
{
    in_gradient.set_matrix_type(gradient.get_matrix_type());
    gradient_cache.set_matrix_type(gradient.get_matrix_type());

    if (gradient.get_matrix_type() == MatrixType::CPU)
    {
        cpu::pool_2d_backward(
            *(in_gradient.get_cpu_instance()),
            *(gradient_cache.get_cpu_instance()),
            in_shape,
            *(diff.get_cpu_instance()),
            *(gradient.get_cpu_instance()),
            stride_h, stride_w, k_h, k_w);
    }
    else if (gradient.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::pool_2d_backward(
            *(in_gradient.get_cuda_instance()),
            *(gradient_cache.get_cuda_instance()),
            in_shape,
            *(diff.get_cuda_instance()),
            *(gradient.get_cuda_instance()),
            stride_h, stride_w, k_h, k_w);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (gradient.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void pool_2d_backward(iMatrix<float> &in_gradient, iMatrix<float> &gradient_cache,
                        const Shape &in_shape, const iMatrix<float> &diff, const iMatrix<float> &gradient,
                        lint stride_h, lint stride_w, lint k_h, lint k_w);
template
void pool_2d_backward(iMatrix<int> &in_gradient, iMatrix<int> &gradient_cache,
                        const Shape &in_shape, const iMatrix<int> &diff, const iMatrix<int> &gradient,
                        lint stride_h, lint stride_w, lint k_h, lint k_w);


} // namespace la
} // namespace julie
