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

#include "Activations_CUDA.hpp"
#include "Matrix_CUDA_func.hpp"
#include "act.cuh"
#include "nsqrt.hpp"

namespace julie
{
namespace la
{
namespace cuda
{

template <typename DT>
void Linear<DT>::operator () (Matrix_CUDA<DT> & output, Matrix_CUDA<DT> & diff, const Matrix_CUDA<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);
    renew_if_shape_not_match(diff, in.m_shape);

    lint size = in.m_shape.size();
    
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        __act_linear_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, diff.m_data, in.m_data, size);
    }
}
template
void Linear<float>::operator () (Matrix_CUDA<float> & output, Matrix_CUDA<float> & diff, const Matrix_CUDA<float> & in);


template <typename DT>
void Linear<DT>::operator () (Matrix_CUDA<DT> & output, const Matrix_CUDA<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);

    lint size = in.m_shape.size();

    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        __act_linear_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, in.m_data, size);
    }
}
template
void Linear<float>::operator () (Matrix_CUDA<float> & output, const Matrix_CUDA<float> & in);


template <typename DT>
void Sigmoid<DT>::operator () (Matrix_CUDA<DT> & output, Matrix_CUDA<DT> & diff, const Matrix_CUDA<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);
    renew_if_shape_not_match(diff, in.m_shape);

    lint size = in.m_shape.size();
    
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        __act_sigmoid_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, diff.m_data, in.m_data, size);
    }
}
template
void Sigmoid<float>::operator () (Matrix_CUDA<float> & output, Matrix_CUDA<float> & diff, const Matrix_CUDA<float> & in);


template <typename DT>
void Sigmoid<DT>::operator () (Matrix_CUDA<DT> & output, const Matrix_CUDA<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);

    lint size = in.m_shape.size();

    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        __act_sigmoid_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, in.m_data, size);
    }
}
template
void Sigmoid<float>::operator () (Matrix_CUDA<float> & output, const Matrix_CUDA<float> & in);


template <typename DT>
void TanH<DT>::operator () (Matrix_CUDA<DT> & output, Matrix_CUDA<DT> & diff, const Matrix_CUDA<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);
    renew_if_shape_not_match(diff, in.m_shape);

    lint size = in.m_shape.size();
    
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        __act_tanh_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, diff.m_data, in.m_data, size);
    }
}
template
void TanH<float>::operator () (Matrix_CUDA<float> & output, Matrix_CUDA<float> & diff, const Matrix_CUDA<float> & in);


template <typename DT>
void TanH<DT>::operator () (Matrix_CUDA<DT> & output, const Matrix_CUDA<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);

    lint size = in.m_shape.size();

    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        __act_tanh_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, in.m_data, size);
    }
}
template
void TanH<float>::operator () (Matrix_CUDA<float> & output, const Matrix_CUDA<float> & in);


template <typename DT>
void ReLU<DT>::operator () (Matrix_CUDA<DT> & output, Matrix_CUDA<DT> & diff, const Matrix_CUDA<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);
    renew_if_shape_not_match(diff, in.m_shape);

    lint size = in.m_shape.size();
    
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        __act_relu_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, diff.m_data, in.m_data, size);
    }
}
template
void ReLU<float>::operator () (Matrix_CUDA<float> & output, Matrix_CUDA<float> & diff, const Matrix_CUDA<float> & in);


template <typename DT>
void ReLU<DT>::operator () (Matrix_CUDA<DT> & output, const Matrix_CUDA<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);

    lint size = in.m_shape.size();

    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        __act_relu_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, in.m_data, size);
    }
}
template
void ReLU<float>::operator () (Matrix_CUDA<float> & output, const Matrix_CUDA<float> & in);


template <typename DT>
void Abs<DT>::operator () (Matrix_CUDA<DT> & output, Matrix_CUDA<DT> & diff, const Matrix_CUDA<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);
    renew_if_shape_not_match(diff, in.m_shape);

    lint size = in.m_shape.size();
    
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        __act_abs_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, diff.m_data, in.m_data, size);
    }
}
template
void Abs<float>::operator () (Matrix_CUDA<float> & output, Matrix_CUDA<float> & diff, const Matrix_CUDA<float> & in);


template <typename DT>
void Abs<DT>::operator () (Matrix_CUDA<DT> & output, const Matrix_CUDA<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);

    lint size = in.m_shape.size();

    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        __act_abs_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, in.m_data, size);
    }
}
template
void Abs<float>::operator () (Matrix_CUDA<float> & output, const Matrix_CUDA<float> & in);


template <typename DT>
void ArcTan<DT>::operator () (Matrix_CUDA<DT> & output, Matrix_CUDA<DT> & diff, const Matrix_CUDA<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);
    renew_if_shape_not_match(diff, in.m_shape);

    lint size = in.m_shape.size();
    
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        __act_arctan_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, diff.m_data, in.m_data, size);
    }
}
template
void ArcTan<float>::operator () (Matrix_CUDA<float> & output, Matrix_CUDA<float> & diff, const Matrix_CUDA<float> & in);


template <typename DT>
void ArcTan<DT>::operator () (Matrix_CUDA<DT> & output, const Matrix_CUDA<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);

    lint size = in.m_shape.size();

    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        __act_arctan_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, in.m_data, size);
    }
}
template
void ArcTan<float>::operator () (Matrix_CUDA<float> & output, const Matrix_CUDA<float> & in);


template <typename DT>
void SoftMax<DT>::operator () (Matrix_CUDA<DT> & output, Matrix_CUDA<DT> & diff, const Matrix_CUDA<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);
    renew_if_shape_not_match(diff, in.m_shape);

    lint size = in.m_shape.size();

    if (size > 0)
    {
        julie::la::Shape l_sh = in.m_shape.sub_shape(0, this->m_axis - 1);
        julie::la::Shape r_sh = in.m_shape.sub_shape(this->m_axis + 1, in.m_shape.dim() - 1);

        lint l_size = std::max<lint>(l_sh.size(), 1);
        lint axis_size = in.m_shape[this->m_axis];
        lint r_size = std::max<lint>(r_sh.size(), 1);

        lint n_blocks = std::min(l_size * r_size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        int64_t grid_y = std::max(nsqrt(n_blocks * l_size / r_size), 1L);
        int64_t grid_x = std::max(n_blocks / grid_y, 1L);
    
        dim3 dimGrid(grid_x, grid_y, 1);
        dim3 dimBlock(BLOCK_WIDTH_1D / 16, 16, 1);

        __act_softmax_2d<<<dimGrid, dimBlock>>> (output.m_data, diff.m_data, in.m_data, l_size, axis_size, r_size);
    }
}
template
void SoftMax<float>::operator () (Matrix_CUDA<float> & output, Matrix_CUDA<float> & diff, const Matrix_CUDA<float> & in);


template <typename DT>
void SoftMax<DT>::operator () (Matrix_CUDA<DT> & output, const Matrix_CUDA<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);

    lint size = in.m_shape.size();

    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        julie::la::Shape l_sh = in.m_shape.sub_shape(0, this->m_axis - 1);
        julie::la::Shape r_sh = in.m_shape.sub_shape(this->m_axis + 1, in.m_shape.dim() - 1);

        lint l_size = std::max<lint>(l_sh.size(), 1);
        lint axis_size = in.m_shape[this->m_axis];
        lint r_size = std::max<lint>(r_sh.size(), 1);

        int64_t grid_y = std::max(nsqrt(n_blocks * l_size / r_size), 1L);
        int64_t grid_x = std::max(n_blocks / grid_y, 1L);
    
        dim3 dimGrid(grid_x, grid_y, 1);
        dim3 dimBlock(BLOCK_WIDTH_1D / 16, 16, 1);

        __act_softmax_2d<<<dimGrid, dimBlock>>> (output.m_data, in.m_data, l_size, axis_size, r_size);
    }
}
template
void SoftMax<float>::operator () (Matrix_CUDA<float> & output, const Matrix_CUDA<float> & in);


}  // namespace cuda
}  // namespace la
}  // namespace julie