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

#include "Losses_CUDA.hpp"
#include "Matrix_CUDA_func.hpp"
#include "loss.cuh"
#include "nsqrt.hpp"

namespace julie
{
namespace la
{
namespace cuda
{

template <typename DT>
void HalfSquareError<DT>::operator()(Matrix_CUDA<DT> &loss, Matrix_CUDA<DT> & diff, const Matrix_CUDA<DT> & target, const Matrix_CUDA<DT> & input)
{
    if (diff.m_shape != input.m_shape)
    {
        diff = Matrix_CUDA<DT> { input.m_shape };
    }

    lint size = input.m_shape.size();
    lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    Shape l_sh = input.m_shape.sub_shape(0, this->m_axis - 1);
    Shape r_sh = input.m_shape.sub_shape(this->m_axis + 1, input.m_shape.dim() - 1);

    renew_if_shape_not_match(loss, l_sh + r_sh);

    lint l_size = std::max<lint>(l_sh.size(), 1);
    lint axis_size = input.m_shape[this->m_axis];
    lint r_size = std::max<lint>(r_sh.size(), 1);

    int64_t grid_y = std::max(nsqrt(n_blocks * l_size / r_size), 1L);
    int64_t grid_x = std::max(n_blocks / grid_y, 1L);

    dim3 dimGrid(grid_x, grid_y, 1);
    dim3 dimBlock(BLOCK_WIDTH_1D / 16, 16, 1);

    __loss_half_square_error_2d<<<dimGrid, dimBlock>>> (
        loss.m_data, diff.m_data, target.m_data, input.m_data, l_size, axis_size, r_size);
}
template
void HalfSquareError<float>::operator()(
    Matrix_CUDA<float> &loss, Matrix_CUDA<float> & diff, const Matrix_CUDA<float> & target, const Matrix_CUDA<float> & input);


template <typename DT>
void Sigmoid_CrossEntropy<DT>::operator()(Matrix_CUDA<DT> &loss, Matrix_CUDA<DT> & diff, const Matrix_CUDA<DT> & target, const Matrix_CUDA<DT> & input)
{
    if (this->m_sigmoid_cache.m_shape != input.m_shape)
    {
        this->m_sigmoid_cache = Matrix_CUDA<DT> { input.m_shape };
    }

    if (diff.m_shape != input.m_shape)
    {
        diff = Matrix_CUDA<DT> { input.m_shape };
    }

    lint size = input.m_shape.size();
    lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    Shape l_sh = input.m_shape.sub_shape(0, this->m_axis - 1);
    Shape r_sh = input.m_shape.sub_shape(this->m_axis + 1, input.m_shape.dim() - 1);

    renew_if_shape_not_match(loss, l_sh + r_sh);

    lint l_size = std::max<lint>(l_sh.size(), 1);
    lint axis_size = input.m_shape[this->m_axis];
    lint r_size = std::max<lint>(r_sh.size(), 1);

    int64_t grid_y = std::max(nsqrt(n_blocks * l_size / r_size), 1L);
    int64_t grid_x = std::max(n_blocks / grid_y, 1L);

    dim3 dimGrid(grid_x, grid_y, 1);
    dim3 dimBlock(BLOCK_WIDTH_1D / 16, 16, 1);

    __loss_sigmoid_crossentropy_2d<<<dimGrid, dimBlock>>> (
        loss.m_data, this->m_sigmoid_cache.m_data, diff.m_data, target.m_data, input.m_data, l_size, axis_size, r_size);
}
template
void Sigmoid_CrossEntropy<float>::operator()(
    Matrix_CUDA<float> &loss, Matrix_CUDA<float> & diff, const Matrix_CUDA<float> & target, const Matrix_CUDA<float> & input);



template <typename DT>
void SoftMax_CrossEntropy<DT>::operator()(Matrix_CUDA<DT> &loss, Matrix_CUDA<DT> & diff, const Matrix_CUDA<DT> & target, const Matrix_CUDA<DT> & input)
{
    if (this->m_softmax_cache.m_shape != input.m_shape)
    {
        this->m_softmax_cache = Matrix_CUDA<DT> { input.m_shape };
    }

    if (diff.m_shape != input.m_shape)
    {
        diff = Matrix_CUDA<DT> { input.m_shape };
    }

    lint size = input.m_shape.size();
    lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    Shape l_sh = input.m_shape.sub_shape(0, this->m_axis - 1);
    Shape r_sh = input.m_shape.sub_shape(this->m_axis + 1, input.m_shape.dim() - 1);

    renew_if_shape_not_match(loss, l_sh + r_sh);

    lint l_size = std::max<lint>(l_sh.size(), 1);
    lint axis_size = input.m_shape[this->m_axis];
    lint r_size = std::max<lint>(r_sh.size(), 1);

    int64_t grid_y = std::max(nsqrt(n_blocks * l_size / r_size), 1L);
    int64_t grid_x = std::max(n_blocks / grid_y, 1L);

    dim3 dimGrid(grid_x, grid_y, 1);
    dim3 dimBlock(BLOCK_WIDTH_1D / 16, 16, 1);

    __loss_softmax_crossentropy_2d<<<dimGrid, dimBlock>>> (
        loss.m_data, this->m_softmax_cache.m_data, diff.m_data, target.m_data, input.m_data, l_size, axis_size, r_size);
}
template
void SoftMax_CrossEntropy<float>::operator()(
    Matrix_CUDA<float> &loss, Matrix_CUDA<float> & diff, const Matrix_CUDA<float> & target, const Matrix_CUDA<float> & input);


}  // namespace cuda
}  // namespace la
}  // namespace julie

