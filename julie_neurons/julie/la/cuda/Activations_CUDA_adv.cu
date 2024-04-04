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

#include "Activations_CUDA_adv.hpp"
#include "Matrix_CUDA_func.hpp"
#include "act_adv.cuh"

namespace julie
{
namespace la
{
namespace cuda
{

template <typename DT>
void PReLU<DT>::operator () (
    Matrix_CUDA<DT> & output, Matrix_CUDA<DT> & diff, Matrix_CUDA<DT> & alpha_diff, const Matrix_CUDA<DT> & in, const DT & alpha)
{
    renew_if_shape_not_match(output, in.m_shape);
    renew_if_shape_not_match(diff, in.m_shape);
    renew_if_shape_not_match(alpha_diff, in.m_shape);

    lint size = in.m_shape.size();
    
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        __act_prelu_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, diff.m_data, alpha_diff.m_data, in.m_data, alpha, size);
    }
}
template
void PReLU<float>::operator () (
    Matrix_CUDA<float> & output, Matrix_CUDA<float> & diff, Matrix_CUDA<float> & alpha_diff, const Matrix_CUDA<float> & in, const float & alpha);


template <typename DT>
void PReLU<DT>::operator () (
    Matrix_CUDA<DT> & output, const Matrix_CUDA<DT> & in,  const DT & alpha)
{
    renew_if_shape_not_match(output, in.m_shape);

    lint size = in.m_shape.size();
    
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

        __act_prelu_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, in.m_data, alpha, size);
    }
}
template
void PReLU<float>::operator () (
    Matrix_CUDA<float> & output, const Matrix_CUDA<float> & in, const float & alpha);

} // namespace cpu
} // namespace la
} // namespace julie