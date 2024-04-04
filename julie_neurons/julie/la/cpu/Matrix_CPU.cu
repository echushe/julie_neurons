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

#include "Matrix_CPU.hpp"
#include "Matrix_CPU_func.hpp"
#include "utilities.hpp"
#include "cuda_utility.cuh"

#include <stdexcept>

namespace julie
{
namespace la
{
namespace cpu
{

// From a GPU matrix
template <typename DT>
Matrix_CPU<DT>::Matrix_CPU (const julie::la::cuda::Matrix_CUDA<DT> & gpu_mat)
    : m_shape {gpu_mat.m_shape}, m_data {nullptr}
{
    if (m_shape.m_size < 1)
    {
        return;
    }

    this->m_data = new DT[this->m_shape.m_size];
    CUDA_CHECK( cudaMemcpy (this->m_data, gpu_mat.m_data, this->m_shape.m_size * sizeof(DT), cudaMemcpyDeviceToHost) );
}
template
Matrix_CPU<int>::Matrix_CPU (const julie::la::cuda::Matrix_CUDA<int> & gpu_mat);
template
Matrix_CPU<float>::Matrix_CPU (const julie::la::cuda::Matrix_CUDA<float> & gpu_mat);


// Get a GPU mode Matrix
template <typename DT>
julie::la::cuda::Matrix_CUDA<DT> Matrix_CPU<DT>::get_CUDA () const
{
    julie::la::cuda::Matrix_CUDA<DT> gpu_mat{ this->m_shape };

    if (this->m_shape.m_size < 1)
    {
        return gpu_mat;
    }

    CUDA_CHECK( cudaMemcpy (gpu_mat.m_data, this->m_data, this->m_shape.m_size * sizeof(DT), cudaMemcpyHostToDevice) );

    return gpu_mat;
}
template
julie::la::cuda::Matrix_CUDA<int> Matrix_CPU<int>::get_CUDA () const;
template
julie::la::cuda::Matrix_CUDA<float> Matrix_CPU<float>::get_CUDA () const;


} // namespace cpu
} // namespace la
} // namespace julie