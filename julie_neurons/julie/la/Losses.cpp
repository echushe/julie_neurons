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

#include "Losses.hpp"

namespace julie
{
namespace la
{

template <typename DT>
LossFunction<DT>::LossFunction(lint axis)
    :
    m_axis {axis},
    m_losses_cpu {nullptr}
#ifdef WITH_CUDA
    ,m_losses_cuda {nullptr}
#endif 
{}
template
LossFunction<float>::LossFunction(lint axis);


template <typename DT>
void LossFunction<DT>::operator () (iMatrix<DT> &loss, iMatrix<DT> & diff, const iMatrix<DT> & target, const iMatrix<DT> & input)
{
    diff.set_matrix_type(input.get_matrix_type());
    loss.set_matrix_type(input.get_matrix_type());
    
    if (input.get_matrix_type() == MatrixType::CPU)
    {
        if (this->m_losses_cpu)
        {
            this->m_losses_cpu->operator()(
                *(loss.get_cpu_instance()), *(diff.get_cpu_instance()), *(target.get_cpu_instance()), *(input.get_cpu_instance()));
        }
        else
        {
            throw std::invalid_argument { std::string{"CPU Matrix type not supported in "} + std::string{__FUNCTION__} };
        } 
    }
    else if (input.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        if (this->m_losses_cuda)
        {
            this->m_losses_cuda->operator()(
                *(loss.get_cuda_instance()), *(diff.get_cuda_instance()), *(target.get_cuda_instance()), *(input.get_cuda_instance()));
        }
        else
        {
            throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
        }
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
void LossFunction<float>::operator () (
    iMatrix<float> &loss, iMatrix<float> & diff, const iMatrix<float> & target, const iMatrix<float> & input);


template <typename DT>
HalfSquareError<DT>::HalfSquareError(lint axis)
    : LossFunction<DT> {axis}
{
    this->m_losses_cpu = std::make_shared<cpu::HalfSquareError<DT>> (axis);
#ifdef WITH_CUDA
    this->m_losses_cuda = std::make_shared<cuda::HalfSquareError<DT>> (axis);
#endif
}
template
HalfSquareError<float>::HalfSquareError(lint axis);


template <typename DT>
Sigmoid_CrossEntropy<DT>::Sigmoid_CrossEntropy(lint axis)
    : LossFunction<DT> {axis}
{
    this->m_losses_cpu = std::make_shared<cpu::Sigmoid_CrossEntropy<DT>> (axis);
#ifdef WITH_CUDA
    this->m_losses_cuda = std::make_shared<cuda::Sigmoid_CrossEntropy<DT>> (axis);
#endif
}
template
Sigmoid_CrossEntropy<float>::Sigmoid_CrossEntropy(lint axis);


template <typename DT>
SoftMax_CrossEntropy<DT>::SoftMax_CrossEntropy(lint axis)
    : LossFunction<DT> {axis}
{
    this->m_losses_cpu = std::make_shared<cpu::SoftMax_CrossEntropy<DT>> (axis);
#ifdef WITH_CUDA
    this->m_losses_cuda = std::make_shared<cuda::SoftMax_CrossEntropy<DT>> (axis);
#endif
}
template
SoftMax_CrossEntropy<float>::SoftMax_CrossEntropy(lint axis);

}  // namespace la
}  // namespace julie

