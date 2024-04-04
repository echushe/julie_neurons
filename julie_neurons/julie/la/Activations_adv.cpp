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

#include "Activations_adv.hpp"

namespace julie
{
namespace la
{

template <typename DT>
Activation_adv<DT>::Activation_adv()
    :
    m_act_cpu {nullptr}
#ifdef WITH_CUDA
    ,m_act_cuda {nullptr}
#endif
{}


template <typename DT>
void Activation_adv<DT>::operator () (
    iMatrix<DT> & output, iMatrix<DT> & diff, iMatrix<DT> &alpha_diff, const iMatrix<DT> & in, const DT &alpha)
{   
    output.set_matrix_type(in.get_matrix_type());
    diff.set_matrix_type(in.get_matrix_type());
    alpha_diff.set_matrix_type(in.get_matrix_type());
    
    if (in.get_matrix_type() == MatrixType::CPU)
    {
        if (this->m_act_cpu)
        {
            this->m_act_cpu->operator()(
                *(output.get_cpu_instance()), *(diff.get_cpu_instance()), *(alpha_diff.get_cpu_instance()),
                *(in.get_cpu_instance()), alpha);
        }
        else
        {
            throw std::invalid_argument { std::string{"CPU Matrix type not supported in "} + std::string{__FUNCTION__} };
        } 
    }
    else if (in.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        if (this->m_act_cuda)
        {
            this->m_act_cuda->operator()(
                *(output.get_cuda_instance()), *(diff.get_cuda_instance()), *(alpha_diff.get_cuda_instance()),
                *(in.get_cuda_instance()), alpha);
        }
        else
        {
            throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
        }
#else
    throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (in.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void Activation_adv<float>::operator () (
    iMatrix<float> & output, iMatrix<float> & diff, iMatrix<float> &alpha_diff, const iMatrix<float> & in, const float &alpha);


template <typename DT>
void Activation_adv<DT>::operator () (iMatrix<DT> & output, const iMatrix<DT> & in, const DT &alpha)
{
    output.set_matrix_type(in.get_matrix_type());
    
    if (in.get_matrix_type() == MatrixType::CPU)
    {
        if (this->m_act_cpu)
        {
            this->m_act_cpu->operator()(*(output.get_cpu_instance()), *(in.get_cpu_instance()), alpha);
        }
        else
        {
            throw std::invalid_argument { std::string{"CPU Matrix type not supported in "} + std::string{__FUNCTION__} };
        } 
    }
    else if (in.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        if (this->m_act_cuda)
        {
            this->m_act_cuda->operator()(*(output.get_cuda_instance()), *(in.get_cuda_instance()), alpha);
        }
        else
        {
            throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
        }
#else
    throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (in.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void Activation_adv<float>::operator () (iMatrix<float> & output, const iMatrix<float> & in, const float &alpha);


template <typename DT>
PReLU<DT>::PReLU()
    : Activation_adv<DT> {}
{
    this->m_act_cpu = std::make_shared<cpu::PReLU<DT>> ();
#ifdef WITH_CUDA
    this->m_act_cuda = std::make_shared<cuda::PReLU<DT>> ();
#endif
}
template
PReLU<float>::PReLU();


} // namespace la
} // namespace julie

