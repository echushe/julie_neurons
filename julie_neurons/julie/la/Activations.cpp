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

#include "Activations.hpp"

namespace julie
{
namespace la
{

template <typename DT>
Activation<DT>::Activation()
    :
    m_act_cpu {nullptr}
#ifdef WITH_CUDA
    ,m_act_cuda {nullptr}
#endif
{}


template <typename DT>
void Activation<DT>::operator () (iMatrix<DT> & output, iMatrix<DT> & diff, const iMatrix<DT> & in)
{   
    output.set_matrix_type(in.get_matrix_type());
    diff.set_matrix_type(in.get_matrix_type());
    
    if (in.get_matrix_type() == MatrixType::CPU)
    {
        if (this->m_act_cpu)
        {
            this->m_act_cpu->operator()(*(output.get_cpu_instance()), *(diff.get_cpu_instance()), *(in.get_cpu_instance()));
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
            this->m_act_cuda->operator()(*(output.get_cuda_instance()), *(diff.get_cuda_instance()), *(in.get_cuda_instance()));
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
void Activation<float>::operator () (iMatrix<float> & output, iMatrix<float> & diff, const iMatrix<float> & in);


template <typename DT>
void Activation<DT>::operator () (iMatrix<DT> & output, const iMatrix<DT> & in)
{
    output.set_matrix_type(in.get_matrix_type());
    
    if (in.get_matrix_type() == MatrixType::CPU)
    {
        if (this->m_act_cpu)
        {
            this->m_act_cpu->operator()(*(output.get_cpu_instance()), *(in.get_cpu_instance()));
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
            this->m_act_cuda->operator()(*(output.get_cuda_instance()), *(in.get_cuda_instance()));
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
void Activation<float>::operator () (iMatrix<float> & output, const iMatrix<float> & in);


template <typename DT>
Linear<DT>::Linear()
    : Activation<DT> {}
{
    this->m_act_cpu = std::make_shared<cpu::Linear<DT>> ();
#ifdef WITH_CUDA
    this->m_act_cuda = std::make_shared<cuda::Linear<DT>> ();
#endif
}
template
Linear<float>::Linear();


template <typename DT>
Sigmoid<DT>::Sigmoid()
    : Activation<DT> {}
{
    this->m_act_cpu = std::make_shared<cpu::Sigmoid<DT>> ();
#ifdef WITH_CUDA
    this->m_act_cuda = std::make_shared<cuda::Sigmoid<DT>> ();
#endif
}
template
Sigmoid<float>::Sigmoid();


template <typename DT>
TanH<DT>::TanH()
    : Activation<DT> {}
{
    this->m_act_cpu = std::make_shared<cpu::TanH<DT>> ();
#ifdef WITH_CUDA
    this->m_act_cuda = std::make_shared<cuda::TanH<DT>> ();
#endif
}
template
TanH<float>::TanH();


template <typename DT>
ReLU<DT>::ReLU()
    : Activation<DT> {}
{
    this->m_act_cpu = std::make_shared<cpu::ReLU<DT>> ();
#ifdef WITH_CUDA
    this->m_act_cuda = std::make_shared<cuda::ReLU<DT>> ();
#endif
}
template
ReLU<float>::ReLU();


template <typename DT>
Abs<DT>::Abs()
    : Activation<DT> {}
{
    this->m_act_cpu = std::make_shared<cpu::Abs<DT>> ();
#ifdef WITH_CUDA
    this->m_act_cuda = std::make_shared<cuda::Abs<DT>> ();
#endif
}
template
Abs<float>::Abs();


template <typename DT>
LeakyReLU<DT>::LeakyReLU()
    : Activation<DT> {}
{
    this->m_act_cpu = std::make_shared<cpu::LeakyReLU<DT>> ();
}
template
LeakyReLU<float>::LeakyReLU();


template <typename DT>
ArcTan<DT>::ArcTan()
    : Activation<DT> {}
{
    this->m_act_cpu = std::make_shared<cpu::ArcTan<DT>> ();
#ifdef WITH_CUDA
    this->m_act_cuda = std::make_shared<cuda::ArcTan<DT>> ();
#endif
}
template
ArcTan<float>::ArcTan();


template <typename DT>
Sin<DT>::Sin()
    : Activation<DT> {}
{
    this->m_act_cpu = std::make_shared<cpu::ArcTan<DT>> ();
}
template
Sin<float>::Sin();


template <typename DT>
SoftSign<DT>::SoftSign()
    : Activation<DT> {}
{
    this->m_act_cpu = std::make_shared<cpu::SoftSign<DT>> ();
}
template
SoftSign<float>::SoftSign();


template <typename DT>
SoftMax<DT>::SoftMax(lint axis)
    : Activation<DT> {}
{
    this->m_act_cpu = std::make_shared<cpu::SoftMax<DT>> (axis);
#ifdef WITH_CUDA
    this->m_act_cuda = std::make_shared<cuda::SoftMax<DT>> (axis);
#endif
}
template
SoftMax<float>::SoftMax(lint axis);

} // namespace la
} // namespace julie

