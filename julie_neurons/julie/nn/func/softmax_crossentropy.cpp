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

#include "softmax_crossentropy.hpp"
#include "iMatrix_func.hpp"

namespace julie
{
namespace nn
{
namespace func
{

SoftMax_CrossEntropy::SoftMax_CrossEntropy(lint axis)
    :
    op::Function {std::string {"SoftMax_CrossEntropy"}, true},
    m_sc {std::make_unique<la::SoftMax_CrossEntropy<float>>(axis)}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

SoftMax_CrossEntropy::SoftMax_CrossEntropy(const SoftMax_CrossEntropy & other)
    :
    op::Function {other},
    m_sc {std::make_unique<la::SoftMax_CrossEntropy<float>>(*(other.m_sc))}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

SoftMax_CrossEntropy::SoftMax_CrossEntropy(SoftMax_CrossEntropy && other)
    :
    op::Function {other},
    m_sc {std::move(other.m_sc)}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

SoftMax_CrossEntropy & SoftMax_CrossEntropy::operator = (const SoftMax_CrossEntropy & other)
{
    op::Function::operator = (other);
    this->m_sc = std::make_unique<la::SoftMax_CrossEntropy<float>>(*(other.m_sc));

    return *this;
}

SoftMax_CrossEntropy & SoftMax_CrossEntropy::operator = (SoftMax_CrossEntropy && other)
{
    op::Function::operator = (other);
    this->m_sc = std::move(other.m_sc);

    return *this;
}

void SoftMax_CrossEntropy::set_inputs(const std::shared_ptr<op::Function> & self, 
                                    const std::vector<std::shared_ptr<op::Variable>> & inputs)
{
    if (inputs.size() != 2)
    {
        throw std::invalid_argument(std::string("Number of inputs for SoftMax_CrossEntropy operation is not 2"));
    }

    op::Function::set_inputs(self, inputs);
}

void SoftMax_CrossEntropy::forward()
{
    var::Tensor<float> *input_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[0].get());
    var::Tensor<float> *target_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[1].get());

    std::shared_ptr<julie::la::iMatrix<float>> input_mat_ptr = input_ptr->val();
    std::shared_ptr<julie::la::iMatrix<float>> target_mat_ptr = target_ptr->val();

    var::Tensor<float> *output_ptr = dynamic_cast<var::Tensor<float>*>(this->m_output.get());
    
    // Forward
    this->m_sc->operator()(*(output_ptr->val()), this->m_diff, *target_mat_ptr, *input_mat_ptr);
}

void SoftMax_CrossEntropy::backward()
{
    var::Tensor<float> *output_ptr = dynamic_cast<var::Tensor<float>*>(this->m_output.get());
    std::shared_ptr<julie::la::iMatrix<float>> out_grad = output_ptr->grad();

    var::Tensor<float> *input_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[0].get());
    //var::Tensor<float> *target_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[1].get());

    std::shared_ptr<julie::la::iMatrix<float>> input_mat_ptr = input_ptr->val();

    // Do chain rule for the input
    this->m_output_grad_cache = *out_grad;

    std::vector<lint> sh;
    for (lint i = 0; i < input_mat_ptr->shape().dim(); ++i)
    {
        if (i == this->m_sc->axis())
        {
            sh.push_back(1);
        }
        else
        {
            sh.push_back(input_mat_ptr->shape()[i]);
        }
    }
    this->m_output_grad_cache.reshape(sh);

    julie::la::repeat(this->m_output_grad_repeat_cache, m_output_grad_cache, this->m_sc->axis(), input_mat_ptr->shape()[this->m_sc->axis()]);
    julie::la::multiply(this->m_input_grad_cache, this->m_diff, this->m_output_grad_repeat_cache);
    input_ptr->add_grad(this->m_input_grad_cache);

    // We DO NOT do chain rule for the target variable
}

void SoftMax_CrossEntropy::clear_cache()
{
    this->m_diff = julie::la::iMatrix<float> {};
    this->m_input_grad_cache = julie::la::iMatrix<float> {};
    this->m_output_grad_cache = julie::la::iMatrix<float> {};
    this->m_output_grad_repeat_cache = julie::la::iMatrix<float> {};
}

} // namespace func
} // namespace nn
} // namespace julie
