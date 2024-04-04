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

#include "prelu.hpp"
#include "iMatrix_func.hpp"

namespace julie
{
namespace nn
{
namespace func
{

PReLU::PReLU()
    :
    op::Function {std::string {"PReLU"}, false},
    m_prelu {std::make_unique<la::PReLU<float>>()}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

PReLU::PReLU(const PReLU & other)
    :
    op::Function {other},
    m_prelu {std::make_unique<la::PReLU<float>>(*(other.m_prelu))}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

PReLU::PReLU(PReLU && other)
    :
    op::Function {other},
    m_prelu {std::move(other.m_prelu)}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

PReLU & PReLU::operator = (const PReLU & other)
{
    op::Function::operator = (other);
    this->m_prelu = std::make_unique<la::PReLU<float>>(*(other.m_prelu));

    return *this;
}

PReLU & PReLU::operator = (PReLU && other)
{
    op::Function::operator = (other);
    this->m_prelu = std::move(other.m_prelu);

    return *this;
}

void PReLU::set_inputs(const std::shared_ptr<op::Function> & self, 
                                    const std::vector<std::shared_ptr<op::Variable>> & inputs)
{
    if (inputs.size() != 2)
    {
        throw std::invalid_argument(std::string("Number of inputs for PReLU operation is not 2"));
    }

    op::Function::set_inputs(self, inputs);
}

void PReLU::forward()
{
    var::Tensor<float> *input_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[0].get());
    std::shared_ptr<julie::la::iMatrix<float>> t_mat_ptr = input_ptr->val();

    var::Scalar<float> *alpha_ptr = dynamic_cast<var::Scalar<float>*>(this->m_inputs[1].get());
    std::shared_ptr<float> alpha_val_ptr = alpha_ptr->val();

    var::Tensor<float> *output_ptr = dynamic_cast<var::Tensor<float>*>(this->m_output.get());

    // Forward
    this->m_prelu->operator()(*(output_ptr->val()), this->m_diff, this->m_alpha_diff, *t_mat_ptr, *alpha_val_ptr);
}

void PReLU::backward()
{
    var::Tensor<float> *output_ptr = dynamic_cast<var::Tensor<float>*>(this->m_output.get());
    std::shared_ptr<julie::la::iMatrix<float>> out_grad = output_ptr->grad();

    var::Tensor<float> *input_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[0].get());
    var::Scalar<float> *alpha_ptr = dynamic_cast<var::Scalar<float>*>(this->m_inputs[1].get());

    std::shared_ptr<julie::la::iMatrix<float>> t_mat_ptr = input_ptr->val();

    // Do chain rule for the input
    julie::la::multiply(this->m_input_grad_cache, this->m_diff, *out_grad);
    input_ptr->add_grad(this->m_input_grad_cache);

    // Do chain rule for the alpha
    alpha_ptr->add_grad(julie::la::dot_product(this->m_alpha_diff_multiply_cache, this->m_alpha_diff, *out_grad));
    // alpha_ptr->add_grad(julie::la::dot_product(this->m_alpha_diff_multiply_cache, this->m_alpha_diff, *out_grad) / this->m_alpha_diff.shape().size());
}

void PReLU::clear_cache()
{
    this->m_diff = julie::la::iMatrix<float> {};
    this->m_alpha_diff = julie::la::iMatrix<float> {};

    this->m_input_grad_cache = julie::la::iMatrix<float> {};
    this->m_alpha_diff_multiply_cache = julie::la::iMatrix<float> {};
}

} // namespace func
} // namespace nn
} // namespace julie
