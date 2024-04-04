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

#include "tanh.hpp"
#include "iMatrix_func.hpp"

namespace julie
{
namespace nn
{
namespace func
{

TanH::TanH()
    :
    op::Function {std::string {"TanH"}, false},
    m_tanh {std::make_unique<la::TanH<float>>()}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

TanH::TanH(const TanH & other)
    :
    op::Function {other},
    m_tanh {std::make_unique<la::TanH<float>>(*(other.m_tanh))}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

TanH::TanH(TanH && other)
    :
    op::Function {other},
    m_tanh {std::move(other.m_tanh)}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

TanH & TanH::operator = (const TanH & other)
{
    op::Function::operator = (other);
    this->m_tanh = std::make_unique<la::TanH<float>>(*(other.m_tanh));

    return *this;
}

TanH & TanH::operator = (TanH && other)
{
    op::Function::operator = (other);
    this->m_tanh = std::move(other.m_tanh);

    return *this;
}

void TanH::set_inputs(const std::shared_ptr<op::Function> & self, 
                                    const std::vector<std::shared_ptr<op::Variable>> & inputs)
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument(std::string("Number of inputs for TanH operation is not 1"));
    }

    op::Function::set_inputs(self, inputs);
}

void TanH::forward()
{
    var::Tensor<float> *input_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[0].get());
    std::shared_ptr<julie::la::iMatrix<float>> t_mat_ptr = input_ptr->val();

    var::Tensor<float> *output_ptr = dynamic_cast<var::Tensor<float>*>(this->m_output.get());

    // Forward
    this->m_tanh->operator()(*(output_ptr->val()), this->m_diff, *t_mat_ptr);
}

void TanH::backward()
{
    var::Tensor<float> *output_ptr = dynamic_cast<var::Tensor<float>*>(this->m_output.get());
    std::shared_ptr<julie::la::iMatrix<float>> out_grad = output_ptr->grad();

    var::Tensor<float> *input_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[0].get());

    std::shared_ptr<julie::la::iMatrix<float>> t_mat_ptr = input_ptr->val();

    // Do chain rule for the input
    julie::la::multiply(this->m_input_grad_cache, this->m_diff, *out_grad);
    input_ptr->add_grad(this->m_input_grad_cache);
}

void TanH::clear_cache()
{
    this->m_diff = julie::la::iMatrix<float> {};
    this->m_input_grad_cache = julie::la::iMatrix<float> {};
}

} // namespace func
} // namespace nn
} // namespace julie
