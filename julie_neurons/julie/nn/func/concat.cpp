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

#include "concat.hpp"
#include "iMatrix_func.hpp"

namespace julie
{
namespace nn
{
namespace func
{

Concat::Concat(lint axis)
    :
    op::Function {std::string {"Concat"}, false},
    m_axis {axis}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

Concat::Concat(const Concat & other)
    : op::Function {other}, m_axis {other.m_axis}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

Concat::Concat(Concat && other)
    : op::Function {other}, m_axis {other.m_axis}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

Concat & Concat::operator = (const Concat & other)
{
    op::Function::operator = (other);
    this->m_axis = other.m_axis;

    return *this;
}

Concat & Concat::operator = (Concat && other)
{
    op::Function::operator = (other);
    this->m_axis = other.m_axis;

    return *this;
}

void Concat::set_inputs(const std::shared_ptr<op::Function> & self, 
                                    const std::vector<std::shared_ptr<op::Variable>> & inputs)
{
    if (inputs.size() != 2)
    {
        throw std::invalid_argument(std::string("Number of inputs for Concat operation should be 2"));
    }

    op::Function::set_inputs(self, inputs);
}

void Concat::forward()
{
    std::shared_ptr<var::Tensor<float>> l_ptr = std::dynamic_pointer_cast<var::Tensor<float>>(this->m_inputs[0]);
    std::shared_ptr<julie::la::iMatrix<float>> l_mat_ptr = l_ptr->val();

    std::shared_ptr<var::Tensor<float>> r_ptr = std::dynamic_pointer_cast<var::Tensor<float>>(this->m_inputs[1]);
    std::shared_ptr<julie::la::iMatrix<float>> r_mat_ptr = r_ptr->val();

    std::shared_ptr<var::Tensor<float>> output_ptr = std::dynamic_pointer_cast<var::Tensor<float>>(this->m_output);

    julie::la::concatenate(*(output_ptr->val()), *l_mat_ptr, *r_mat_ptr, this->m_axis);

}

void Concat::backward()
{
    std::shared_ptr<var::Tensor<float>> output_ptr = std::dynamic_pointer_cast<var::Tensor<float>>(this->m_output);
    std::shared_ptr<julie::la::iMatrix<float>> out_grad = output_ptr->grad();

    std::shared_ptr<var::Tensor<float>> l_ptr = std::dynamic_pointer_cast<var::Tensor<float>>(this->m_inputs[0]);
    std::shared_ptr<julie::la::iMatrix<float>> l_mat_ptr = l_ptr->val();

    std::shared_ptr<var::Tensor<float>> r_ptr = std::dynamic_pointer_cast<var::Tensor<float>>(this->m_inputs[1]);
    std::shared_ptr<julie::la::iMatrix<float>> r_mat_ptr = r_ptr->val();

    // Do chain rule for left hand side
    julie::la::slice(this->m_left_grad_cache, *out_grad, this->m_axis, 0, l_mat_ptr->shape()[this->m_axis]);
    l_ptr->add_grad(this->m_left_grad_cache);

    // Do chain rule for right hand side
    julie::la::slice(this->m_right_grad_cache, *out_grad, this->m_axis, l_mat_ptr->shape()[this->m_axis], r_mat_ptr->shape()[this->m_axis]);
    r_ptr->add_grad(this->m_right_grad_cache);
}

void Concat::clear_cache()
{
    this->m_left_grad_cache = julie::la::iMatrix<float> {};
    this->m_right_grad_cache = julie::la::iMatrix<float> {};
}

} // namespace func
} // namespace nn
} // namespace julie
