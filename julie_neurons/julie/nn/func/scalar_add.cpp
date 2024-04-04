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

#include "scalar_add.hpp"
#include "iMatrix_func.hpp"

namespace julie
{
namespace nn
{
namespace func
{

ScalarAdd::ScalarAdd()
    :
    op::Function {std::string {"ScalarAdd"}, false}
{
    this->m_output = std::make_shared<var::Scalar<float>> ();
}

ScalarAdd::ScalarAdd(const ScalarAdd & other)
    : op::Function {other}
{
    this->m_output = std::make_shared<var::Scalar<float>> ();
}

ScalarAdd::ScalarAdd(ScalarAdd && other)
    : op::Function {other}
{
    this->m_output = std::make_shared<var::Scalar<float>> ();
}

ScalarAdd & ScalarAdd::operator = (const ScalarAdd & other)
{
    op::Function::operator = (other);

    return *this;
}

ScalarAdd & ScalarAdd::operator = (ScalarAdd && other)
{
    op::Function::operator = (other);

    return *this;
}

void ScalarAdd::set_inputs(const std::shared_ptr<op::Function> & self, 
                                    const std::vector<std::shared_ptr<op::Variable>> & inputs)
{
    if (inputs.size() != 2)
    {
        throw std::invalid_argument(std::string("Number of inputs for add operation is not 2"));
    }

    op::Function::set_inputs(self, inputs);
}

void ScalarAdd::forward()
{
    std::shared_ptr<var::Scalar<float>> l_ptr = std::dynamic_pointer_cast<var::Scalar<float>>(this->m_inputs[0]);
    std::shared_ptr<float> l_val_ptr = l_ptr->val();

    std::shared_ptr<var::Scalar<float>> r_ptr = std::dynamic_pointer_cast<var::Scalar<float>>(this->m_inputs[1]);
    std::shared_ptr<float> r_val_ptr = r_ptr->val();

    std::shared_ptr<var::Scalar<float>> output_ptr = std::dynamic_pointer_cast<var::Scalar<float>>(this->m_output);
    std::shared_ptr<float> output_val_ptr = output_ptr->val();

    *output_val_ptr = *l_val_ptr + *r_val_ptr;
}

void ScalarAdd::backward()
{
    std::shared_ptr<var::Scalar<float>> output_ptr = std::dynamic_pointer_cast<var::Scalar<float>>(this->m_output);
    std::shared_ptr<float> out_grad = output_ptr->grad();

    std::shared_ptr<var::Scalar<float>> l_ptr = std::dynamic_pointer_cast<var::Scalar<float>>(this->m_inputs[0]);
    std::shared_ptr<var::Scalar<float>> r_ptr = std::dynamic_pointer_cast<var::Scalar<float>>(this->m_inputs[1]);

    // Do chain rule for left hand side
    l_ptr->add_grad(*out_grad);

    // Do chain rule for right hand side
    r_ptr->add_grad(*out_grad);
}

void ScalarAdd::clear_cache()
{
}

} // namespace func
} // namespace nn
} // namespace julie
