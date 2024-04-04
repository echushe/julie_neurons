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

#include "scale.hpp"
#include "iMatrix_func.hpp"

namespace julie
{
namespace nn
{
namespace func
{

Scale::Scale()
    :
    op::Function {std::string {"Scale"}, false}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

Scale::Scale(const Scale & other)
    : op::Function {other}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

Scale::Scale(Scale && other)
    : op::Function {other}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

Scale & Scale::operator = (const Scale & other)
{
    op::Function::operator = (other);

    return *this;
}

Scale & Scale::operator = (Scale && other)
{
    op::Function::operator = (other);

    return *this;
}

void Scale::set_inputs(const std::shared_ptr<op::Function> & self, 
                                    const std::vector<std::shared_ptr<op::Variable>> & inputs)
{
    if (inputs.size() != 2)
    {
        throw std::invalid_argument(std::string("Number of inputs for scale operation is not 2"));
    }

    op::Function::set_inputs(self, inputs);
}

void Scale::forward()
{
    var::Tensor<float> *l_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[0].get());
    var::Scalar<float> *r_ptr = dynamic_cast<var::Scalar<float>*>(this->m_inputs[1].get());

    std::shared_ptr<julie::la::iMatrix<float>> l_mat_ptr = l_ptr->val();
    std::shared_ptr<float> scalar_ptr = r_ptr->val();

    var::Tensor<float> *output_ptr = dynamic_cast<var::Tensor<float>*>(this->m_output.get());

    // forward
    julie::la::multiply(*(output_ptr->val()), *l_mat_ptr, *scalar_ptr);
}

void Scale::backward()
{
    var::Tensor<float> *output = dynamic_cast<var::Tensor<float>*>(this->m_output.get());
    std::shared_ptr<julie::la::iMatrix<float>> out_grad = output->grad();

    var::Tensor<float> *l_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[0].get());
    var::Scalar<float> *r_ptr = dynamic_cast<var::Scalar<float>*>(this->m_inputs[1].get());

    std::shared_ptr<julie::la::iMatrix<float>> l_mat_ptr = l_ptr->val();
    std::shared_ptr<float> scale_ptr = r_ptr->val();

    // Do chain rule for left hand side
    julie::la::multiply(this->m_left_grad_cache, *out_grad, *scale_ptr);
    l_ptr->add_grad(this->m_left_grad_cache);

    // Do chain rule for right hand side
    r_ptr->add_grad(julie::la::dot_product(this->m_right_grad_multiply_cache, *l_mat_ptr, *out_grad));
    //r_ptr->add_grad(julie::la::dot_product(this->m_right_grad_multiply_cache, *l_mat_ptr, *out_grad) / l_mat_ptr->shape().size());
}

void Scale::clear_cache()
{
    this->m_left_grad_cache = julie::la::iMatrix<float> {};
    this->m_right_grad_multiply_cache = julie::la::iMatrix<float> {};
}

} // namespace func
} // namespace nn
} // namespace julie
