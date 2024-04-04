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

#include "matmul.hpp"
#include "iMatrix_func.hpp"

namespace julie
{
namespace nn
{
namespace func
{

MatMul::MatMul()
    :
    op::Function {std::string {"MatMul"}, false}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

MatMul::MatMul(const MatMul & other)
    : op::Function {other}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

MatMul::MatMul(MatMul && other)
    : op::Function {other}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

MatMul & MatMul::operator = (const MatMul & other)
{
    op::Function::operator = (other);

    return *this;
}

MatMul & MatMul::operator = (MatMul && other)
{
    op::Function::operator = (other);

    return *this;
}

void MatMul::set_inputs(const std::shared_ptr<op::Function> & self, 
                                    const std::vector<std::shared_ptr<op::Variable>> & inputs)
{
    if (inputs.size() != 2)
    {
        throw std::invalid_argument(std::string("Number of inputs for MatMul operation is not 2"));
    }

    op::Function::set_inputs(self, inputs);
}

void MatMul::forward()
{
    var::Tensor<float> *l_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[0].get());
    var::Tensor<float> *r_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[1].get());

    std::shared_ptr<julie::la::iMatrix<float>> l_mat_ptr = l_ptr->val();
    std::shared_ptr<julie::la::iMatrix<float>> r_mat_ptr = r_ptr->val();

    if (l_mat_ptr->shape().dim() != 2 || r_mat_ptr->shape().dim() != 2)
    {
        throw std::invalid_argument{ std::string{"MatMul operation: All matrices should be 2-dimensional in "} + std::string{__FUNCTION__} };
    }

    var::Tensor<float> *output_ptr = dynamic_cast<var::Tensor<float>*>(this->m_output.get());

    julie::la::matmul(*(output_ptr->val()), *l_mat_ptr, *r_mat_ptr, 1, 1);
}

void MatMul::backward()
{
    var::Tensor<float> *output_ptr = dynamic_cast<var::Tensor<float>*>(this->m_output.get());
    std::shared_ptr<julie::la::iMatrix<float>> out_grad = output_ptr->grad();

    var::Tensor<float> *l_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[0].get());
    var::Tensor<float> *r_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[1].get());

    std::shared_ptr<julie::la::iMatrix<float>> l_mat_ptr = l_ptr->val();
    std::shared_ptr<julie::la::iMatrix<float>> r_mat_ptr = r_ptr->val();

    if (l_mat_ptr->shape().dim() != 2 || r_mat_ptr->shape().dim() != 2)
    {
        throw std::invalid_argument{ std::string{"MatMul operation: All matrices should be 2-dimensional in "} + std::string{__FUNCTION__} };
    }

    // Do chain rule for left hand side
    julie::la::transpose(this->m_right_transpose_cache, *r_mat_ptr, 1);
    julie::la::matmul(this->m_left_grad_cache, *out_grad, this->m_right_transpose_cache, 1, 1);
    l_ptr->add_grad(this->m_left_grad_cache);

    // Do chain rule for right hand side
    julie::la::transpose(this->m_left_transpose_cache, *l_mat_ptr, 1);
    julie::la::matmul(this->m_right_grad_cache, this->m_left_transpose_cache, *out_grad, 1, 1);
    r_ptr->add_grad(this->m_right_grad_cache);
}

void MatMul::clear_cache()
{
    this->m_left_transpose_cache = julie::la::iMatrix<float> {};
    this->m_left_grad_cache = julie::la::iMatrix<float> {};

    this->m_right_transpose_cache = julie::la::iMatrix<float> {};
    this->m_right_grad_cache = julie::la::iMatrix<float> {};
}

} // namespace func
} // namespace nn
} // namespace julie
