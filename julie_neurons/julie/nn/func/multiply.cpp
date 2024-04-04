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

#include "multiply.hpp"
#include "iMatrix_func.hpp"

namespace julie
{
namespace nn
{
namespace func
{

Multiply::Multiply()
    :
    op::Function {std::string {"Multiply"}, false}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();   
}

Multiply::Multiply(const Multiply & other)
    : op::Function {other}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

Multiply::Multiply(Multiply && other)
    : op::Function {other}
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

Multiply & Multiply::operator = (const Multiply & other)
{
    op::Function::operator = (other);

    return *this;
}

Multiply & Multiply::operator = (Multiply && other)
{
    op::Function::operator = (other);

    return *this;
}

void Multiply::set_inputs(const std::shared_ptr<op::Function> & self, 
                                    const std::vector<std::shared_ptr<op::Variable>> & inputs)
{
    if (inputs.size() != 2)
    {
        throw std::invalid_argument(std::string("Number of inputs for multiply operation is not 2"));
    }

    op::Function::set_inputs(self, inputs);
}

void Multiply::forward()
{
    var::Tensor<float> *l_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[0].get());
    var::Tensor<float> *r_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[1].get());

    std::shared_ptr<julie::la::iMatrix<float>> l_mat_ptr = l_ptr->val();
    std::shared_ptr<julie::la::iMatrix<float>> r_mat_ptr = r_ptr->val();

    var::Tensor<float> *output_ptr = dynamic_cast<var::Tensor<float>*>(this->m_output.get());

    julie::la::broadcast_multiply(*(output_ptr->val()), *l_mat_ptr, *r_mat_ptr);
}

void Multiply::backward()
{
    var::Tensor<float> *output_ptr = dynamic_cast<var::Tensor<float>*>(this->m_output.get());
    std::shared_ptr<julie::la::iMatrix<float>> out_grad = output_ptr->grad();

    var::Tensor<float> *l_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[0].get());
    var::Tensor<float> *r_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[1].get());

    std::shared_ptr<julie::la::iMatrix<float>> l_mat_ptr = l_ptr->val();
    std::shared_ptr<julie::la::iMatrix<float>> r_mat_ptr = r_ptr->val();

    // Do chain rule for left hand side
    julie::la::broadcast_multiply(*(l_ptr->grad()), *out_grad, *r_mat_ptr);

    lint r_mat_size = r_mat_ptr->shape().size();

    size_t count = 0;
    if (this->m_fused_cache.empty())
    {
        this->m_fused_cache.push_back(julie::la::iMatrix<float> {});
    }
    
    julie::la::multiply(this->m_fused_cache[0], *l_mat_ptr, *out_grad);
    while (this->m_fused_cache[count].shape().size() > r_mat_size)
    {
        if (this->m_fused_cache.size() <= count + 1)
        {
            this->m_fused_cache.push_back(julie::la::iMatrix<float> {});
        }

        this->m_fused_cache[count].get_reduce_sum(this->m_fused_cache[count + 1], 0);

        ++count;
    }

    // Do chain rule for right hand side
    r_ptr->add_grad(this->m_fused_cache[count]);
}

void Multiply::clear_cache()
{
    this->m_left_grad_cache = julie::la::iMatrix<float> {};
    for (auto &mat : this->m_fused_cache)
    {
        mat = julie::la::iMatrix<float> {};
    }
}

} // namespace func
} // namespace nn
} // namespace julie
