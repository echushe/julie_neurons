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

#include "sgd.hpp"
#include "scalar.hpp"
#include "tensor.hpp"
#include "iMatrix_func.hpp"

namespace julie
{
namespace nn
{
namespace opt
{

SGD::SGD (
    const std::vector<std::shared_ptr<op::Variable>> & params,
    float learning_rate,
    float momentum_rate)
    :
    Optimizer {params},
    m_lr {learning_rate},
    m_mr {momentum_rate}
{
    this->init_momentum();
}

SGD::SGD (const op::Graph & graph, float learning_rate, float momentum_rate)
    :
    Optimizer {graph},
    m_lr {learning_rate},
    m_mr {momentum_rate}
{
    this->init_momentum();
}

void SGD::init_momentum()
{
    // std::cout << "There are " << this->m_params.size() << " trainable variables in this graph" << std::endl;
    for (size_t i = 0; i < this->m_params.size(); ++i)
    {
        std::shared_ptr<op::Variable> param = this->m_params[i];
        
        if (param->data_type() == op::Variable::VariableType::TENSOR)
        {
            std::shared_ptr<julie::la::iMatrix<float>> w_ptr = dynamic_cast<var::Tensor<float>*> (param.get())->val();
            this->m_momentum.push_back(
                std::make_shared<var::Tensor<float>> (
                    julie::la::iMatrix<float>(0, w_ptr->shape(), w_ptr->get_matrix_type())
                    ));
        }
        else if (param->data_type() == op::Variable::VariableType::SCALAR)
        {
            this->m_momentum.push_back(std::make_shared<var::Scalar<float>> (0));
        }

        this->m_cache.push_back(std::make_shared<SGD_cache> ());
    }
}

SGD::SGD (const SGD & other)
    :
    Optimizer {other},
    m_lr {other.m_lr},
    m_mr {other.m_mr},
    m_momentum {other.m_momentum},
    m_cache {other.m_cache}
{}


SGD::SGD (SGD && other)
    :
    Optimizer {other},
    m_lr {other.m_lr},
    m_mr {other.m_mr},
    m_momentum {std::move(other.m_momentum)},
    m_cache {std::move(other.m_cache)}
{}


SGD & SGD::operator = (const SGD & other)
{
    Optimizer::operator = (other);
    m_lr = other.m_lr;
    m_mr = other.m_mr;

    this->m_momentum = other.m_momentum;
    this->m_cache = other.m_cache;

    return *this;
}

SGD & SGD::operator = (SGD && other)
{
    Optimizer::operator = (other);
    m_lr = other.m_lr;
    m_mr = other.m_mr;

    this->m_momentum = std::move(other.m_momentum);
    this->m_cache = std::move(other.m_cache);

    return *this;
}

void SGD::tensor_step(const julie::la::iMatrix<float> & gradient, julie::la::iMatrix<float> & momentum, julie::la::iMatrix<float> & weight, size_t idx)
{
    // std::cout << "The gradient is: \n" << gradient << std::endl;

    julie::la::multiply(this->m_cache[idx]->m_momentum_part, momentum, this->m_mr);
    julie::la::multiply(this->m_cache[idx]->m_gradient_part, gradient, (1.0f - this->m_mr));

    julie::la::add(momentum, this->m_cache[idx]->m_momentum_part, this->m_cache[idx]->m_gradient_part);

    julie::la::multiply(this->m_cache[idx]->m_momentum_learning_rated, momentum, this->m_lr);
    weight -= this->m_cache[idx]->m_momentum_learning_rated;

    // weight -= this->m_lr * gradient;
}

void SGD::scalar_step(const float & gradient, float & momentum, float & weight)
{
    momentum = this->m_mr * momentum + (1.0 - this->m_mr) * gradient;
    weight -= this->m_lr * momentum;

    // weight -= this->m_lr * gradient;
}

void SGD::step()
{
    for (size_t i = 0; i < this->m_params.size(); ++i)
    {
        std::shared_ptr<op::Variable> param = this->m_params[i];
        std::shared_ptr<op::Variable> momentum = this->m_momentum[i];

        if (param)
        {
            if (param->data_type() == op::Variable::VariableType::TENSOR)
            {
                if (param.get()->forward_visited() && param.get()->backward_visited())
                {
                    std::shared_ptr<julie::la::iMatrix<float>> w_ptr = dynamic_cast<var::Tensor<float>*> (param.get())->val();
                    std::shared_ptr<julie::la::iMatrix<float>> g_ptr = dynamic_cast<var::Tensor<float>*> (param.get())->grad();

                    std::shared_ptr<julie::la::iMatrix<float>> m_ptr;

                    m_ptr = dynamic_cast<var::Tensor<float>*> (momentum.get())->val();
                    
                    this->tensor_step(*g_ptr, *m_ptr, *w_ptr, i);
                    // std::cout << "Update weights!" << std::endl;
                }
            }
            else if (param->data_type() == op::Variable::VariableType::SCALAR)
            {
                if (param.get()->forward_visited() && param.get()->backward_visited())
                {
                    std::shared_ptr<float> w_ptr = dynamic_cast<var::Scalar<float>*> (param.get())->val();
                    std::shared_ptr<float> g_ptr = dynamic_cast<var::Scalar<float>*> (param.get())->grad();

                    std::shared_ptr<float> m_ptr;
                    
                    m_ptr = dynamic_cast<var::Scalar<float>*> (momentum.get())->val();

                    this->scalar_step(*g_ptr, *m_ptr, *w_ptr);
                }
            }
            else
            {
                /* code */
            }
        }
    }
}

} // namespace opt
} // namespace nn
} // namespace julie
