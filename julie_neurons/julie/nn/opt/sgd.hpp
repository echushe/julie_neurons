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

#pragma once
#include "optimizer.hpp"
#include "iMatrix.hpp"
namespace julie
{
namespace nn
{
namespace opt
{

/**********************************************************************************
 * SGD is abbreviation of "Stochastic Gradient Descent".
 * 
 * This optimizer applies momentum in stochastic gradient descent.
 * There will be no momentum if you set momentum_rate to zero.
 **********************************************************************************/
class SGD : public Optimizer
{
    // This is definition of intermediate variables as cache in optimization calculations.
    struct SGD_cache
    {
        julie::la::iMatrix<float> m_momentum_part;
        julie::la::iMatrix<float> m_gradient_part;
        julie::la::iMatrix<float> m_momentum_learning_rated;
    };

public:

    // Constructor
    SGD (const std::vector<std::shared_ptr<op::Variable>> & params, float learning_rate, float momentum_rate);

    // Constructor
    SGD (const op::Graph & graph, float learning_rate, float momentum_rate);

    // Copy constructor
    SGD (const SGD & other);

    // Move constructor
    SGD (SGD && other);

    // Copy assignment
    SGD & operator = (const SGD & other);

    // Move assignment
    SGD & operator = (SGD && other);

    // A step of gradient descent applying SGD
    virtual void step();

private:

    // Initialize momentum for each trainable variable
    void init_momentum();
    
    // A step of gradient descent for a tensor
    void tensor_step(const julie::la::iMatrix<float> & gradient, julie::la::iMatrix<float> & momentum, julie::la::iMatrix<float> & weight, size_t idx);

    // A step of gradient descent for a scalar
    void scalar_step(const float & gradient, float & momentum, float & weight);

private:

    // Learning rate
    float m_lr;

    // Momentum rate
    float m_mr;

    // A list of reference of momentum variables that are initialized each for each trainable variable.
    std::vector<std::shared_ptr<op::Variable>> m_momentum;

    // Intermediate variables as cache in optimization calculations.
    std::vector<std::shared_ptr<SGD_cache>> m_cache;

};

} // namespace opt
} // namespace nn
} // namespace julie
