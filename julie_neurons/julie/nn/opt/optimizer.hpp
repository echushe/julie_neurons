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
#include "graph.hpp"

namespace julie
{
namespace nn
{
namespace opt
{

/**********************************************************************************
 * An optimizer applies gradients generated by back-propagation to trainable variables 
 * using a certain gradient descent strategy.
 * 
 * The class Optimizer is an abstract class that is base class of all optimizers.
 * An offspring class of optimizer will implement a kind of gradient descent strategy. 
 **********************************************************************************/
class Optimizer
{
public:

    // Construct an optimizer binding with a list of trainable variables.
    Optimizer (const std::vector<std::shared_ptr<op::Variable>> & params);

    // Construct an optimizer binding with a neural network graph.
    // All trainable variables of this neural network are binded to the optimizer.
    Optimizer (const op::Graph & graph);

    // Copy constructor.
    // The new copy of optimizer will use the same binding of trainable variables
    // as the original one.
    Optimizer (const Optimizer & other);

    // Move constructor.
    // The original optimizer will handover all trainable variables to the new one.
    Optimizer (Optimizer && other);

    // Copy assignment.
    // The optimizer which is left operand will use the same binding of trainable variables
    // as the right operand.
    Optimizer & operator = (const Optimizer & other);

    // Move assignment.
    // The optimizer which is right operand will handover all trainable variables to the left operand.
    Optimizer & operator = (Optimizer && other);

    // A step of gradient descent will be executed by this method.
    virtual void step() = 0;

protected:

    // Trainable variables the optimizer will deal with.
    std::vector<std::shared_ptr<op::Variable>> m_params;
};

} // namespace opt
} // namespace nn
} // namespace julie
