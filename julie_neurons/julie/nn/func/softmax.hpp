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
#include "function.hpp"
#include "tensor.hpp"
#include "scalar.hpp"

#include "Activations.hpp"

namespace julie
{
namespace nn
{
namespace func
{

/******************************************************************************
 * "SoftMax" is a function which does forward-propagation and back-propagation of
 * the following operation:
 * 
 *     SoftMax ( tensor_a )
 * 
 ******************************************************************************/
class SoftMax : public op::Function
{
public:

    // Constructor
    // SoftMax function should be created with a specified dimension
    // softmax will work on.
    SoftMax(lint axis);

    // Copy constructor
    SoftMax(const SoftMax & other);

    // Move constructor
    SoftMax(SoftMax && other);

    // Copy assignment
    SoftMax & operator = (const SoftMax & other);

    // Move assignment
    SoftMax & operator = (SoftMax && other);

public:

    // This method executes forward-propagation of this function.
    virtual void forward();

    // This method executes back-propagation of this function.
    virtual void backward();

    // There may be some temporary buffers in the function to speed up calculations.
    // Execute this function can clear them to save memory.
    virtual void clear_cache();

    // Set inputs of this function
    // Arguments:
    //     self: smart pointer of the function self. It should be the same as C++ pointer "this".
    //     inputs: A list of references (smart pointers) of variables as inputs of this function.
    //             SoftMax function accepts only 1 input.
    // Returns: void
    virtual void set_inputs(const std::shared_ptr<op::Function> & self, 
                    const std::vector<std::shared_ptr<op::Variable>> & inputs);

private:

    // The raw activation function of SoftMax
    std::unique_ptr<la::SoftMax<float>> m_softmax;

    /*
     * All cache items that are intermediate values will be defined here
     * */

    julie::la::iMatrix<float> m_diff;
    julie::la::iMatrix<float> m_input_grad_cache;
};


} // namespace func
} // namespace nn
} // namespace julie
