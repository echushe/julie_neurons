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

namespace julie
{
namespace nn
{
namespace func
{

/******************************************************************************
 * "Concat" is a function which does forward-propagation and back-propagation of
 * the following operation:
 * 
 *     Concatenation of multiple matrices: tensor_a, tensor_b, tensor_c, etc.
 * 
 ******************************************************************************/
class Concat : public op::Function
{
public:

    // Default constructor
    Concat(lint axis);
    
    // Copy constructor
    Concat(const Concat & other);

    // Move constructor
    Concat(Concat && other);

    // Copy assignment
    Concat & operator = (const Concat & other);

    // Move assignment
    Concat & operator = (Concat && other);

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
    //             Concat function accepts 2 inputs.
    // Returns: void
    virtual void set_inputs(const std::shared_ptr<op::Function> & self, 
                        const std::vector<std::shared_ptr<op::Variable>> & inputs);

private:

    lint m_axis;

    /*
     * All cache items that are intermediate values will be defined here
     * */
    
    julie::la::iMatrix<float> m_left_grad_cache;
    julie::la::iMatrix<float> m_right_grad_cache;
    
};


} // namespace func
} // namespace nn
} // namespace julie
