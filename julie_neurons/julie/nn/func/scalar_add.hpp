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

namespace julie
{
namespace nn
{
namespace func
{

/******************************************************************************
 * "ScalarAdd" is a function which does forward-propagation and back-propagation of
 * the following operation:
 * 
 *     scalar_a + scalar_b
 * 
 ******************************************************************************/
class ScalarAdd : public op::Function
{
public:

    // Default constructor
    ScalarAdd();
    
    // Copy constructor
    ScalarAdd(const ScalarAdd & other);

    // Move constructor
    ScalarAdd(ScalarAdd && other);

    // Copy assignment
    ScalarAdd & operator = (const ScalarAdd & other);

    // Move assignment
    ScalarAdd & operator = (ScalarAdd && other);

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
    //             Add function accepts 2 inputs.
    // Returns: void
    virtual void set_inputs(const std::shared_ptr<op::Function> & self, 
                            const std::vector<std::shared_ptr<op::Variable>> & inputs);

private:

    /*
     * All cache items that are intermediate values will be defined here
     * */

};


} // namespace func
} // namespace nn
} // namespace julie
