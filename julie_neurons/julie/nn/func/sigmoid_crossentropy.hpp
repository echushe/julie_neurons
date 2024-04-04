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

#include "Losses.hpp"

namespace julie
{
namespace nn
{
namespace func
{

/******************************************************************************
 * "Sigmoid_CrossEntropy" is a function which does forward-propagation and
 * back-propagation of the following operation:
 * 
 *      CrossEntropy ( Sigmoid (tensor_a), tensor_b )
 * 
 ******************************************************************************/
class Sigmoid_CrossEntropy : public op::Function
{
public:

    // Default constructor
    Sigmoid_CrossEntropy(lint axis);

    // Copy constructor
    Sigmoid_CrossEntropy(const Sigmoid_CrossEntropy & other);

    // Move constructor
    Sigmoid_CrossEntropy(Sigmoid_CrossEntropy && other);

    // Copy assignment
    Sigmoid_CrossEntropy & operator = (const Sigmoid_CrossEntropy & other);

    // Move assignment
    Sigmoid_CrossEntropy & operator = (Sigmoid_CrossEntropy && other);

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
    //             Sigmoid_CrossEntropy function accepts 2 inputs.
    // Returns: void
    virtual void set_inputs(const std::shared_ptr<op::Function> & self, 
                    const std::vector<std::shared_ptr<op::Variable>> & inputs);

private:

    // The raw loss function of Sigmoid_CrossEntropy
    std::unique_ptr<la::Sigmoid_CrossEntropy<float>> m_sc;

    /*
     * All cache items that are intermediate values will be defined here
     * */

    julie::la::iMatrix<float> m_diff;
    julie::la::iMatrix<float> m_input_grad_cache;
    julie::la::iMatrix<float> m_output_grad_cache;
    julie::la::iMatrix<float> m_output_grad_repeat_cache;
};


} // namespace func
} // namespace nn
} // namespace julie
