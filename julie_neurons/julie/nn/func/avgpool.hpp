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
#ifdef WITH_ONEDNN
#include "Pool2dOneDNN.hpp"
#endif

namespace julie
{
namespace nn
{
namespace func
{

/******************************************************************************
 * "AvgPool" is a function which does forward-propagation and back-propagation
 * of the following operation:
 * 
 *     AvgPool ( tensor_a )
 * 
 ******************************************************************************/
class AvgPool : public op::Function
{
public:

    // Constructor
    // Arguments:
    //     pad_h:    padding size at the height dimension
    //     pad_w:    padding size at the width dimension
    //     kernel_h: height of pooling kernel
    //     kernel_w: width of pooling kernel
    //     stride_h: stride size at the height dimension
    //     stride_w: stride size at the width dimension
    AvgPool(lint pad_h, lint pad_w, lint kernel_h, lint kernel_w, lint stride_h, lint stride_w);

    // Copy constructor
    AvgPool(const AvgPool & other);

    // Move constructor
    AvgPool(AvgPool && other);

    // Copy assignment
    AvgPool & operator = (const AvgPool & other);

    // Move assignment
    AvgPool & operator = (AvgPool && other);

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
    //             Sigmoid function accepts only 1 input.
    // Returns: void
    virtual void set_inputs(const std::shared_ptr<op::Function> & self, 
                    const std::vector<std::shared_ptr<op::Variable>> & inputs);

private:

    lint m_pad_h;
    lint m_pad_w;
    lint m_kernel_h;
    lint m_kernel_w;
    lint m_stride_h;
    lint m_stride_w;

#ifdef WITH_ONEDNN
    std::unique_ptr<julie::la::Pool2dOneDNN<float>> m_onednn_pool2d;
#endif

    /*
     * All cache items that are intermediate values will be defined here
     * */
    
    julie::la::iMatrix<float> m_padding_cache;
    julie::la::iMatrix<float> m_pool_diff;
    julie::la::iMatrix<float> m_pool_grad_cache;
    julie::la::iMatrix<float> m_pad_grad_cache;
    julie::la::iMatrix<float> m_input_grad_cache;
};


} // namespace func
} // namespace nn
} // namespace julie
