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

#include "Conv2d.hpp"
#ifdef WITH_ONEDNN
#include "Conv2dOneDNN.hpp"
#endif

#ifdef WITH_CUDNN
#include "Conv2dCuDNN.hpp"
#endif
namespace julie
{
namespace nn
{
namespace func
{

/******************************************************************************
 * "Conv2d" is a function which does forward-propagation and back-propagation of
 * the following operation:
 * 
 *     convolution_2d ( input, weights, bias )
 * 
 ******************************************************************************/
class Conv2d : public op::Function
{
public:

    // Constructor
    // Conv2d function should hold the folloing arguments when created:
    // Arguments:
    //     pad_h:    padding size at the height dimension
    //     pad_w:    padding size at the width dimension
    //     stride_h: stride size at the height dimension
    //     stride_w: stride size at the width dimension
    Conv2d(lint pad_h, lint pad_w, lint stride_h, lint stride_w);
    
    // Copy constructor
    Conv2d(const Conv2d & other);

    // Move constructor
    Conv2d(Conv2d && other);

    // Copy assignment
    Conv2d & operator = (const Conv2d & other);

    // Move assignment
    Conv2d & operator = (Conv2d && other);

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
    //             Conv2d function accepts 3 inputs.
    // Returns: void
    virtual void set_inputs(const std::shared_ptr<op::Function> & self, 
                        const std::vector<std::shared_ptr<op::Variable>> & inputs);

private:
    
    // Raw convolution 2d functions
    std::unique_ptr<julie::la::Conv2d<float>> m_conv2d;
#ifdef WITH_ONEDNN
    std::unique_ptr<julie::la::Conv2dOneDNN<float>> m_onednn_conv2d;
#endif
#ifdef WITH_CUDNN
    std::unique_ptr<julie::la::Conv2dCuDNN<float>> m_cudnn_conv2d;
#endif

    /*
     * All cache items that are intermediate values will be defined here
     * */

    julie::la::iMatrix<float> m_feature_grad_cache;
    julie::la::iMatrix<float> m_weight_grad_cache;
    julie::la::iMatrix<float> m_bias_grad_cache;
};


} // namespace func
} // namespace nn
} // namespace julie
