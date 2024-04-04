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
#include "iMatrix.hpp"
#include "oneapi/dnnl/dnnl.hpp"

namespace julie
{
namespace la
{

/********************************************************************************
 * The Conv2dOneDNN class implements forward-propagation and back-propagation of
 * 2-dimensional convolution calculations.
 * 
 * This Conv2dOneDNN class depends on Intel oneDNN (Original MKL) library
 ********************************************************************************/
template <typename DT>
class Conv2dOneDNN
{
private:

    //static std::shared_ptr<dnnl::engine> the_dnnl_engine;

public:

    // Constructor of a 2-dimensional convoltion operation
    // Arguments:
    //     pad_h:    Padding size at the height dimension
    //     pad_w:    Padding size at the width dimension
    //     stride_h: Stride size along the height dimension
    //     stride_w: Stride size along the width dimension
    Conv2dOneDNN (lint pad_h, lint pad_w, lint stride_h, lint stride_w);

    // This method executes forward-propagation of Conv2dOneDNN (padding + convolution)
    // Arguments:
    //     output:  Output of forward-propagation
    //     input:   Input of this method
    //     weights: Kernel weights of convolutional operation
    //     bias:    Bias weights of convolutional operation
    // Returns: void
    void forward (julie::la::iMatrix<DT> &output, const julie::la::iMatrix<DT> & input, const julie::la::iMatrix<DT> & weights, const julie::la::iMatrix<DT> & bias);

    // This method executes back-propagation of Conv2dOneDNN (padding + convolution)
    // Arguments:
    //     in_gradient_out: Gradient of Conv2dOneDNN input. It is output of this method.
    //     w_gradient_out:  Gradient of kernel weights. It is output of this method.
    //     b_gradient_out:  Gradient of bias weights. It is output of this method.
    //     gradient_in:     Gradient of Conv2dOneDNN output. It is input of this method.
    //     input_sh:        Shape of Conv2dOneDNN input. It is input of this method.
    //     weights:         Kernel weights. It is input of this method.
    // Returns: void
    void backward (
        julie::la::iMatrix<DT> & in_gradient_out,
        julie::la::iMatrix<DT> & w_gradient_out,
        julie::la::iMatrix<DT> & b_gradient_out,
        const julie::la::iMatrix<DT> & gradient_in, const Shape & input_sh, const julie::la::iMatrix<DT> & weights);

    // Clear memory of all intermediate variables for calculations
    void clear_cache();

//private:

    //void asyn_forward(iMatrix<DT> &output, const iMatrix<DT> & input, const iMatrix<DT> & weights, const iMatrix<DT> & bias, const dnnl::stream & st);

private:

    // Padding size at the height dimension
    lint m_pad_h;
    // Padding size at the width dimension
    lint m_pad_w;
    // Stride size along the height dimension
    lint m_stride_h;
    // Stride size along the width dimension
    lint m_stride_w;

    /*
     * All cache items that are intermediate variables will be defined here
     * */

    julie::la::iMatrix<DT> m_input;
    std::vector<dnnl::primitive> m_net_fwd;
    std::vector<dnnl::primitive> m_net_bwd;
    std::vector<std::unordered_map<int, dnnl::memory>> m_net_fwd_args;
    std::vector<std::unordered_map<int, dnnl::memory>> m_net_bwd_args;

};

} // la
} // julie
