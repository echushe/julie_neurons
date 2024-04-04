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
 * "HalfSquareError" is a function which does forward-propagation and
 * back-propagation of the following operation:
 * 
 *      0.5 * sum ( square (tensor_a - tensor_b))
 * 
 ******************************************************************************/
class HalfSquareError : public op::Function
{
public:
    HalfSquareError(lint axis);
    
    HalfSquareError(const HalfSquareError & other);
    HalfSquareError(HalfSquareError && other);

    HalfSquareError & operator = (const HalfSquareError & other);
    HalfSquareError & operator = (HalfSquareError && other);

public:
    virtual void forward();
    virtual void backward();
    virtual void clear_cache();

    // Set inputs of this function
    // Arguments:
    //     self: smart pointer of the function self. It should be the same as C++ pointer "this".
    //     inputs: A list of references (smart pointers) of variables as inputs of this function.
    //             Half_SquareError function accepts 2 inputs.
    // Returns: void
    virtual void set_inputs(const std::shared_ptr<op::Function> & self, 
                    const std::vector<std::shared_ptr<op::Variable>> & inputs);

private:

    // The raw loss function of Half_SquareError
    std::unique_ptr<la::HalfSquareError<float>> m_hse;

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
