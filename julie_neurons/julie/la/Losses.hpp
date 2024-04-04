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
#include "Losses_CPU.hpp"

#ifdef WITH_CUDA
#include "Losses_CUDA.hpp"
#endif


namespace julie
{
namespace la
{

/*********************************************************************************
 * This is the base class of all raw loss functions.
 *********************************************************************************/
template <typename DT>
class LossFunction
{
public:

    // Constructor
    // A loss function should work on a specified dimension.
    LossFunction(lint axis);

    // Execution of the loss function.
    // Arguments:
    //     loss:   Error output of the loss function
    //     diff:   The derivative: d (loss) / d (input)
    //     target: Target input of this loss function
    //     input:  Input of this loss function
    // Example:
    //     Suppose there is an input and a target, they are both of shape (64, 32, 100), and the loss function
    //     works on the second (index == 1) dimension. Then the loss output's shape is (64, 100), shape of the
    //     differentiation d (loss) / d (input) is (64, 32, 100).
    void operator () (iMatrix<DT> &loss, iMatrix<DT> & diff, const iMatrix<DT> & target, const iMatrix<DT> & input);

    // Get the dimension index this loss function works on.
    lint axis() const { return this->m_axis; }

protected:

    // The axis (dimension index) this loss function works on.
    lint m_axis;

    // Reference to the naked CPU type loss function.
    // This reference will be instantiated by derived classes, for example: Sigmoid_CrossEntropy.
    std::shared_ptr<cpu::LossFunction_CPU<DT>> m_losses_cpu;

#ifdef WITH_CUDA
    // Reference to the naket CUDA type loss function.
    // This reference will be instantiated by derived classes, for example: Sigmoid_CrossEntropy.
    std::shared_ptr<cuda::LossFunction_CUDA<DT>> m_losses_cuda;
#endif
};


/*********************************************************************************
 * This loss function is raw implementation of the following operation:
 * 
 *     0.5 * sum ((a1 - t1)^2, (a2 - t2)^2, ... (an - tn)^2)
 * 
 *********************************************************************************/
template <typename DT>
class HalfSquareError : public LossFunction<DT>
{
public:

    // Constructor with a specified dimension
    HalfSquareError(lint axis);
};

/*********************************************************************************
 * This loss function is raw implementation of the following operation:
 * 
 *     CrossEntropy ( Sigmoid (input), target )
 * 
 *********************************************************************************/
template <typename DT>
class Sigmoid_CrossEntropy : public LossFunction<DT>
{
public:

    // Constructor with a specified dimension
    Sigmoid_CrossEntropy(lint axis);
};

/*********************************************************************************
 * This loss function is raw implementation of the following operation:
 * 
 *     CrossEntropy ( SoftMax (input), target )
 * 
 *********************************************************************************/
template <typename DT>
class SoftMax_CrossEntropy : public LossFunction<DT>
{
public:

    // Constructor with a specified dimension
    SoftMax_CrossEntropy(lint axis);
};

}  // namespace la
}  // namespace julie

