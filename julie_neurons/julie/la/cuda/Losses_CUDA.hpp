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
#include "Matrix_CUDA.hpp"
#include "Activations_CUDA.hpp"


namespace julie
{
namespace la
{
namespace cuda
{

/*********************************************************************************
 * This is the base class of all CUDA mode loss functions.
 *********************************************************************************/
template <typename DT>
class LossFunction_CUDA
{
public:

    // Constructor
    // A loss function should work on a specified dimension.
    LossFunction_CUDA(lint axis) : m_axis {axis} {}

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
    virtual void operator () (Matrix_CUDA<DT> &loss, Matrix_CUDA<DT> &diff, const Matrix_CUDA<DT> &target, const Matrix_CUDA<DT> &input) = 0;

    // Get the dimension index this loss function works on.
    lint axis() const { return this->m_axis; }

protected:

    // The axis (dimension index) this loss function works on.
    lint m_axis;

};

/*********************************************************************************
 * This loss function is CUDA implementation of the following operation:
 * 
 *     0.5 * sum ((a1 - t1)^2, (a2 - t2)^2, ... (an - tn)^2)
 * 
 *********************************************************************************/
template <typename DT>
class HalfSquareError : public LossFunction_CUDA<DT>
{
private:

public:

    // Constructor with a specified dimension
    HalfSquareError(lint axis) : LossFunction_CUDA<DT> {axis} {};

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
    virtual void operator () (Matrix_CUDA<DT> &loss, Matrix_CUDA<DT> & diff, const Matrix_CUDA<DT> & target, const Matrix_CUDA<DT> & input);
};

/*********************************************************************************
 * This loss function is CUDA implementation of the following operation:
 * 
 *     CrossEntropy ( Sigmoid (input), target )
 * 
 *********************************************************************************/
template <typename DT>
class Sigmoid_CrossEntropy : public LossFunction_CUDA<DT>
{
private:

public:

    // Constructor with a specified dimension
    Sigmoid_CrossEntropy(lint axis) : LossFunction_CUDA<DT> {axis} {};

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
    virtual void operator () (Matrix_CUDA<DT> &loss, Matrix_CUDA<DT> & diff, const Matrix_CUDA<DT> & target, const Matrix_CUDA<DT> & input);

private:

    // We need to cache intermediate sigmoid values here to prevent frequent free and malloc operations
    Matrix_CUDA<DT> m_sigmoid_cache;
};

/*********************************************************************************
 * This loss function is CUDA implementation of the following operation:
 * 
 *     CrossEntropy ( SoftMax (input), target )
 * 
 *********************************************************************************/
template <typename DT>
class SoftMax_CrossEntropy : public LossFunction_CUDA<DT>
{
private:

public:

    // Constructor with a specified dimension
    SoftMax_CrossEntropy(lint axis) : LossFunction_CUDA<DT> {axis} {};

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
    virtual void operator () (Matrix_CUDA<DT> &loss, Matrix_CUDA<DT> &diff, const Matrix_CUDA<DT> &target, const Matrix_CUDA<DT> &input);

private:

    // We need to cache intermediate softmax values here to prevent frequent free and malloc operations
    Matrix_CUDA<DT> m_softmax_cache;
};

}  // namespace cuda
}  // namespace la
}  // namespace julie

