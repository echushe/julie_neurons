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
#include "variable.hpp"
#include "iMatrix.hpp"


namespace julie
{
namespace nn
{
namespace var
{

/******************************************************************************
 * Tensor is one kind of variable that holds an iMatrix as value for
 * forward-propagation, and another iMatrix as a gradient for back-propagation.
 * 
 * The other kind of variable is called "Scalar" which holds a single integer or
 * float-point number only for forward-propagation, and another single number as
 * a gradient for back-propagation.
 ******************************************************************************/
template <typename DT>
class Tensor : public op::Variable
{
public:

    // Default constructor
    Tensor(julie::MatrixType mtype = julie::CPU);

    // Construct a tensor with an iMatrix as its value.
    Tensor(const julie::la::iMatrix<DT> & val);

    // Construct a tensor with an iMatrix as its value with move semantics.
    Tensor(julie::la::iMatrix<DT> && val);

    // Copy constructor
    // Copy constructor of Variable (base class) is called here as well.
    Tensor (const Tensor & other);
    
    // Move constructor
    // Move construcsor of Variable (base class) is called here as well.
    Tensor (Tensor && other);

    // Copy assignment
    // Copy assignment of Variable (base class) is called here as well.
    Tensor & operator = (const Tensor & other);

    // Move assignment
    // Move assignment if Variable (base class) is called here as well.
    Tensor & operator = (Tensor && other);

public:

    // Get variable type of this variable.
    // This method will always return VariableType::TENSOR here. 
    virtual VariableType data_type() const;

    // This method is to set MatrixType of this tensor.
    // Both value and gradient of this tensor hold the same matrix type.
    // Matrix types:
    //     CPU:  A matrix of this type is supposed to run on a CPU
    //     CUDA: A matrix of this type is supposed to run on an nvidia GPU via CUDA APIs
    //     CL:   A matrix of this type is supposed to run via openCL APIs (under development)
    virtual void set_device(julie::MatrixType mtype);
    
    // This method is to get MatrixType of this tensor
    // Both value and gradient of this tensor hold the same matrix type.
    // Matrix types:
    //     CPU:  A matrix of this type is supposed to run on a CPU
    //     CUDA: A matrix of this type is supposed to run on an nvidia GPU via CUDA APIs
    //     CL:   A matrix of this type is supposed to run via openCL APIs (under development)
    julie::MatrixType get_device() const;

    // This method is to make this tensor visited in forward-propagation.
    virtual void forward_visit_finish();

    // This method is to make this tensor visited in back-propagation.
    virtual void backward_visit_finish();

    // Set gradient of this tensor to zero.
    // This method is usually called when one iteration of back-propagation is refreshed.
    virtual void set_grad_to_zero();

public:

    // Get reference of this tensor's value
    std::shared_ptr<julie::la::iMatrix<DT>> val();

    // Get reference of this tensor's gradient
    std::shared_ptr<julie::la::iMatrix<DT>> grad();

    // Set this tensor's value with deep copy
    void val(const julie::la::iMatrix<DT> & val);

    // Set this tensor's value with move semantics
    void val(julie::la::iMatrix<DT> && val);

    // This method is to do += operation of this tensor's gradient.
    // One tensor may receive gradients provided by multiple functions in back-propagation.
    void add_grad(const julie::la::iMatrix<DT> & grad);

private:

    // Value of this tensor
    std::shared_ptr<julie::la::iMatrix<DT>> m_val;
    
    // Gradient of this tensor
    std::shared_ptr<julie::la::iMatrix<DT>> m_grad;

    julie::MatrixType m_device;
};

} // namespace var
} // namespace nn
} // namespace julie
