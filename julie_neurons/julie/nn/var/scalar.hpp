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


namespace julie
{
namespace nn
{
namespace var
{

/******************************************************************************
 * Scalar is one kind of variable that holds a single number as value for
 * forward-propagation, and another single number as a gradient for back-propagation.
 * 
 * The other kind of variable is called "Tensor" which holds an iMatrix for
 * forward-propagation, and another iMatrix as a gradient for back-propagation.
 ******************************************************************************/
template <typename DT>
class Scalar : public op::Variable
{
public:

    // Default constructor
    Scalar();

    // Construct a scalar with an initialized value
    Scalar(DT val);

    // Copy constructor
    Scalar (const Scalar & other);

    // Move constructor
    Scalar (Scalar && other);

    // Copy assignment
    Scalar & operator = (const Scalar & other);

    // Move assignment
    Scalar & operator = (Scalar && other);

public:

    // Get variable type of this variable.
    // This method will always return VariableType::SCALAR here. 
    virtual VariableType data_type() const;

    virtual void set_device(julie::MatrixType mtype);
    
    // This method is to make this scalar visited in forward-propagation.
    virtual void forward_visit_finish();

    // This method is to make this scalar visited in back-propagation.
    virtual void backward_visit_finish();

    // Set gradient of this scalar to zero.
    // This method is usually called when one iteration of back-propagation is refreshed.
    virtual void set_grad_to_zero();

public:

    // Get reference of this scalar's value
    std::shared_ptr<DT> val();

    // Get reference of this scalar's gradient
    std::shared_ptr<DT> grad();

    // Set this scalar's value
    void val(DT val);

    // This method is to do += operation of this scalar's gradient.
    // One scalar may receive gradients provided by multiple functions in back-propagation.
    void add_grad(DT grad);

private:

    // Value of this scalar
    std::shared_ptr<DT> m_val;

    // Gradient of this scalar
    std::shared_ptr<DT> m_grad;
};

} // namespace var
} // namespace nn
} // namespace julie
