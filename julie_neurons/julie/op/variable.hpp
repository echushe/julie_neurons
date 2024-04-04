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
#include "utilities.hpp"

#include <memory>
#include <vector>
#include <mutex>
#include <ostream>

namespace julie
{

namespace op
{

class Function;

/******************************************************************************
 * Variables are edges of a neural network graph where matrix or scalar data
 * is stored as values or gradients.
 * 
 * Variables can be inputs or outputs of functions. They can be trainable or
 * untrainable. Variables which are trainable are treated as weights of the
 * network while untrainable variables are considered as input and output data
 * of neural net layers.
 ******************************************************************************/
class Variable
{
public:

    // A variable can be a tensor (matrix holder) or a scalar (single number holder)
    enum VariableType
    {
        TENSOR = 0,
        SCALAR = 1
    };

public:

    // Default constructor
    Variable ();
    
    // Copy constructor
    // Copy constructor does not copy topological data of another variable such as
    // the variable's provider (function node) or its receivers (function nodes).
    Variable (const Variable & other);
    
    // Move constructor
    // Copy constructor does not move topological data of another variable such as
    // the variable's provider (function node) or its receivers (function nodes).
    Variable (Variable && other);

    // Copy assignment
    // Copy assignment does not copy topological data of another variable such as
    // the variable's provider (function node) or its receivers (function nodes).
    Variable & operator = (const Variable & other);

    // Move assignment
    // Move assignment does not move topological data of another variable such as
    // the variable's provider (function node) or its receivers (function nodes).
    Variable & operator = (Variable && other);

    // This method is to check whether this variable is trainable or not.
    bool trainable() const;
    
    // This method is to mark whether this variable is trainable or not.
    void trainable(bool trainable);

    // Get reference of this variable's provider (function node).
    std::shared_ptr<Function> get_provider() const;

    // Get all this variable's receivers (function nodes) as a list of references. 
    std::vector<std::shared_ptr<Function>> get_receivers() const;

    // Specify provider (a function node) of this variable.
    // It means this variable will become the function node's output. 
    void set_provider(const std::shared_ptr<Function> & provider);

    // Add a receiver to the variable's receiver list.
    // A receiver is a function node which uses this variable as one of its inputs.
    // A varible can be a input of multiple receivers (function nodes)
    void add_receiver(const std::shared_ptr<Function> & receiver);

    // Remove a receiver to the variable's receiver list.
    // A receiver is a function node which uses this variable as one of its inputs.
    // A varible can be a input of multiple receivers (function nodes).
    // This method returns false if the receiver to be removed cannot be found.
    bool remove_receiver(const std::shared_ptr<Function> & receiver);

    // Disconnect this variable from its provider
    void clear_provider();

    // Each variable has a unique ID.
    // The ID is automatically generated when the variable is created.
    // Generation of variable ID is multi-thread safe.
    int id() const;

    virtual void set_device(julie::MatrixType mtype) = 0;

public:

    // This method is to make the variable forward-propagation unvisited.
    void clear_forward_visit();

    // This method is to make the variable back-propagation unvisited.
    void clear_backward_visit();

    bool forward_visit_started() const;

    // This method is to check if the variable is forward-propagation visited.
    bool forward_visited() const;

    bool backward_visit_started() const;

    // This method is to check if the variable is back-propagation visited.
    bool backward_visited() const;

    // This method executes visitation in forward-propagation
    virtual void forward_visit_start();

    virtual void forward_visit_finish() = 0;

    // This method executes visitation in back-propagation
    virtual void backward_visit_start();

    virtual void backward_visit_finish() = 0;

    // This method is to get type of this variable.
    // There are 2 types of variables: Tensor and Scalar
    virtual VariableType data_type() const = 0;

    // A method to set gradient of this varible to zero
    // This method will do nothing if gradient is a NULL reference.
    virtual void set_grad_to_zero() = 0;

protected:

    // A boolean value to specify if this variable is trainable or not.
    bool m_trainable;

    bool m_forward_visit_started;

    // A boolean value to specify if this variable is forward-propagation visited.
    bool m_forward_visited;

    bool m_backward_visit_started; 
    
    // A boolean value to specify if this variable is back-propagation visited.
    bool m_backward_visited;

    // The variable's provider is a function node which uses this variable as output.
    std::shared_ptr<Function> m_provider;
    
    // This is a list of receivers' references.
    // Each receiver is a function node using the variable as one of its inputs.
    std::vector<std::shared_ptr<Function>> m_receivers;

private:

    // A global mutex to guarantee that variable IDs are synchronously generated.
    static std::mutex MUTEX;

    // A global ID counter for all variables
    static int ID_COUNT;
    
    // Unique ID for this variable
    int m_id;

    // A method to generate a unique ID for this variable
    static int new_id();
};

// Standard output stream to print some basic information of this variable
std::ostream & operator << (std::ostream & os, const Variable & var);

} // namespace op
} // namespace julie
