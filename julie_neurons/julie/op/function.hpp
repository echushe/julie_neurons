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

#include <vector>
#include <memory>

namespace julie
{

namespace op
{

/******************************************************************************
 * Functions use variables as inputs and outputs.
 * 
 * A function can take one or multiple variables as inputs, and hold one variable
 * as its output.
 * 
 * As building blocks of forward-propagation and back-propagation, functions are
 * nodes of a neural network graph where each function is a certain kind of
 * operation or calculation executed in a forward or a backward way. 
 ******************************************************************************/
class Function
{
public:

    // Construct a new function
    // Arguments:
    //     type: function type as a string, for example: "MatMul", "Sigmoid", etc
    //     is_loss_function: a flag to confirm this function is a loss function or a normal function
    Function(const std::string & type, bool is_loss_function);
    
    // Copy constructor
    // Copy constructor does not copy topological data of another function such as
    // the functions's inputs (variables) or its output (a variable).
    Function(const Function & other);

    // Move constructor
    // Move constructor does not move topological data of another function such as
    // the functions's inputs (variables) or its output (a variable).
    Function(Function && other);
    
    // Copy assignment
    // Copy assignment does not copy topological data of another function such as
    // the functions's inputs (variables) or its output (a variable).
    Function & operator = (const Function & other);

    // Move assignment
    // Move assignment does not move topological data of another function such as
    // the functions's inputs (variables) or its output (a variable).
    Function & operator = (Function && other);

    // This method executes forward-propagation of this function.
    virtual void forward() = 0;
    
    // This method executes back-propagation of this function.
    virtual void backward() = 0;

    // This method is to make the function forward-propagation unvisited.
    void clear_forward_visit();

    // This method is to make the function back-propagation unvisited.
    void clear_backward_visit();

    // This method is to start visiting the function node in forward-propagation
    void forward_visit_start();

    // This method is to finish visiting the function node in forward-propagation
    void forward_visit_finish();

    // This method is to start visiting the function node in back-propagation
    void backward_visit_start();

    // This method is to finish visiting the function node in back-propagation
    void backward_visit_finish();

    bool forward_visit_started() const;

    // This method is to check if the function is forward-propagation executed.
    bool forward_visited() const;

    bool backward_visit_started() const;

    // This method is to check if the function is back-propagation executed.
    bool backward_visited() const;
    
    // There may be some temporary buffers in the function to speed up calculations.
    // Execute this function can clear them to save memory.
    virtual void clear_cache() = 0;

    // Get all references of inputs (variables) of this function
    std::vector<std::shared_ptr<Variable>> get_inputs() const;
    
    // Get output (variable) of this function
    std::shared_ptr<Variable> get_output() const;

    // Set inputs of this function
    // Arguments:
    //     self: smart pointer of the function self. It should be the same as C++ pointer "this"
    //     inputs: A list of references (smart pointers) of variables as inputs of this function
    // Returns: void
    virtual void set_inputs(const std::shared_ptr<Function> & self, const std::vector<std::shared_ptr<Variable>> & inputs);

    // Clear inputs of this function
    void clear_inputs(const std::shared_ptr<Function> & self);

public:

    // Each function node has a unique ID.
    // The ID is automatically generated when the function node is created.
    // Generation of function ID is multi-thread safe.
    int id() const;

    // There are many types functions, for example: "ReLU", "SoftMax", etc.
    // Each function node should hold a string indicating its function type.
    // This method is to get function type of this function node.
    std::string type() const;
    
    // This method is to check whether this function is a loss function or a normal function.
    bool is_loss_function() const;

    // Set this function to loss function
    Function & to_loss_function();

    // Set this function to non loss function
    Function & to_non_loss_function();

protected: // attributes

    // Unique ID for this function node
    int m_id;

    // There are many types functions, for example: "ReLU", "SoftMax", etc.
    // Each function node should hold a string indicating its function type.
    std::string m_type;

    bool m_forward_visit_started;

    // A boolean value to specify if this function is forward-propagation executed.
    bool m_forward_visited;

    bool m_backward_visit_started;

    // A boolean value to specify if this function is back-propagation executed.
    bool m_backward_visited;
    
    // A flag indicating that this function is a loss function or not
    bool m_is_loss_function;

    // A list storing references (smart pointers) of all this function's input variables 
    std::vector<std::shared_ptr<Variable>> m_inputs;
    
    // Output of this function, held by a smart pointer
    std::shared_ptr<Variable> m_output;

protected:

    // A global mutex to guarantee that function IDs are synchronously generated.
    static std::mutex MUTEX;

    // A global ID counter for all function nodes
    static int ID_COUNT;

    // A method to generate a unique ID for this function node
    static int new_id();
};

// Standard output stream to print some basic information of this function node
std::ostream & operator << (std::ostream & os, const Function & func);

} // namespace op
} // namespace julie