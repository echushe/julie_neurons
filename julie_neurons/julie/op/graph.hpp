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

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <ostream>

namespace julie
{

namespace op
{

/******************************************************************************
 * A Graph is used to hold the entire neural network structure where graph
 * searches are implemented for forward and back-propagations.
 * 
 * Functions are handled as Graph nodes and variables as Graph edges.
 * 
 * You can add functions with variables as inputs directly into the graph.
 * But variables cannot be added into graph directly because they have to be
 * carried by functions.
 ******************************************************************************/
class Graph
{
public:

    // Default constructor 
    Graph () {}

    // Comment out copy constructor to use default copy constructor
    //Graph (const Graph & other);

    // Comment out move constructor to use default move constructor
    //Graph (Graph && other);

    // Comment out copy assignment to use default copy assignment
    //Graph & operator = (const Graph & other);

    // Comment out move assignment to use default move 
    //Graph & operator = (Graph && other); 

    // Add a new function node to the Graph.
    // Arguments:
    //     func:   reference to the function
    //     inputs: inputs of this function as a list of variable references
    // Returns: void
    void add_node(const std::shared_ptr<Function> & func, const std::vector<std::shared_ptr<Variable>> & inputs);

    // Remove a function node from the Graph.
    // This function to be removed should be an endpoint function, which means
    // the function's output is not input of any function node.
    // Otherwise, this method will fail and return false.
    bool remove_node(const std::shared_ptr<Function> & func);

    // Remove a function node from the Graph.
    // This function to be removed should be an endpoint function, which means
    // the function's output is not input of any function node.
    // Otherwise, this method will fail and return false.
    bool remove_node(int id);

    // Add a variable to the Graph as one of the inputs.
    // This input variable should be part of this graph already.
    void add_input(const std::shared_ptr<Variable> & input);

    // Add a variable to the Graph as one of the outputs.
    // This output variable should be part of this graph already.
    void add_output(const std::shared_ptr<Variable> & output);

    // Execute forward-propagation ending with a variable.
    void forward(const std::shared_ptr<Variable> &var);

    // Execute back-propagation ending with a variable.
    // This variable is usually a trainable variable (weights).
    bool backward(const std::shared_ptr<Variable> &var);
    
    // Execute back-propagation ending with all trainable variables. 
    void backward();

    // Clear visitations of forward-propagation of all variables.
    void clear_forwards();

    // Clear visitations of back-propagation of all variables.
    void clear_backwards();

    // Clear all function cache
    void clear_cache();

    // Get a list of references of all variables in this Graph.
    std::vector<std::shared_ptr<Variable>> all_variables() const;

    // Get a list of references of all trainable variables (weights) in this Graph.
    std::vector<std::shared_ptr<Variable>> all_trainable_variables() const;

public:
    // Get a string which describes structure of this Graph.
    std::string to_string() const;

    void set_device(julie::MatrixType mtype);

public:
    // Destructor
    ~Graph() {};

private:

    // Execute forward-propagation ending with a function node.
    void func_forward(const std::shared_ptr<Function> & func);

    // Execute back-propagation ending with a function node.
    bool func_backward(const std::shared_ptr<Function> & func);
    
    // Add information of a function node into output stream.
    void print_node(std::ostream & os, const std::shared_ptr<Function> & func, int depth) const;
    
    // Recursively add information of function nodes into output stream.
    void print_node_recursive(
        std::ostream & os,
        std::unordered_set<int> & funcs_visited,
        const std::shared_ptr<Function> & func,
        int depth) const;

private:

    // A hashmap of all variables in this Graph using variable IDs as keys
    std::unordered_map<int, std::shared_ptr<Variable>> m_variables;
    // A hashmap of all functions in this Graph using function IDs as keys
    std::unordered_map<int, std::shared_ptr<Function>> m_functions;

    // A hashmap of all variables as inputs
    // All inputs should already exist in the Graph before inserting into this hashmap.
    std::unordered_map<int, std::shared_ptr<Variable>> m_inputs;
    // A hashmap of all variables as outputs
    // All outputs should already exist in the Graph before inserting into this hashmap.
    std::unordered_map<int, std::shared_ptr<Function>> m_outputs;
};


} // namespace op    
} // namespace julie
