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

#include "graph.hpp"
#include <sstream>
#include <iostream>

namespace julie
{

namespace op
{

void Graph::add_node(const std::shared_ptr<Function> & func, const std::vector<std::shared_ptr<Variable>> & inputs)
{
    func->set_inputs(func, inputs);

    for (std::shared_ptr<Variable> input : inputs)
    {
        if (!this->m_variables.count(input->id()))
        {
            this->m_variables[input->id()] = input;
        }
    }

    if (!this->m_functions.count(func->id()))
    {
        this->m_functions[func->id()] = func;
    }

    std::shared_ptr<Variable> output = func->get_output();
    if (!this->m_variables.count(output->id()))
    {
        this->m_variables[output->id()] = output;
    }
}

bool Graph::remove_node(const std::shared_ptr<Function> & func)
{
    if (!this->m_functions.count(func->id()))
    {
        return false;
    }

    // This function node cannot be removed if its output is used by other functions
    if (!func->get_output()->get_receivers().empty())
    {
        return false;
    }

    auto inputs = func->get_inputs();
    for (auto &input : inputs)
    {
        // Remove orphan inputs from the graph
        if (!input->get_provider())
        {
            this->m_variables.erase(input->id());
            this->m_inputs.erase(input->id());
            this->m_outputs.erase(input->id());
        }
    }

    // Disconnect this function node from other parts of the graph
    func->clear_inputs(func);

    // Remove this function's output from graph
    this->m_variables.erase(func->get_output()->id());
    this->m_inputs.erase(func->get_output()->id());
    this->m_outputs.erase(func->get_output()->id());

    // Remove this function from graph
    this->m_functions.erase(func->id());

    return true;
}

bool Graph::remove_node(int id)
{
    if (!this->m_functions.count(id))
    {
        return false;
    }

    auto func = this->m_functions.find(id)->second;
    return this->remove_node(func);
}

void Graph::add_input(const std::shared_ptr<Variable> & input)
{
    if (this->m_variables.count(input->id()) && !this->m_inputs.count(input->id()))
    {
        this->m_inputs[input->id()] = input;
    }
}

void Graph::add_output(const std::shared_ptr<Variable> & output)
{
    if (this->m_variables.count(output->id()) && !this->m_outputs.count(output->id()))
    {
        this->m_inputs[output->id()] = output;
    }
}

void Graph::forward(const std::shared_ptr<Variable> &var)
{
    if (var->forward_visit_started())
    {
        // Circle is detected here
        std::cout << "var forward: Circle is detected here" << std::endl;
        return;
    }

    if (var->forward_visited())
    {
        return;
    }

    var->forward_visit_start();

    // Get the function which is provider of this variable
    std::shared_ptr<Function> provider = var->get_provider();

    if (provider)
    {
        // Run forward operation of this function
        this->func_forward(provider);
    }

    var->forward_visit_finish();
}

void Graph::func_forward(const std::shared_ptr<Function> & func)
{
    //std::cout << func->type() << std::endl;
    if (func->forward_visit_started())
    {
        // Circle is detected here
        std::cout << "func_forward: Circle is detected here" << std::endl;
        return;
    }

    if (func->forward_visited())
    {
        return;
    }

    func->forward_visit_start();

    // Get all variables required by this function
    std::vector<std::shared_ptr<Variable>> inputs_required = func->get_inputs();
    // Go through all inputs and do forward operation for them
    for (std::shared_ptr<Variable> input : inputs_required)
    {
        this->forward(input);
    }

    // Run forward of this function
    func->forward();
    func->forward_visit_finish();
}

void Graph::backward()
{
    for (auto & key_value : this->m_variables)
    {
        // pave backward route for all trainable variables in the graph
        if (key_value.second->trainable())
        {
            this->backward(key_value.second);
        }
    }
}

bool Graph::backward(const std::shared_ptr<Variable> &var)
{
    bool backward_success = false;

    if (var->backward_visit_started())
    {
        // Circle is detected here
        std::cout << "var backward: Circle is detected here" << std::endl;
        return backward_success;
    }

    if (var->backward_visited())
    {
        backward_success = true;
        return backward_success;
    }

    var->backward_visit_start();

    // Find all receivers of this variable
    std::vector<std::shared_ptr<Function>> receivers = var->get_receivers();

    // Run backward of all receivers
    for (auto & receiver : receivers)
    {
        if (this->func_backward(receiver))
        {
            backward_success = true;
        }
    }

    if (backward_success)
    {
        var->backward_visit_finish();
    }

    return backward_success;
}

bool Graph::func_backward(const std::shared_ptr<Function> & func)
{
    bool backward_success = false;

    if (func->backward_visit_started())
    {
        // Circle is detected here
        std::cout << "func backward: Circle is detected here" << std::endl;
        return backward_success;
    }

    if (func->backward_visited())
    {
        backward_success = true;
        return backward_success;
    }

    func->backward_visit_start();

    // Get all input variables
    std::vector<std::shared_ptr<Variable>> inputs_required = func->get_inputs();

    // Go through all inputs of this function
    for (std::shared_ptr<Variable> input : inputs_required)
    {
        if (!input->forward_visited())
        {
            throw std::invalid_argument("Input variable should be forward visited while executing backward operation");
        }
    }

    std::shared_ptr<Variable> output = func->get_output();

    // Recursively do back-propagation for the output variable
    backward_success = this->backward(output);

    std::vector<std::shared_ptr<Function>> receivers = output->get_receivers();
    // If the recursive backwards are unsuccessful, there are 2 possibilities
    // 1. This function is a loss function (back-propagation can be successfully done)
    // 2. There is a dead end for back-propagation
    if (!backward_success)
    {
        // If this function is a loss function
        if (func->is_loss_function())
        {
            // Do back-propagation of this function
            func->backward();
            func->backward_visit_finish();
            backward_success = true;
            return backward_success;
        }
        else // There is a dead end of route search if this function is not loss function
        {
            return backward_success;
        }
    }
    else // Recursive backwards are successful
    {
        // Do back-propagation of this function
        func->backward();
        func->backward_visit_finish();
        return backward_success;
    }
}

void Graph::clear_forwards()
{
    for (auto & key_value : this->m_variables)
    {
        key_value.second->clear_forward_visit();
    }

    for (auto & key_value : this->m_functions)
    {
        key_value.second->clear_forward_visit();
    }
}

void Graph::clear_backwards()
{
    for (auto & key_value : this->m_variables)
    {
        key_value.second->clear_backward_visit();
    }

    for (auto & key_value : this->m_functions)
    {
        key_value.second->clear_backward_visit();
    }
}

void Graph::clear_cache()
{
    for (auto & key_value : this->m_functions)
    {
        key_value.second->clear_cache();
    }
}

std::vector<std::shared_ptr<Variable>> Graph::all_variables() const
{
    std::vector<std::shared_ptr<Variable>> var_list;

    for (auto & key_value : this->m_variables)
    {
        var_list.push_back(key_value.second);
    }

    return var_list;
}

std::vector<std::shared_ptr<Variable>> Graph::all_trainable_variables() const
{
    std::vector<std::shared_ptr<Variable>> var_list;

    for (auto & key_value : this->m_variables)
    {
        if (key_value.second->trainable())
        {
            var_list.push_back(key_value.second);
        }
    }

    return var_list;
}

void Graph::print_node(std::ostream & os, const std::shared_ptr<Function> & func, int depth) const
{
    for (int i = 0; i < depth; ++i)
    {
        os << "  ";
    }
    os << "[" << func->id() << "] <" << func->type() << ">\n";

    for (int i = 0; i < depth; ++i)
    {
        os << "  ";
    }
    os << "inputs: ";
    auto inputs = func->get_inputs();
    for (auto & input : inputs)
    {
        os << "(" << input->id() << ") " << input->data_type()
        << " trainable: " << input->trainable() 
        << " forward visited: " << input->forward_visited()
        << " backward visited: " << input->backward_visited() << " ";
    }
    os << "\n";

    for (int i = 0; i < depth; ++i)
    {
        os << "  ";
    }
    os << "output: ";
    auto output = func->get_output();
    os << "(" << output->id() << ") " << output->data_type()
    << " trainable: " << output->trainable()
    << " forward visited: " << output->forward_visited()
    << " backward visited: " << output->backward_visited() << " ";
    os << "\n" << std::endl;
}

void Graph::print_node_recursive(
    std::ostream & os,
    std::unordered_set<int> & funcs_visited,
    const std::shared_ptr<Function> & func,
    int depth) const
{
    // Stop if this function node is already visited
    if (funcs_visited.count(func->id()))
    {
        return;
    }

    funcs_visited.insert(func->id());

    this->print_node(os, func, depth);

    std::shared_ptr<Variable> output = func->get_output();

    // Find all functions who are receivers of this output
    std::vector<std::shared_ptr<Function>> receivers = output->get_receivers();

    // Do the same backward recursively
    for (auto & receiver : receivers)
    {
        this->print_node_recursive(os, funcs_visited, receiver, depth + 1);
    }    
}

std::string Graph::to_string() const
{
    std::ostringstream o_stream;

    std::unordered_map<int, std::shared_ptr<Function>> funcs_begin_with;

    for (auto & key_value : this->m_inputs)
    {
        auto func_list = key_value.second->get_receivers();

        for (auto & func : func_list)
        {
            if (!funcs_begin_with.count(func->id()))
            {
                funcs_begin_with[func->id()] = func;
            }
        }
    }

    std::unordered_set<int> funcs_visited;

    for (auto & key_value : funcs_begin_with)
    {
        print_node_recursive(o_stream, funcs_visited, key_value.second, 0);
    }

    return o_stream.str();
}

void Graph::set_device(julie::MatrixType mtype)
{
    for (auto & key_value : this->m_variables)
    {
        key_value.second->set_device(mtype);
    }
}

} // namespace op    
} // namespace julie
