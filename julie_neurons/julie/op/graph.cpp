#include "graph.hpp"

void julie::op::Graph::new_function(const std::shared_ptr<Function> & func)
{
    std::vector<std::shared_ptr<Variable>> inputs = func->get_inputs();
    for (std::shared_ptr<Variable> input : inputs)
    {
        if (!this->m_variables.count(input))
        {
            this->m_variables.insert(input);
        }
    }

    if (!this->m_functions.count(func))
    {
        this->m_functions.insert(func);
    }

    std::shared_ptr<Variable> output = func->get_output();
    if (!this->m_variables.count(output))
    {
        this->m_variables.insert(output);
    }
}

void julie::op::Graph::func_forward(const std::shared_ptr<Function> & func)
{
    this->func_forward(func.get());
}

void julie::op::Graph::func_forward(Function *func)
{
    // Get all variables required by this function
    std::vector<std::shared_ptr<Variable>> inputs_required = func->get_inputs();
    // Container of predecessor functions
    std::vector<Function*> funcs_to_run;

    // In case there are variables that are not initialized
    for (std::shared_ptr<Variable> input : inputs_required)
    {
        if (!input->has_val() && input->get_provider())
        {
            funcs_to_run.push_back(input->get_provider());
        }
    }

    // Run required predecessor functions
    for (Function *f : funcs_to_run)
    {
        this->func_forward(f);
    }

    // Run this function
    if (func->forward_precheck())
    {
        func->forward();
    }
}

void julie::op::Graph::destroy_backward_route()
{
    for (std::shared_ptr<Variable> var_ptr : this->m_variables)
    {
        var_ptr->needs_grad(false);
    }
}

void julie::op::Graph::pave_backward_route(const std::shared_ptr<Variable> & the_trainable)
{
    // The end point of back propagation should be trainable
    if (!the_trainable->trainable())
    {
        return;
    }

    // "Needs grad is true" means backward route is already paved
    if (the_trainable->needs_grad())
    {
        return;
    }

    the_trainable->needs_grad(true);

    // Find all receivers of this trainable variable
    std::vector<Function*> receivers = the_trainable->get_receivers();

    // Pave backward route for all receivers
    for (Function *receiver : receivers)
    {
        this->func_pave_backward_route(receiver);
    }
}

void julie::op::Graph::func_pave_backward_route(Function *func)
{
    // Output of this function needs gradient
    std::shared_ptr<Variable> output = func->get_output();
    
    // "Needs grad is true" means backward route is already paved
    if (output->needs_grad())
    {
        return;
    }

    output->needs_grad(true);

    // Find all receivers of this trainable variable
    std::vector<Function*> receivers = output->get_receivers();

    // Pave backward route for all receivers
    for (Function *receiver : receivers)
    {
        this->func_pave_backward_route(receiver);
    }
}

void julie::op::Graph::func_backward(Function *func)
{
    std::shared_ptr<Variable> output = func->get_output();

    // If output needs gradient data
    if (output->needs_grad())
    {
        // If output does not have gradient data
        if (!output->has_grad())
        {
            // Find all functions who are receivers of this output
            std::vector<Function*> receivers = output->get_receivers();

            // Do the same backward recursively
            for (Function *receiver : receivers)
            {
                this->func_backward(receiver);
            }
        }

        // Do back propagation for this function
        if (func->backward_precheck())
        {
            func->backward();
        }
    }
}

void julie::op::Graph::func_backward(const std::shared_ptr<Function> & func)
{
    this->func_backward(func.get());
}

void julie::op::Graph::clear_forwards()
{
    for (std::shared_ptr<Variable> var_ptr : this->m_variables)
    {
        // Clear values of non-trainable variables
        if (!var_ptr->trainable())
        {
            var_ptr->clear_val();
        }
    }
}

void julie::op::Graph::clear_backwards()
{
    for (std::shared_ptr<Variable> var_ptr : this->m_variables)
    {
        var_ptr->clear_grad();
    }
}
