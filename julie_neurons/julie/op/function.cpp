#include "function.hpp"

julie::op::Function::Function(const Function & other)
{}

julie::op::Function::Function(Function && other)
{}
    
julie::op::Function & julie::op::Function::operator = (const Function & other)
{
    return *this;
}

julie::op::Function & julie::op::Function::operator = (Function && other)
{    
    return *this;
}

std::vector<std::shared_ptr<julie::op::Variable>> julie::op::Function::get_inputs() const
{
    return this->m_inputs;
}

std::shared_ptr<julie::op::Variable> julie::op::Function::get_output() const
{
    return this->m_output;
}

bool julie::op::Function::forward_precheck() const
{
    for (const std::shared_ptr<Variable> & var_ptr : this->m_inputs)
    {
        if (!var_ptr->has_val())
        {
            // throw std::invalid_argument(std::string("Input is expected to have value but it actually does not."));
            return false;
        }
    }

    return true;
}

bool julie::op::Function::backward_precheck() const
{
    // There should exist gradient of the output
    if (!this->m_output->has_grad())
    {
        // throw std::invalid_argument(std::string("Output of this function is expected to have gradient but it actually does not."));
        return false;
    }

    return true;
}