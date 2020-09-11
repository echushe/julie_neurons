#pragma once
#include "variable.hpp"

#include <vector>
#include <memory>

namespace julie
{

namespace op
{

class Function
{
public:

    Function() {}
    Function(const Function & other);
    Function(Function && other);
    
    Function & operator = (const Function & other);
    Function & operator = (Function && other);

    virtual void forward() = 0;
    virtual void backward() = 0;

    std::vector<std::shared_ptr<Variable>> get_inputs() const;
    std::shared_ptr<Variable> get_output() const;

    // virtual void set_inputs(const std::vector<std::shared_ptr<Variable>> & inputs) = 0;
    // virtual void set_output(const std::shared_ptr<Variable> & output) = 0;

    bool forward_precheck() const;
    bool backward_precheck() const;

/*
public:
    virtual ~Function() {};
*/

protected:

    std::vector<std::shared_ptr<Variable>> m_inputs;
    std::shared_ptr<Variable> m_output;
};

} // namespace op
} // namespace julie