#pragma once
#include "function.hpp"
#include "tensor.hpp"

namespace julie
{
namespace nn
{
namespace func
{

class Add : public op::Function
{
public:
    Add(const std::shared_ptr<op::Variable> & l_ptr, const std::shared_ptr<op::Variable> & r_ptr);
    
    Add(const Add & other);
    Add(Add && other);

    Add & operator = (const Add & other);
    Add & operator = (Add && other);

public:
    virtual void forward();
    virtual void backward();

private:

};


} // namespace func
} // namespace nn
} // namespace julie
