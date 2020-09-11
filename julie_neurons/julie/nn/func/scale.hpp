#pragma once
#include "function.hpp"
#include "tensor.hpp"
#include "scalar.hpp"

namespace julie
{
namespace nn
{
namespace func
{

class Scale : public op::Function
{
public:
    Scale(const std::shared_ptr<op::Variable> & l_ptr, const std::shared_ptr<op::Variable> & r_ptr);
    
    Scale(const Scale & other);
    Scale(Scale && other);

    Scale & operator = (const Scale & other);
    Scale & operator = (Scale && other);

public:
    virtual void forward();
    virtual void backward();

private:

};


} // namespace func
} // namespace nn
} // namespace julie
