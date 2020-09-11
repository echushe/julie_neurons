#pragma once
#include "function.hpp"
#include "tensor.hpp"

namespace julie
{
namespace nn
{
namespace func
{

class MatMul : public op::Function
{
public:
    MatMul(const std::shared_ptr<op::Variable> & l_ptr, const std::shared_ptr<op::Variable> & r_ptr);
    
    MatMul(const MatMul & other);
    MatMul(MatMul && other);

    MatMul & operator = (const MatMul & other);
    MatMul & operator = (MatMul && other);

public:
    virtual void forward();
    virtual void backward();

private:

};


} // namespace func
} // namespace nn
} // namespace julie
