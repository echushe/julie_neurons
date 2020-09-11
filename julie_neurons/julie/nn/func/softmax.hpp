#pragma once
#include "function.hpp"
#include "tensor.hpp"
#include "scalar.hpp"

#include "Activations.hpp"

namespace julie
{
namespace nn
{
namespace func
{

class SoftMax : public op::Function
{
public:
    SoftMax(const std::shared_ptr<op::Variable> & t_ptr, lint axis);
    
    SoftMax(const SoftMax & other);
    SoftMax(SoftMax && other);

    SoftMax & operator = (const SoftMax & other);
    SoftMax & operator = (SoftMax && other);

public:
    virtual void forward();
    virtual void backward();

private:
    std::unique_ptr<la::Softmax<double>> m_softmax;
    la::DMatrix<double> m_diff;

};


} // namespace func
} // namespace nn
} // namespace julie
