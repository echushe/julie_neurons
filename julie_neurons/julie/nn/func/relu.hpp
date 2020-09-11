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

class ReLU : public op::Function
{
public:
    ReLU(const std::shared_ptr<op::Variable> & t_ptr);
    
    ReLU(const ReLU & other);
    ReLU(ReLU && other);

    ReLU & operator = (const ReLU & other);
    ReLU & operator = (ReLU && other);

public:
    virtual void forward();
    virtual void backward();

private:
    std::unique_ptr<la::ReLU<double>> m_relu;
    la::DMatrix<double> m_diff;

};


} // namespace func
} // namespace nn
} // namespace julie
