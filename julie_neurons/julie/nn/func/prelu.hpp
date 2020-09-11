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

class PReLU : public op::Function
{
public:
    PReLU(const std::shared_ptr<op::Variable> & t_ptr, const std::shared_ptr<op::Variable> & alpha);
    
    PReLU(const PReLU & other);
    PReLU(PReLU && other);

    PReLU & operator = (const PReLU & other);
    PReLU & operator = (PReLU && other);

public:
    virtual void forward();
    virtual void backward();

private:
    std::unique_ptr<la::PReLU<double>> m_relu;
    la::DMatrix<double> m_diff;
    la::DMatrix<double> m_alpha_diff;

};


} // namespace func
} // namespace nn
} // namespace julie
