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

class TanH : public op::Function
{
public:
    TanH(const std::shared_ptr<op::Variable> & t_ptr);
    
    TanH(const TanH & other);
    TanH(TanH && other);

    TanH & operator = (const TanH & other);
    TanH & operator = (TanH && other);

public:
    virtual void forward();
    virtual void backward();

private:
    std::unique_ptr<la::TanH<double>> m_tanh;
    la::DMatrix<double> m_diff;

};


} // namespace func
} // namespace nn
} // namespace julie
