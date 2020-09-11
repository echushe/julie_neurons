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

class Sigmoid : public op::Function
{
public:
    Sigmoid(const std::shared_ptr<op::Variable> & t_ptr);
    
    Sigmoid(const Sigmoid & other);
    Sigmoid(Sigmoid && other);

    Sigmoid & operator = (const Sigmoid & other);
    Sigmoid & operator = (Sigmoid && other);

public:
    virtual void forward();
    virtual void backward();

private:
    std::unique_ptr<la::Sigmoid<double>> m_sigmoid;
    la::DMatrix<double> m_diff;

};


} // namespace func
} // namespace nn
} // namespace julie
