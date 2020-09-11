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

class ArcTan : public op::Function
{
public:
    ArcTan(const std::shared_ptr<op::Variable> & t_ptr);
    
    ArcTan(const ArcTan & other);
    ArcTan(ArcTan && other);

    ArcTan & operator = (const ArcTan & other);
    ArcTan & operator = (ArcTan && other);

public:
    virtual void forward();
    virtual void backward();

private:
    std::unique_ptr<la::ArcTan<double>> m_arctan;
    la::DMatrix<double> m_diff;

};


} // namespace func
} // namespace nn
} // namespace julie
