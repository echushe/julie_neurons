#pragma once
#include "function.hpp"
#include "tensor.hpp"

#include "Conv2d.hpp"

namespace julie
{
namespace nn
{
namespace func
{

class Conv2d : public op::Function
{
public:
    Conv2d(
        const std::shared_ptr<op::Variable> & feature,
        const std::shared_ptr<op::Variable> & weight,
        const std::shared_ptr<op::Variable> & bias,
        lint pad_h,
        lint pad_w,
        lint stride_h,
        lint stride_w);
    
    Conv2d(const Conv2d & other);
    Conv2d(Conv2d && other);

    Conv2d & operator = (const Conv2d & other);
    Conv2d & operator = (Conv2d && other);

public:
    virtual void forward();
    virtual void backward();

private:

    std::unique_ptr<la::Conv2d<double>> m_conv2d;
    
    la::DMatrix<double> m_b_diff;

};


} // namespace func
} // namespace nn
} // namespace julie
