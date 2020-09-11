#pragma once
#include "function.hpp"
#include "tensor.hpp"
#include "scalar.hpp"

#include "Losses.hpp"

namespace julie
{
namespace nn
{
namespace func
{

class Sigmoid_CrossEntropy : public op::Function
{
public:
    Sigmoid_CrossEntropy(const std::shared_ptr<op::Variable> & input_ptr, const std::shared_ptr<op::Variable> & target_ptr, lint axis);
    
    Sigmoid_CrossEntropy(const Sigmoid_CrossEntropy & other);
    Sigmoid_CrossEntropy(Sigmoid_CrossEntropy && other);

    Sigmoid_CrossEntropy & operator = (const Sigmoid_CrossEntropy & other);
    Sigmoid_CrossEntropy & operator = (Sigmoid_CrossEntropy && other);

public:
    virtual void forward();
    virtual void backward();

private:
    std::unique_ptr<la::ErrorFunction<double>> m_sc;
    la::DMatrix<double> m_diff;
};


} // namespace func
} // namespace nn
} // namespace julie
