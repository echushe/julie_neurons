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

class SoftMax_CrossEntropy : public op::Function
{
public:
    SoftMax_CrossEntropy(const std::shared_ptr<op::Variable> & input_ptr, const std::shared_ptr<op::Variable> & target_ptr, lint axis);
    
    SoftMax_CrossEntropy(const SoftMax_CrossEntropy & other);
    SoftMax_CrossEntropy(SoftMax_CrossEntropy && other);

    SoftMax_CrossEntropy & operator = (const SoftMax_CrossEntropy & other);
    SoftMax_CrossEntropy & operator = (SoftMax_CrossEntropy && other);

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
