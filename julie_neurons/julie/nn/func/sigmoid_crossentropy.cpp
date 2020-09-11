#include "sigmoid_crossentropy.hpp"


julie::nn::func::Sigmoid_CrossEntropy::Sigmoid_CrossEntropy(
    const std::shared_ptr<op::Variable> & input_ptr,
    const std::shared_ptr<op::Variable> & target_ptr,
    lint axis)
    :
    op::Function {},
    m_sc {std::make_unique<la::Sigmoid_CrossEntropy<double>>(axis)},
    m_diff {}
{
    // double var = static_cast<double>(100) / w_mat.shape().size();
    // w_mat.gaussian_random(0, var);

    this->m_inputs.push_back(input_ptr);
    this->m_inputs.push_back(target_ptr);

    input_ptr->add_receiver(this);
    target_ptr->add_receiver(this);

    this->m_output = std::make_shared<var::Tensor<double>> ();
    this->m_output->set_provider(this);
}

julie::nn::func::Sigmoid_CrossEntropy::Sigmoid_CrossEntropy(const Sigmoid_CrossEntropy & other)
    :
    op::Function {other},
    m_sc {other.m_sc->clone()},
    m_diff {other.m_diff}
{}

julie::nn::func::Sigmoid_CrossEntropy::Sigmoid_CrossEntropy(Sigmoid_CrossEntropy && other)
    :
    op::Function {other},
    m_sc {std::move(other.m_sc)},
    m_diff {std::move(other.m_diff)}
{}

julie::nn::func::Sigmoid_CrossEntropy & julie::nn::func::Sigmoid_CrossEntropy::operator = (const Sigmoid_CrossEntropy & other)
{
    op::Function::operator = (other);
    this->m_sc = other.m_sc->clone();
    this->m_diff = other.m_diff;

    return *this;
}

julie::nn::func::Sigmoid_CrossEntropy & julie::nn::func::Sigmoid_CrossEntropy::operator = (Sigmoid_CrossEntropy && other)
{
    op::Function::operator = (other);
    this->m_sc = std::move(other.m_sc);
    this->m_diff = std::move(other.m_diff);

    return *this;
}

void julie::nn::func::Sigmoid_CrossEntropy::forward()
{
    var::Tensor<double> *input_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[0].get());
    var::Tensor<double> *target_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[1].get());

    std::shared_ptr<la::DMatrix<double>> input_mat_ptr = input_ptr->val();
    std::shared_ptr<la::DMatrix<double>> target_mat_ptr = target_ptr->val();

    var::Tensor<double> *output_ptr = dynamic_cast<var::Tensor<double>*>(this->m_output.get());

    output_ptr->val(this->m_sc->operator()(this->m_diff, *target_mat_ptr, *input_mat_ptr));
    output_ptr->grad(la::DMatrix<double> {1, input_mat_ptr->shape()});
}

void julie::nn::func::Sigmoid_CrossEntropy::backward()
{
    var::Tensor<double> *output_ptr = dynamic_cast<var::Tensor<double>*>(this->m_output.get());
    std::shared_ptr<la::DMatrix<double>> out_grad = output_ptr->grad();

    var::Tensor<double> *input_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[0].get());
    var::Tensor<double> *target_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[1].get());


    if (input_ptr->needs_grad())
    {
        // Do chain rule for the input
        input_ptr->grad(this->m_diff);
    }

    if (target_ptr->needs_grad())
    {
        // We DO NOT do chain rule for the target variable
    }
}
