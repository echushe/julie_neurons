#include "sigmoid.hpp"


julie::nn::func::Sigmoid::Sigmoid(const std::shared_ptr<op::Variable> & t_ptr)
    :
    op::Function {},
    m_sigmoid {std::make_unique<la::Sigmoid<double>>()},
    m_diff {}
{
    // double var = static_cast<double>(100) / w_mat.shape().size();
    // w_mat.gaussian_random(0, var);

    this->m_inputs.push_back(t_ptr);
    t_ptr->add_receiver(this);

    this->m_output = std::make_shared<var::Tensor<double>> ();
    this->m_output->set_provider(this);
}

julie::nn::func::Sigmoid::Sigmoid(const Sigmoid & other)
    :
    op::Function {other},
    m_sigmoid {std::make_unique<la::Sigmoid<double>>(*(other.m_sigmoid))}
{}

julie::nn::func::Sigmoid::Sigmoid(Sigmoid && other)
    :
    op::Function {other},
    m_sigmoid {std::move(other.m_sigmoid)}
{}

julie::nn::func::Sigmoid & julie::nn::func::Sigmoid::operator = (const Sigmoid & other)
{
    op::Function::operator = (other);
    this->m_sigmoid = std::make_unique<la::Sigmoid<double>>(*(other.m_sigmoid));

    return *this;
}

julie::nn::func::Sigmoid & julie::nn::func::Sigmoid::operator = (Sigmoid && other)
{
    op::Function::operator = (other);
    this->m_sigmoid = std::move(other.m_sigmoid);

    return *this;
}

void julie::nn::func::Sigmoid::forward()
{
    var::Tensor<double> *input_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[0].get());
    std::shared_ptr<la::DMatrix<double>> t_mat_ptr = input_ptr->val();

    var::Tensor<double> *output_ptr = dynamic_cast<var::Tensor<double>*>(this->m_output.get());

    la::DMatrix<double> output_mat;

    if (input_ptr->needs_grad())
    {
        this->m_sigmoid->operator()(output_mat, this->m_diff, *t_mat_ptr);
    }
    else
    {
        this->m_sigmoid->operator()(output_mat, *t_mat_ptr);
    }

    output_ptr->val(std::move(output_mat));
}

void julie::nn::func::Sigmoid::backward()
{
    var::Tensor<double> *output_ptr = dynamic_cast<var::Tensor<double>*>(this->m_output.get());
    std::shared_ptr<la::DMatrix<double>> out_grad = output_ptr->grad();

    var::Tensor<double> *input_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[0].get());

    std::shared_ptr<la::DMatrix<double>> t_mat_ptr = input_ptr->val();

    if (input_ptr->needs_grad())
    {
        // Do chain rule for the input
        input_ptr->grad(la::multiply(this->m_diff, *out_grad));
    }
}
