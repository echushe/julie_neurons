#include "arctan.hpp"


julie::nn::func::ArcTan::ArcTan(const std::shared_ptr<op::Variable> & t_ptr)
    :
    op::Function {},
    m_arctan {std::make_unique<la::ArcTan<double>>()},
    m_diff {}
{
    // double var = static_cast<double>(100) / w_mat.shape().size();
    // w_mat.gaussian_random(0, var);

    this->m_inputs.push_back(t_ptr);
    t_ptr->add_receiver(this);

    this->m_output = std::make_shared<var::Tensor<double>> ();
    this->m_output->set_provider(this);
}

julie::nn::func::ArcTan::ArcTan(const ArcTan & other)
    : op::Function {other}
{}

julie::nn::func::ArcTan::ArcTan(ArcTan && other)
    : op::Function {other}
{}

julie::nn::func::ArcTan & julie::nn::func::ArcTan::operator = (const ArcTan & other)
{
    op::Function::operator = (other);

    return *this;
}

julie::nn::func::ArcTan & julie::nn::func::ArcTan::operator = (ArcTan && other)
{
    op::Function::operator = (other);

    return *this;
}

void julie::nn::func::ArcTan::forward()
{
    var::Tensor<double> *input_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[0].get());
    std::shared_ptr<la::DMatrix<double>> t_mat_ptr = input_ptr->val();

    var::Tensor<double> *output_ptr = dynamic_cast<var::Tensor<double>*>(this->m_output.get());

    la::DMatrix<double> output_mat;

    if (input_ptr->needs_grad())
    {
        this->m_arctan->operator()(output_mat, this->m_diff, *t_mat_ptr);
    }
    else
    {
        this->m_arctan->operator()(output_mat, *t_mat_ptr);
    }

    output_ptr->val(std::move(output_mat));
}

void julie::nn::func::ArcTan::backward()
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
