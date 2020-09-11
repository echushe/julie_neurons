#include "scale.hpp"


julie::nn::func::Scale::Scale(const std::shared_ptr<op::Variable> & l_ptr, const std::shared_ptr<op::Variable> & r_ptr)
    :
    op::Function {}
{
    // double var = static_cast<double>(100) / w_mat.shape().size();
    // w_mat.gaussian_random(0, var);

    this->m_inputs.push_back(l_ptr);
    this->m_inputs.push_back(r_ptr);

    l_ptr->add_receiver(this);
    r_ptr->add_receiver(this);

    this->m_output = std::make_shared<var::Tensor<double>> ();
    this->m_output->set_provider(this);
}

julie::nn::func::Scale::Scale(const Scale & other)
    : op::Function {other}
{}

julie::nn::func::Scale::Scale(Scale && other)
    : op::Function {other}
{}

julie::nn::func::Scale & julie::nn::func::Scale::operator = (const Scale & other)
{
    op::Function::operator = (other);

    return *this;
}

julie::nn::func::Scale & julie::nn::func::Scale::operator = (Scale && other)
{
    op::Function::operator = (other);

    return *this;
}

void julie::nn::func::Scale::forward()
{
    var::Tensor<double> *l_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[0].get());
    var::Scalar<double> *r_ptr = dynamic_cast<var::Scalar<double>*>(this->m_inputs[1].get());

    std::shared_ptr<la::DMatrix<double>> l_mat_ptr = l_ptr->val();
    std::shared_ptr<double> scalar_ptr = r_ptr->val();

    var::Tensor<double> *output_ptr = dynamic_cast<var::Tensor<double>*>(this->m_output.get());
    output_ptr->val(*l_mat_ptr * *scalar_ptr);
}

void julie::nn::func::Scale::backward()
{
    var::Tensor<double> *output = dynamic_cast<var::Tensor<double>*>(this->m_output.get());
    std::shared_ptr<la::DMatrix<double>> out_grad = output->grad();

    var::Tensor<double> *l_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[0].get());
    var::Scalar<double> *r_ptr = dynamic_cast<var::Scalar<double>*>(this->m_inputs[1].get());

    std::shared_ptr<la::DMatrix<double>> l_mat_ptr = l_ptr->val();
    std::shared_ptr<double> scale_ptr = r_ptr->val();

    if (l_ptr->needs_grad())
    {
        // Do chain rule for left hand side
        l_ptr->grad(*out_grad * *scale_ptr);
    }

    if (r_ptr->needs_grad())
    {
        // Do chain rule for right hand side
        r_ptr->grad(la::dot_product(*l_mat_ptr, *out_grad) / l_mat_ptr->shape().size());
    }
}
