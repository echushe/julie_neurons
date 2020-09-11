#include "matmul.hpp"


julie::nn::func::MatMul::MatMul(const std::shared_ptr<op::Variable> & l_ptr, const std::shared_ptr<op::Variable> & r_ptr)
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

julie::nn::func::MatMul::MatMul(const MatMul & other)
    : op::Function {other}
{}

julie::nn::func::MatMul::MatMul(MatMul && other)
    : op::Function {other}
{}

julie::nn::func::MatMul & julie::nn::func::MatMul::operator = (const MatMul & other)
{
    op::Function::operator = (other);

    return *this;
}

julie::nn::func::MatMul & julie::nn::func::MatMul::operator = (MatMul && other)
{
    op::Function::operator = (other);

    return *this;
}

void julie::nn::func::MatMul::forward()
{
    var::Tensor<double> *l_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[0].get());
    var::Tensor<double> *r_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[1].get());

    std::shared_ptr<la::DMatrix<double>> l_mat_ptr = l_ptr->val();
    std::shared_ptr<la::DMatrix<double>> r_mat_ptr = r_ptr->val();

    var::Tensor<double> *output_ptr = dynamic_cast<var::Tensor<double>*>(this->m_output.get());
    output_ptr->val(la::matmul(*l_mat_ptr, *r_mat_ptr));
}

void julie::nn::func::MatMul::backward()
{
    var::Tensor<double> *output_ptr = dynamic_cast<var::Tensor<double>*>(this->m_output.get());
    std::shared_ptr<la::DMatrix<double>> out_grad = output_ptr->grad();

    var::Tensor<double> *l_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[0].get());
    var::Tensor<double> *r_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[1].get());

    std::shared_ptr<la::DMatrix<double>> l_mat_ptr = l_ptr->val();
    std::shared_ptr<la::DMatrix<double>> r_mat_ptr = r_ptr->val();

    // std::cout << *l_mat_ptr << std::endl;
    // std::cout << *r_mat_ptr << std::endl;
    // std::cout << *out_grad << std::endl;

    if (l_ptr->needs_grad())
    {
        // Do chain rule for left hand side
        l_ptr->grad(la::matmul(*out_grad, r_mat_ptr->get_transpose(r_mat_ptr->shape().dim() - 1)));
    }

    if (r_ptr->needs_grad())
    {
        // Do chain rule for right hand side
        r_ptr->grad(la::matmul(l_mat_ptr->get_transpose(1), *out_grad));
    }
}
