#include "add.hpp"


julie::nn::func::Add::Add(const std::shared_ptr<op::Variable> & l_ptr, const std::shared_ptr<op::Variable> & r_ptr)
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

julie::nn::func::Add::Add(const Add & other)
    : op::Function {other}
{}

julie::nn::func::Add::Add(Add && other)
    : op::Function {other}
{}

julie::nn::func::Add & julie::nn::func::Add::operator = (const Add & other)
{
    op::Function::operator = (other);

    return *this;
}

julie::nn::func::Add & julie::nn::func::Add::operator = (Add && other)
{
    op::Function::operator = (other);

    return *this;
}

void julie::nn::func::Add::forward()
{
    var::Tensor<double> *l_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[0].get());
    var::Tensor<double> *r_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[1].get());

    std::shared_ptr<la::DMatrix<double>> l_mat_ptr = l_ptr->val();
    std::shared_ptr<la::DMatrix<double>> r_mat_ptr = r_ptr->val();

    var::Tensor<double> *output_ptr = dynamic_cast<var::Tensor<double>*>(this->m_output.get());
    output_ptr->val(la::broadcast_add(*l_mat_ptr, *r_mat_ptr));
}

void julie::nn::func::Add::backward()
{
    var::Tensor<double> *output_ptr = dynamic_cast<var::Tensor<double>*>(this->m_output.get());
    std::shared_ptr<la::DMatrix<double>> out_grad = output_ptr->grad();

    var::Tensor<double> *l_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[0].get());
    var::Tensor<double> *r_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[1].get());

    // std::cout << *out_grad << std::endl;

    if (l_ptr->needs_grad())
    {
        // Do chain rule for left hand side
        l_ptr->grad(*out_grad);
    }

    if (r_ptr->needs_grad())
    {
        std::shared_ptr<la::DMatrix<double>> r_mat_ptr = r_ptr->val();
        lint r_mat_size = r_mat_ptr->shape().size();

        la::DMatrix<double> fused = *out_grad;
        while (fused.shape().size() > r_mat_size)
        {
            fused = fused.get_fused(0);
        }

        // std::cout << fused << std::endl;

        // Do chain rule for right hand side
        r_ptr->grad(fused);
    }
}
