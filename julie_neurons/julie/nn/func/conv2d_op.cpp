#include "conv2d_op.hpp"


julie::nn::func::Conv2d::Conv2d(
    const std::shared_ptr<op::Variable> & feature,
    const std::shared_ptr<op::Variable> & weight,
    const std::shared_ptr<op::Variable> & bias,
    lint pad_h,
    lint pad_w,
    lint stride_h,
    lint stride_w)
    :
    op::Function {},
    m_conv2d {std::make_unique<la::Conv2d<double>>(pad_h, pad_w, stride_h, stride_w)}
{
    // double var = static_cast<double>(100) / w_mat.shape().size();
    // w_mat.gaussian_random(0, var);

    this->m_inputs.push_back(feature);
    this->m_inputs.push_back(weight);
    this->m_inputs.push_back(bias);

    feature->add_receiver(this);
    weight->add_receiver(this);
    bias->add_receiver(this);

    this->m_output = std::make_shared<var::Tensor<double>> ();
    this->m_output->set_provider(this);
}

julie::nn::func::Conv2d::Conv2d(const Conv2d & other)
    :
    op::Function {other},
    m_conv2d {std::make_unique<la::Conv2d<double>>(*(other.m_conv2d))}
{}

julie::nn::func::Conv2d::Conv2d(Conv2d && other)
    :
    op::Function {other},
    m_conv2d {std::move(other.m_conv2d)}
{}

julie::nn::func::Conv2d & julie::nn::func::Conv2d::operator = (const Conv2d & other)
{
    op::Function::operator = (other);
    this->m_conv2d = std::make_unique<la::Conv2d<double>>(*(other.m_conv2d));

    return *this;
}

julie::nn::func::Conv2d & julie::nn::func::Conv2d::operator = (Conv2d && other)
{
    op::Function::operator = (other);
    this->m_conv2d = std::move(other.m_conv2d);

    return *this;
}

void julie::nn::func::Conv2d::forward()
{
    var::Tensor<double> *featrue_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[0].get());
    var::Tensor<double> *weight_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[1].get());
    var::Tensor<double> *bias_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[2].get());

    std::shared_ptr<la::DMatrix<double>> f_mat_ptr = featrue_ptr->val();
    std::shared_ptr<la::DMatrix<double>> w_mat_ptr = weight_ptr->val();
    std::shared_ptr<la::DMatrix<double>> b_mat_ptr = bias_ptr->val();

    var::Tensor<double> *output_ptr = dynamic_cast<var::Tensor<double>*>(this->m_output.get());

    la::DMatrix<double> output_mat = this->m_conv2d->forward(*f_mat_ptr, *w_mat_ptr, *b_mat_ptr);

    output_ptr->val(std::move(output_mat));
}

void julie::nn::func::Conv2d::backward()
{
    var::Tensor<double> *output_ptr = dynamic_cast<var::Tensor<double>*>(this->m_output.get());
    std::shared_ptr<la::DMatrix<double>> out_grad = output_ptr->grad();

    // std::cout << "Output gradient of this conv layer: \n" << *out_grad << std::endl;

    var::Tensor<double> *featrue_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[0].get());
    var::Tensor<double> *weight_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[1].get());
    var::Tensor<double> *bias_ptr = dynamic_cast<var::Tensor<double>*>(this->m_inputs[2].get());
    
    std::shared_ptr<la::DMatrix<double>> f_mat_ptr = featrue_ptr->val();
    std::shared_ptr<la::DMatrix<double>> w_mat_ptr = weight_ptr->val();
    std::shared_ptr<la::DMatrix<double>> b_mat_ptr = bias_ptr->val();

    la::Shape out_grad_sh = out_grad->shape();

    la::DMatrix<double> feature_grad;
    la::DMatrix<double> weight_grad;
    la::DMatrix<double> bias_grad;

    // std::cout << " ----------- Conv2d backward: -------------- " << std::endl;
    // std::cout << "Shape of conv input: " << f_mat_ptr->shape() << std::endl;
    // std::cout << "Shape of conv weight: " << w_mat_ptr->shape() << std::endl;
    // std::cout << "Shape of conv bias: " << b_mat_ptr->shape() << std::endl;

    this->m_conv2d->backward(feature_grad, weight_grad, bias_grad, *out_grad, f_mat_ptr->shape(), *w_mat_ptr, *b_mat_ptr);

    if (featrue_ptr->needs_grad())
    {
        featrue_ptr->grad(std::move(feature_grad));
    }

    if (weight_ptr->needs_grad())
    {
        weight_ptr->grad(std::move(weight_grad));
    }

    if (bias_ptr->needs_grad())
    {
        bias_ptr->grad(std::move(bias_grad));
    }
}
