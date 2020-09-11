#include "sgd.hpp"

#include "scalar.hpp"
#include "tensor.hpp"

julie::nn::opt::SGD::SGD (
    const std::vector<std::shared_ptr<op::Variable>> & params,
    double learning_rate,
    double momentum_rate)
    :
    Optimizer {params},
    m_lr {learning_rate},
    m_mr {momentum_rate}
{
    for (size_t i = 0; i < this->m_params.size(); ++i)
    {
        this->m_momentum.push_back(std::shared_ptr<op::Variable>(nullptr));
    }
}


julie::nn::opt::SGD::SGD (const SGD & other)
    :
    Optimizer {other},
    m_lr {other.m_lr},
    m_mr {other.m_mr},
    m_momentum {other.m_momentum}
{}


julie::nn::opt::SGD::SGD (SGD && other)
    :
    Optimizer {other},
    m_lr {other.m_lr},
    m_mr {other.m_mr},
    m_momentum {std::move(other.m_momentum)}
{}


julie::nn::opt::SGD & julie::nn::opt::SGD::operator = (const SGD & other)
{
    Optimizer::operator = (other);
    m_lr = other.m_lr;
    m_mr = other.m_mr;
    this->m_momentum = other.m_momentum;
}

julie::nn::opt::SGD & julie::nn::opt::SGD::operator = (SGD && other)
{
    Optimizer::operator = (other);
    m_lr = other.m_lr;
    m_mr = other.m_mr;

    this->m_momentum = std::move(other.m_momentum);
}

void julie::nn::opt::SGD::tensor_step(const la::DMatrix<double> & gradient, la::DMatrix<double> & momentum, la::DMatrix<double> & weight)
{
    // std::cout << "The gradient is: \n" << gradient << std::endl;
    momentum = this->m_mr * momentum + (1.0 - this->m_mr) * gradient;
    weight -= this->m_lr * momentum;

    // weight -= this->m_lr * gradient;
}

void julie::nn::opt::SGD::scalar_step(const double & gradient, double & momentum, double & weight)
{
    momentum = this->m_mr * momentum + (1.0 - this->m_mr) * gradient;
    weight -= this->m_lr * momentum;

    // weight -= this->m_lr * gradient;
}

void julie::nn::opt::SGD::step()
{
    for (size_t i = 0; i < this->m_params.size(); ++i)
    {
        std::shared_ptr<op::Variable> param = this->m_params[i];
        std::shared_ptr<op::Variable> momentum = this->m_momentum[i];

        if (param)
        {
            if (param->data_type() == op::Variable::VariableType::TENSOR)
            {
                if (param.get()->has_val() && param.get()->has_grad())
                {
                    std::shared_ptr<la::DMatrix<double>> w_ptr = dynamic_cast<var::Tensor<double>*> (param.get())->val();
                    std::shared_ptr<la::DMatrix<double>> g_ptr = dynamic_cast<var::Tensor<double>*> (param.get())->grad();

                    std::shared_ptr<la::DMatrix<double>> m_ptr;

                    if (!momentum)
                    {
                        momentum = std::make_shared<var::Tensor<double>> (julie::la::DMatrix<double>(0, w_ptr->shape()));
                    }

                    m_ptr = dynamic_cast<var::Tensor<double>*> (momentum.get())->val();
                    
                    this->tensor_step(*g_ptr, *m_ptr, *w_ptr);
                }
            }
            else if (param->data_type() == op::Variable::VariableType::SCALAR)
            {
                if (param.get()->has_val() && param.get()->has_grad())
                {
                    std::shared_ptr<double> w_ptr = dynamic_cast<var::Scalar<double>*> (param.get())->val();
                    std::shared_ptr<double> g_ptr = dynamic_cast<var::Scalar<double>*> (param.get())->grad();

                    std::shared_ptr<double> m_ptr;
                    
                    if (!momentum)
                    {
                        momentum = std::make_shared<var::Scalar<double>> (0);
                    }
                    
                    m_ptr = dynamic_cast<var::Scalar<double>*> (param.get())->val();

                    this->scalar_step(*g_ptr, *m_ptr, *w_ptr);
                }
            }
            else
            {
                /* code */
            }
        }
    }
}