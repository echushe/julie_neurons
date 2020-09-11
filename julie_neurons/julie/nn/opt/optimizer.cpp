#include "optimizer.hpp"

julie::nn::opt::Optimizer::Optimizer (const std::vector<std::shared_ptr<op::Variable>> & params)
    : m_params {params}
{}

julie::nn::opt::Optimizer::Optimizer (const Optimizer & other)
    : m_params {other.m_params}
{}

julie::nn::opt::Optimizer::Optimizer (Optimizer && other)
    : m_params {std::move(other.m_params)}
{}

julie::nn::opt::Optimizer & julie::nn::opt::Optimizer::operator = (const Optimizer & other)
{
    this->m_params = other.m_params;
}

julie::nn::opt::Optimizer & julie::nn::opt::Optimizer::operator = (Optimizer && other)
{
    this->m_params = std::move(other.m_params);
}

