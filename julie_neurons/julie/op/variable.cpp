#include "variable.hpp"

julie::op::Variable::Variable ()
    :
    m_trainable {false},
    m_needs_grad {false},
    m_provider {nullptr},
    m_receivers {}
{}

julie::op::Variable::Variable (const Variable & other)
    :
    m_trainable {other.m_trainable},
    m_needs_grad {other.m_needs_grad},
    m_provider {nullptr},
    m_receivers {}
{}

julie::op::Variable::Variable (Variable && other)
    :
    m_trainable {other.m_trainable},
    m_needs_grad {other.m_needs_grad},
    m_provider {nullptr},
    m_receivers {}
{}

julie::op::Variable & julie::op::Variable::operator = (const Variable & other)
{
    this->m_trainable = other.m_trainable;
    this->m_needs_grad = other.m_needs_grad;

    return *this;
}


julie::op::Variable & julie::op::Variable::operator = (Variable && other)
{
    this->m_trainable = other.m_trainable;
    this->m_needs_grad = other.m_needs_grad;

    return *this;
}

bool julie::op::Variable::trainable() const
{
    return this->m_trainable;
}

void julie::op::Variable::trainable(bool trainable)
{
    this->m_trainable = trainable;
}

bool julie::op::Variable::needs_grad() const
{
    return this->m_needs_grad;
}

void julie::op::Variable::needs_grad(bool grad)
{
    this->m_needs_grad = true;
}

julie::op::Function* julie::op::Variable::get_provider() const
{
    return this->m_provider;
}

std::vector<julie::op::Function*> julie::op::Variable::get_receivers() const
{
    return this->m_receivers;
}

void julie::op::Variable::set_provider(Function *provider)
{
    this->m_provider = provider;
}

void julie::op::Variable::add_receiver(Function *receiver)
{
    this->m_receivers.push_back(receiver);
}

