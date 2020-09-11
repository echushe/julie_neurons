#pragma once
#include "variable.hpp"
#include "DMatrix.hpp"


namespace julie
{
namespace nn
{
namespace var
{

template <typename DT>
class Scalar : public op::Variable
{
public:
    Scalar();
    Scalar(DT val);

    Scalar (const Scalar & other);
    Scalar (Scalar && other);

    Scalar & operator = (const Scalar & other);
    Scalar & operator = (Scalar && other);

public:
    virtual void clear_grad();
    virtual void clear_val();

    virtual bool has_grad() const;
    virtual bool has_val() const;

    virtual VariableType data_type() const;

public:

    inline std::shared_ptr<DT> val() const;
    inline std::shared_ptr<DT> grad() const;

    inline void val(DT val);
    inline void grad(DT grad);

private:
    std::shared_ptr<DT> m_val;
    std::shared_ptr<DT> m_grad;
};

} // namespace var
} // namespace nn
} // namespace julie

template <typename DT>
julie::nn::var::Scalar<DT>::Scalar()
    :
    op::Variable {},
    m_val {nullptr},
    m_grad {nullptr}
{}

template <typename DT>
julie::nn::var::Scalar<DT>::Scalar(DT val)
    :
    op::Variable {},
    m_val {std::make_shared<DT>(val)},
    m_grad {nullptr}
{}

template <typename DT>
julie::nn::var::Scalar<DT>::Scalar(const Scalar<DT> & other)
    :
    op::Variable {other},
    m_val {std::make_shared<DT>(*(other.m_val))},
    m_grad {std::make_shared<DT>(*(other.m_grad))}
{}

template <typename DT>
julie::nn::var::Scalar<DT>::Scalar(Scalar<DT> && other)
    :
    op::Variable {other},
    m_val {std::move(other.m_val)},
    m_grad {std::move(other.m_grad)}
{}

template <typename DT>
julie::nn::var::Scalar<DT> & julie::nn::var::Scalar<DT>::operator = (const Scalar<DT> & other)
{
    // Call copy assignment of base class
    op::Variable::operator = (other);

    this->m_val = std::make_shared<DT> (*(other.m_val));
    this->m_grad = std::make_shared<DT> (*(other.m_grad));

    return *this;
}

template <typename DT>
julie::nn::var::Scalar<DT> & julie::nn::var::Scalar<DT>::operator = (Scalar<DT> && other)
{
    // Call move assignment of base class
    op::Variable::operator = (other);

    this->m_val = std::move(other.m_val);
    this->m_grad = std::move(other.m_grad);

    return *this;
}

template <typename DT>
void julie::nn::var::Scalar<DT>::clear_grad()
{
    this->m_grad.reset();
}

template <typename DT>
void julie::nn::var::Scalar<DT>::clear_val()
{
    this->m_val.reset();
}

template <typename DT>
bool julie::nn::var::Scalar<DT>::has_grad() const
{
    if (this->m_grad)
    {
        return true;
    }

    return false;
}

template <typename DT>
bool julie::nn::var::Scalar<DT>::has_val() const
{
    if (this->m_val)
    {
        return true;
    }

    return false;
}

template <typename DT>
julie::op::Variable::VariableType julie::nn::var::Scalar<DT>::data_type() const
{
    return op::Variable::VariableType::SCALAR;
}

template <typename DT>
std::shared_ptr<DT> julie::nn::var::Scalar<DT>::val() const
{
    return this->m_val;
}

template <typename DT>
std::shared_ptr<DT> julie::nn::var::Scalar<DT>::grad() const
{
    return this->m_grad;
}

template <typename DT>
void julie::nn::var::Scalar<DT>::val(DT val)
{
    this->m_val = std::make_shared<DT>(val);
}

template <typename DT>
void julie::nn::var::Scalar<DT>::grad(DT grad)
{
    this->m_grad = std::make_shared<DT>(grad);
}
