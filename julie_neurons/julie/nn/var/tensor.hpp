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
class Tensor : public op::Variable
{
public:
    Tensor();
    Tensor(const la::DMatrix<DT> & val);
    Tensor(la::DMatrix<DT> && val);

    Tensor (const Tensor & other);
    Tensor (Tensor && other);

    Tensor & operator = (const Tensor & other);
    Tensor & operator = (Tensor && other);

public:
    virtual void clear_grad();
    virtual void clear_val();

    virtual bool has_grad() const;
    virtual bool has_val() const;

    virtual VariableType data_type() const;

public:

    inline std::shared_ptr<la::DMatrix<DT>> val() const;
    inline std::shared_ptr<la::DMatrix<DT>> grad() const;

    inline void val(const la::DMatrix<DT> & val);
    inline void val(la::DMatrix<DT> && val);

    inline void grad(const la::DMatrix<DT> & grad);
    inline void grad(la::DMatrix<DT> && grad);

private:
    std::shared_ptr<la::DMatrix<DT>> m_val;
    std::shared_ptr<la::DMatrix<DT>> m_grad;
};

} // namespace var
} // namespace nn
} // namespace julie

template <typename DT>
julie::nn::var::Tensor<DT>::Tensor()
    :
    op::Variable {},
    m_val {nullptr},
    m_grad {nullptr}
{}

template <typename DT>
julie::nn::var::Tensor<DT>::Tensor(const la::DMatrix<DT> & val)
    :
    op::Variable {},
    m_val {std::make_shared<la::DMatrix<DT>>(val)},
    m_grad {nullptr}
{}

template <typename DT>
julie::nn::var::Tensor<DT>::Tensor(la::DMatrix<DT> && val)
    :
    op::Variable {},
    m_val {std::make_shared<la::DMatrix<DT>>(std::move(val))},
    m_grad {nullptr}
{}

template <typename DT>
julie::nn::var::Tensor<DT>::Tensor(const Tensor<DT> & other)
    :
    op::Variable {other},
    m_val {std::make_shared<la::DMatrix<DT>>(*(other.m_val))},
    m_grad {std::make_shared<la::DMatrix<DT>>(*(other.m_grad))}
{}

template <typename DT>
julie::nn::var::Tensor<DT>::Tensor(Tensor<DT> && other)
    :
    op::Variable {other},
    m_val {std::move(other.m_val)},
    m_grad {std::move(other.m_grad)}
{}

template <typename DT>
julie::nn::var::Tensor<DT> & julie::nn::var::Tensor<DT>::operator = (const Tensor<DT> & other)
{
    // Call copy assignment of base class
    op::Variable::operator = (other);

    this->m_val = std::make_shared<la::DMatrix<DT>> (*(other.m_val));
    this->m_grad = std::make_shared<la::DMatrix<DT>> (*(other.m_grad));

    return *this;
}

template <typename DT>
julie::nn::var::Tensor<DT> & julie::nn::var::Tensor<DT>::operator = (Tensor<DT> && other)
{
    // Call move assignment of base class
    op::Variable::operator = (other);

    this->m_val = std::move(other.m_val);
    this->m_grad = std::move(other.m_grad);

    return *this;
}

template <typename DT>
void julie::nn::var::Tensor<DT>::clear_grad()
{
    this->m_grad.reset();
}

template <typename DT>
void julie::nn::var::Tensor<DT>::clear_val()
{
    this->m_val.reset();
}

template <typename DT>
bool julie::nn::var::Tensor<DT>::has_grad() const
{
    if (this->m_grad)
    {
        return true;
    }

    return false;
}

template <typename DT>
bool julie::nn::var::Tensor<DT>::has_val() const
{
    if (this->m_val)
    {
        return true;
    }

    return false;
}

template <typename DT>
julie::op::Variable::VariableType julie::nn::var::Tensor<DT>::data_type() const
{
    return op::Variable::VariableType::TENSOR;
}

template <typename DT>
std::shared_ptr<julie::la::DMatrix<DT>> julie::nn::var::Tensor<DT>::val() const
{
    return this->m_val;
}

template <typename DT>
std::shared_ptr<julie::la::DMatrix<DT>> julie::nn::var::Tensor<DT>::grad() const
{
    return this->m_grad;
}

template <typename DT>
void julie::nn::var::Tensor<DT>::val(const la::DMatrix<DT> & val)
{
    this->m_val = std::make_shared<la::DMatrix<DT>>(val);
}

template <typename DT>
void julie::nn::var::Tensor<DT>::val(la::DMatrix<DT> && val)
{
    this->m_val = std::make_shared<la::DMatrix<DT>>(std::move(val));
}

template <typename DT>
void julie::nn::var::Tensor<DT>::grad(const la::DMatrix<DT> & grad)
{
    this->m_grad = std::make_shared<la::DMatrix<DT>>(grad);
}

template <typename DT>
void julie::nn::var::Tensor<DT>::grad(la::DMatrix<DT> && grad)
{
    this->m_grad = std::make_shared<la::DMatrix<DT>>(std::move(grad));
}
