#include "scalar.hpp"
#include <iostream>

namespace julie
{
namespace nn
{
namespace var
{

template <typename DT>
Scalar<DT>::Scalar()
    :
    op::Variable {},
    m_val {nullptr},
    m_grad {nullptr}
{}
template
Scalar<float>::Scalar();


template <typename DT>
Scalar<DT>::Scalar(DT val)
    :
    op::Variable {},
    m_val {std::make_shared<DT>(val)},
    m_grad {nullptr}
{}
template
Scalar<float>::Scalar(float val);


template <typename DT>
Scalar<DT>::Scalar(const Scalar<DT> & other)
    :
    op::Variable {other},
    m_val {std::make_shared<DT>(*(other.m_val))},
    m_grad {std::make_shared<DT>(*(other.m_grad))}
{}
template
Scalar<float>::Scalar(const Scalar<float> & other);


template <typename DT>
Scalar<DT>::Scalar(Scalar<DT> && other)
    :
    op::Variable {other},
    m_val {std::move(other.m_val)},
    m_grad {std::move(other.m_grad)}
{}
template
Scalar<float>::Scalar(Scalar<float> && other);


template <typename DT>
Scalar<DT> & Scalar<DT>::operator = (const Scalar<DT> & other)
{
    // Call copy assignment of base class
    op::Variable::operator = (other);

    this->m_val = std::make_shared<DT> (*(other.m_val));
    this->m_grad = std::make_shared<DT> (*(other.m_grad));

    return *this;
}
template
Scalar<float> & Scalar<float>::operator = (const Scalar<float> & other);


template <typename DT>
Scalar<DT> & Scalar<DT>::operator = (Scalar<DT> && other)
{
    // Call move assignment of base class
    op::Variable::operator = (other);

    this->m_val = std::move(other.m_val);
    this->m_grad = std::move(other.m_grad);

    return *this;
}
template
Scalar<float> & Scalar<float>::operator = (Scalar<float> && other);


template <typename DT>
julie::op::Variable::VariableType Scalar<DT>::data_type() const
{
    return op::Variable::VariableType::SCALAR;
}
template
julie::op::Variable::VariableType Scalar<float>::data_type() const;


template <typename DT>
std::shared_ptr<DT> Scalar<DT>::val()
{
    if (!this->m_val)
    {
        this->m_val = std::make_shared<DT>(0);
    }

    return this->m_val;
}
template
std::shared_ptr<float> Scalar<float>::val();


template <typename DT>
std::shared_ptr<DT> Scalar<DT>::grad()
{
    if (!this->m_grad)
    {
        // There are no receivers for this tensor, 
        // Set gradient to one
        if (this->m_receivers.empty())
        {
            this->m_grad = std::make_shared<DT>(1);
        }
        else // There are receivers for this tensor, init gradient to zero
        {
            this->m_grad = std::make_shared<DT>(0);
        }
    }
    
    return this->m_grad;
}
template
std::shared_ptr<float> Scalar<float>::grad();


template <typename DT>
void Scalar<DT>::val(DT val)
{
    this->m_val = std::make_shared<DT>(val);
}
template
void Scalar<float>::val(float val);


template <typename DT>
void Scalar<DT>::add_grad(DT grad)
{
    if (this->m_grad)
    {
        *(this->m_grad) += grad;
    }
    else
    {
        this->m_grad = std::make_shared<DT>(grad);
    }
}
template
void Scalar<float>::add_grad(float grad);


template <typename DT>
void Scalar<DT>::set_grad_to_zero()
{
    if (this->m_grad)
    {
        *(this->m_grad) = 0;
    }
}
template
void Scalar<float>::set_grad_to_zero();


template <typename DT>
void Scalar<DT>::forward_visit_finish()
{
    if (!this->m_val)
    {
        throw std::invalid_argument(std::string{"Value of this scalar should exist in "} + std::string{__FUNCTION__} );
    }

    if (!this->m_grad)
    {
        this->m_grad = std::make_shared<DT> (0);
    }

    // There are no receivers for this tensor, 
    // Set gradient of this tensor to one
    if (this->m_receivers.empty())
    {
        *(this->m_grad) = 1;
    }

    this->m_forward_visit_started = false;
    this->m_forward_visited = true;
}
template
void Scalar<float>::forward_visit_finish();


template <typename DT>
void Scalar<DT>::backward_visit_finish()
{
    if (!this->m_val)
    {
        throw std::invalid_argument(std::string{"Value of this scalar should exist in "} + std::string{__FUNCTION__} );
    }

    this->m_backward_visit_started = false;
    this->m_backward_visited = true;
}
template
void Scalar<float>::backward_visit_finish();

template <typename DT>
void Scalar<DT>::set_device(julie::MatrixType mtype)
{}
template
void Scalar<float>::set_device(julie::MatrixType mtype);

} // namespace var
} // namespace nn
} // namespace julie
