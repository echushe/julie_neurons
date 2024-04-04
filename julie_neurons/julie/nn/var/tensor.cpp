#include "tensor.hpp"
#include "iMatrix_func.hpp"

namespace julie
{
namespace nn
{
namespace var
{

template <typename DT>
Tensor<DT>::Tensor(julie::MatrixType mtype)
    :
    op::Variable {},
    m_val {nullptr},
    m_grad {nullptr},
    m_device {mtype}
{}
template
Tensor<float>::Tensor(julie::MatrixType mtype);


template <typename DT>
Tensor<DT>::Tensor(const julie::la::iMatrix<DT> & val)
    :
    op::Variable {},
    m_val {std::make_shared<julie::la::iMatrix<DT>>(val)},
    m_grad {nullptr},
    m_device {m_val->get_matrix_type()}
{}
template
Tensor<float>::Tensor(const julie::la::iMatrix<float> & val);


template <typename DT>
Tensor<DT>::Tensor(julie::la::iMatrix<DT> && val)
    :
    op::Variable {},
    m_val {std::make_shared<julie::la::iMatrix<DT>>(std::move(val))},
    m_grad {nullptr},
    m_device {m_val->get_matrix_type()}
{}
template
Tensor<float>::Tensor(julie::la::iMatrix<float> && val);


template <typename DT>
Tensor<DT>::Tensor(const Tensor<DT> & other)
    :
    op::Variable {other},
    m_val {nullptr},
    m_grad {nullptr},
    m_device {other.m_device}
{
    if (other.m_val)
    {
        std::make_shared<julie::la::iMatrix<DT>>(*(other.m_val));
    }

    if (other.m_grad)
    {
        std::make_shared<julie::la::iMatrix<DT>>(*(other.m_grad));
    }
}
template
Tensor<float>::Tensor(const Tensor<float> & other);


template <typename DT>
Tensor<DT>::Tensor(Tensor<DT> && other)
    :
    op::Variable {other},
    m_val {std::move(other.m_val)},
    m_grad {std::move(other.m_grad)},
    m_device {other.m_device}
{
    other.m_val = nullptr;
    other.m_grad = nullptr;
}
template
Tensor<float>::Tensor(Tensor<float> && other);


template <typename DT>
Tensor<DT> & Tensor<DT>::operator = (const Tensor<DT> & other)
{
    // Call copy assignment of base class
    op::Variable::operator = (other);
    this->m_val = nullptr;
    this->m_grad = nullptr;

    if (other.m_val)
    {
        this->m_val = std::make_shared<julie::la::iMatrix<DT>> (*(other.m_val));
    }

    if (other.m_grad)
    {
        this->m_grad = std::make_shared<julie::la::iMatrix<DT>> (*(other.m_grad));
    }

    this->m_device = other.m_device;

    return *this;
}
template
Tensor<float> & Tensor<float>::operator = (const Tensor<float> & other);


template <typename DT>
Tensor<DT> & Tensor<DT>::operator = (Tensor<DT> && other)
{
    // Call move assignment of base class
    op::Variable::operator = (other);

    this->m_val = std::move(other.m_val);
    this->m_grad = std::move(other.m_grad);
    this->m_device = other.m_device;

    other.m_val = nullptr;
    other.m_grad = nullptr;

    return *this;
}
template
Tensor<float> & Tensor<float>::operator = (Tensor<float> && other);


template <typename DT>
julie::op::Variable::VariableType Tensor<DT>::data_type() const
{
    return op::Variable::VariableType::TENSOR;
}
template
julie::op::Variable::VariableType Tensor<float>::data_type() const;


template <typename DT>
void Tensor<DT>::set_device(julie::MatrixType mtype)
{
    if (this->m_val)
    {
        this->m_val->set_matrix_type(mtype);
    }
    
    if (this->m_grad)
    {
        this->m_grad->set_matrix_type(mtype);
    }

    this->m_device = mtype;
}
template
void Tensor<float>::set_device(julie::MatrixType mtype);


template <typename DT>
julie::MatrixType Tensor<DT>::get_device() const
{
    return this->m_device;
}
template
julie::MatrixType Tensor<float>::get_device() const;


template <typename DT>
std::shared_ptr<julie::la::iMatrix<DT>> Tensor<DT>::val()
{
    if (!this->m_val)
    {
        this->m_val = std::make_shared<julie::la::iMatrix<DT>>(0, julie::la::Shape{1}, this->m_device);
    }

    return this->m_val;
}
template
std::shared_ptr<julie::la::iMatrix<float>> Tensor<float>::val();


template <typename DT>
std::shared_ptr<julie::la::iMatrix<DT>> Tensor<DT>::grad()
{
    if (!this->m_grad)
    {
        if (!this->m_val)
        {
            if (this->m_receivers.empty())
            {
                // There are no receivers for this tensor, 
                // Set gradient of this tensor to one
                this->m_grad = std::make_shared<julie::la::iMatrix<DT>>(1, julie::la::Shape{1}, this->m_device);
            }
            else
            {
                // There are receivers for this tensor, init gradient to zero
                this->m_grad = std::make_shared<julie::la::iMatrix<DT>>(0, julie::la::Shape{1}, this->m_device);
            }
        }
        else
        {
            if (this->m_receivers.empty())
            {
                // There are no receivers for this tensor, 
                // Set gradient of this tensor to one
                this->m_grad = std::make_shared<julie::la::iMatrix<DT>>(1, this->m_val->shape(), this->m_device);
            }
            else
            {
                // There are receivers for this tensor, init gradient to zero
                this->m_grad = std::make_shared<julie::la::iMatrix<DT>>(0, this->m_val->shape(), this->m_device);
            }
        }
    }

    return this->m_grad;
}
template
std::shared_ptr<julie::la::iMatrix<float>> Tensor<float>::grad();


template <typename DT>
void Tensor<DT>::val(const julie::la::iMatrix<DT> & val)
{
    if (this->m_device != val.get_matrix_type())
    {
        this->set_device(val.get_matrix_type());
    }

    if (this->m_val)
    {
        *(this->m_val) = val;
    }
    else
    {
        this->m_val = std::make_shared<julie::la::iMatrix<DT>>(val);
    }
}
template
void Tensor<float>::val(const julie::la::iMatrix<float> & val);


template <typename DT>
void Tensor<DT>::val(julie::la::iMatrix<DT> && val)
{
    if (this->m_device != val.get_matrix_type())
    {
        this->set_device(val.get_matrix_type());
    }

    this->m_val = std::make_shared<julie::la::iMatrix<DT>>(std::move(val));
}
template
void Tensor<float>::val(julie::la::iMatrix<float> && val);


template <typename DT>
void Tensor<DT>::set_grad_to_zero()
{
    if (this->m_grad)
    {
        *(this->m_grad) = 0;
    }
}
template
void Tensor<float>::set_grad_to_zero();


template <typename DT>
void Tensor<DT>::add_grad(const julie::la::iMatrix<DT> & grad)
{
    if (this->m_device != grad.get_matrix_type())
    {
        this->set_device(grad.get_matrix_type());
    }

    if (!this->m_grad || this->m_grad->shape().size() != grad.shape().size())
    {
        this->m_grad = std::make_shared<julie::la::iMatrix<DT>>(grad);
    }
    else
    {
        if (this->m_grad->shape() != grad.shape())
        {
            this->m_grad->reshape(grad.shape());
        }

        *(this->m_grad) += grad;
    }
}
template
void Tensor<float>::add_grad(const julie::la::iMatrix<float> & grad);


template <typename DT>
void Tensor<DT>::forward_visit_finish()
{
    if (!this->m_val)
    {
        throw std::invalid_argument(std::string{"Value of this tensor should exist in "} + std::string{__FUNCTION__} );
    }

    if (!this->m_grad)
    {
        this->m_grad = std::make_shared<julie::la::iMatrix<DT>> (this->m_val->shape(), this->m_device);
    }
    else
    {
        julie::la::renew_if_shape_not_match(*(this->m_grad), this->m_val->shape());
    }

    // There are no receivers for this tensor, 
    // Set gradient of this tensor to one
    if (this->m_receivers.empty())
    {
        *(this->m_grad) = 1;
    }
    else // There are receivers for this tensor, init gradient of this tensor to zero
    {
        *(this->m_grad) = 0;
    }

    this->m_forward_visit_started = false;
    this->m_forward_visited = true;
}
template
void Tensor<float>::forward_visit_finish();


template <typename DT>
void Tensor<DT>::backward_visit_finish()
{
    if (!this->m_val)
    {
        throw std::invalid_argument(std::string{"Value of this tensor should exist in "} + std::string{__FUNCTION__} );
    }

    this->m_backward_visit_started = false;
    this->m_backward_visited = true;
}
template
void Tensor<float>::backward_visit_finish();


} // namespace var
} // namespace nn
} // namespace julie
