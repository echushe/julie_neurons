/******************************************************************************
 *             Copyright 2020 DeepFrame AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "avgpool.hpp"
#include "iMatrix_func.hpp"
#include "iMatrix_func_adv.hpp"

namespace julie
{
namespace nn
{
namespace func
{

AvgPool::AvgPool(lint pad_h, lint pad_w, lint kernel_h, lint kernel_w, lint stride_h, lint stride_w)
    :
    op::Function {std::string {"AvgPool"}, false},
    m_pad_h {pad_h},
    m_pad_w {pad_w},
    m_kernel_h {kernel_h},
    m_kernel_w {kernel_w},
    m_stride_h {stride_h},
    m_stride_w {stride_w}
#ifdef WITH_ONEDNN
    , m_onednn_pool2d {std::make_unique<julie::la::Pool2dOneDNN<float>>(
        pad_h, pad_w, kernel_h, kernel_w, stride_h, stride_w, julie::la::Pool2dOneDNN<float>::PoolType::Avg)}
#endif
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

AvgPool::AvgPool(const AvgPool & other)
    :
    op::Function {other},
    m_pad_h {other.m_pad_h},
    m_pad_w {other.m_pad_w},
    m_kernel_h {other.m_kernel_h},
    m_kernel_w {other.m_kernel_w},
    m_stride_h {other.m_stride_h},
    m_stride_w {other.m_stride_w}
#ifdef WITH_ONEDNN
    , m_onednn_pool2d {std::make_unique<julie::la::Pool2dOneDNN<float>>(*(other.m_onednn_pool2d))}
#endif
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

AvgPool::AvgPool(AvgPool && other)
    :
    op::Function {other},
    m_pad_h {other.m_pad_h},
    m_pad_w {other.m_pad_w},
    m_kernel_h {other.m_kernel_h},
    m_kernel_w {other.m_kernel_w},
    m_stride_h {other.m_stride_h},
    m_stride_w {other.m_stride_w}
#ifdef WITH_ONEDNN
    , m_onednn_pool2d {std::move(other.m_onednn_pool2d)}
#endif
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

AvgPool & AvgPool::operator = (const AvgPool & other)
{
    op::Function::operator = (other);

    m_pad_h = other.m_pad_h;
    m_pad_w = other.m_pad_w;
    m_kernel_h = other.m_kernel_h;
    m_kernel_w = other.m_kernel_w;
    m_stride_h = other.m_stride_h;
    m_stride_w = other.m_stride_w;
#ifdef WITH_ONEDNN
    m_onednn_pool2d = std::make_unique<julie::la::Pool2dOneDNN<float>>(*(other.m_onednn_pool2d));
#endif

    return *this;
}

AvgPool & AvgPool::operator = (AvgPool && other)
{
    op::Function::operator = (other);

    m_pad_h = other.m_pad_h;
    m_pad_w = other.m_pad_w;
    m_kernel_h = other.m_kernel_h;
    m_kernel_w = other.m_kernel_w;
    m_stride_h = other.m_stride_h;
    m_stride_w = other.m_stride_w;
#ifdef WITH_ONEDNN
    m_onednn_pool2d = std::move(other.m_onednn_pool2d);
#endif

    return *this;
}

void AvgPool::set_inputs(const std::shared_ptr<op::Function> & self, 
                                    const std::vector<std::shared_ptr<op::Variable>> & inputs)
{
    if (inputs.size() != 1)
    {
        throw std::invalid_argument(std::string("Number of inputs for AvgPool operation is not 1"));
    }

    op::Function::set_inputs(self, inputs);
}

void AvgPool::forward()
{
    var::Tensor<float> *input_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[0].get());
    std::shared_ptr<julie::la::iMatrix<float>> t_mat_ptr = input_ptr->val();

    var::Tensor<float> *output_ptr = dynamic_cast<var::Tensor<float>*>(this->m_output.get());

    if (t_mat_ptr->get_matrix_type() == julie::MatrixType::CPU)
    {
#ifdef WITH_ONEDNN
        this->m_onednn_pool2d->forward(*(output_ptr->val()), *t_mat_ptr);
#else
        // Forward of padding
        julie::la::pad_2d(this->m_padding_cache, *t_mat_ptr, this->m_pad_h, this->m_pad_w);
        // Forward of pooling
        julie::la::avgpool_2d(
            *(output_ptr->val()), this->m_pool_diff, this->m_padding_cache, 
            this->m_stride_h, this->m_stride_w, this->m_kernel_h, this->m_kernel_w);
#endif
    }
    else if (t_mat_ptr->get_matrix_type() == julie::MatrixType::CUDA)
    {
        // Forward of padding
        julie::la::pad_2d(this->m_padding_cache, *t_mat_ptr, this->m_pad_h, this->m_pad_w);
        // Forward of pooling
        julie::la::avgpool_2d(
            *(output_ptr->val()), this->m_pool_diff, this->m_padding_cache, 
            this->m_stride_h, this->m_stride_w, this->m_kernel_h, this->m_kernel_w);
    }
    else
    {
        throw std::invalid_argument { std::string{"Matrix type inconsistent or not supported for DNN boost mode "} + 
            std::string{__FUNCTION__} };
    }
}

void AvgPool::backward()
{
    var::Tensor<float> *output_ptr = dynamic_cast<var::Tensor<float>*>(this->m_output.get());
    std::shared_ptr<julie::la::iMatrix<float>> out_grad = output_ptr->grad();

    var::Tensor<float> *input_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[0].get());

    //std::shared_ptr<julie::la::iMatrix<float>> t_mat_ptr = input_ptr->val();

    // Do chain rule for the input
    if (out_grad->get_matrix_type() == julie::MatrixType::CPU)
    {
#ifdef WITH_ONEDNN
        this->m_onednn_pool2d->backward(this->m_input_grad_cache, *(out_grad), *(input_ptr->val()));
#else
        // Backward for pooling
        julie::la::pool_2d_backward(
            this->m_pad_grad_cache, this->m_pool_grad_cache, this->m_padding_cache.shape(), this->m_pool_diff, *(out_grad),
            this->m_stride_h, this->m_stride_w, this->m_kernel_h, this->m_kernel_w);
        // Backward for padding
        julie::la::pad_2d_backward(this->m_input_grad_cache, this->m_pad_grad_cache, this->m_pad_h, this->m_pad_w);
#endif
    }
    else if (out_grad->get_matrix_type() == julie::MatrixType::CUDA)
    {
        // Backward for pooling
        julie::la::pool_2d_backward(
            this->m_pad_grad_cache, this->m_pool_grad_cache, this->m_padding_cache.shape(), this->m_pool_diff, *(out_grad),
            this->m_stride_h, this->m_stride_w, this->m_kernel_h, this->m_kernel_w);
        // Backward for padding
        julie::la::pad_2d_backward(this->m_input_grad_cache, this->m_pad_grad_cache, this->m_pad_h, this->m_pad_w);
    }
    else
    {
        throw std::invalid_argument { std::string{"Matrix type inconsistent or not supported for DNN boost mode "} + 
            std::string{__FUNCTION__} };
    }

    //std::cout << this->m_input_grad_cache << std::endl;
    input_ptr->add_grad(this->m_input_grad_cache);
}

void AvgPool::clear_cache()
{
    this->m_padding_cache = julie::la::iMatrix<float> {};
    this->m_pool_diff = julie::la::iMatrix<float> {};
    this->m_pool_grad_cache = julie::la::iMatrix<float> {};
    this->m_pad_grad_cache = julie::la::iMatrix<float> {};
    this->m_input_grad_cache = julie::la::iMatrix<float> {};
}

} // namespace func
} // namespace nn
} // namespace julie
