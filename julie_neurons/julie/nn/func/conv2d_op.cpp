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

#include "conv2d_op.hpp"

namespace julie
{
namespace nn
{
namespace func
{


Conv2d::Conv2d(lint pad_h, lint pad_w, lint stride_h, lint stride_w)
    :
    op::Function {std::string {"Conv2d"}, false},
    m_conv2d {std::make_unique<julie::la::Conv2d<float>>(pad_h, pad_w, stride_h, stride_w)}
#ifdef WITH_ONEDNN
    ,m_onednn_conv2d {std::make_unique<julie::la::Conv2dOneDNN<float>>(pad_h, pad_w, stride_h, stride_w)}
#endif
#ifdef WITH_CUDNN
    ,m_cudnn_conv2d {std::make_unique<julie::la::Conv2dCuDNN<float>>(pad_h, pad_w, stride_h, stride_w)}
#endif
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

Conv2d::Conv2d(const Conv2d & other)
    :
    op::Function {other},
    m_conv2d {std::make_unique<julie::la::Conv2d<float>>(*(other.m_conv2d))}
#ifdef WITH_ONEDNN
    ,m_onednn_conv2d {std::make_unique<julie::la::Conv2dOneDNN<float>>(*(other.m_onednn_conv2d))}
#endif
#ifdef WITH_CUDNN
    ,m_cudnn_conv2d {std::make_unique<julie::la::Conv2dCuDNN<float>>(*(other.m_cudnn_conv2d))}
#endif
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

Conv2d::Conv2d(Conv2d && other)
    :
    op::Function {other},
    m_conv2d {std::move(other.m_conv2d)}
#ifdef WITH_ONEDNN
    ,m_onednn_conv2d {std::move(other.m_onednn_conv2d)}
#endif
#ifdef WITH_CUDNN
    ,m_cudnn_conv2d {std::move(other.m_cudnn_conv2d)}
#endif
{
    this->m_output = std::make_shared<var::Tensor<float>> ();
}

Conv2d & Conv2d::operator = (const Conv2d & other)
{
    op::Function::operator = (other);
    this->m_conv2d = std::make_unique<julie::la::Conv2d<float>>(*(other.m_conv2d));
#ifdef WITH_ONEDNN
    this->m_onednn_conv2d = std::make_unique<julie::la::Conv2dOneDNN<float>>(*(other.m_onednn_conv2d));
#endif
#ifdef WITH_CUDNN
    this->m_cudnn_conv2d = std::make_unique<julie::la::Conv2dCuDNN<float>>(*(other.m_cudnn_conv2d));
#endif

    return *this;
}

Conv2d & Conv2d::operator = (Conv2d && other)
{
    op::Function::operator = (other);
    this->m_conv2d = std::move(other.m_conv2d);
#ifdef WITH_ONEDNN
    this->m_onednn_conv2d = std::move(other.m_onednn_conv2d);
#endif
#ifdef WITH_CUDNN
    this->m_cudnn_conv2d = std::move(other.m_cudnn_conv2d);
#endif

    return *this;
}

void Conv2d::set_inputs(const std::shared_ptr<op::Function> & self, 
                                    const std::vector<std::shared_ptr<op::Variable>> & inputs)
{
    if (inputs.size() != 3)
    {
        throw std::invalid_argument(std::string("Number of inputs for Conv2d operation is not 3"));
    }

    op::Function::set_inputs(self, inputs);
}

void Conv2d::forward()
{
    var::Tensor<float> *featrue_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[0].get());
    var::Tensor<float> *weight_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[1].get());
    var::Tensor<float> *bias_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[2].get());

    std::shared_ptr<julie::la::iMatrix<float>> f_mat_ptr = featrue_ptr->val();
    std::shared_ptr<julie::la::iMatrix<float>> w_mat_ptr = weight_ptr->val();
    std::shared_ptr<julie::la::iMatrix<float>> b_mat_ptr = bias_ptr->val();

    var::Tensor<float> *output_ptr = dynamic_cast<var::Tensor<float>*>(this->m_output.get());

    if (f_mat_ptr->get_matrix_type() == julie::MatrixType::CPU &&
             w_mat_ptr->get_matrix_type() == julie::MatrixType::CPU &&
             b_mat_ptr->get_matrix_type() == julie::MatrixType::CPU)
    {
#ifdef WITH_ONEDNN
        this->m_onednn_conv2d->forward(*(output_ptr->val()), *f_mat_ptr, *w_mat_ptr, *b_mat_ptr);
#else
        this->m_conv2d->forward(*(output_ptr->val()), *f_mat_ptr, *w_mat_ptr, *b_mat_ptr);
#endif
    }
    else if (f_mat_ptr->get_matrix_type() == julie::MatrixType::CUDA &&
             w_mat_ptr->get_matrix_type() == julie::MatrixType::CUDA &&
             b_mat_ptr->get_matrix_type() == julie::MatrixType::CUDA)
    {
#ifdef WITH_CUDNN
        this->m_cudnn_conv2d->forward(*(output_ptr->val()), *f_mat_ptr, *w_mat_ptr, *b_mat_ptr);
#else
        this->m_conv2d->forward(*(output_ptr->val()), *f_mat_ptr, *w_mat_ptr, *b_mat_ptr);
#endif
    }
    else
    {
        throw std::invalid_argument { std::string{"Matrix type inconsistent or not supported for DNN boost mode "} + 
            std::string{__FUNCTION__} };
    }
}

void Conv2d::backward()
{
    var::Tensor<float> *output_ptr = dynamic_cast<var::Tensor<float>*>(this->m_output.get());
    std::shared_ptr<julie::la::iMatrix<float>> out_grad = output_ptr->grad();

    // std::cout << "Output gradient of this conv layer: \n" << *out_grad << std::endl;

    var::Tensor<float> *featrue_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[0].get());
    var::Tensor<float> *weight_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[1].get());
    var::Tensor<float> *bias_ptr = dynamic_cast<var::Tensor<float>*>(this->m_inputs[2].get());
    
    std::shared_ptr<julie::la::iMatrix<float>> f_mat_ptr = featrue_ptr->val();
    std::shared_ptr<julie::la::iMatrix<float>> w_mat_ptr = weight_ptr->val();
    std::shared_ptr<julie::la::iMatrix<float>> b_mat_ptr = bias_ptr->val();

    la::Shape out_grad_sh = out_grad->shape();

    // std::cout << " ----------- Conv2d backward: -------------- " << std::endl;
    // std::cout << "Shape of conv input: " << f_mat_ptr->shape() << std::endl;
    // std::cout << "Shape of conv weight: " << w_mat_ptr->shape() << std::endl;
    // std::cout << "Shape of conv bias: " << b_mat_ptr->shape() << std::endl;

    if (out_grad->get_matrix_type() == julie::MatrixType::CPU &&
             w_mat_ptr->get_matrix_type() == julie::MatrixType::CPU)
    {
#ifdef WITH_ONEDNN
        this->m_onednn_conv2d->backward(
            this->m_feature_grad_cache, this->m_weight_grad_cache, this->m_bias_grad_cache,
            *out_grad, f_mat_ptr->shape(), *w_mat_ptr);
#else
        this->m_conv2d->backward(
            this->m_feature_grad_cache, this->m_weight_grad_cache, this->m_bias_grad_cache,
            *out_grad, f_mat_ptr->shape(), *w_mat_ptr);
#endif
    }
    else if (out_grad->get_matrix_type() == julie::MatrixType::CUDA &&
             w_mat_ptr->get_matrix_type() == julie::MatrixType::CUDA)
    {
#ifdef WITH_CUDNN
        this->m_cudnn_conv2d->backward(
            this->m_feature_grad_cache, this->m_weight_grad_cache, this->m_bias_grad_cache,
            *out_grad, f_mat_ptr->shape(), *w_mat_ptr);
#else
        this->m_conv2d->backward(
            this->m_feature_grad_cache, this->m_weight_grad_cache, this->m_bias_grad_cache,
            *out_grad, f_mat_ptr->shape(), *w_mat_ptr);
#endif
    }
    else
    {
        throw std::invalid_argument { std::string{"Matrix type inconsistent or not supported for DNN boost mode "} + 
            std::string{__FUNCTION__} };
    }

    featrue_ptr->add_grad(this->m_feature_grad_cache);
    weight_ptr->add_grad(this->m_weight_grad_cache);
    bias_ptr->add_grad(this->m_bias_grad_cache);
}

void Conv2d::clear_cache()
{
    this->m_conv2d->clear_cache();

    this->m_feature_grad_cache = julie::la::iMatrix<float> {};
    this->m_weight_grad_cache = julie::la::iMatrix<float> {};
    this->m_bias_grad_cache = julie::la::iMatrix<float> {};
}

} // namespace func
} // namespace nn
} // namespace julie
