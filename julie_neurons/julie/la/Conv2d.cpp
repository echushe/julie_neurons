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

#include "Conv2d.hpp"
#include "iMatrix_func.hpp"
#include "iMatrix_func_adv.hpp"

namespace julie
{
namespace la
{

template <typename DT>
Conv2d<DT>::Conv2d (lint pad_h, lint pad_w, lint stride_h, lint stride_w)
    :
    m_pad_h {pad_h},
    m_pad_w {pad_w},
    m_stride_h {stride_h},
    m_stride_w {stride_w},
    m_fused_cache {iMatrix<DT>{}, iMatrix<DT>{}}
{}
template
Conv2d<int>::Conv2d (lint pad_h, lint pad_w, lint stride_h, lint stride_w);
template
Conv2d<float>::Conv2d (lint pad_h, lint pad_w, lint stride_h, lint stride_w);


template <typename DT>
void Conv2d<DT>::forward (
    iMatrix<DT> &output, const iMatrix<DT> & input, const iMatrix<DT> & weights, const iMatrix<DT> & bias)
{
    if (input.shape().dim() != 4)
    {
        throw std::invalid_argument(std::string("Conv2d input should be 4-dimensional."));
    }

    if (weights.shape().dim() != 4)
    {
        throw std::invalid_argument(std::string("Conv2d weight should be 4-dimensional."));
    }

    if (input.shape()[1] != weights.shape()[1])
    {
        throw std::invalid_argument(std::string("Number of channel should be consistent between input and conv filters."));
    }

    if (weights.shape()[0] != bias.shape().size())
    {
        throw std::invalid_argument(std::string("Number of filters and numbers of bias should be identical."));
    }

    if (this->m_pad_h == 0 && this->m_pad_w == 0)
    {
        // Pure convolution
        this->convolute(output, input, weights, bias);
    }
    else
    {
        // padding
        pad_2d(this->m_pad_output, input, this->m_pad_h, this->m_pad_w);

        // std::cout << m_pad_cache << std::endl;

        // Pure convolution
        this->convolute(output, this->m_pad_output, weights, bias);
    }
}
template
void Conv2d<int>::forward (
    iMatrix<int> &output, const iMatrix<int> & input, const iMatrix<int> & weights, const iMatrix<int> & bias);
template
void Conv2d<float>::forward (
    iMatrix<float> &output, const iMatrix<float> & input, const iMatrix<float> & weights, const iMatrix<float> & bias);


template <typename DT>
void Conv2d<DT>::backward (
        iMatrix<DT> & in_gradient_out,
        iMatrix<DT> & w_gradient_out,
        iMatrix<DT> & b_gradient_out,
        const iMatrix<DT> & gradient_in, const Shape & input_sh, const iMatrix<DT> & weights)
{
    if (input_sh.dim() != 4)
    {
        throw std::invalid_argument(std::string("Conv2d input should be 4-dimensional."));
    }

    if (weights.shape().dim() != 4)
    {
        throw std::invalid_argument(std::string("Conv2d weight should be 4-dimensional."));
    }

    if (input_sh[1] != weights.shape()[1])
    {
        throw std::invalid_argument(std::string("Number of channel should be consistent between input and conv filters."));
    }

    if (this->m_pad_h == 0 && this->m_pad_w == 0)
    {
        // Pure convolution
        this->convolute_backward(in_gradient_out, w_gradient_out, b_gradient_out, gradient_in, input_sh, weights);
    }
    else
    {
        Shape sh_after_pad {input_sh[0], input_sh[1], input_sh[2] + this->m_pad_h * 2, input_sh[3] + this->m_pad_w * 2};

        this->convolute_backward(this->m_gradient_after_pad, w_gradient_out, b_gradient_out, gradient_in, sh_after_pad, weights);
        pad_2d_backward(in_gradient_out, this->m_gradient_after_pad, this->m_pad_h, this->m_pad_w);
    }
/*
    std::cout << "gradient_in: " << gradient_in << std::endl;
    std::cout << "in_gradient_out: " << in_gradient_out << std::endl;
    std::cout << "w_gradient_out: " << w_gradient_out << std::endl;
    std::cout << "b_gradient_out: " << b_gradient_out << std::endl;
*/
}
template
void Conv2d<int>::backward (
        iMatrix<int> & in_gradient_out,
        iMatrix<int> & w_gradient_out,
        iMatrix<int> & b_gradient_out,
        const iMatrix<int> & gradient_in, const Shape & input_sh, const iMatrix<int> & weights);
template
void Conv2d<float>::backward (
        iMatrix<float> & in_gradient_out,
        iMatrix<float> & w_gradient_out,
        iMatrix<float> & b_gradient_out,
        const iMatrix<float> & gradient_in, const Shape & input_sh, const iMatrix<float> & weights);


template <typename DT>
void Conv2d<DT>::convolute (
    iMatrix<DT> &output, const iMatrix<DT> & input, const iMatrix<DT> & weights, const iMatrix<DT> & bias)
{
    lint w_n = weights.shape()[0];
    lint w_ch = weights.shape()[1];
    lint w_h = weights.shape()[2];
    lint w_w = weights.shape()[3];

    lint in_bat = input.shape()[0];
    lint in_ch = input.shape()[1];
    lint in_h = input.shape()[2];
    lint in_w = input.shape()[3];

    lint out_h = (in_h - w_h) / this->m_stride_h + 1;
    lint out_w = (in_w - w_w) / this->m_stride_w + 1;

    // [w_n, out_h]
    bias.get_right_extended(this->m_bias_ch_out_h, out_h);
    // [w_n, out_h, out_w]
    this->m_bias_ch_out_h.get_right_extended(this->m_bias_ch_out, out_w);
    // [w_n, out_h, out_w]
    this->m_bias_ch_out.reshape(julie::la::Shape {w_n, out_h, out_w});

    img2row_2d(this->m_img2row_output, input, this->m_stride_h, this->m_stride_w, w_h, w_w);

    transpose(this->m_weights_trans, weights, 1);
    matmul(this->m_bat_out_ch, this->m_img2row_output, this->m_weights_trans, 1, 3);
    transpose_neighboring_dims(output, this->m_bat_out_ch, 1, 1, 2, 2);

    output += this->m_bias_ch_out;
    output.reshape(Shape {in_bat, w_n, out_h, out_w});
}
template
void Conv2d<int>::convolute (
    iMatrix<int> &output, const iMatrix<int> & input, const iMatrix<int> & weights, const iMatrix<int> & bias);
template
void Conv2d<float>::convolute (
    iMatrix<float> &output, const iMatrix<float> & input, const iMatrix<float> & weights, const iMatrix<float> & bias);


template <typename DT>
void Conv2d<DT>::convolute_backward (
    iMatrix<DT> & in_gradient_out,
    iMatrix<DT> & w_gradient_out,
    iMatrix<DT> & b_gradient_out,
    const iMatrix<DT> & gradient_in, const Shape & input_shape, const iMatrix<DT> & weights)
{
    lint g_bat = gradient_in.shape()[0];
    lint g_ch = gradient_in.shape()[1];
    lint g_h = gradient_in.shape()[2];
    lint g_w = gradient_in.shape()[3];

    transpose_neighboring_dims(this->m_gradient_in_bat_out_ch, gradient_in, 1, 1, 2, 3);

    matmul(this->m_img2row_gradient, this->m_gradient_in_bat_out_ch, weights, 1, 1);
    
    this->m_img2row_gradient.reshape(
        Shape {
            this->m_img2row_gradient.shape()[0],
            this->m_img2row_gradient.shape()[1] * this->m_img2row_gradient.shape()[2],
            this->m_img2row_gradient.shape()[3] * this->m_img2row_gradient.shape()[4] * this->m_img2row_gradient.shape()[5]
        }
    );

    // std::cout << "Padding: " << this->m_pad_h << " " << this->m_pad_w << std::endl;
    
    img2row_2d_backward(
        in_gradient_out, input_shape, this->m_img2row_gradient,
        this->m_stride_h, this->m_stride_w, weights.shape()[2], weights.shape()[3]);
    
    // from [bat, out_h, out_w, out_ch] to [out_ch, bat, out_h, out_w]
    transpose(this->m_gradient_in_ch_bat_out, this->m_gradient_in_bat_out_ch, 3);
    
    //                     [out_ch, bat, out_h, out_w]     [bat, out_h * out_w, in_ch * w_h * w_w]
    matmul(w_gradient_out, this->m_gradient_in_ch_bat_out, this->m_img2row_output, 3, 2);
    w_gradient_out.reshape(weights.shape());

    gradient_in.get_reduce_sum(this->m_fused_cache[0], 0);
    this->m_fused_cache[0].get_reduce_sum(this->m_fused_cache[1], 1);
    this->m_fused_cache[1].get_reduce_sum(b_gradient_out, 1);
    // b_gradient_out = gradient_in.get_reduce_sum(0).get_reduce_sum(1).get_reduce_sum(1);
}
template
void Conv2d<int>::convolute_backward (
    iMatrix<int> & in_gradient_out,
    iMatrix<int> & w_gradient_out,
    iMatrix<int> & b_gradient_out,
    const iMatrix<int> & gradient_in, const Shape & input_shape, const iMatrix<int> & weights);
template
void Conv2d<float>::convolute_backward (
    iMatrix<float> & in_gradient_out,
    iMatrix<float> & w_gradient_out,
    iMatrix<float> & b_gradient_out,
    const iMatrix<float> & gradient_in, const Shape & input_shape, const iMatrix<float> & weights);


template <typename DT>
void Conv2d<DT>::clear_cache()
{
    m_pad_output = iMatrix<DT> {};
    m_img2row_output = iMatrix<DT> {};

    m_gradient_after_pad = iMatrix<DT> {};
    m_weights_trans = iMatrix<DT> {};

    m_bat_out_ch = iMatrix<DT> {};

    m_bias_ch_out_h = iMatrix<DT> {};
    m_bias_ch_out = iMatrix<DT> {};

    //////////////////////
    m_gradient_in_bat_out_ch = iMatrix<DT> {};
    m_gradient_in_ch_bat_out = iMatrix<DT> {};

    m_img2row_gradient = iMatrix<DT> {};

    for (auto &ele : this->m_fused_cache)
    {
        ele = iMatrix<DT> {};
    }
}
template
void Conv2d<int>::clear_cache();
template
void Conv2d<float>::clear_cache();


} // la
} // julie
