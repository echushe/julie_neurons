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

#include "Matrix_CPU_func_adv.hpp"
#include "Matrix_CPU_func.hpp"

#include <algorithm>

namespace julie
{
namespace la
{
namespace cpu
{

/*
Do padding for a 4-dimensional array like this:
from [b, c, h, w] to [b, c, h + pad_h, w + pad_w]
*/
template <typename DT>
void pad_2d(Matrix_CPU<DT> & output, const Matrix_CPU<DT> & input, lint pad_h, lint pad_w)
{
    julie::la::Shape output_sh{input.m_shape[0], input.m_shape[1], input.m_shape[2] + pad_h * 2, input.m_shape[3] + pad_w * 2};

    renew_if_shape_not_match(output, output_sh);

    lint in_bat = input.m_shape[0];
    lint in_ch = input.m_shape[1];
    lint in_h = input.m_shape[2];
    lint in_w = input.m_shape[3];

    lint in_size = in_ch * in_h * in_w;
    lint in_ch_size = in_h * in_w;

    lint out_bat = in_bat;
    lint out_ch = in_ch;
    lint out_h = in_h + 2 * pad_h;
    lint out_w = in_w + 2 * pad_w;

    lint out_size = out_ch * out_h * out_w;
    lint out_ch_size = out_h * out_w;

    DT *in_begin = input.m_data;
    DT *out_begin = output.m_data;

    lint in_h_plus_pad = in_h + pad_h;
    lint in_w_plus_pad = in_w + pad_w;

    for (lint bat_i = 0; bat_i < in_bat; ++bat_i)
    {
        DT *in_ch_begin = in_begin;
        DT *out_ch_begin = out_begin;

        for (lint ch_i = 0; ch_i < in_ch; ++ch_i)
        {
            DT *in_h_begin = in_ch_begin;
            DT *out_h_begin = out_ch_begin;

            for (lint out_h_i = 0; out_h_i < out_h; ++out_h_i)
            {
                DT *out_pos = out_h_begin;

                if (out_h_i < pad_h || out_h_i >= in_h_plus_pad)
                {
                    for (lint out_w_i = 0; out_w_i < out_w; ++out_w_i)
                    {
                        *out_pos = 0;
                        ++out_pos;
                    }
                }
                else
                {
                    DT *in_pos = in_h_begin;

                    for (lint out_w_i = 0; out_w_i < out_w; ++out_w_i)
                    {
                        if (out_w_i >= pad_w && out_w_i < in_w_plus_pad)
                        {
                            *out_pos = *in_pos;
                            ++in_pos;
                        }
                        else
                        {
                            *out_pos = 0;
                        }

                        ++out_pos;
                    }
                }

                if (out_h_i >= pad_h && out_h_i < in_h_plus_pad)
                {
                    in_h_begin += in_w;
                }
                out_h_begin += out_w;
            }

            in_ch_begin += in_ch_size;
            out_ch_begin += out_ch_size;
        }

        in_begin += in_size;
        out_begin += out_size;
    }
}
template
void pad_2d(Matrix_CPU<float> & output, const Matrix_CPU<float> & input, lint pad_h, lint pad_w);
template
void pad_2d(Matrix_CPU<int> & output, const Matrix_CPU<int> & input, lint pad_h, lint pad_w);


/*
Do back-propagation for a 4-dimensional array like this:
from [b, c, h + pad_h, w + pad_w] to [b, c, h, w]
*/
template <typename DT>
void pad_2d_backward(Matrix_CPU<DT> & in_gradient, const Matrix_CPU<DT> & gradient, lint pad_h, lint pad_w)
{
    julie::la::Shape in_gradient_sh {gradient.m_shape[0], gradient.m_shape[1], gradient.m_shape[2] - pad_h * 2, gradient.m_shape[3] - pad_w * 2};

    renew_if_shape_not_match(in_gradient, in_gradient_sh);
    
    lint g_bat = gradient.m_shape[0];
    lint g_ch = gradient.m_shape[1];
    lint g_h = gradient.m_shape[2];
    lint g_w = gradient.m_shape[3];

    lint g_size = g_ch * g_h * g_w;
    lint g_ch_size = g_h * g_w;

    lint out_bat = g_bat;
    lint out_ch = g_ch;
    lint out_h = g_h - 2 * pad_h;
    lint out_w = g_w - 2 * pad_w;

    lint out_size = out_ch * out_h * out_w;
    lint out_ch_size = out_h * out_w;

    DT *g_begin = gradient.m_data;
    DT *out_begin = in_gradient.m_data;

    lint g_h_sub_pad = g_h - pad_h;
    lint g_w_sub_pad = g_w - pad_w;

    for (lint bat_i = 0; bat_i < g_bat; ++bat_i)
    {
        DT *g_ch_begin = g_begin;
        DT *out_ch_begin = out_begin;

        for (lint ch_i = 0; ch_i < g_ch; ++ch_i)
        {
            DT *g_h_begin = g_ch_begin;
            DT *out_h_begin = out_ch_begin;

            for (lint g_h_i = 0; g_h_i < g_h; ++g_h_i)
            {
                if (g_h_i >= pad_h && g_h_i < g_h_sub_pad)
                {
                    DT *g_pos = g_h_begin + pad_w;
                    DT *out_pos = out_h_begin;

                    for (lint g_w_i = pad_w; g_w_i < g_w_sub_pad; ++g_w_i)
                    {
                        *out_pos = *g_pos;
                        ++g_pos;
                        ++out_pos;
                    }

                    out_h_begin += out_w;
                }

                g_h_begin += g_w;                
                
            }

            g_ch_begin += g_ch_size;
            out_ch_begin += out_ch_size;
        }

        g_begin += g_size;
        out_begin += out_size;
    }
}
template
void pad_2d_backward(Matrix_CPU<float> & in_gradient, const Matrix_CPU<float> & gradient, lint pad_h, lint pad_w);
template
void pad_2d_backward(Matrix_CPU<int> & in_gradient, const Matrix_CPU<int> & gradient, lint pad_h, lint pad_w);


template <typename DT>
void __img2row_2d_row(lint in_ch_size, lint in_w, DT *in_begin, DT *out_begin, lint in_ch, lint w_h, lint w_w)
{
    DT *in_ch_begin = in_begin;
    DT *out_pos = out_begin;

    for (lint ch_i = 0; ch_i < in_ch; ++ch_i)
    {
        DT *in_w_begin = in_ch_begin;

        for (lint w_h_i = 0; w_h_i < w_h; ++w_h_i)
        {
            DT *in_pos = in_w_begin;

            for (lint w_w_i = 0; w_w_i < w_w; ++w_w_i)
            {
                *out_pos = *in_pos;
                ++in_pos;
                ++out_pos;
            }

            in_w_begin += in_w;
        }

        in_ch_begin += in_ch_size;
    }
}


// Convert an array of [b, c, h, w] into an array of [b, n_conv_outputs, c * w_h * w_w]
// where n_conv_outputs == conv_output_h * conv_output_w
template <typename DT>
void img2row_2d(Matrix_CPU<DT> & output, const Matrix_CPU<DT> & input, lint stride_h, lint stride_w, lint w_h, lint w_w)
{
    if (input.m_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("img2row_2d input should be 4-dimensional."));
    }

    lint w_ch = input.m_shape[1];

    lint w_size = w_ch * w_h * w_w;
    lint w_ch_size = w_h * w_w;

    lint in_bat = input.m_shape[0];
    lint in_ch = input.m_shape[1];
    lint in_h = input.m_shape[2];
    lint in_w = input.m_shape[3];

    lint in_size = in_ch * in_h * in_w;
    lint in_ch_size = in_h * in_w;

    lint out_bat = in_bat;

    lint conv_out_h = (in_h - w_h) / stride_h + 1;
    lint conv_out_w = (in_w - w_w) / stride_w + 1;

    lint out_h = conv_out_h * conv_out_w;
    lint out_w = in_ch * w_h * w_w;

    lint out_size = out_h * out_w;

    renew_if_shape_not_match(output, julie::la::Shape{out_bat, out_h, out_w});

    DT *in_bat_begin = input.m_data;
    DT *out_pos = output.m_data;

    for (lint bat_i = 0; bat_i < in_bat; ++bat_i)
    {
        DT *in_filter_begin = in_bat_begin;

        for (lint conv_out_h_i = 0; conv_out_h_i < conv_out_h; ++conv_out_h_i)
        {
            DT *in_filter_pos = in_filter_begin;

            for (lint conv_out_w_i = 0; conv_out_w_i < conv_out_w; ++conv_out_w_i)
            {
                __img2row_2d_row(in_ch_size, in_w, in_filter_pos, out_pos, w_ch, w_h, w_w);
                
                in_filter_pos += stride_w;
                out_pos += out_w;
            }

            in_filter_begin += in_w * stride_h;
        }

        in_bat_begin += in_size;
    }   
}
template
void img2row_2d(Matrix_CPU<float> & output, const Matrix_CPU<float> & input, lint stride_h, lint stride_w, lint w_h, lint w_w);
template
void img2row_2d(Matrix_CPU<int> & output, const Matrix_CPU<int> & input, lint stride_h, lint stride_w, lint w_h, lint w_w);


template <typename DT>
void __img2row_2d_row_backward(lint in_ch_size, lint in_w, DT *in_begin, DT *out_begin, lint in_ch, lint w_h, lint w_w)
{
    DT *in_ch_begin = in_begin;
    DT *out_pos = out_begin;

    for (lint ch_i = 0; ch_i < in_ch; ++ch_i)
    {
        DT *in_w_begin = in_ch_begin;

        for (lint w_h_i = 0; w_h_i < w_h; ++w_h_i)
        {
            DT *in_pos = in_w_begin;

            for (lint w_w_i = 0; w_w_i < w_w; ++w_w_i)
            {
                *in_pos += *out_pos;
                ++in_pos;
                ++out_pos;
            }

            in_w_begin += in_w;
        }

        in_ch_begin += in_ch_size;
    }
}


template <typename DT>
void img2row_2d_backward(Matrix_CPU<DT> & in_gradient, const Shape & in_shape, const Matrix_CPU<DT> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w)
{
    if (gradient.m_shape.dim() != 3)
    {
        throw std::invalid_argument(std::string("Gradient of img2row output should be 3-dimensional."));
    }

    if (in_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("Input for conv2d should be 4-dimensional."));
    }

    Shape out_shape = gradient.shape();
    lint out_bat = out_shape[0];
    lint out_h = out_shape[1];
    lint out_w = out_shape[2];
    lint in_ch = out_w / (w_h * w_w);

    if (in_shape[0] != out_bat || in_shape[1] != in_ch)
    {
        throw std::invalid_argument(std::string("Batch size for input and ouput for img2row should be she same."));
    }

    
    lint conv_out_h = (in_shape[2] - w_h) / stride_h + 1;
    lint conv_out_w = (in_shape[3] - w_w) / stride_w + 1;

    // std::cout << "stride: " << stride_h << " " << stride_w << std::endl;
    // std::cout << "Shape of input: " << in_shape << std::endl;
    // std::cout << "Shape of img2row output gradient: " << gradient.shape() << std::endl;

    if (conv_out_h * conv_out_w != out_h)
    {
        throw std::invalid_argument(std::string("Shape of conv2d input does not match shape of img2row"));
    }

    renew_if_shape_not_match(in_gradient, in_shape);

    // Re-assign in_gradient into zero values  
    in_gradient = 0;

    lint in_gradient_bat_size = in_shape[1] * in_shape[2] * in_shape[3];
    lint in_gradient_ch_size = in_shape[2] * in_shape[3];
    lint in_gradient_w = in_shape[3];

    DT *in_gradient_bat_pos = in_gradient.m_data;
    DT *out_pos = gradient.m_data;

    for (lint bat_i = 0; bat_i < out_bat; ++bat_i)
    {
        DT *in_filter_begin = in_gradient_bat_pos;

        for (lint conv_out_h_i = 0; conv_out_h_i < conv_out_h; ++conv_out_h_i)
        {
            DT *in_filter_pos = in_filter_begin;

            for (lint conv_out_w_i = 0; conv_out_w_i < conv_out_w; ++conv_out_w_i)
            {
                __img2row_2d_row_backward(in_gradient_ch_size, in_gradient_w, in_filter_pos, out_pos, in_ch, w_h, w_w);
                in_filter_pos += stride_w;
                out_pos += out_w;
            }

            in_filter_begin += in_gradient_w * stride_h;
        }

        in_gradient_bat_pos += in_gradient_bat_size;
    }
}
template
void img2row_2d_backward(Matrix_CPU<float> & in_gradient, const Shape & in_shape, const Matrix_CPU<float> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w);
template
void img2row_2d_backward(Matrix_CPU<int> & in_gradient, const Shape & in_shape, const Matrix_CPU<int> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w);


// Convert an array of [b, c, h, w] into an array of [b, c * w_h * w_w, n_conv_outputs]
// where n_conv_outputs == conv_output_h * conv_output_w
template <typename DT>
void img2col_2d(Matrix_CPU<DT> & output, const Matrix_CPU<DT> & input,
                                lint stride_h, lint stride_w, lint w_h, lint w_w)
{
    if (input.m_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("img2row_2d input should be 4-dimensional."));
    }

    Matrix_CPU<DT> img2row_output;
    img2row_2d(img2row_output, input, stride_h, stride_w, w_h, w_w);

    transpose_neighboring_dims(output, img2row_output, 1, 1, 2, 2);
}
template
void img2col_2d(Matrix_CPU<float> & output, const Matrix_CPU<float> & input,
                                lint stride_h, lint stride_w, lint w_h, lint w_w);
template
void img2col_2d(Matrix_CPU<int> & output, const Matrix_CPU<int> & input,
                                lint stride_h, lint stride_w, lint w_h, lint w_w);


template <typename DT>
void img2col_2d_backward(Matrix_CPU<DT> & in_gradient, const Shape & in_shape, const Matrix_CPU<DT> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w)
{
    Matrix_CPU<DT> trans_gradient;
    transpose_neighboring_dims(trans_gradient, gradient, 1, 1, 2, 2);

    img2row_2d_backward(in_gradient, in_shape, trans_gradient, stride_h, stride_w, w_h, w_w);
}
template
void img2col_2d_backward(Matrix_CPU<float> & in_gradient, const Shape & in_shape, const Matrix_CPU<float> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w);
template
void img2col_2d_backward(Matrix_CPU<int> & in_gradient, const Shape & in_shape, const Matrix_CPU<int> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w);


template <typename DT>
void maxpool_2d(Matrix_CPU<DT> &output, Matrix_CPU<DT> &diff, const Matrix_CPU<DT> &input,
                lint stride_h, lint stride_w, lint k_h, lint k_w)
{
    if (input.m_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("maxpool_2d input should be 4-dimensional."));
    }

    lint in_bat = input.m_shape[0];
    lint in_ch = input.m_shape[1];
    lint in_h = input.m_shape[2];
    lint in_w = input.m_shape[3];

    lint in_ch_size = in_h * in_w;

    lint out_h = (in_h - k_h) / stride_h + 1;
    lint out_w = (in_w - k_w) / stride_w + 1;

    lint diff_h = out_h * k_h;
    lint diff_w = out_w * k_w;

    lint diff_ch_size = diff_h * diff_w;

    renew_if_shape_not_match(output, julie::la::Shape{in_bat, in_ch, out_h, out_w});
    renew_if_shape_not_match(diff, julie::la::Shape{in_bat, in_ch, diff_h, diff_w});

    diff = 0;

    DT *input_ch_pos = input.m_data;
    DT *diff_ch_pos = diff.m_data;
    DT *output_pos = output.m_data;

    lint input_out_h_jump = in_w * stride_h;
    lint diff_out_h_jump = diff_w * k_h;
    DT global_max = std::numeric_limits<DT>::max() * (-1);

    for (lint bat_i = 0; bat_i < in_bat; ++bat_i)
    {
        for (lint ch_i = 0; ch_i < in_ch; ++ch_i)
        {
            DT *input_kernel_h_pos = input_ch_pos;
            DT *diff_kernel_h_pos = diff_ch_pos;

            for (lint out_h_i = 0; out_h_i < out_h; ++out_h_i)
            {
                DT *input_kernel_pos = input_kernel_h_pos;
                DT *diff_kernel_pos = diff_kernel_h_pos;

                for (lint out_w_i = 0; out_w_i < out_w; ++out_w_i)
                {
                    //DT *input_kernel_pos = input_ch_pos[(out_h_i * stride_h) * in_w + out_w_i * stride_w];
                    // Search for maximum value in each kernel
                    DT *input_pos = input_kernel_pos;
                    DT max = global_max;
                    lint max_k_h_i = 0;
                    lint max_k_w_i = 0;

                    for (lint k_h_i = 0; k_h_i < k_h; ++k_h_i)
                    {
                        for (lint k_w_i = 0; k_w_i < k_w; ++k_w_i)
                        {
                            if (input_pos[k_w_i] > max)
                            {
                                max = input_pos[k_w_i];
                                max_k_h_i = k_h_i;
                                max_k_w_i = k_w_i;
                            }
                        }

                        input_pos += in_w;
                    }

                    diff_kernel_pos[diff_w * max_k_h_i + max_k_w_i] = 1;
                    *(output_pos) = max;

                    input_kernel_pos += stride_w;
                    diff_kernel_pos += k_w;
                    ++output_pos;
                }

                input_kernel_h_pos += input_out_h_jump;
                diff_kernel_h_pos += diff_out_h_jump;
            }

            input_ch_pos += in_ch_size;
            diff_ch_pos += diff_ch_size;
        }
    }
}
template
void maxpool_2d(Matrix_CPU<float> &output, Matrix_CPU<float> &diff, const Matrix_CPU<float> &input,
                lint stride_h, lint stride_w, lint k_h, lint k_w);
template
void maxpool_2d(Matrix_CPU<int> &output, Matrix_CPU<int> &diff, const Matrix_CPU<int> &input,
                lint stride_h, lint stride_w, lint k_h, lint k_w);


template <typename DT>
void avgpool_2d(Matrix_CPU<DT> &output, Matrix_CPU<DT> &diff, const Matrix_CPU<DT> &input,
                lint stride_h, lint stride_w, lint k_h, lint k_w)
{
    if (input.m_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("maxpool_2d input should be 4-dimensional."));
    }

    lint in_bat = input.m_shape[0];
    lint in_ch = input.m_shape[1];
    lint in_h = input.m_shape[2];
    lint in_w = input.m_shape[3];

    lint in_ch_size = in_h * in_w;

    lint out_h = (in_h - k_h) / stride_h + 1;
    lint out_w = (in_w - k_w) / stride_w + 1;

    lint diff_h = out_h * k_h;
    lint diff_w = out_w * k_w;

    lint diff_ch_size = diff_h * diff_w;

    renew_if_shape_not_match(output, julie::la::Shape{in_bat, in_ch, out_h, out_w});
    renew_if_shape_not_match(diff, julie::la::Shape{in_bat, in_ch, diff_h, diff_w});

    DT avg_coe = static_cast<DT>(1) / (k_h * k_w);
    diff = avg_coe;

    DT *input_ch_pos = input.m_data;
    DT *output_pos = output.m_data;

    lint input_out_h_jump = in_w * stride_h;

    for (lint bat_i = 0; bat_i < in_bat; ++bat_i)
    {
        for (lint ch_i = 0; ch_i < in_ch; ++ch_i)
        {
            DT *input_kernel_h_pos = input_ch_pos;

            for (lint out_h_i = 0; out_h_i < out_h; ++out_h_i)
            {
                DT *input_kernel_pos = input_kernel_h_pos;

                for (lint out_w_i = 0; out_w_i < out_w; ++out_w_i)
                {
                    //DT *input_kernel_pos = input_ch_pos[(out_h_i * stride_h) * in_w + out_w_i * stride_w];
                    // Search for maximum value in each kernel
                    DT *input_pos = input_kernel_pos;
                    DT sum = 0;

                    for (lint k_h_i = 0; k_h_i < k_h; ++k_h_i)
                    {
                        for (lint k_w_i = 0; k_w_i < k_w; ++k_w_i)
                        {
                            sum += input_pos[k_w_i];
                        }

                        input_pos += in_w;
                    }

                    *(output_pos) = sum * avg_coe;

                    input_kernel_pos += stride_w;
                    ++output_pos;
                }

                input_kernel_h_pos += input_out_h_jump;
            }

            input_ch_pos += in_ch_size;
        }
    }
}
template
void avgpool_2d(Matrix_CPU<float> &output, Matrix_CPU<float> &diff, const Matrix_CPU<float> &input,
                lint stride_h, lint stride_w, lint k_h, lint k_w);
template
void avgpool_2d(Matrix_CPU<int> &output, Matrix_CPU<int> &diff, const Matrix_CPU<int> &input,
                lint stride_h, lint stride_w, lint k_h, lint k_w);


template <typename DT>
void pool_2d_backward(Matrix_CPU<DT> &in_gradient, Matrix_CPU<DT> &gradient_cache,
                        const Shape &in_shape, const Matrix_CPU<DT> &diff, const Matrix_CPU<DT> &gradient,
                        lint stride_h, lint stride_w, lint k_h, lint k_w)
{
    if (gradient.m_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("Gradient of maxpool_2d output should be 4-dimensional."));
    }

    if (diff.m_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("Derivative of maxpool_2d should be 4-dimensional."));
    }

    if (in_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("maxpool_2d input should be 4-dimensional."));
    }

    Shape out_shape = gradient.shape();
    lint out_bat = out_shape[0];
    lint out_ch = out_shape[1];

    if (in_shape[0] != out_bat || in_shape[1] != out_ch)
    {
        throw std::invalid_argument(std::string("Batch size and channel size of input and ouput gradients for max pooling 2d should be she same."));
    }

    lint in_bat = in_shape[0];
    lint in_ch = in_shape[1];
    lint in_h = in_shape[2];
    lint in_w = in_shape[3];

    lint out_h = (in_h - k_h) / stride_h + 1;
    lint out_w = (in_w - k_w) / stride_w + 1;

    if (out_h != out_shape[2] || out_w != out_shape[3])
    {
        throw std::invalid_argument(std::string("Height and width of input and output gradients for max pooling 2d do not match."));
    }

    lint diff_h = out_h * k_h;
    lint diff_w = out_w * k_w;

    Shape diff_sh = julie::la::Shape{in_bat, in_ch, diff_h, diff_w};
    if (diff_sh != diff.m_shape)
    {
        throw std::invalid_argument(std::string("Shape of derivative does not match the shape required for max pooling 2d."));
    }

    renew_if_shape_not_match(in_gradient, in_shape);
    renew_if_shape_not_match(gradient_cache, diff_sh);

    in_gradient = 0;

    lint in_ch_size = in_h * in_w;
    lint diff_ch_size = diff_h * diff_w;
    lint out_ch_size = out_h * out_w;
    
    DT *g_cache_ch_pos = gradient_cache.m_data;
    DT *g_ch_pos = gradient.m_data;

    /*
    for (lint bat_i = 0; bat_i < in_bat; ++bat_i)
    {
        for (lint ch_i = 0; ch_i < in_ch; ++ch_i)
        {
            for (lint d_h_i = 0; d_h_i < diff_h; ++d_h_i)
            {
                DT *g_cache_h_pos = g_cache_ch_pos + d_h_i * diff_w;
                DT *g_h_pos = g_ch_pos + (d_h_i / k_h) * out_w;

                for (lint d_w_i = 0; d_w_i < diff_w; ++d_w_i)
                {
                    g_cache_h_pos[d_w_i] = g_h_pos[d_w_i / k_w];
                }
            }

            g_cache_ch_pos += diff_ch_size;
            g_ch_pos += out_ch_size;
        }
    }
    */
    
    lint k_h__diff_w = k_h * diff_w;

    for (lint bat_i = 0; bat_i < in_bat; ++bat_i)
    {
        for (lint ch_i = 0; ch_i < in_ch; ++ch_i)
        {
            DT *g_cache_h_pos = g_cache_ch_pos;
            DT *g_h_pos = g_ch_pos;

            for (lint o_h_i = 0; o_h_i < out_h; ++o_h_i)
            {
                for (lint i = 0; i < k_h; ++i)
                {
                    DT *g_cache_pos = g_cache_h_pos;
                    DT *g_pos = g_h_pos;
                    for (lint o_w_i = 0; o_w_i < out_w; ++o_w_i)
                    {
                        for (lint j = 0; j < k_w; ++j)
                        {
                            g_cache_pos[j] = *g_pos;
                        }

                        g_cache_pos += k_w;
                        ++g_pos;
                    }

                    g_cache_h_pos += diff_w;
                }
                g_h_pos += out_w;
            }

            g_cache_ch_pos += diff_ch_size;
            g_ch_pos += out_ch_size;
        }
    }

    // The chain rule
    gradient_cache *= diff;

    DT *in_g_ch_pos = in_gradient.m_data;
    g_cache_ch_pos = gradient_cache.m_data;
    lint input_out_h_jump = in_w * stride_h;
    lint diff_out_h_jump = diff_w * k_h;

    for (lint bat_i = 0; bat_i < in_bat; ++bat_i)
    {
        for (lint ch_i = 0; ch_i < in_ch; ++ch_i)
        {
            DT *in_g_k_h_pos = in_g_ch_pos;
            DT *g_cache_k_h_pos = g_cache_ch_pos;

            for (lint o_h_i = 0; o_h_i < out_h; ++o_h_i)
            {
                DT *in_g_k_pos = in_g_k_h_pos;
                DT *g_cache_k_pos = g_cache_k_h_pos;

                for (lint o_w_i = 0; o_w_i < out_w; ++o_w_i)
                {
                    // Calculation for one kernel
                    lint in_g_h_idx = 0;
                    lint g_cache_h_idx = 0;
                    for (lint k_h_i = 0; k_h_i < k_h; ++k_h_i)
                    {   
                        lint in_g_idx = in_g_h_idx;
                        lint g_cache_idx = g_cache_h_idx;
                        for (lint k_w_i = 0; k_w_i < k_w; ++k_w_i)
                        {
                            in_g_k_pos[in_g_idx++] += g_cache_k_pos[g_cache_idx++];
                        }

                        in_g_h_idx += in_w;
                        g_cache_h_idx += diff_w;
                    }

                    in_g_k_pos += stride_w;
                    g_cache_k_pos += k_w;
                }

                in_g_k_h_pos += input_out_h_jump;
                g_cache_k_h_pos += diff_out_h_jump;
            }

            in_g_ch_pos += in_ch_size;
            g_cache_ch_pos += diff_ch_size;
        }
    }
}
template
void pool_2d_backward(Matrix_CPU<float> &in_gradient, Matrix_CPU<float> &gradient_cache,
                        const Shape &in_shape, const Matrix_CPU<float> &diff, const Matrix_CPU<float> &gradient,
                        lint stride_h, lint stride_w, lint k_h, lint k_w);
template
void pool_2d_backward(Matrix_CPU<int> &in_gradient, Matrix_CPU<int> &gradient_cache,
                        const Shape &in_shape, const Matrix_CPU<int> &diff, const Matrix_CPU<int> &gradient,
                        lint stride_h, lint stride_w, lint k_h, lint k_w);




} // namespace cpu
} // namespace la
} // namespace julie