#pragma once
#include "DMatrix.hpp"

namespace julie
{
namespace la
{

/*
Do padding for a 4-dimensional array like this:
from [b, c, h, w] to [b, c, h + pad_h, w + pad_w]
*/
template <typename DT>
DMatrix<DT> pad_2d(const DMatrix<DT> & input, lint pad_h, lint pad_w)
{
    DMatrix<DT> output{Shape{input.m_shape[0], input.m_shape[1], input.m_shape[2] + pad_h * 2, input.m_shape[3] + pad_w * 2 }};

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

    return output;
}


/*
Do back propagation for a 4-dimensional array like this:
from [b, c, h + pad_h, w + pad_w] to [b, c, h, w]
*/
template <typename DT>
DMatrix<DT> pad_2d_backward(const DMatrix<DT> & gradient, lint pad_h, lint pad_w)
{
    DMatrix<DT> output{Shape{gradient.m_shape[0], gradient.m_shape[1], gradient.m_shape[2] - pad_h * 2, gradient.m_shape[3] - pad_w * 2 }};

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
    DT *out_begin = output.m_data;

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

    return output;
}


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
DMatrix<DT> img2row_2d(const DMatrix<DT> & input, lint stride_h, lint stride_w, lint w_h, lint w_w)
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


    DMatrix<DT> output {Shape{out_bat, out_h, out_w}};

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
    //DMatrix<DT> output {}

    return output;
}


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
DMatrix<DT> img2row_2d_backward(const Shape & in_shape, const DMatrix<DT> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w)
{
    if (gradient.m_shape.dim() != 3)
    {
        throw std::invalid_argument(std::string("Input gradient should be 3-dimensional."));
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

    DMatrix<DT> l_gradient = gradient;
    l_gradient.reshape(Shape{out_bat, out_h, in_ch, w_h, w_w});

    DMatrix<DT> in_gradient {0, in_shape};

    lint in_gradient_bat_size = in_shape[1] * in_shape[2] * in_shape[3];
    lint in_gradient_ch_size = in_shape[2] * in_shape[3];
    lint in_gradient_w = in_shape[3];

    DT *in_gradient_bat_pos = in_gradient.m_data;
    DT *out_pos = l_gradient.m_data;

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

    return in_gradient;

}


// Convert an array of [b, c, h, w] into an array of [b, c * w_h * w_w, n_conv_outputs]
// where n_conv_outputs == conv_output_h * conv_output_w
template <typename DT>
DMatrix<DT> img2col_2d(const DMatrix<DT> & input, lint stride_h, lint stride_w, lint w_h, lint w_w)
{
    if (input.m_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("img2row_2d input should be 4-dimensional."));
    }

    DMatrix<DT> batch_of_mat = img2row_2d(input, stride_h, stride_w, w_h, w_w);
    std::vector<DMatrix<DT>> list_of_mat = batch_of_mat.get_collapsed(0);

    for (DMatrix<DT> & mat : list_of_mat)
    {
        mat = transpose(mat, 1);
    }

    return DMatrix<DT> {list_of_mat};
}


template <typename DT>
DMatrix<DT> img2col_2d_backward(const Shape & in_shape, const DMatrix<DT> & gradient,
                                lint stride_h, lint stride_w, lint w_h, lint w_w)
{
    std::vector<DMatrix<DT>> list_of_mat = gradient.get_collapsed(0);

    for (DMatrix<DT> & mat : list_of_mat)
    {
        mat = transpose(mat, 1);
    }

    DMatrix<DT> trans_gradient {list_of_mat};

    return img2row_2d_backward(in_shape, trans_gradient, stride_h, stride_w, w_h, w_w);

}


} // namespace julie
} // namespace la