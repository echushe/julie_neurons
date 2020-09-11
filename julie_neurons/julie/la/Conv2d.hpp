#pragma once
#include "DMatrix.hpp"
#include "DMatrix_adv.hpp"

namespace julie
{

namespace la
{

template <typename DT>
class Conv2d
{
public:
    Conv2d (lint pad_h, lint pad_w, lint stride_h, lint stride_w);

    DMatrix<DT> forward (const DMatrix<DT> & input, const DMatrix<DT> & weights, const DMatrix<DT> & bias);

    void backward (
        DMatrix<DT> & in_gradient_out,
        DMatrix<DT> & w_gradient_out,
        DMatrix<DT> & b_gradient_out,
        const DMatrix<DT> & gradient_in, const Shape & input_sh, const DMatrix<DT> & weights, const DMatrix<DT> & bias);

private:

    DMatrix<DT> convolute (const DMatrix<DT> & input, const DMatrix<DT> & weights, const DMatrix<DT> & bias);

    void convolute_backward (
        DMatrix<DT> & in_gradient_out,
        DMatrix<DT> & w_gradient_out,
        DMatrix<DT> & b_gradient_out,
        const DMatrix<DT> & gradient_in, const Shape & input_shape, const DMatrix<DT> & weights, const DMatrix<DT> & bias);


private:

    lint m_pad_h;
    lint m_pad_w;
    lint m_stride_h;
    lint m_stride_w;

    DMatrix<DT> m_img2row_cache;
};

} // la
} // julie


template <typename DT>
julie::la::Conv2d<DT>::Conv2d (lint pad_h, lint pad_w, lint stride_h, lint stride_w)
    :
    m_pad_h (pad_h),
    m_pad_w (pad_w),
    m_stride_h (stride_h),
    m_stride_w (stride_w)
{}


template <typename DT>
julie::la::DMatrix<DT> julie::la::Conv2d<DT>::forward (const DMatrix<DT> & input, const DMatrix<DT> & weights, const DMatrix<DT> & bias)
{
    if (input.m_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("Conv2d input should be 4-dimensional."));
    }

    if (weights.m_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("Conv2d weight should be 4-dimensional."));
    }

    if (input.m_shape[1] != weights.m_shape[1])
    {
        throw std::invalid_argument(std::string("Number of channel should be consistent between input and conv filters."));
    }

    if (weights.m_shape[0] != bias.m_shape.size())
    {
        throw std::invalid_argument(std::string("Number of filters and numbers of bias should be identical."));
    }

    if (this->m_pad_h == 0 && this->m_pad_w == 0)
    {
        // Pure convolution
        return this->convolute(input, weights, bias);
    }
    else
    {
        // padding
        DMatrix<DT> pad_input = pad_2d(input, this->m_pad_h, this->m_pad_w);

        // Pure convolution
        return this->convolute(pad_input, weights, bias);
    }
}

template <typename DT>
void julie::la::Conv2d<DT>::backward (
        DMatrix<DT> & in_gradient_out,
        DMatrix<DT> & w_gradient_out,
        DMatrix<DT> & b_gradient_out,
        const DMatrix<DT> & gradient_in, const Shape & input_sh, const DMatrix<DT> & weights, const DMatrix<DT> & bias)
{
    if (input_sh.dim() != 4)
    {
        throw std::invalid_argument(std::string("Conv2d input should be 4-dimensional."));
    }

    if (weights.m_shape.dim() != 4)
    {
        throw std::invalid_argument(std::string("Conv2d weight should be 4-dimensional."));
    }

    if (input_sh[1] != weights.m_shape[1])
    {
        throw std::invalid_argument(std::string("Number of channel should be consistent between input and conv filters."));
    }

    if (weights.m_shape[0] != bias.m_shape.size())
    {
        throw std::invalid_argument(std::string("Number of filters and numbers of bias should be identical."));
    }

    if (this->m_pad_h == 0 && this->m_pad_w == 0)
    {
        // Pure convolution
        this->convolute_backward(in_gradient_out, w_gradient_out, b_gradient_out, gradient_in, input_sh, weights, bias);
    }
    else
    {
        DMatrix<DT> g_after_pad;
        Shape sh_after_pad {input_sh[0], input_sh[1], input_sh[2] + this->m_pad_h * 2, input_sh[3] + this->m_pad_w * 2};

        this->convolute_backward(g_after_pad, w_gradient_out, b_gradient_out, gradient_in, sh_after_pad, weights, bias);
        in_gradient_out = pad_2d_backward(g_after_pad, this->m_pad_h, this->m_pad_w);
    }
}


template <typename DT>
julie::la::DMatrix<DT> julie::la::Conv2d<DT>::convolute (const DMatrix<DT> & input, const DMatrix<DT> & weights, const DMatrix<DT> & bias)
{
    lint w_n = weights.m_shape[0];
    lint w_ch = weights.m_shape[1];
    lint w_h = weights.m_shape[2];
    lint w_w = weights.m_shape[3];

    lint in_bat = input.m_shape[0];
    lint in_ch = input.m_shape[1];
    lint in_h = input.m_shape[2];
    lint in_w = input.m_shape[3];

    lint out_h = (in_h - w_h) / this->m_stride_h + 1;
    lint out_w = (in_w - w_w) / this->m_stride_w + 1;

    DT *bias_ele = bias.m_data;
    std::vector<DMatrix<DT>> bias_list {static_cast<size_t>(bias.m_shape.size())};
    for (DMatrix<DT> & b : bias_list)
    {
        b = DMatrix<DT> {*bias_ele, Shape{out_h, out_w}};
    }
    DMatrix<DT> l_bias {bias_list};

    this->m_img2row_cache = img2row_2d(input, this->m_stride_h, this->m_stride_w, w_h, w_w);

    // std::cout << "img2row's shape: " << this->m_img2row_cache.shape() << std::endl;
    // std::cout << "transposed w shape: " << transpose(weights, 1).shape() << std::endl;
    DMatrix<DT> bat_out_ch = matmul(this->m_img2row_cache, transpose(weights, 1), 1, 3);
    std::vector<DMatrix<DT>> bat_list = bat_out_ch.get_collapsed(0);

    for (DMatrix<DT> & mat : bat_list)
    {
        mat = transpose(mat, 1);
        mat += l_bias;
    }

    DMatrix<DT> bat_out {bat_list};
    bat_out.reshape(Shape {in_bat, w_n, out_h, out_w});

    return bat_out;
}

template <typename DT>
void julie::la::Conv2d<DT>::convolute_backward (
    DMatrix<DT> & in_gradient_out,
    DMatrix<DT> & w_gradient_out,
    DMatrix<DT> & b_gradient_out,
    const DMatrix<DT> & gradient_in, const Shape & input_shape, const DMatrix<DT> & weights, const DMatrix<DT> & bias)
{
    lint g_bat = gradient_in.m_shape[0];
    lint g_ch = gradient_in.m_shape[1];
    lint g_h = gradient_in.m_shape[2];
    lint g_w = gradient_in.m_shape[3];
    
    DMatrix<DT> l_gradient_in {gradient_in};
    l_gradient_in.reshape(Shape {g_bat, g_ch, g_h * g_w});

    std::vector<DMatrix<DT>> bat_list = l_gradient_in.get_collapsed(0);
    for (DMatrix<DT> & mat : bat_list)
    {
        mat = transpose(mat, 1);
    }

    DMatrix<DT> matmul_out_g {bat_list};  
    DMatrix<DT> img2row_gradient = matmul(matmul_out_g, weights, 1, 1);
    
    img2row_gradient.reshape(
        Shape {
            img2row_gradient.m_shape[0],
            img2row_gradient.m_shape[1],
            img2row_gradient.m_shape[2] * img2row_gradient.m_shape[3] * img2row_gradient.m_shape[4]
        }
    );

    // std::cout << "Padding: " << this->m_pad_h << " " << this->m_pad_w << std::endl;
    
    in_gradient_out = img2row_2d_backward(
        input_shape, img2row_gradient,
        this->m_stride_h, this->m_stride_w, weights.m_shape[2], weights.m_shape[3]);
    
    DMatrix<DT> img2row_out_g_trans = transpose(matmul_out_g, 2);
    w_gradient_out = matmul(img2row_out_g_trans, this->m_img2row_cache, 2, 2);

    b_gradient_out = l_gradient_in.get_fused(0).get_fused(1);
}
