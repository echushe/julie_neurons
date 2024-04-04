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

#include "Conv2dOneDNN.hpp"
#include "iMatrix_func.hpp"
#include "iMatrix_func_adv.hpp"
#include "OneDNNHelper.hpp"
//#include "dnnl_utilities.hpp"

namespace julie
{
namespace la
{

//template <typename DT>
//std::shared_ptr<dnnl::engine> Conv2dOneDNN<DT>::the_dnnl_engine = nullptr;


template <typename DT>
Conv2dOneDNN<DT>::Conv2dOneDNN (lint pad_h, lint pad_w, lint stride_h, lint stride_w)
    :
    m_pad_h {pad_h},
    m_pad_w {pad_w},
    m_stride_h {stride_h},
    m_stride_w {stride_w}
{}
template
Conv2dOneDNN<int>::Conv2dOneDNN (lint pad_h, lint pad_w, lint stride_h, lint stride_w);
template
Conv2dOneDNN<float>::Conv2dOneDNN (lint pad_h, lint pad_w, lint stride_h, lint stride_w);


template <typename DT>
void Conv2dOneDNN<DT>::forward (
    iMatrix<DT> &output, const iMatrix<DT> & input, const iMatrix<DT> & weights, const iMatrix<DT> & bias)
{
    if (input.shape().dim() != 4)
    {
        throw std::invalid_argument(std::string("Conv2dOneDNN input should be 4-dimensional."));
    }

    if (weights.shape().dim() != 4)
    {
        throw std::invalid_argument(std::string("Conv2dOneDNN weight should be 4-dimensional."));
    }

    if (input.shape()[1] != weights.shape()[1])
    {
        throw std::invalid_argument(std::string("Number of channel should be consistent between input and conv filters."));
    }

    if (weights.shape()[0] != bias.shape().size())
    {
        throw std::invalid_argument(std::string("Number of filters and numbers of bias should be identical."));
    }

    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;

    auto the_dnnl_engine = OneDNNEngineHelper::get_dnnl_engine();
    dnnl::stream st(*the_dnnl_engine);

    //[Create network]
    this->m_net_fwd.clear();
    this->m_net_fwd_args.clear();
    //[Create network]

    // Resolve shapes
    lint w_n = weights.shape()[0];
    lint w_ch = weights.shape()[1];
    lint w_h = weights.shape()[2];
    lint w_w = weights.shape()[3];

    lint in_bat = input.shape()[0];
    lint in_ch = input.shape()[1];
    lint in_h = input.shape()[2];
    lint in_w = input.shape()[3];

    this->m_input = input;

    lint out_h = (in_h + this->m_pad_h * 2 - w_h) / this->m_stride_h + 1;
    lint out_w = (in_w + this->m_pad_w * 2 - w_w) / this->m_stride_w + 1;

    la::renew_if_shape_not_match(output, la::Shape{in_bat, w_n, out_h, out_w});

    dnnl::memory::dims conv_src_tz = {in_bat, in_ch, in_h, in_w};
    dnnl::memory::dims conv_weights_tz = {w_n, w_ch, w_h, w_w};
    dnnl::memory::dims conv_bias_tz = {w_n};
    dnnl::memory::dims conv_dst_tz = {in_bat, w_n, out_h, out_w};
    dnnl::memory::dims conv_strides = {this->m_stride_h, this->m_stride_w};
    dnnl::memory::dims conv_padding = {this->m_pad_h, this->m_pad_w};

    //[Create convolution memory descriptors]
    auto conv_src_md = dnnl::memory::desc({conv_src_tz}, dt::f32, tag::nchw);
    auto conv_bias_md = dnnl::memory::desc({conv_bias_tz}, dt::f32, tag::x);
    auto conv_weights_md = dnnl::memory::desc({conv_weights_tz}, dt::f32, tag::oihw);
    auto conv_dst_md = dnnl::memory::desc({conv_dst_tz}, dt::f32, tag::nchw);
    //[Create convolution memory descriptors]

    //[Create user memory]
    dnnl::memory conv_src_memory;
    dnnl::memory conv_weights_memory;
    dnnl::memory conv_bias_memory;
    dnnl::memory conv_dst_memory;

    auto input_cpu_ptr = input.get_cpu_instance();
    auto weight_cpu_ptr = weights.get_cpu_instance();
    auto bias_cpu_ptr = bias.get_cpu_instance();
    auto output_cpu_ptr = output.get_cpu_instance();
    
    if (std::is_same<DT, float>::value)
    {
        conv_src_memory = dnnl::memory(conv_src_md, *the_dnnl_engine, input_cpu_ptr->m_data);
        conv_weights_memory = dnnl::memory(conv_weights_md, *the_dnnl_engine, weight_cpu_ptr->m_data);
        conv_bias_memory = dnnl::memory(conv_bias_md, *the_dnnl_engine, bias_cpu_ptr->m_data);
        conv_dst_memory = dnnl::memory(conv_dst_md, *the_dnnl_engine, output_cpu_ptr->m_data);
    }
    else
    {
        conv_src_memory = dnnl::memory(conv_src_md, *the_dnnl_engine);
        conv_weights_memory = dnnl::memory(conv_weights_md, *the_dnnl_engine);
        conv_bias_memory = dnnl::memory(conv_bias_md, *the_dnnl_engine);
        conv_dst_memory = dnnl::memory(conv_dst_md, *the_dnnl_engine);

        //[Assign inputs to buffers]
        float *user_src_handle = static_cast<float *>(conv_src_memory.get_data_handle());
        lint lint_idx_size = input.shape().size();
        for (lint i = 0; i < lint_idx_size; ++i)
        {
            user_src_handle[i] = input_cpu_ptr->m_data[i];
        }

        float *user_weights_handle = static_cast<float *>(conv_weights_memory.get_data_handle());
        lint_idx_size = weights.shape().size();
        for (lint i = 0; i < lint_idx_size; ++i)
        {
            user_weights_handle[i] = weight_cpu_ptr->m_data[i];
        }

        float *user_bias_handle = static_cast<float *>(conv_bias_memory.get_data_handle());
        lint_idx_size = bias.shape().size();
        for (lint i = 0; i < lint_idx_size; ++i)
        {
            user_bias_handle[i] = bias_cpu_ptr->m_data[i];
        }
        //[Assign inputs to buffers]
    }

    //[Create convolution descriptor]
    auto conv_desc = 
        dnnl::convolution_forward::desc(dnnl::prop_kind::forward,
        dnnl::algorithm::convolution_direct, conv_src_md, conv_weights_md,
        conv_bias_md, conv_dst_md, conv_strides, conv_padding, conv_padding);
    //[Create convolution descriptor]

    //[Create convolution primitive descriptor]
    auto conv_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, *the_dnnl_engine);
    //[Create convolution primitive descriptor]

    //[Create memory for output]
    //auto conv_dst_memory = dnnl::memory(conv_prim_desc.dst_desc(), *the_dnnl_engine);
    //[Create memory for output]

    //[Create convolution primitive]
    this->m_net_fwd.push_back(dnnl::convolution_forward(conv_prim_desc));
    this->m_net_fwd_args.push_back({{DNNL_ARG_SRC, conv_src_memory},
            {DNNL_ARG_WEIGHTS, conv_weights_memory},
            {DNNL_ARG_BIAS, conv_bias_memory},
            {DNNL_ARG_DST, conv_dst_memory}});

    //[Execute model]
    assert(m_net_fwd.size() == m_net_fwd_args.size() && "something is missing");
    for (size_t i = 0; i < m_net_fwd.size(); ++i)
        m_net_fwd.at(i).execute(st, m_net_fwd_args.at(i));
    //[Execute model]

    // Synchronize convolution inference
    st.wait();

    if (std::is_same<DT, float>::value)
    {
    }
    else
    {
        //[Assign outputs to buffers]
        float *dst_handle = static_cast<float *>(conv_dst_memory.get_data_handle());
        lint lint_idx_size = output.shape().size();
        for (lint i = 0; i < lint_idx_size; ++i)
        {
            output_cpu_ptr->m_data[i] = dst_handle[i];
        }
        //[Assign outputs to buffers]
    }
}
template
void Conv2dOneDNN<int>::forward (
    iMatrix<int> &output, const iMatrix<int> & input, const iMatrix<int> & weights, const iMatrix<int> & bias);
template
void Conv2dOneDNN<float>::forward (
    iMatrix<float> &output, const iMatrix<float> & input, const iMatrix<float> & weights, const iMatrix<float> & bias);


template <typename DT>
void Conv2dOneDNN<DT>::backward (
        iMatrix<DT> & in_gradient_out,
        iMatrix<DT> & w_gradient_out,
        iMatrix<DT> & b_gradient_out,
        const iMatrix<DT> & gradient_in, const Shape & input_sh, const iMatrix<DT> & weights)
{
    if (input_sh.dim() != 4)
    {
        throw std::invalid_argument(std::string("Conv2dOneDNN input should be 4-dimensional."));
    }

    if (weights.shape().dim() != 4)
    {
        throw std::invalid_argument(std::string("Conv2dOneDNN weight should be 4-dimensional."));
    }

    if (input_sh[1] != weights.shape()[1])
    {
        throw std::invalid_argument(std::string("Number of channel should be consistent between input and conv filters."));
    }

    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;

    auto the_dnnl_engine = OneDNNEngineHelper::get_dnnl_engine();
    dnnl::stream st(*the_dnnl_engine);

    //[Create network]
    this->m_net_bwd.clear();
    this->m_net_bwd_args.clear();
    //[Create network]

    // Resolve shapes
    lint w_n = weights.shape()[0];
    lint w_ch = weights.shape()[1];
    lint w_h = weights.shape()[2];
    lint w_w = weights.shape()[3];

    lint in_bat = input_sh[0];
    lint in_ch = input_sh[1];
    lint in_h = input_sh[2];
    lint in_w = input_sh[3];

    lint out_h = (in_h + this->m_pad_h * 2 - w_h) / this->m_stride_h + 1;
    lint out_w = (in_w + this->m_pad_w * 2 - w_w) / this->m_stride_w + 1;
    
    la::renew_if_shape_not_match(in_gradient_out, la::Shape{in_bat, in_ch, in_h, in_w});
    la::renew_if_shape_not_match(w_gradient_out, la::Shape{w_n, w_ch, w_h, w_w});
    la::renew_if_shape_not_match(b_gradient_out, la::Shape{w_n});

    dnnl::memory::dims conv_src_tz = {in_bat, in_ch, in_h, in_w};
    dnnl::memory::dims conv_weights_tz = {w_n, w_ch, w_h, w_w};
    dnnl::memory::dims conv_bias_tz = {w_n};
    dnnl::memory::dims conv_dst_tz = {in_bat, w_n, out_h, out_w};
    dnnl::memory::dims conv_strides = {this->m_stride_h, this->m_stride_w};
    dnnl::memory::dims conv_padding = {this->m_pad_h, this->m_pad_w};

    //[Create convolution memory descriptors]
    auto conv_src_md = dnnl::memory::desc({conv_src_tz}, dt::f32, tag::nchw);
    auto conv_bias_md = dnnl::memory::desc({conv_bias_tz}, dt::f32, tag::x);
    auto conv_weights_md = dnnl::memory::desc({conv_weights_tz}, dt::f32, tag::oihw);
    auto conv_dst_md = dnnl::memory::desc({conv_dst_tz}, dt::f32, tag::nchw);
    //[Create convolution memory descriptors]

    auto conv_bwd_src_md = dnnl::memory::desc({conv_src_tz}, dt::f32, tag::nchw);
    auto conv_diff_bias_md = dnnl::memory::desc({conv_bias_tz}, dt::f32, tag::x);
    auto conv_diff_weights_md = dnnl::memory::desc({conv_weights_tz}, dt::f32, tag::oihw);
    // create memory descriptors for convolution
    auto conv_diff_src_md = dnnl::memory::desc({conv_src_tz}, dt::f32, tag::nchw);
    auto conv_diff_dst_md = dnnl::memory::desc({conv_dst_tz}, dt::f32, tag::nchw);

    dnnl::memory conv_diff_weights_memory;
    dnnl::memory conv_diff_bias_memory;
    dnnl::memory conv_diff_dst_memory;
    dnnl::memory conv_diff_src_memory;

    auto w_gradient_out_cpu_ptr = w_gradient_out.get_cpu_instance();
    auto b_gradient_out_cpu_ptr = b_gradient_out.get_cpu_instance();
    auto gradient_in_cpu_ptr = gradient_in.get_cpu_instance();
    auto in_gradient_out_cpu_ptr = in_gradient_out.get_cpu_instance();

    if (std::is_same<DT, float>::value)
    {
        conv_diff_weights_memory = dnnl::memory(conv_diff_weights_md, *the_dnnl_engine, w_gradient_out_cpu_ptr->m_data);
        conv_diff_bias_memory = dnnl::memory(conv_diff_bias_md, *the_dnnl_engine, b_gradient_out_cpu_ptr->m_data);
        conv_diff_dst_memory = dnnl::memory(conv_diff_dst_md, *the_dnnl_engine, gradient_in_cpu_ptr->m_data);
        conv_diff_src_memory = dnnl::memory(conv_diff_src_md, *the_dnnl_engine, in_gradient_out_cpu_ptr->m_data);
    }
    else
    {
        conv_diff_weights_memory = dnnl::memory(conv_diff_weights_md, *the_dnnl_engine);
        conv_diff_bias_memory = dnnl::memory(conv_diff_bias_md, *the_dnnl_engine);
        conv_diff_dst_memory = dnnl::memory(conv_diff_dst_md, *the_dnnl_engine);
        conv_diff_src_memory = dnnl::memory(conv_diff_src_md, *the_dnnl_engine);
        
        float *diff_dst_handle = static_cast<float *>(conv_diff_dst_memory.get_data_handle());
        lint lint_idx_size = gradient_in.shape().size();
        for (lint i = 0; i < lint_idx_size; ++i)
        {
            diff_dst_handle[i] = gradient_in_cpu_ptr->m_data[i];
        }
    }

    //[Create convolution descriptor]
    auto conv_desc = 
        dnnl::convolution_forward::desc(dnnl::prop_kind::forward,
        dnnl::algorithm::convolution_direct, conv_src_md, conv_weights_md,
        conv_bias_md, conv_dst_md, conv_strides, conv_padding, conv_padding);
    //[Create convolution descriptor]

    //[Create convolution primitive descriptor]
    auto conv_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, *the_dnnl_engine);
    //[Create convolution primitive descriptor]

    // create data backward descriptor
    auto conv_bwd_data_desc = dnnl::convolution_backward_data::desc(dnnl::algorithm::convolution_direct,
            conv_diff_src_md, conv_diff_weights_md, conv_diff_dst_md, conv_strides,
            conv_padding, conv_padding);

    // backward primitive descriptor needs to hint forward descriptor
    auto conv_bwd_data_prim_desc
            = dnnl::convolution_backward_data::primitive_desc(conv_bwd_data_desc, *the_dnnl_engine, conv_prim_desc);
            
    // create weights backward descriptor
    auto conv_bwd_weights_desc
            = dnnl::convolution_backward_weights::desc(dnnl::algorithm::convolution_direct,
                    conv_bwd_src_md, conv_diff_weights_md, conv_diff_bias_md,
                    conv_diff_dst_md, conv_strides, conv_padding, conv_padding);
    
    auto conv_bwd_weights_prim_desc = dnnl::convolution_backward_weights::primitive_desc(
            conv_bwd_weights_desc, *the_dnnl_engine, conv_prim_desc);  
    
    // create backward convolution weights primitive
    this->m_net_bwd.push_back(dnnl::convolution_backward_weights(conv_bwd_weights_prim_desc));
    this->m_net_bwd_args.push_back({{DNNL_ARG_SRC, this->m_net_fwd_args[0][DNNL_ARG_SRC]},
            {DNNL_ARG_DIFF_DST, conv_diff_dst_memory},
            // delay putting DIFF_WEIGHTS until reorder (if needed)
            {DNNL_ARG_DIFF_WEIGHTS, conv_diff_weights_memory},
            {DNNL_ARG_DIFF_BIAS, conv_diff_bias_memory}});

    // create backward convolution data primitive
    this->m_net_bwd.push_back(dnnl::convolution_backward_data(conv_bwd_data_prim_desc));
    this->m_net_bwd_args.push_back({
            {DNNL_ARG_DIFF_DST, conv_diff_dst_memory},
            {DNNL_ARG_WEIGHTS, this->m_net_fwd_args[0][DNNL_ARG_WEIGHTS]},
            {DNNL_ARG_DIFF_SRC, conv_diff_src_memory}});

//////////////////////////////////////////
//       Forward & Backward Action      //
//////////////////////////////////////////

    //[Execute model]
    assert(this->m_net_bwd.size() == this->m_net_bwd_args.size() && "something is missing");
    for (size_t i = 0; i < this->m_net_bwd.size(); ++i)
    {
        this->m_net_bwd.at(i).execute(st, this->m_net_bwd_args.at(i));
    }
    //[Execute model]

    // Synchronize convolution inference
    st.wait();

    //[Assign outputs to buffers]
    if (std::is_same<DT, float>::value)
    {
    }
    else
    {
        float *net_diff_src_handle = static_cast<float *>(conv_diff_src_memory.get_data_handle());
        lint lint_idx_size = in_gradient_out.shape().size();
        for (lint i = 0; i < lint_idx_size; ++i)
        {
            in_gradient_out_cpu_ptr->m_data[i] = net_diff_src_handle[i];
        }

        float *net_diff_weights_handle = static_cast<float *>(conv_diff_weights_memory.get_data_handle());
        lint_idx_size = w_gradient_out.shape().size();
        for (lint i = 0; i < lint_idx_size; ++i)
        {
            w_gradient_out_cpu_ptr->m_data[i] = net_diff_weights_handle[i];
        }

        float *net_diff_bias_handle = static_cast<float *>(conv_diff_bias_memory.get_data_handle());
        lint_idx_size = b_gradient_out.shape().size();
        for (lint i = 0; i < lint_idx_size; ++i)
        {
            b_gradient_out_cpu_ptr->m_data[i] = net_diff_bias_handle[i];
        }
    }
    //[Assign outputs to buffers]
/*
    std::cout << "gradient_in: " << gradient_in << std::endl;
    std::cout << "in_gradient_out: " << in_gradient_out << std::endl;
    std::cout << "w_gradient_out: " << w_gradient_out << std::endl;
    std::cout << "b_gradient_out: " << b_gradient_out << std::endl;
*/
}
template
void Conv2dOneDNN<int>::backward (
        iMatrix<int> & in_gradient_out,
        iMatrix<int> & w_gradient_out,
        iMatrix<int> & b_gradient_out,
        const iMatrix<int> & gradient_in, const Shape & input_sh, const iMatrix<int> & weights);
template
void Conv2dOneDNN<float>::backward (
        iMatrix<float> & in_gradient_out,
        iMatrix<float> & w_gradient_out,
        iMatrix<float> & b_gradient_out,
        const iMatrix<float> & gradient_in, const Shape & input_sh, const iMatrix<float> & weights);



template <typename DT>
void Conv2dOneDNN<DT>::clear_cache()
{
    this->m_input = iMatrix<DT> {};
    this->m_net_fwd.clear();
    this->m_net_fwd_args.clear();
    this->m_net_bwd.clear();
    this->m_net_bwd_args.clear();
}
template
void Conv2dOneDNN<int>::clear_cache();
template
void Conv2dOneDNN<float>::clear_cache();


} // la
} // julie
