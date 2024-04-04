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

#include "Pool2dOneDNN.hpp"
#include "iMatrix_func.hpp"
#include "OneDNNHelper.hpp"
//#include "dnnl_utilities.hpp"

namespace julie
{
namespace la
{

//template <typename DT>
//std::shared_ptr<dnnl::engine> Pool2dOneDNN<DT>::the_dnnl_engine = nullptr;


template <typename DT>
Pool2dOneDNN<DT>::Pool2dOneDNN (
    lint pad_h, lint pad_w, lint kernel_h, lint kernel_w, lint stride_h, lint stride_w,
    PoolType type)
    :
    m_pad_h {pad_h},
    m_pad_w {pad_w},
    m_kernel_h {kernel_h},
    m_kernel_w {kernel_w},
    m_stride_h {stride_h},
    m_stride_w {stride_w},
    m_type {type}
{}
template
Pool2dOneDNN<int>::Pool2dOneDNN (
    lint pad_h, lint pad_w, lint kernel_h, lint kernel_w, lint stride_h, lint stride_w,
    PoolType type);
template
Pool2dOneDNN<float>::Pool2dOneDNN (
    lint pad_h, lint pad_w, lint kernel_h, lint kernel_w, lint stride_h, lint stride_w,
    PoolType type);


template <typename DT>
void Pool2dOneDNN<DT>::forward (iMatrix<DT> &output, const iMatrix<DT> & input)
{
    if (input.shape().dim() != 4)
    {
        throw std::invalid_argument(std::string("pool_2d input should be 4-dimensional."));
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
    lint in_bat = input.shape()[0];
    lint in_ch = input.shape()[1];
    lint in_h = input.shape()[2];
    lint in_w = input.shape()[3];

    lint out_h = (in_h + this->m_pad_h * 2 - this->m_kernel_h) / this->m_stride_h + 1;
    lint out_w = (in_w + this->m_pad_w * 2 - this->m_kernel_w) / this->m_stride_w + 1;

    la::renew_if_shape_not_match(output, la::Shape{in_bat, in_ch, out_h, out_w});

    this->m_input = input;

    dnnl::memory::dims pool_src_tz = {in_bat, in_ch, in_h, in_w};
    dnnl::memory::dims pool_dst_tz = {in_bat, in_ch, out_h, out_w};
    dnnl::memory::dims pool_kernel = {this->m_kernel_h, this->m_kernel_w};
    dnnl::memory::dims pool_strides = {this->m_stride_h, this->m_stride_w};
    dnnl::memory::dims pool_padding = {this->m_pad_h, this->m_pad_w};
    
    auto pool_src_md = dnnl::memory::desc({pool_src_tz}, dt::f32, tag::nchw);
    auto pool_dst_md = dnnl::memory::desc({pool_dst_tz}, dt::f32, tag::nchw);
    
    dnnl::algorithm pool_type = dnnl::algorithm::pooling_max;
    if (PoolType::Max == this->m_type)
    {
        pool_type = dnnl::algorithm::pooling_max;
    }
    else if (PoolType::Avg == this->m_type)
    {
        pool_type = dnnl::algorithm::pooling_avg;
    }
    //[Create pooling primitive]
    auto pool_desc = dnnl::pooling_forward::desc(
        dnnl::prop_kind::forward,
        pool_type,
        pool_src_md, pool_dst_md,
        pool_strides, pool_kernel, pool_padding, pool_padding);

    auto pool_pd = dnnl::pooling_forward::primitive_desc(pool_desc, *the_dnnl_engine);

    dnnl::memory pool_src_memory;
    dnnl::memory pool_dst_memory;

    auto input_cpu_ptr = input.get_cpu_instance();
    auto output_cpu_ptr = output.get_cpu_instance();

    if (std::is_same<DT, float>::value)
    {
        pool_src_memory = dnnl::memory(pool_src_md, *the_dnnl_engine, input_cpu_ptr->m_data);
        pool_dst_memory = dnnl::memory(pool_dst_md, *the_dnnl_engine, output_cpu_ptr->m_data);
    }
    else
    {
        pool_src_memory = dnnl::memory(pool_src_md, *the_dnnl_engine);
        pool_dst_memory = dnnl::memory(pool_dst_md, *the_dnnl_engine);

        float *user_src_handle = static_cast<float *>(pool_src_memory.get_data_handle());
        lint lint_idx_size = input.shape().size();
        for (lint i = 0; i < lint_idx_size; ++i)
        {
            user_src_handle[i] = input_cpu_ptr->m_data[i];
        }
    }

    this->m_pool_workspace_memory = dnnl::memory(pool_pd.workspace_desc(), *the_dnnl_engine);
    
    this->m_net_fwd.push_back(dnnl::pooling_forward(pool_pd));
    this->m_net_fwd_args.push_back({
        {DNNL_ARG_SRC, pool_src_memory},
        {DNNL_ARG_WORKSPACE, this->m_pool_workspace_memory},
        {DNNL_ARG_DST, pool_dst_memory}});
    //[Create pooling primitive]

    //[Execute model]
    assert(this->m_net_fwd.size() == this->m_net_fwd_args.size() && "something is missing");
    for (size_t i = 0; i < this->m_net_fwd.size(); ++i)
        this->m_net_fwd.at(i).execute(st, this->m_net_fwd_args.at(i));
    //[Execute model]

    // Synchronize convolution inference
    st.wait();

    if (std::is_same<DT, float>::value)
    {
    }
    else
    {
        //[Assign outputs to buffers]
        float *dst_handle = static_cast<float *>(pool_dst_memory.get_data_handle());
        lint lint_idx_size = output.shape().size();
        for (lint i = 0; i < lint_idx_size; ++i)
        {
            output_cpu_ptr->m_data[i] = dst_handle[i];
        }
        //[Assign outputs to buffers]
    }
}
template
void Pool2dOneDNN<int>::forward (iMatrix<int> &output, const iMatrix<int> & input);
template
void Pool2dOneDNN<float>::forward (iMatrix<float> &output, const iMatrix<float> & input);


template <typename DT>
void Pool2dOneDNN<DT>::backward (iMatrix<DT> & in_gradient_out, const iMatrix<DT> & gradient_in, const iMatrix<DT> &input)
{
    if (gradient_in.shape().dim() != 4)
    {
        throw std::invalid_argument(std::string("pool_2d input should be 4-dimensional."));
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
    lint in_bat = input.shape()[0];
    lint in_ch = input.shape()[1];
    lint in_h = input.shape()[2];
    lint in_w = input.shape()[3];

    lint out_h = (in_h + this->m_pad_h * 2 - this->m_kernel_h) / this->m_stride_h + 1;
    lint out_w = (in_w + this->m_pad_w * 2 - this->m_kernel_w) / this->m_stride_w + 1;
    
    la::renew_if_shape_not_match(in_gradient_out, la::Shape{in_bat, in_ch, in_h, in_w});

    dnnl::memory::dims pool_src_tz = {in_bat, in_ch, in_h, in_w};
    dnnl::memory::dims pool_dst_tz = {in_bat, in_ch, out_h, out_w};
    dnnl::memory::dims pool_kernel = {this->m_kernel_h, this->m_kernel_w};
    dnnl::memory::dims pool_strides = {this->m_stride_h, this->m_stride_w};
    dnnl::memory::dims pool_padding = {this->m_pad_h, this->m_pad_w};

    // Backward pooling
    // create memory descriptors for pooling
    auto pool_src_md = dnnl::memory::desc({pool_src_tz}, dt::f32, tag::nchw);
    auto pool_dst_md = dnnl::memory::desc({pool_dst_tz}, dt::f32, tag::nchw);
    auto pool_diff_src_md = dnnl::memory::desc({pool_src_tz}, dt::f32, tag::nchw);
    auto pool_diff_dst_md = dnnl::memory::desc({pool_dst_tz}, dt::f32, tag::nchw);

    dnnl::algorithm pool_type = dnnl::algorithm::pooling_max;
    if (PoolType::Max == this->m_type)
    {
        pool_type = dnnl::algorithm::pooling_max;
    }
    else if (PoolType::Avg == this->m_type)
    {
        pool_type = dnnl::algorithm::pooling_avg;
    }

    auto pool_desc = dnnl::pooling_forward::desc(
        dnnl::prop_kind::forward,
        pool_type,
        pool_src_md, pool_dst_md,
        pool_strides, pool_kernel, pool_padding, pool_padding);

    auto pool_pd = dnnl::pooling_forward::primitive_desc(pool_desc, *the_dnnl_engine);
    
    // create backward pooling descriptor
    auto pool_bwd_desc = dnnl::pooling_backward::desc(
        pool_type,
        pool_diff_src_md, pool_diff_dst_md,
        pool_strides, pool_kernel, pool_padding, pool_padding);
    
    // backward primitive descriptor needs to hint forward descriptor
    auto pool_bwd_pd = dnnl::pooling_backward::primitive_desc(pool_bwd_desc, *the_dnnl_engine, pool_pd);

    dnnl::memory pool_diff_src_memory;
    dnnl::memory pool_diff_dst_memory;

    auto gradient_dst_cpu_ptr = gradient_in.get_cpu_instance();
    auto gradient_src_cpu_ptr = in_gradient_out.get_cpu_instance();

    if (std::is_same<DT, float>::value)
    {
        pool_diff_src_memory = dnnl::memory(pool_diff_src_md, *the_dnnl_engine, gradient_src_cpu_ptr->m_data);
        pool_diff_dst_memory = dnnl::memory(pool_diff_dst_md, *the_dnnl_engine, gradient_dst_cpu_ptr->m_data);
    }
    else
    {
        pool_diff_src_memory = dnnl::memory(pool_diff_src_md, *the_dnnl_engine);
        pool_diff_dst_memory = dnnl::memory(pool_diff_dst_md, *the_dnnl_engine);

        float *diff_dst_handle = static_cast<float *>(pool_diff_dst_memory.get_data_handle());
        lint lint_idx_size = gradient_in.shape().size();
        for (lint i = 0; i < lint_idx_size; ++i)
        {
            diff_dst_handle[i] = gradient_dst_cpu_ptr->m_data[i];
        }
    }

    //auto pool_workspace_memory = dnnl::memory(pool_pd.workspace_desc(), *the_dnnl_engine);

    // finally create backward pooling primitive
    this->m_net_bwd.push_back(dnnl::pooling_backward(pool_bwd_pd));
    this->m_net_bwd_args.push_back({
        {DNNL_ARG_DIFF_DST, pool_diff_dst_memory},
        {DNNL_ARG_DIFF_SRC, pool_diff_src_memory},
        {DNNL_ARG_WORKSPACE, this->m_pool_workspace_memory}
        });

    //[Execute model]
    assert(this->m_net_bwd.size() == this->m_net_bwd_args.size() && "something is missing");
    for (size_t i = 0; i < this->m_net_bwd.size(); ++i)
        this->m_net_bwd.at(i).execute(st, this->m_net_bwd_args.at(i));
    //[Execute model]

    // Synchronize pooling backward
    st.wait();

    if (std::is_same<DT, float>::value)
    {
    }
    else
    {
        //[Assign outputs to buffers]
        float *diff_src_handle = static_cast<float *>(pool_diff_src_memory.get_data_handle());
        lint lint_idx_size = in_gradient_out.shape().size();
        for (lint i = 0; i < lint_idx_size; ++i)
        {
            gradient_src_cpu_ptr->m_data[i] = diff_src_handle[i];
        }
        //[Assign outputs to buffers]
    }
}
template
void Pool2dOneDNN<int>::backward (iMatrix<int> & in_gradient_out, const iMatrix<int> & gradient_in, const iMatrix<int> &input);
template
void Pool2dOneDNN<float>::backward (iMatrix<float> & in_gradient_out, const iMatrix<float> & gradient_in, const iMatrix<float> &input);



template <typename DT>
void Pool2dOneDNN<DT>::clear_cache()
{
    this->m_input = iMatrix<DT> {};
    this->m_net_fwd.clear();
    this->m_net_fwd_args.clear();
    this->m_net_bwd.clear();
    this->m_net_bwd_args.clear();
}
template
void Pool2dOneDNN<int>::clear_cache();
template
void Pool2dOneDNN<float>::clear_cache();


} // la
} // julie
