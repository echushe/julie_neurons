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

#include "Losses_CPU.hpp"
#include "Matrix_CPU_func.hpp"
#include <cmath>
#include <algorithm>

namespace julie
{
namespace la
{
namespace cpu
{

template <typename DT>
DT HalfSquareError<DT>::elementary_HalfSquareError(DT *diff, const DT *target, const DT *input, lint size, lint r_sh_size)
{
    const DT *input_p = input;
    const DT *target_p = target;
    DT *diff_p = diff;
    
    DT sum = 0;

    for (lint i = 0; i < size; ++i)
    {
        DT sub = *input_p - *target_p;
        sum += sub * sub;

        *diff_p = sub;

        input_p += r_sh_size;
        target_p += r_sh_size;
        diff_p += r_sh_size;
    }

    return sum / 2.0;
}


template <typename DT>
void HalfSquareError<DT>::operator()(Matrix_CPU<DT> &loss, Matrix_CPU<DT> & diff, const Matrix_CPU<DT> & target, const Matrix_CPU<DT> & input)
{
    if (diff.m_shape != input.m_shape)
    {
        diff = Matrix_CPU<DT> { input.m_shape };
    }

    Shape l_sh = input.m_shape.sub_shape(0, this->m_axis - 1);
    Shape r_sh = input.m_shape.sub_shape(this->m_axis + 1, input.m_shape.dim() - 1);

    lint n_ele = std::max<lint>(l_sh.size(), 1);
    lint my_axis_size = input.m_shape[this->m_axis];
    lint r_sh_size = std::max<lint>(r_sh.size(), 1);
    lint step = my_axis_size * r_sh_size;

    DT *diff_data = diff.m_data;
    DT *t_data = target.m_data;
    DT *in_data = input.m_data;

    renew_if_shape_not_match(loss, l_sh + r_sh);

    DT *loss_mat_data = loss.m_data;

    for (lint i = 0; i < n_ele; ++i)
    {
        DT *r_diff_data = diff_data;
        DT *r_t_data = t_data;
        DT *r_in_data = in_data;

        for (lint j = 0; j < r_sh_size; ++j)
        {
            *loss_mat_data = this->elementary_HalfSquareError(r_diff_data, r_t_data, r_in_data, my_axis_size, r_sh_size);
            ++r_diff_data;
            ++r_t_data;
            ++r_in_data;

            ++loss_mat_data;
        }

        diff_data += step;
        t_data += step;
        in_data += step;
    }
}
template
void HalfSquareError<float>::operator()(
    Matrix_CPU<float> &loss, Matrix_CPU<float> & diff, const Matrix_CPU<float> & target, const Matrix_CPU<float> & input);


template <typename DT>
DT Sigmoid_CrossEntropy<DT>::elementary_sigmoid_crossentropy(
    DT *sigmoid, DT *diff, const DT *target, const DT *input, lint size, lint r_sh_size)
{
    const DT *input_p = input;
    DT *diff_p = diff;
    DT *sigmoid_p = sigmoid;

    for (lint i = 0; i < size; ++i)
    {
        *sigmoid_p = 1.0 / (1.0 + exp(*input_p * (-1)));

        input_p += r_sh_size;
        sigmoid_p += r_sh_size;        
    }

    DT centropy_sum = 0;

    const DT *target_p = target;
    sigmoid_p = sigmoid;
    diff_p = diff;
    for (lint i = 0; i < size; ++i)
    {
        centropy_sum += *target_p * log(*sigmoid_p);
        *diff_p = *sigmoid_p - *target_p;

        target_p += r_sh_size;
        sigmoid_p += r_sh_size;
        diff_p += r_sh_size;
    }

    centropy_sum *= -1;

    return centropy_sum;
}


template <typename DT>
void Sigmoid_CrossEntropy<DT>::operator()(Matrix_CPU<DT> &loss, Matrix_CPU<DT> & diff, const Matrix_CPU<DT> & target, const Matrix_CPU<DT> & input)
{
    if (this->m_sigmoid_cache.m_shape != input.m_shape)
    {
        this->m_sigmoid_cache = Matrix_CPU<DT> { input.m_shape };
    }

    if (diff.m_shape != input.m_shape)
    {
        diff = Matrix_CPU<DT> { input.m_shape };
    }

    Shape l_sh = input.m_shape.sub_shape(0, this->m_axis - 1);
    Shape r_sh = input.m_shape.sub_shape(this->m_axis + 1, input.m_shape.dim() - 1);

    lint n_ele = std::max<lint>(l_sh.size(), 1);
    lint my_axis_size = input.m_shape[this->m_axis];
    lint r_sh_size = std::max<lint>(r_sh.size(), 1);
    lint step = my_axis_size * r_sh_size;

    DT *sigmoid = this->m_sigmoid_cache.m_data;
    DT *diff_data = diff.m_data;
    DT *t_data = target.m_data;
    DT *in_data = input.m_data;

    renew_if_shape_not_match(loss, l_sh + r_sh);

    DT *loss_mat_data = loss.m_data;

    for (lint i = 0; i < n_ele; ++i)
    {
        DT *r_sigmoid = sigmoid;
        DT *r_diff_data = diff_data;
        DT *r_t_data = t_data;
        DT *r_in_data = in_data;

        for (lint j = 0; j < r_sh_size; ++j)
        {
            *loss_mat_data = this->elementary_sigmoid_crossentropy(r_sigmoid, r_diff_data, r_t_data, r_in_data, my_axis_size, r_sh_size);
            ++r_sigmoid;
            ++r_diff_data;
            ++r_t_data;
            ++r_in_data;

            ++loss_mat_data;
        }

        sigmoid += step;
        diff_data += step;
        t_data += step;
        in_data += step;
    }
}
template
void Sigmoid_CrossEntropy<float>::operator()(
    Matrix_CPU<float> &loss, Matrix_CPU<float> & diff, const Matrix_CPU<float> & target, const Matrix_CPU<float> & input);


template <typename DT>
DT SoftMax_CrossEntropy<DT>::elementary_softmax_crossentropy(
    DT *softmax, DT *diff, const DT *target, const DT *input, lint size, lint r_sh_size)
{
    const DT *input_p = input;
    DT *diff_p = diff;
    DT *softmax_p = softmax;

    DT sum = 0;
    for (lint i = 0; i < size; ++i)
    {
        *softmax_p = exp(*input_p);
        sum += *softmax_p;

        input_p += r_sh_size;
        softmax_p += r_sh_size;        
    }

    DT centropy_sum = 0;

    const DT *target_p = target;
    softmax_p = softmax;
    diff_p = diff;
    for (lint i = 0; i < size; ++i)
    {
        *softmax_p /= sum;

        centropy_sum += *target_p * log(*softmax_p);
        *diff_p = *softmax_p - *target_p;

        target_p += r_sh_size;
        softmax_p += r_sh_size;
        diff_p += r_sh_size;
    }

    centropy_sum *= -1;

    return centropy_sum;
}

template <typename DT>
void SoftMax_CrossEntropy<DT>::operator()(Matrix_CPU<DT> &loss, Matrix_CPU<DT> & diff, const Matrix_CPU<DT> & target, const Matrix_CPU<DT> & input)
{
    if (this->m_softmax_cache.m_shape != input.m_shape)
    {
        this->m_softmax_cache = Matrix_CPU<DT> { input.m_shape };
    }

    if (diff.m_shape != input.m_shape)
    {
        diff = Matrix_CPU<DT> { input.m_shape };
    }

    Shape l_sh = input.m_shape.sub_shape(0, this->m_axis - 1);
    Shape r_sh = input.m_shape.sub_shape(this->m_axis + 1, input.m_shape.dim() - 1);

    lint n_ele = std::max<lint>(l_sh.size(), 1);
    lint my_axis_size = input.m_shape[this->m_axis];
    lint r_sh_size = std::max<lint>(r_sh.size(), 1);
    lint step = my_axis_size * r_sh_size;

    DT *softmax = this->m_softmax_cache.m_data;
    DT *diff_data = diff.m_data;
    DT *t_data = target.m_data;
    DT *in_data = input.m_data;

    renew_if_shape_not_match(loss, l_sh + r_sh);

    DT *loss_mat_data = loss.m_data;

    for (lint i = 0; i < n_ele; ++i)
    {
        DT *r_softmax = softmax;
        DT *r_diff_data = diff_data;
        DT *r_t_data = t_data;
        DT *r_in_data = in_data;

        for (lint j = 0; j < r_sh_size; ++j)
        {
            *loss_mat_data = this->elementary_softmax_crossentropy(r_softmax, r_diff_data, r_t_data, r_in_data, my_axis_size, r_sh_size);
            ++r_softmax;
            ++r_diff_data;
            ++r_t_data;
            ++r_in_data;

            ++loss_mat_data;
        }

        softmax += step;
        diff_data += step;
        t_data += step;
        in_data += step;
    }

    // std::cout << this->m_softmax_cache << std::endl;
}
template
void SoftMax_CrossEntropy<float>::operator()(
    Matrix_CPU<float> &loss, Matrix_CPU<float> & diff, const Matrix_CPU<float> & target, const Matrix_CPU<float> & input);


}  // namespace cpu
}  // namespace la
}  // namespace julie

