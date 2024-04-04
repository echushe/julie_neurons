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

#include "Activations_CPU.hpp"
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
void Linear<DT>::operator () (Matrix_CPU<DT> & output, Matrix_CPU<DT> & diff, const Matrix_CPU<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);
    renew_if_shape_not_match(diff, in.m_shape);

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = in.m_data[i];
        diff.m_data[i] = 1;
    }
}
template
void Linear<float>::operator () (Matrix_CPU<float> & output, Matrix_CPU<float> & diff, const Matrix_CPU<float> & in);


template <typename DT>
void Linear<DT>::operator () (Matrix_CPU<DT> & output, const Matrix_CPU<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = in.m_data[i];
    }
}
template
void Linear<float>::operator () (Matrix_CPU<float> & output, const Matrix_CPU<float> & in);


template <typename DT>
void Sigmoid<DT>::operator () (Matrix_CPU<DT> & output, Matrix_CPU<DT> & diff, const Matrix_CPU<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);
    renew_if_shape_not_match(diff, in.m_shape);

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        DT y = 1.0 / (1.0 + exp(-in.m_data[i]));
        output.m_data[i] = y;
        diff.m_data[i] = y * (1 - y);
    }
}
template
void Sigmoid<float>::operator () (Matrix_CPU<float> & output, Matrix_CPU<float> & diff, const Matrix_CPU<float> & in);


template <typename DT>
void Sigmoid<DT>::operator () (Matrix_CPU<DT> & output, const Matrix_CPU<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = 1.0 / (1.0 + exp(-in.m_data[i]));
    }
}
template
void Sigmoid<float>::operator () (Matrix_CPU<float> & output, const Matrix_CPU<float> & in);


template <typename DT>
void TanH<DT>::operator () (Matrix_CPU<DT> & output, Matrix_CPU<DT> & diff, const Matrix_CPU<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);
    renew_if_shape_not_match(diff, in.m_shape);

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        DT y = tanh(in.m_data[i]);
        output.m_data[i] = y;
        diff.m_data[i] = 1 - y * y;
    }
}
template
void TanH<float>::operator () (Matrix_CPU<float> & output, Matrix_CPU<float> & diff, const Matrix_CPU<float> & in);


template <typename DT>
void TanH<DT>::operator () (Matrix_CPU<DT> & output, const Matrix_CPU<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = tanh(in.m_data[i]);
    }
}
template
void TanH<float>::operator () (Matrix_CPU<float> & output, const Matrix_CPU<float> & in);


template <typename DT>
void ReLU<DT>::operator () (Matrix_CPU<DT> & output, Matrix_CPU<DT> & diff, const Matrix_CPU<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);
    renew_if_shape_not_match(diff, in.m_shape);

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        DT x = in.m_data[i];

        if (x >= 0)
        {
            output.m_data[i] = x;
            diff.m_data[i] = 1;
        }
        else
        {
            output.m_data[i] = 0;
            diff.m_data[i] = 0;
        }
    }
}
template
void ReLU<float>::operator () (Matrix_CPU<float> & output, Matrix_CPU<float> & diff, const Matrix_CPU<float> & in);


template <typename DT>
void ReLU<DT>::operator () (Matrix_CPU<DT> & output, const Matrix_CPU<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        DT x = in.m_data[i];

        if (x >= 0)
        {
            output.m_data[i] = x;
        }
        else
        {
            output.m_data[i] = 0;
        }
    }
}
template
void ReLU<float>::operator () (Matrix_CPU<float> & output, const Matrix_CPU<float> & in);


template <typename DT>
void Abs<DT>::operator () (Matrix_CPU<DT> & output, Matrix_CPU<DT> & diff, const Matrix_CPU<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);
    renew_if_shape_not_match(diff, in.m_shape);

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        DT x = in.m_data[i];

        if (x >= 0)
        {
            output.m_data[i] = x;
            diff.m_data[i] = 1;
        }
        else
        {
            output.m_data[i] = -x;
            diff.m_data[i] = -1;
        }
    }
}
template
void Abs<float>::operator () (Matrix_CPU<float> & output, Matrix_CPU<float> & diff, const Matrix_CPU<float> & in);


template <typename DT>
void Abs<DT>::operator () (Matrix_CPU<DT> & output, const Matrix_CPU<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        DT x = in.m_data[i];

        if (x >= 0)
        {
            output.m_data[i] = x;
        }
        else
        {
            output.m_data[i] = -x;
        }
    }
}
template
void Abs<float>::operator () (Matrix_CPU<float> & output, const Matrix_CPU<float> & in);


template <typename DT>
void LeakyReLU<DT>::operator () (Matrix_CPU<DT> & output, Matrix_CPU<DT> & diff, const Matrix_CPU<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);
    renew_if_shape_not_match(diff, in.m_shape);

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        DT x = in.m_data[i];

        if (x >= 0)
        {
            output.m_data[i] = x;
            diff.m_data[i] = 1;
        }
        else
        {
            output.m_data[i] = 0.01 * x;
            diff.m_data[i] = 0.01;
        }
    }
}
template
void LeakyReLU<float>::operator () (Matrix_CPU<float> & output, Matrix_CPU<float> & diff, const Matrix_CPU<float> & in);


template <typename DT>
void LeakyReLU<DT>::operator () (Matrix_CPU<DT> & output, const Matrix_CPU<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        DT x = in.m_data[i];

        if (x >= 0)
        {
            output.m_data[i] = x;
        }
        else
        {
            output.m_data[i] = 0.01 * x;
        }
    }
}
template
void LeakyReLU<float>::operator () (Matrix_CPU<float> & output, const Matrix_CPU<float> & in);


template <typename DT>
void ArcTan<DT>::operator()(Matrix_CPU<DT>& output, Matrix_CPU<DT>& diff, const Matrix_CPU<DT>& in)
{
    renew_if_shape_not_match(output, in.m_shape);
    renew_if_shape_not_match(diff, in.m_shape);

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = atan(in.m_data[i]);
        diff.m_data[i] = 1 / (1 + in.m_data[i] * in.m_data[i]);
    }
}
template
void ArcTan<float>::operator()(Matrix_CPU<float>& output, Matrix_CPU<float>& diff, const Matrix_CPU<float>& in);


template <typename DT>
void ArcTan<DT>::operator()(Matrix_CPU<DT>& output, const Matrix_CPU<DT>& in)
{
    renew_if_shape_not_match(output, in.m_shape);

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = atan(in.m_data[i]);
    }
}
template
void ArcTan<float>::operator()(Matrix_CPU<float>& output, const Matrix_CPU<float>& in);


template <typename DT>
void Sin<DT>::operator()(Matrix_CPU<DT>& output, Matrix_CPU<DT>& diff, const Matrix_CPU<DT>& in)
{
    renew_if_shape_not_match(output, in.m_shape);
    renew_if_shape_not_match(diff, in.m_shape);

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = sin(in.m_data[i]);
        diff.m_data[i] = cos(in.m_data[i]);
    }
}
template
void Sin<float>::operator()(Matrix_CPU<float>& output, Matrix_CPU<float>& diff, const Matrix_CPU<float>& in);


template <typename DT>
void Sin<DT>::operator()(Matrix_CPU<DT>& output, const Matrix_CPU<DT>& in)
{
    renew_if_shape_not_match(output, in.m_shape);

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = sin(in.m_data[i]);
    }
}
template
void Sin<float>::operator()(Matrix_CPU<float>& output, const Matrix_CPU<float>& in);


template <typename DT>
void SoftSign<DT>::operator()(Matrix_CPU<DT>& output, Matrix_CPU<DT>& diff, const Matrix_CPU<DT>& in)
{
    renew_if_shape_not_match(output, in.m_shape);
    renew_if_shape_not_match(diff, in.m_shape);

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        DT d = 1 + fabs(in.m_data[i]);
        output.m_data[i] = in.m_data[i] / d;
        diff.m_data[i] = 1 / (d * d);
    }
}
template
void SoftSign<float>::operator()(Matrix_CPU<float>& output, Matrix_CPU<float>& diff, const Matrix_CPU<float>& in);


template <typename DT>
void SoftSign<DT>::operator()(Matrix_CPU<DT>& output, const Matrix_CPU<DT>& in)
{
    renew_if_shape_not_match(output, in.m_shape);

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = in.m_data[i] / (1 + fabs(in.m_data[i]));
    }
}
template
void SoftSign<float>::operator()(Matrix_CPU<float>& output, const Matrix_CPU<float>& in);


template <typename DT>
void SoftMax<DT>::elementary_softmax(DT *output, const DT *input, lint size, lint r_sh_size)
{
    const DT *input_p = input;
    DT *output_p = output;

    DT sum = 0;
    for (lint i = 0; i < size; ++i)
    {
        *output_p = exp(*input_p);
        sum += *output_p;

        input_p += r_sh_size;
        output_p += r_sh_size;
    }

    output_p = output;
    for (lint i = 0; i < size; ++i)
    {
        *output_p /= sum;

        output_p += r_sh_size;
    }
}


template <typename DT>
void SoftMax<DT>::elementary_softmax(DT *output, DT *diff, const DT *input, lint size, lint r_sh_size)
{
    const DT *input_p = input;
    DT *diff_p = diff;
    DT *output_p = output;

    DT sum = 0;
    for (lint i = 0; i < size; ++i)
    {
        *output_p = exp(*input_p);
        sum += *output_p;

        input_p += r_sh_size;
        output_p += r_sh_size;
    }

    output_p = output;
    diff_p = diff;
    for (lint i = 0; i < size; ++i)
    {
        *output_p /= sum;
        *diff_p = *output_p * (1 - *output_p);

        output_p += r_sh_size;
        diff_p += r_sh_size;
    }
}


template <typename DT>
void SoftMax<DT>::operator () (Matrix_CPU<DT> & output, Matrix_CPU<DT> & diff, const Matrix_CPU<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);
    renew_if_shape_not_match(diff, in.m_shape);

    Shape l_sh = in.m_shape.sub_shape(0, this->m_axis - 1);
    Shape r_sh = in.m_shape.sub_shape(this->m_axis + 1, in.m_shape.dim() - 1);

    lint n_ele = std::max<lint>(l_sh.size(), 1);
    lint my_axis_size = in.m_shape[this->m_axis];
    lint r_sh_size = std::max<lint>(r_sh.size(), 1);
    lint step = my_axis_size * r_sh_size;

    DT *in_data = in.m_data;
    DT *diff_data = diff.m_data;
    DT *out_data = output.m_data;

    for (lint i = 0; i < n_ele; ++i)
    {
        DT *r_in_data = in_data;
        DT *r_diff_data = diff_data;
        DT *r_out_data = out_data;

        for (lint j = 0; j < r_sh_size; ++j)
        {
            this->elementary_softmax(r_out_data, r_diff_data, r_in_data, my_axis_size, r_sh_size);
            ++r_in_data;
            ++r_diff_data;
            ++r_out_data;
        }

        in_data += step;
        diff_data += step;
        out_data += step;
    }
}
template
void SoftMax<float>::operator () (Matrix_CPU<float> & output, Matrix_CPU<float> & diff, const Matrix_CPU<float> & in);


template <typename DT>
void SoftMax<DT>::operator () (Matrix_CPU<DT> & output, const Matrix_CPU<DT> & in)
{
    renew_if_shape_not_match(output, in.m_shape);

    Shape l_sh = in.m_shape.sub_shape(0, this->m_axis - 1);
    Shape r_sh = in.m_shape.sub_shape(this->m_axis + 1, in.m_shape.dim() - 1);

    lint n_ele = std::max<lint>(l_sh.size(), 1);
    lint my_axis_size = in.m_shape[this->m_axis];
    lint r_sh_size = std::max<lint>(r_sh.size(), 1);
    lint step = my_axis_size * r_sh_size;

    DT *in_data = in.m_data;
    DT *out_data = output.m_data;

    for (lint i = 0; i < n_ele; ++i)
    {
        DT *r_in_data = in_data;
        DT *r_out_data = out_data;

        for (lint j = 0; j < r_sh_size; ++j)
        {
            this->elementary_softmax(r_out_data, r_in_data, my_axis_size, r_sh_size);
            ++r_in_data;
            ++r_out_data;
        }

        in_data += step;
        out_data += step;
    }
}
template
void SoftMax<float>::operator () (Matrix_CPU<float> & output, const Matrix_CPU<float> & in);

} // namespace cpu
} // namespace la
} // namespace julie