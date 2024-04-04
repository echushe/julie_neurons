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

#include "Activations_CPU_adv.hpp"
#include "Matrix_CPU_func.hpp"
#include <cmath>

namespace julie
{
namespace la
{
namespace cpu
{

template <typename DT>
void PReLU<DT>::operator () (
    Matrix_CPU<DT> & output, Matrix_CPU<DT> & diff, Matrix_CPU<DT> & alpha_diff, const Matrix_CPU<DT> & in, const DT & alpha)
{
    renew_if_shape_not_match(output, in.m_shape);
    renew_if_shape_not_match(diff, in.m_shape);
    renew_if_shape_not_match(alpha_diff, in.m_shape);

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        DT x = in.m_data[i];

        if (x >= 0)
        {
            output.m_data[i] = x;
            diff.m_data[i] = 1;
            alpha_diff.m_data[i] = 0;
        }
        else
        {
            output.m_data[i] = alpha * x;
            diff.m_data[i] = alpha;
            alpha_diff.m_data[i] = x;
        }
    }
}
template
void PReLU<float>::operator () (
    Matrix_CPU<float> & output, Matrix_CPU<float> & diff, Matrix_CPU<float> & alpha_diff, const Matrix_CPU<float> & in, const float & alpha);


template <typename DT>
void PReLU<DT>::operator () (
    Matrix_CPU<DT> & output, const Matrix_CPU<DT> & in,  const DT & alpha)
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
            output.m_data[i] = alpha * x;
        }
    }
}
template
void PReLU<float>::operator () (
    Matrix_CPU<float> & output, const Matrix_CPU<float> & in, const float & alpha);

} // namespace cpu
} // namespace la
} // namespace julie