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

#pragma once


template <typename DT>
__global__ void __act_prelu_1d(DT *out_data, DT *diff_data, DT *a_diff_data, DT *in_data, DT a_data, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        DT x = in_data[i];
        if (x >= 0)
        {
            out_data[i] = x;
            diff_data[i] = 1.0;
            a_diff_data[i] = 0;
        }
        else
        {
            out_data[i] = a_data * x;
            diff_data[i] = a_data;
            a_diff_data[i] = x;
        }
    }
}

template <typename DT>
__global__ void __act_prelu_1d(DT *out_data, DT *in_data, DT a_data, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        DT x = in_data[i];
        if (x >= 0)
        {
            out_data[i] = x;
        }
        else
        {
            out_data[i] = a_data * x;
        }
    }
}

