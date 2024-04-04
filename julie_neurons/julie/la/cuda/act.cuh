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
__global__ void __act_linear_1d(DT *out_data, DT *diff_data, DT *in_data, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        diff_data[i] = 1.0;
        out_data[i] = in_data[i];
    }
}

template <typename DT>
__global__ void __act_linear_1d(DT *out_data, DT *in_data, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        out_data[i] = in_data[i];
    }
}

template <typename DT>
__global__ void __act_relu_1d(DT *out_data, DT *diff_data, DT *in_data, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        if (in_data[i] >= 0)
        {
            diff_data[i] = 1.0;
            out_data[i] = in_data[i];
        }
        else
        {
            diff_data[i] = 0;
            out_data[i] = 0;
        }
    }
}

template <typename DT>
__global__ void __act_relu_1d(DT *out_data, DT *in_data, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        if (in_data[i] >= 0)
        {
            out_data[i] = in_data[i];
        }
        else
        {
            out_data[i] = 0;
        }
    }
}

template <typename DT>
__global__ void __act_abs_1d(DT *out_data, DT *diff_data, DT *in_data, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        if (in_data[i] < 0)
        {
            diff_data[i] = -1;
            out_data[i] = in_data[i] * (-1);
        }
        else
        {
            diff_data[i] = 1;
            out_data[i] = in_data[i];
        }
    }
}

template <typename DT>
__global__ void __act_abs_1d(DT *out_data, DT *in_data, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        if (in_data[i] < 0)
        {
            out_data[i] = in_data[i] * (-1);
        }
        else
        {
            out_data[i] = in_data[i];
        }
    }
}

template <typename DT>
__global__ void __act_sigmoid_1d(DT *out_data, DT *diff_data, DT *in_data, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        DT tmp = 1.0 / (1.0 + exp(-in_data[i]));
        out_data[i] = tmp;
        diff_data[i] = tmp * (1.0 - tmp); 
    }
}

template <typename DT>
__global__ void __act_sigmoid_1d(DT *out_data, DT *in_data, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        out_data[i] = 1.0 / (1.0 + exp(-in_data[i]));
    }
}

template <typename DT>
__global__ void __act_arctan_1d(DT *out_data, DT *diff_data, DT *in_data, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        DT tmp = in_data[i];
        out_data[i] = atan(tmp);
        diff_data[i] = 1.0 / (1.0 + tmp * tmp);
    }
}

template <typename DT>
__global__ void __act_arctan_1d(DT *out_data, DT *in_data, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        out_data[i] = atan(in_data[i]);
    }
}

template <typename DT>
__global__ void __act_tanh_1d(DT *out_data, DT *diff_data, DT *in_data, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        DT tmp = tanh(in_data[i]);
        out_data[i] = tmp;
        diff_data[i] = 1.0 - tmp * tmp;
    }
}

template <typename DT>
__global__ void __act_tanh_1d(DT *out_data, DT *in_data, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        out_data[i] = tanh(in_data[i]);
    }
}

template <typename DT>
__global__ void __act_softmax_2d(DT *out_data, DT *diff_data, DT *in_data, int64_t left_size, int64_t size, int64_t right_size)
{
    int64_t offset_x = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads_x = gridDim.x * blockDim.x;

    int64_t offset_y = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t n_threads_y = gridDim.y * blockDim.y;

    for (int64_t left_i = offset_y; left_i < left_size; left_i += n_threads_y)
    {
        for (int64_t right_i = offset_x; right_i < right_size; right_i += n_threads_x)
        {
            DT sum = 0;
            int64_t first_ele_pos = left_i * right_size * size + right_i;

            int64_t pos = first_ele_pos;
            for (int64_t i = 0; i < size; ++i)
            {
                out_data[pos] = exp(in_data[pos]);
                sum += out_data[pos];
        
                pos += right_size;
            }

            pos = first_ele_pos;
            for (int64_t i = 0; i < size; ++i)
            {
                out_data[pos] /= sum;
                diff_data[pos] = out_data[pos] * (1 - out_data[pos]);

                pos += right_size;
            }
        }
    }
}

template <typename DT>
__global__ void __act_softmax_2d(DT *out_data, DT *in_data, int64_t left_size, int64_t size, int64_t right_size)
{
    int64_t offset_x = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads_x = gridDim.x * blockDim.x;

    int64_t offset_y = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t n_threads_y = gridDim.y * blockDim.y;

    for (int64_t left_i = offset_y; left_i < left_size; left_i += n_threads_y)
    {
        for (int64_t right_i = offset_x; right_i < right_size; right_i += n_threads_x)
        {
            DT sum = 0;
            int64_t first_ele_pos = left_i * right_size * size + right_i;

            int64_t pos = first_ele_pos;
            for (int64_t i = 0; i < size; ++i)
            {
                out_data[pos] = exp(in_data[pos]);
                sum += out_data[pos];
        
                pos += right_size;
            }

            pos = first_ele_pos;
            for (int64_t i = 0; i < size; ++i)
            {
                out_data[pos] /= sum;

                pos += right_size;
            }
        }
    }
}

