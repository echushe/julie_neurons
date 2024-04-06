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

#include "cuda_utility.cuh"
#include <iostream>
#include <limits>

#define KERNEL_SLICE_LEN 64

template <typename DT>
__global__ void __max_in_slice_1d(DT *out_data, DT *in_data, DT max, int64_t len)
{
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    int64_t slice_len = len / n_threads + 1;

    int64_t begin = thread_id * slice_len;
    int64_t end = (thread_id + 1) * slice_len;

    for (int64_t i = begin; i < end && i < len; ++i)
    {
        if (max < in_data[i])
        {
            max = in_data[i];
        }
    }

    out_data[thread_id] = max;
}


template <typename DT>
DT __max(DT *in_data, int64_t n_blocks, int64_t block_width, int64_t len)
{
    DT *gpu_buffer_in = in_data;
    DT *gpu_buffer_out = in_data;

    DT max = std::numeric_limits<DT>::max() * (-1);

    while (n_blocks > 1)
    {
        CUDA_CHECK( cudaMalloc ((void**) &gpu_buffer_out, n_blocks * block_width * sizeof(DT)) );
        __max_in_slice_1d<<<n_blocks, block_width>>>(gpu_buffer_out, gpu_buffer_in, max, len);
        
        if (gpu_buffer_in != in_data)
        {
            CUDA_CHECK( cudaFree(gpu_buffer_in) );
        }

        gpu_buffer_in = gpu_buffer_out;

        len = n_blocks * block_width;
        n_blocks = n_blocks / KERNEL_SLICE_LEN + 1;
    }

    DT *cpu_buffer_out = new DT[len];
    CUDA_CHECK( cudaMemcpy (cpu_buffer_out, gpu_buffer_out, len * sizeof(DT), cudaMemcpyDeviceToHost) );
    if (gpu_buffer_out != in_data)
    {
        CUDA_CHECK( cudaFree(gpu_buffer_out) );
    }

    for (int64_t i = 0; i < len; ++i)
    {
        // std::cout << cpu_buffer_out[i] << std::endl;
        if (max < cpu_buffer_out[i])
        {
            max = cpu_buffer_out[i];
        }
    }

    delete[] cpu_buffer_out;

    return max;
}


template <typename DT>
__global__ void __argmax_in_slice_1d(DT *out_data, int64_t *out_data_idx, DT *in_data, int64_t *in_data_idx, DT max, int64_t len)
{
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    int64_t slice_len = len / n_threads + 1;

    int64_t begin = thread_id * slice_len;
    int64_t end = (thread_id + 1) * slice_len;

    int64_t argmax = 0;

    for (int64_t i = begin; i < end && i < len; ++i)
    {
        if (max < in_data[i])
        {
            max = in_data[i];
            argmax = in_data_idx[i];
        }
    }

    out_data[thread_id] = max;
    out_data_idx[thread_id] = argmax;
}


template <typename DT>
__global__ void __initialize_idx(DT *data, int64_t len)
{
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < len)
    {
        data[thread_id] = thread_id;
    }
}


template <typename DT>
int64_t __argmax(DT *in_data, int64_t n_blocks, int64_t block_width, int64_t len)
{
    DT *gpu_buffer_in = in_data;
    int64_t *gpu_buffer_in_idx;
    DT *gpu_buffer_out = in_data;
    int64_t *gpu_buffer_out_idx;

    CUDA_CHECK( cudaMalloc ((void**) &gpu_buffer_in_idx, len * sizeof(int64_t)) );
    __initialize_idx<<<len / block_width + 1, block_width>>>(gpu_buffer_in_idx, len);
    gpu_buffer_out_idx = gpu_buffer_in_idx;

    DT max = std::numeric_limits<DT>::max() * (-1);
    int64_t argmax = 0;

    while (n_blocks > 1)
    {
        CUDA_CHECK( cudaMalloc ((void**) &gpu_buffer_out, n_blocks * block_width * sizeof(DT)) );
        CUDA_CHECK( cudaMalloc ((void**) &gpu_buffer_out_idx, n_blocks * block_width * sizeof(int64_t)) );
        __argmax_in_slice_1d<<<n_blocks, block_width>>>(gpu_buffer_out, gpu_buffer_out_idx, gpu_buffer_in, gpu_buffer_in_idx, max, len);
        
        if (gpu_buffer_in != in_data)
        {
            CUDA_CHECK( cudaFree(gpu_buffer_in) );
        }
        CUDA_CHECK( cudaFree(gpu_buffer_in_idx) );

        gpu_buffer_in = gpu_buffer_out;
        gpu_buffer_in_idx = gpu_buffer_out_idx;

        len = n_blocks * block_width;
        n_blocks = n_blocks / KERNEL_SLICE_LEN + 1;
    }

    DT *cpu_buffer_out = new DT[len];
    int64_t *cpu_buffer_out_idx = new int64_t[len];
    CUDA_CHECK( cudaMemcpy (cpu_buffer_out, gpu_buffer_out, len * sizeof(DT), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy (cpu_buffer_out_idx, gpu_buffer_out_idx, len * sizeof(int64_t), cudaMemcpyDeviceToHost) );
    if (gpu_buffer_out != in_data)
    {
        CUDA_CHECK( cudaFree(gpu_buffer_out) );
    }
    CUDA_CHECK( cudaFree(gpu_buffer_out_idx) );

    for (int64_t i = 0; i < len; ++i)
    {
        if (max < cpu_buffer_out[i])
        {
            max = cpu_buffer_out[i];
            argmax = cpu_buffer_out_idx[i];
        }
    }

    delete[] cpu_buffer_out;
    delete[] cpu_buffer_out_idx;

    return argmax;
}


template <typename DT>
__global__ void __argmax_2d(int64_t *indexes, DT *in_data, DT max, int64_t left_size, int64_t current_size, int64_t right_size)
{
    int64_t offset_left = blockIdx.x * blockDim.x + threadIdx.x; // Parallel for left dimensions
    int64_t n_threads_left = gridDim.x * blockDim.x;

    int64_t offset_right = blockIdx.y * blockDim.y + threadIdx.y; // Parallel for right dimensions
    int64_t n_threads_right = gridDim.y * blockDim.y;

    for (int64_t l_i = offset_left; l_i < left_size; l_i += n_threads_left)
    {
        for (int64_t r_i = offset_right; r_i < right_size; r_i += n_threads_right)
        {
            DT m = max;
            int64_t max_c = 0;

            for (int64_t c_i = 0; c_i < current_size; ++c_i)
            {
                DT value = in_data[ right_size * (current_size * l_i + c_i) + r_i ];
                if (value > m)
                {
                    m = value;
                    max_c = c_i;
                }
            }

            indexes[ l_i * right_size + r_i ] = max_c;
        }
    }
}


template <typename DT>
__global__ void __argmin_2d(int64_t *indexes, DT *in_data, DT min, int64_t left_size, int64_t current_size, int64_t right_size)
{
    int64_t offset_left = blockIdx.x * blockDim.x + threadIdx.x; // Parallel for left dimensions
    int64_t n_threads_left = gridDim.x * blockDim.x;

    int64_t offset_right = blockIdx.y * blockDim.y + threadIdx.y; // Parallel for right dimensions
    int64_t n_threads_right = gridDim.y * blockDim.y;

    for (int64_t l_i = offset_left; l_i < left_size; l_i += n_threads_left)
    {
        for (int64_t r_i = offset_right; r_i < right_size; r_i += n_threads_right)
        {
            DT m = min;
            int64_t min_c = 0;

            for (int64_t c_i = 0; c_i < current_size; ++c_i)
            {
                DT value = in_data[ right_size * (current_size * l_i + c_i) + r_i ];
                if (value < m)
                {
                    m = value;
                    min_c = c_i;
                }
            }

            indexes[ l_i * right_size + r_i ] = min_c;
        }
    }
}


template <typename DT>
__global__ void __min_in_slice_1d(DT *out_data, DT *in_data, DT min, int64_t len)
{
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    int64_t slice_len = len / n_threads + 1;

    int64_t begin = thread_id * slice_len;
    int64_t end = (thread_id + 1) * slice_len;

    for (int64_t i = begin; i < end && i < len; ++i)
    {
        if (min > in_data[i])
        {
            min = in_data[i];
        }
    }

    out_data[thread_id] = min;
}


template <typename DT>
DT __min(DT *in_data, int64_t n_blocks, int64_t block_width, int64_t len)
{
    DT *gpu_buffer_in = in_data;
    DT *gpu_buffer_out = in_data;

    DT min = std::numeric_limits<DT>::max();

    while (n_blocks > 1)
    {
        CUDA_CHECK( cudaMalloc ((void**) &gpu_buffer_out, n_blocks * block_width * sizeof(DT)) );
        __min_in_slice_1d<<<n_blocks, block_width>>>(gpu_buffer_out, gpu_buffer_in, min, len);

        if (gpu_buffer_in != in_data)
        {
            CUDA_CHECK( cudaFree(gpu_buffer_in) );
        }

        gpu_buffer_in = gpu_buffer_out;

        len = n_blocks * block_width;
        n_blocks = n_blocks / KERNEL_SLICE_LEN + 1;
    }

    DT *cpu_buffer_out = new DT[len];
    CUDA_CHECK( cudaMemcpy (cpu_buffer_out, gpu_buffer_out, len * sizeof(DT), cudaMemcpyDeviceToHost) );
    if (gpu_buffer_out != in_data)
    {
        CUDA_CHECK( cudaFree(gpu_buffer_out) );
    }

    for (int64_t i = 0; i < len; ++i)
    {
        if (min > cpu_buffer_out[i])
        {
            min = cpu_buffer_out[i];
        }
    }

    delete[] cpu_buffer_out;

    return min;
}


template <typename DT>
__global__ void __argmin_in_slice_1d(DT *out_data, int64_t *out_data_idx, DT *in_data, int64_t *in_data_idx, DT min, int64_t len)
{
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    int64_t slice_len = len / n_threads + 1;

    int64_t begin = thread_id * slice_len;
    int64_t end = (thread_id + 1) * slice_len;

    int64_t argmin = 0;

    for (int64_t i = begin; i < end && i < len; ++i)
    {
        if (min > in_data[i])
        {
            min = in_data[i];
            argmin = in_data_idx[i];
        }
    }

    out_data[thread_id] = min;
    out_data_idx[thread_id] = argmin;
}


template <typename DT>
int64_t __argmin(DT *in_data, int64_t n_blocks, int64_t block_width, int64_t len)
{
    DT *gpu_buffer_in = in_data;
    int64_t *gpu_buffer_in_idx;
    DT *gpu_buffer_out = in_data;
    int64_t *gpu_buffer_out_idx;

    CUDA_CHECK( cudaMalloc ((void**) &gpu_buffer_in_idx, len * sizeof(int64_t)) );
    __initialize_idx<<<len / block_width + 1, block_width>>>(gpu_buffer_in_idx, len);
    gpu_buffer_out_idx = gpu_buffer_in_idx;

    DT min = std::numeric_limits<DT>::max();
    int64_t argmin = 0;

    while (n_blocks > 1)
    {
        CUDA_CHECK( cudaMalloc ((void**) &gpu_buffer_out, n_blocks * block_width * sizeof(DT)) );
        CUDA_CHECK( cudaMalloc ((void**) &gpu_buffer_out_idx, n_blocks * block_width * sizeof(int64_t)) );
        __argmin_in_slice_1d<<<n_blocks, block_width>>>(gpu_buffer_out, gpu_buffer_out_idx, gpu_buffer_in, gpu_buffer_in_idx, min, len);
        
        if (gpu_buffer_in != in_data)
        {
            CUDA_CHECK( cudaFree(gpu_buffer_in) );
        }
        CUDA_CHECK( cudaFree(gpu_buffer_in_idx) );

        gpu_buffer_in = gpu_buffer_out;
        gpu_buffer_in_idx = gpu_buffer_out_idx;

        len = n_blocks * block_width;
        n_blocks = n_blocks / KERNEL_SLICE_LEN + 1;
    }

    DT *cpu_buffer_out = new DT[len];
    int64_t *cpu_buffer_out_idx = new int64_t[len];
    CUDA_CHECK( cudaMemcpy (cpu_buffer_out, gpu_buffer_out, len * sizeof(DT), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy (cpu_buffer_out_idx, gpu_buffer_out_idx, len * sizeof(int64_t), cudaMemcpyDeviceToHost) );
    if (gpu_buffer_out != in_data)
    {
        CUDA_CHECK( cudaFree(gpu_buffer_out) );
    }
    CUDA_CHECK( cudaFree(gpu_buffer_out_idx) );

    for (int64_t i = 0; i < len; ++i)
    {
        if (min > cpu_buffer_out[i])
        {
            min = cpu_buffer_out[i];
            argmin = cpu_buffer_out_idx[i];
        }
    }

    delete[] cpu_buffer_out;
    delete[] cpu_buffer_out_idx;

    return argmin;
}


template <typename DT>
__global__ void __normalize_1d(DT *in_out_data, DT old_min, DT old_range, DT new_min, DT new_range, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        in_out_data[i] = new_min + ((in_out_data[i] - old_min) / old_range) * new_range;
    }
}


template <typename DT>
__global__ void __sum_in_slice_1d(DT *out_data, DT *in_data, int64_t len)
{
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    int64_t slice_len = len / n_threads + 1;

    int64_t begin = thread_id * slice_len;
    int64_t end = (thread_id + 1) * slice_len;

    DT sum = 0;

    for (int64_t i = begin; i < end && i < len; ++i)
    {
        sum += in_data[i];
    }

    out_data[thread_id] = sum;
}


template <typename DT>
DT __sum(DT *in_data, int64_t n_blocks, int64_t block_width, int64_t len)
{
    DT *gpu_buffer_in = in_data;
    DT *gpu_buffer_out = in_data;

    while (n_blocks > 1)
    {
        CUDA_CHECK( cudaMalloc ((void**) &gpu_buffer_out, n_blocks * block_width * sizeof(DT)) );
        __sum_in_slice_1d<<<n_blocks, block_width>>>(gpu_buffer_out, gpu_buffer_in, len);

        if (gpu_buffer_in != in_data)
        {
            CUDA_CHECK( cudaFree(gpu_buffer_in) );
        }

        gpu_buffer_in = gpu_buffer_out;

        len = n_blocks * block_width;
        n_blocks = n_blocks / KERNEL_SLICE_LEN + 1;
    }

    DT *cpu_buffer_out = new DT[len];
    CUDA_CHECK( cudaMemcpy (cpu_buffer_out, gpu_buffer_out, len * sizeof(DT), cudaMemcpyDeviceToHost) );
    if (gpu_buffer_out != in_data)
    {
        CUDA_CHECK( cudaFree(gpu_buffer_out) );
    }

    DT sum = 0;
    for (int64_t i = 0; i < len; ++i)
    {
        sum += cpu_buffer_out[i];
    }

    delete[] cpu_buffer_out;

    return sum;
}


template <typename DT>
__global__ void __square_1d(DT *out_data, DT *in_data, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        DT tmp = in_data[i];
        out_data[i] = tmp * tmp;
    }
}


template <typename DT>
__global__ void __square_1d(DT *in_out_data, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i += n_threads)
    {
        DT tmp = in_out_data[i];
        in_out_data[i] = tmp * tmp;
    }
}


template <typename DT>
__global__ void __at_1d(DT *value, DT *data, int64_t idx)
{
    *value = data[idx];
}


template <typename DT>
__global__ void __threshold_1d(DT *out_data, DT *in_data, DT th, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i+= n_threads)
    {
        if (in_data[i] < th)
        {
            out_data[i] = 0;
        }
        else
        {
            out_data[i] = 1;
        }
    }
}


template <typename DT>
__global__ void __abs_threshold_1d(DT *out_data, DT *in_data, DT th, int64_t len)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < len; i+= n_threads)
    {
        if (in_data[i] < th && in_data[i] > -th)
        {
            out_data[i] = 0;
        }
        else
        {
            out_data[i] = 1;
        }
    }
}
