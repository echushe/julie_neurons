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
#include "multiply.cuh"
#include "stat.cuh"
#include "transpose.cuh"
#include "assign.cuh"
#include "nsqrt.hpp"
#include <algorithm>

template <typename DT>
DT __dot(DT *l_data, DT *r_data, int64_t n_blocks_mul, int64_t n_blocks_sum, int64_t block_width, int64_t len)
{
    DT *buffer;

    CUDA_CHECK( cudaMalloc((void**) &buffer, len * sizeof(DT)) );
    __multiply_1d<<<n_blocks_mul, block_width>>>(buffer, l_data, r_data, len);
    DT sum = __sum(buffer, n_blocks_sum, block_width, len);
    CUDA_CHECK( cudaFree(buffer) );

    return sum;
}

template <typename DT>
DT __device__ __dot(DT *l_data, DT *r_data, int64_t len)
{
    DT sum = 0;
    for(int64_t i = 0; i < len; ++i)
    {
        sum += l_data[i] * r_data[i];
    }

    return sum;
}

template <typename DT>
__global__ void __matmul_2d_kernel_2d(
    DT *out_data,
    DT *l_data,
    DT *r_data_trans,
    int64_t l_rows, int64_t l_cols_r_rows, int64_t r_cols)
{
    int64_t offset_x = blockIdx.x * blockDim.x + threadIdx.x;    // right columns
    int64_t n_threads_x = gridDim.x * blockDim.x;

    int64_t offset_y = blockIdx.y * blockDim.y + threadIdx.y;    // left rows
    int64_t n_threads_y = gridDim.y * blockDim.y;

    int64_t l_pos = l_cols_r_rows * offset_y;
    int64_t l_pos_stride = l_cols_r_rows * n_threads_y;

    int64_t r_pos_begin = l_cols_r_rows * offset_x;
    int64_t r_pos_stride = l_cols_r_rows * n_threads_x;

    int64_t out_row_pos = r_cols * offset_y;
    int64_t out_row_stride = r_cols * n_threads_y;

    for(int64_t y = offset_y; y < l_rows; y += n_threads_y)
    {
        int64_t out_pos = out_row_pos + offset_x;
        int64_t r_pos = r_pos_begin;

        for(int64_t x = offset_x; x < r_cols; x += n_threads_x)
        {
            out_data[out_pos] = __dot(l_data + l_pos, r_data_trans + r_pos, l_cols_r_rows);

            out_pos += n_threads_x;
            r_pos += r_pos_stride;
        }

        out_row_pos += out_row_stride;
        l_pos += l_pos_stride;
    }
}

template <typename DT>
__global__ void __matmul_2d_kernel_3d(
    DT *out_data,
    DT *l_data,
    DT *r_data_trans,
    int64_t l_rows, int64_t l_cols_r_rows, int64_t r_cols)
{
    int64_t offset_x = blockIdx.x * blockDim.x + threadIdx.x;    // right columns
    int64_t n_threads_x = gridDim.x * blockDim.x;

    int64_t offset_y = blockIdx.y * blockDim.y + threadIdx.y;    // left rows
    int64_t n_threads_y = gridDim.y * blockDim.y;

    int64_t offset_z = blockIdx.z * blockDim.z + threadIdx.z;    // left columns and right rows
    int64_t n_threads_z = gridDim.z * blockDim.z;

    int64_t l_pos = l_cols_r_rows * offset_y;
    int64_t l_pos_stride = l_cols_r_rows * n_threads_y;

    int64_t r_pos_begin = l_cols_r_rows * offset_x;
    int64_t r_pos_stride = l_cols_r_rows * n_threads_x;

    int64_t out_row_pos = r_cols * offset_y;
    int64_t out_row_stride = r_cols * n_threads_y;

    for(int64_t y = offset_y; y < l_rows; y += n_threads_y)
    {
        int64_t out_pos = out_row_pos + offset_x;
        int64_t r_pos = r_pos_begin;

        for(int64_t x = offset_x; x < r_cols; x += n_threads_x)
        {
            DT *l_col = l_data + l_pos;
            DT *r_trans_col = r_data_trans + r_pos;

            for (int64_t z = offset_z; z < l_cols_r_rows; z += n_threads_z)
            {
                DT plus = l_col[z] * r_trans_col[z];
                // Synchronization problem here
                // Should use atomic operation to add
                atomicAdd(out_data + out_pos, plus);
                // out_data[out_data] += plus;
            }

            out_pos += n_threads_x;
            r_pos += r_pos_stride;
        }

        out_row_pos += out_row_stride;
        l_pos += l_pos_stride;
    }
}


template <typename DT>
void __matmul_2d(
    DT *out_data,
    DT *l_data,
    DT *r_data,
    int64_t n_blocks,
    int64_t block_width,
    int64_t l_rows, int64_t l_cols_r_rows, int64_t r_cols)
{
    DT *trans_buffer;
    CUDA_CHECK( cudaMalloc((void**) &trans_buffer, l_cols_r_rows * r_cols * sizeof(DT)) );

    __transpose_1d<<<n_blocks, block_width>>>(trans_buffer, r_data, l_cols_r_rows, r_cols);

    int64_t grid_y = std::max(nsqrt(n_blocks * l_rows / r_cols), 1L);
    int64_t grid_x = std::max(n_blocks / grid_y, 1L);

    dim3 dimGrid(grid_x, grid_y, 1);
    dim3 dimBlock(block_width / 16, 16, 1);

    __matmul_2d_kernel_2d<<<dimGrid, dimBlock>>>(out_data, l_data, trans_buffer, l_rows, l_cols_r_rows, r_cols);
    
    CUDA_CHECK( cudaFree(trans_buffer) );
}

/*
template <typename DT>
void __matmul_3d(
    DT *out_data,
    DT *l_data,
    DT *r_data,
    int64_t n_blocks,
    int64_t block_width,
    int64_t l_rows, int64_t l_cols_r_rows, int64_t r_cols)
{
    DT *trans_buffer;
    CUDA_CHECK( cudaMalloc((void**) &trans_buffer, l_cols_r_rows * r_cols * sizeof(DT)) );

    __assign_1d<<<n_blocks, block_width>>>(out_data, static_cast<DT>(0), l_rows * r_cols);
    __transpose_1d<<<n_blocks, block_width>>>(trans_buffer, r_data, l_cols_r_rows, r_cols);

    lint size = l_rows * l_cols_r_rows * r_cols;
    lint coe = static_cast<lint>(pow(size / n_blocks, 1.0f/3.0f));
    lint x = std::max(r_cols / coe, 1L);
    lint y = std::max(l_rows / coe, 1L);
    lint z = std::max(l_cols_r_rows / coe, 1L);

    dim3 dimGrid(x, y, z);
    dim3 dimBlock(block_width / 64, 8, 8);

    // std::cout << x << " " << y << " " << z << std::endl;

    __matmul_2d_kernel_3d<<<dimGrid, dimBlock>>>(out_data, l_data, trans_buffer, l_rows, l_cols_r_rows, r_cols);
    
    CUDA_CHECK( cudaFree(trans_buffer) );
}
*/