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
__global__ void __transpose_1d(DT *out_data, DT *in_data, int64_t in_h, int64_t in_w)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;
    int64_t len = in_h * in_w;

    for (int64_t idx = offset; idx < len; idx += n_threads)
    {
        int64_t y = idx / in_w;
        int64_t x = idx % in_w;

        out_data[x * in_h + y] = in_data[idx];
    }
}


template <typename DT>
__global__ void __transpose_neighboring_dim_pair_1d(DT *out_data, DT *in_data, int64_t left_size, int64_t l_size, int64_t r_size, int64_t right_size)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;
    
    int64_t len = left_size * l_size * r_size * right_size;


    for (int64_t idx = offset; idx < len; idx += n_threads)
    {
        int64_t in_right_idx = idx % right_size;

        int64_t i = idx / right_size;
        int64_t in_r_idx = i % r_size;

        i /= r_size;
        int64_t in_l_idx = i % l_size;
        
        i /= l_size;
        int64_t in_left_idx = i % left_size;

        out_data[right_size * (l_size * (r_size * in_left_idx + in_r_idx) + in_l_idx) + in_right_idx] = in_data[idx];
    }
}