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
__global__ void __concat_1d(DT *out_data, DT *mat1_data, DT *mat2_data, 
    int64_t cat1_right_size, int64_t cat2_right_size, int64_t cat1_cat2_right_size, int64_t output_size)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < output_size; i += n_threads)
    {
        int64_t l_idx = i / cat1_cat2_right_size;
        int64_t cat_idx = i % cat1_cat2_right_size;

        if (cat_idx < cat1_right_size)
        {
            out_data[i] = mat1_data[l_idx * cat1_right_size + cat_idx];
        }
        else
        {
            out_data[i] = mat2_data[l_idx * cat2_right_size + cat_idx - cat1_right_size];
        }
    }
}


template <typename DT>
__global__ void __slice_1d(DT *out_data, DT *input_data, int64_t shift, int64_t input_right_size, int64_t output_right_size, int64_t output_size)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < output_size; i += n_threads)
    {
        int64_t l_idx = i / output_right_size;
        int64_t slice_idx = i % output_right_size;

        out_data[i] = input_data[l_idx * input_right_size + shift + slice_idx];
    }
}


template <typename DT>
__global__ void __repeat_1d(DT *out_data, DT *input_data, 
    int64_t input_re_right_size, int64_t output_re_right_size, int64_t output_size)
{
    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads = gridDim.x * blockDim.x;

    for (int64_t i = offset; i < output_size; i += n_threads)
    {
        int64_t l_idx = i / output_re_right_size;
        int64_t output_re_idx = i % output_re_right_size;
        int64_t input_re_idx = output_re_idx % input_re_right_size;

        out_data[i] = input_data[l_idx * input_re_right_size + input_re_idx];
    }
}