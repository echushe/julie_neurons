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
__global__ void __loss_half_square_error_2d(DT *out_data, DT *diff_data, DT *target_data, DT *in_data, int64_t left_size, int64_t size, int64_t right_size)
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
            int64_t out_pos = left_i * right_size + right_i;

            int64_t pos = first_ele_pos;
            for (int64_t i = 0; i < size; ++i)
            {
                DT sub = in_data[pos] - target_data[pos];
                sum += sub * sub;

                diff_data[pos] = sub;
        
                pos += right_size;
            }

            out_data[out_pos] = sum / 2.0;
        }
    }
}


template <typename DT>
__global__ void __loss_sigmoid_crossentropy_2d(
    DT *out_data, DT *sigmoid_data, DT *diff_data, DT *target_data, DT *in_data, int64_t left_size, int64_t size, int64_t right_size)
{
    int64_t offset_x = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_threads_x = gridDim.x * blockDim.x;

    int64_t offset_y = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t n_threads_y = gridDim.y * blockDim.y;

    for (int64_t left_i = offset_y; left_i < left_size; left_i += n_threads_y)
    {
        for (int64_t right_i = offset_x; right_i < right_size; right_i += n_threads_x)
        {
            int64_t first_ele_pos = left_i * right_size * size + right_i;
            int64_t out_pos = left_i * right_size + right_i;

            int64_t pos = first_ele_pos;
            for (int64_t i = 0; i < size; ++i)
            {
                sigmoid_data[pos] = 1.0 / (1.0 + exp(-in_data[pos]));
        
                pos += right_size;
            }

            DT centropy_sum = 0;

            pos = first_ele_pos;
            for (int64_t i = 0; i < size; ++i)
            {
                centropy_sum += target_data[pos] * log(sigmoid_data[pos]);
                diff_data[pos] = sigmoid_data[pos] - target_data[pos];

                pos += right_size;
            }

            out_data[out_pos] = centropy_sum * (-1);
        }
    }
}


template <typename DT>
__global__ void __loss_softmax_crossentropy_2d(
    DT *out_data, DT *softmax_data, DT *diff_data, DT *target_data, DT *in_data, int64_t left_size, int64_t size, int64_t right_size)
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
            int64_t out_pos = left_i * right_size + right_i;

            int64_t pos = first_ele_pos;

            for (int64_t i = 0; i < size; ++i)
            {
                softmax_data[pos] = exp(in_data[pos]);
                sum += softmax_data[pos];
        
                pos += right_size;
            }

            DT centropy_sum = 0;

            pos = first_ele_pos;
            for (int64_t i = 0; i < size; ++i)
            {
                softmax_data[pos] /= sum;

                centropy_sum += target_data[pos] * log(softmax_data[pos]);
                diff_data[pos] = softmax_data[pos] - target_data[pos];

                pos += right_size;
            }

            out_data[out_pos] = centropy_sum * (-1);
        }
    }
}
