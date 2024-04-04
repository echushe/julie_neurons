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

#include "Matrix_CUDA.hpp"
#include "Matrix_CUDA_func.hpp"
#include "Matrix_CPU.hpp"
#include "utilities.hpp"
#include "assign.cuh"
#include "add.cuh"
#include "subtract.cuh"
#include "multiply.cuh"
#include "divide.cuh"
#include "stat.cuh"
#include "memcpy.cuh"
#include "cuda_utility.cuh"
#include "nsqrt.hpp"


#include <random>
#include <algorithm>
#include <stdexcept>
#include <iterator>
#include <sstream>
#include <cstring>
#include <functional>


namespace julie
{
namespace la
{
namespace cuda
{

template <typename DT>
Matrix_CUDA<DT>::Matrix_CUDA()
    : m_shape{}, m_data{ nullptr }
{}
template
Matrix_CUDA<int>::Matrix_CUDA();
template
Matrix_CUDA<float>::Matrix_CUDA();


template <typename DT>
Matrix_CUDA<DT>::Matrix_CUDA(const Shape & shape)
    : m_shape{ shape }, m_data{ nullptr }
{
    if (shape.m_size < 1)
    {
        return;
    }

    lint size = this->m_shape.m_size;
    lint cu_bytes = size * sizeof(DT);
    
    // Apply memory for cuda
    CUDA_CHECK( cudaMalloc ((void**) &this->m_data, cu_bytes) );
    // initialize cuda memory to zero
    CUDA_CHECK( cudaMemset (this->m_data, 0, cu_bytes) );

}
template
Matrix_CUDA<int>::Matrix_CUDA(const Shape & shape);
template
Matrix_CUDA<float>::Matrix_CUDA(const Shape & shape);


template <typename DT>
Matrix_CUDA<DT>::Matrix_CUDA(DT value, const Shape & shape)
    : Matrix_CUDA{ shape }
{
    if (shape.m_size < 1)
    {
        return;
    }

    lint size = this->m_shape.m_size;
    lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    __assign_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(this->m_data, value, size);
}
template
Matrix_CUDA<int>::Matrix_CUDA(int value, const Shape & shape);
template
Matrix_CUDA<float>::Matrix_CUDA(float value, const Shape & shape);


template <typename DT>
Matrix_CUDA<DT>::Matrix_CUDA(const std::vector<Matrix_CUDA<DT>>& matrices)
    : Matrix_CUDA{}
{
    lint array_size = matrices.size();

    if (array_size > 0)
    {
        Shape mat_sh = matrices[0].m_shape;
        lint mat_size = mat_sh.m_size;

        if (mat_size > 0)
        {
            for (lint i = 1; i < array_size; ++i)
            {
                if (matrices[i].m_shape != mat_sh)
                {
                    throw std::invalid_argument(
                        std::string("Matrix_CUDA::Matrix_CUDA: invalid matrix array because of different shapes of matrices"));
                }
            }

            this->m_shape = Shape{ array_size } +mat_sh;

            // Get size info
            lint size = this->m_shape.m_size;
            lint cu_bytes = size * sizeof(DT);

            // Apply memory for cuda
            CUDA_CHECK( cudaMalloc ((void**) &this->m_data, cu_bytes) );
            // initialize cuda memory to zero
            CUDA_CHECK( cudaMemset (this->m_data, 0, cu_bytes) );

            DT *this_pos = this->m_data;
            DT *that_pos;

            for (lint i = 0; i < array_size; ++i)
            {
                that_pos = matrices[i].m_data;
                CUDA_CHECK( cudaMemcpy (this_pos, that_pos, mat_size * sizeof(DT), cudaMemcpyDeviceToDevice) );
                this_pos += mat_size;
            }
        }
    }
}
template
Matrix_CUDA<int>::Matrix_CUDA(const std::vector<Matrix_CUDA<int>>& matrices);
template
Matrix_CUDA<float>::Matrix_CUDA(const std::vector<Matrix_CUDA<float>>& matrices);


template <typename DT>
Matrix_CUDA<DT>::Matrix_CUDA(const std::vector<std::shared_ptr<Matrix_CUDA<DT>>> &matrices)
    : Matrix_CUDA{}
{
    lint array_size = matrices.size();

    if (array_size > 0)
    {
        Shape mat_sh = matrices[0]->m_shape;
        lint mat_size = mat_sh.m_size;

        if (mat_size > 0)
        {
            for (lint i = 1; i < array_size; ++i)
            {
                if (matrices[i]->m_shape != mat_sh)
                {
                    throw std::invalid_argument(
                        std::string("Matrix_CUDA::Matrix_CUDA: invalid matrix array because of different shapes of matrices"));
                }
            }

            this->m_shape = Shape{ array_size } +mat_sh;

            // Get size info
            lint size = this->m_shape.m_size;
            lint cu_bytes = size * sizeof(DT);

            // Apply memory for cuda
            CUDA_CHECK( cudaMalloc ((void**) &this->m_data, cu_bytes) );
            // initialize cuda memory to zero
            CUDA_CHECK( cudaMemset (this->m_data, 0, cu_bytes) );

            DT *this_pos = this->m_data;
            DT *that_pos;

            for (lint i = 0; i < array_size; ++i)
            {
                that_pos = matrices[i]->m_data;
                CUDA_CHECK( cudaMemcpy (this_pos, that_pos, mat_size * sizeof(DT), cudaMemcpyDeviceToDevice) );
                this_pos += mat_size;
            }
        }
    }
}
template
Matrix_CUDA<int>::Matrix_CUDA(const std::vector<std::shared_ptr<Matrix_CUDA<int>>> &matrices);
template
Matrix_CUDA<float>::Matrix_CUDA(const std::vector<std::shared_ptr<Matrix_CUDA<float>>> &matrices);


template <typename DT>
Matrix_CUDA<DT>::Matrix_CUDA(const Matrix_CUDA<DT> & other)
    : m_shape{ other.m_shape }, m_data{ nullptr }
{
    lint size = m_shape.m_size;

    if (size > 0)
    {
        lint cu_bytes = size * sizeof(DT);

        // Apply memory for cuda
        CUDA_CHECK( cudaMalloc ((void**) &this->m_data, cu_bytes) );
        // initialize cuda memory to zero
        CUDA_CHECK( cudaMemset (this->m_data, 0, cu_bytes) );

        CUDA_CHECK( cudaMemcpy (this->m_data, other.m_data, cu_bytes, cudaMemcpyDeviceToDevice) );
    }
}
template
Matrix_CUDA<int>::Matrix_CUDA(const Matrix_CUDA<int> & other);
template
Matrix_CUDA<float>::Matrix_CUDA(const Matrix_CUDA<float> & other);


template <typename DT>
Matrix_CUDA<DT>::Matrix_CUDA(Matrix_CUDA<DT> && other)
    : m_shape{ std::move(other.m_shape) }, m_data{ other.m_data }
{
    other.m_data = nullptr;
}
template
Matrix_CUDA<int>::Matrix_CUDA(Matrix_CUDA<int> && other);
template
Matrix_CUDA<float>::Matrix_CUDA(Matrix_CUDA<float> && other);


template <typename DT>
Matrix_CUDA<DT>::Matrix_CUDA(const std::vector<DT> & vec, bool horizontal)
    : m_shape{ static_cast<lint>(vec.size()), 1 }, m_data{ nullptr }
{
    if (horizontal)
    {
        m_shape.m_data[0] = 1;
        m_shape.m_data[1] = vec.size();
    }

    lint size = m_shape.m_size;

    if (size > 0)
    {
        lint cu_bytes = size * sizeof(DT);

        // Apply memory for cuda
        CUDA_CHECK( cudaMalloc ((void**) &this->m_data, cu_bytes) );
        // initialize cuda memory to zero
        CUDA_CHECK( cudaMemset (this->m_data, 0, cu_bytes) );

        // Create a cpu_matrix from 1d array
        julie::la::cpu::Matrix_CPU <DT> cpu_mat {vec, horizontal};
        // Copy cpu_matrix to gpu_matrix
        CUDA_CHECK( cudaMemcpy (this->m_data, cpu_mat.m_data, cu_bytes, cudaMemcpyHostToDevice) );
    }
}
template
Matrix_CUDA<int>::Matrix_CUDA(const std::vector<int> & vec, bool horizontal);
template
Matrix_CUDA<float>::Matrix_CUDA(const std::vector<float> & vec, bool horizontal);


template <typename DT>
Matrix_CUDA<DT>::Matrix_CUDA(const std::vector<DT> & vec, const Shape &shape)
    : m_shape{ static_cast<lint>(vec.size()) }, m_data{ nullptr }
{
    if (shape.m_size != this->m_shape.m_size)
    {
        throw std::invalid_argument(std::string("Matrix_CUDA::Matrix_CUDA: Size of shape argument does not match vector length"));
    }

    this->m_shape = shape;
    lint size = m_shape.m_size;

    if (size > 0)
    {
        lint cu_bytes = size * sizeof(DT);

        // Apply memory for cuda
        CUDA_CHECK( cudaMalloc ((void**) &this->m_data, cu_bytes) );
        // initialize cuda memory to zero
        CUDA_CHECK( cudaMemset (this->m_data, 0, cu_bytes) );

        // Create a cpu_matrix from 1d array
        julie::la::cpu::Matrix_CPU <DT> cpu_mat {vec, shape};
        // Copy cpu_matrix to gpu_matrix
        CUDA_CHECK( cudaMemcpy (this->m_data, cpu_mat.m_data, cu_bytes, cudaMemcpyHostToDevice) );
    }
}
template
Matrix_CUDA<int>::Matrix_CUDA(const std::vector<int> & vec, const Shape &shape);
template
Matrix_CUDA<float>::Matrix_CUDA(const std::vector<float> & vec, const Shape &shape);


template <typename DT>
Matrix_CUDA<DT>::Matrix_CUDA(const std::vector<std::vector<DT>> & array)
    :
    m_shape{ static_cast<lint>(array.size()),
               static_cast<lint>(array.size() > 0 ? array[0].size() : 0) },
    m_data{ nullptr }
{
    //Check validity of list
    lint n_cols = this->m_shape.m_data[1];
    for (auto itr = array.begin(); itr != array.end(); ++itr)
    {
        if (itr->size() != n_cols)
        {
            throw std::invalid_argument(std::string("Each row should have the same number of elements!"));
        }
    }

    lint size = this->m_shape.m_size;

    if (size > 0)
    {
        lint cu_bytes = size * sizeof(DT);

        // Apply memory for cuda
        CUDA_CHECK( cudaMalloc ((void**) &this->m_data, cu_bytes) );
        // initialize cuda memory to zero
        CUDA_CHECK( cudaMemset (this->m_data, 0, cu_bytes) );

        // Create a cpu_matrix from 2d array
        julie::la::cpu::Matrix_CPU<DT> cpu_mat {array};
        // Copy cpu_matrix to gpu_matrix
        CUDA_CHECK( cudaMemcpy (this->m_data, cpu_mat.m_data, cu_bytes, cudaMemcpyHostToDevice) );
    }
}
template
Matrix_CUDA<int>::Matrix_CUDA(const std::vector<std::vector<int>> & array);
template
Matrix_CUDA<float>::Matrix_CUDA(const std::vector<std::vector<float>> & array);


template <typename DT>
Matrix_CUDA<DT>::~Matrix_CUDA()
{
    // std::cout << "Matrix_CUDA destroyed, shape: " << this->m_shape << std::endl;
    // Release cuda memory
    CUDA_CHECK( cudaFree (this->m_data) );
}
template
Matrix_CUDA<int>::~Matrix_CUDA();
template
Matrix_CUDA<float>::~Matrix_CUDA();


template <typename DT>
void Matrix_CUDA<DT>::get_full_transpose(Matrix_CUDA<DT> & out) const
{
    full_transpose(out, *this);
}
template
void Matrix_CUDA<int>::get_full_transpose(Matrix_CUDA<int> & out) const;
template
void Matrix_CUDA<float>::get_full_transpose(Matrix_CUDA<float> & out) const;


template <typename DT>
void Matrix_CUDA<DT>::get_transpose(Matrix_CUDA<DT> & out, lint left_dims) const
{
    transpose(out, *this, left_dims);
}
template
void Matrix_CUDA<int>::get_transpose(Matrix_CUDA<int> & out, lint left_dims) const;
template
void Matrix_CUDA<float>::get_transpose(Matrix_CUDA<float> & out, lint left_dims) const;


template <typename DT>
Matrix_CUDA<DT> & Matrix_CUDA<DT>::operator = (const Matrix_CUDA<DT> & other)
{
    bool renew = false;
    if (this->m_shape.m_size != other.m_shape.m_size)
    {
        renew = true;
    }

    if (renew)
    {
        CUDA_CHECK( cudaFree (this->m_data) );
        this->m_data = nullptr;
    }

    this->m_shape = other.m_shape;
    lint size = this->m_shape.m_size;
    if (size < 1)
    {
        return *this;
    }

    lint cu_bytes = size * sizeof(DT);

    if (renew)
    {
        // Apply memory for cuda
        CUDA_CHECK( cudaMalloc ((void**) &this->m_data, cu_bytes) );
        // initialize cuda memory to zero
        CUDA_CHECK( cudaMemset (this->m_data, 0, cu_bytes) );
    }

    CUDA_CHECK( cudaMemcpy (this->m_data, other.m_data, cu_bytes, cudaMemcpyDeviceToDevice) );

    return *this;
}
template
Matrix_CUDA<int> & Matrix_CUDA<int>::operator = (const Matrix_CUDA<int> & other);
template
Matrix_CUDA<float> & Matrix_CUDA<float>::operator = (const Matrix_CUDA<float> & other);


template <typename DT>
Matrix_CUDA<DT> & Matrix_CUDA<DT>::operator = (Matrix_CUDA<DT> && other)
{
    CUDA_CHECK( cudaFree (this->m_data) );
    
    this->m_shape = std::move(other.m_shape);

    this->m_data = other.m_data;
    other.m_data = nullptr;

    return *this;
}
template
Matrix_CUDA<int> & Matrix_CUDA<int>::operator = (Matrix_CUDA<int> && other);
template
Matrix_CUDA<float> & Matrix_CUDA<float>::operator = (Matrix_CUDA<float> && other);


template <typename DT>
Matrix_CUDA<DT> & Matrix_CUDA<DT>::operator = (DT scalar)
{
    lint size = this->m_shape.m_size;
    lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    __assign_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(this->m_data, scalar, size);

    return *this;
}
template
Matrix_CUDA<int> & Matrix_CUDA<int>::operator = (int scalar);
template
Matrix_CUDA<float> & Matrix_CUDA<float>::operator = (float scalar);


template <typename DT>
DT Matrix_CUDA<DT>::at(const Coordinate & pos) const
{
    if (this->m_shape != pos.m_shape)
    {
        throw std::invalid_argument(invalid_coordinate);
    }

    DT *value_device;
    DT value_host;
    lint cu_bytes = sizeof(DT);
    CUDA_CHECK( cudaMalloc ((void**) &value_device, cu_bytes) );

    __at_1d<<<1, 1>>>(value_device, this->m_data, pos.index());
    CUDA_CHECK( cudaMemcpy (&value_host, value_device, cu_bytes, cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaFree (value_device) );

    return value_host;
}
template
int Matrix_CUDA<int>::at(const Coordinate & pos) const;
template
float Matrix_CUDA<float>::at(const Coordinate & pos) const;


template <typename DT>
DT Matrix_CUDA<DT>::operator [] (const Coordinate & pos) const
{
    return this->at(pos);
}
template
int Matrix_CUDA<int>::operator [] (const Coordinate & pos) const;
template
float Matrix_CUDA<float>::operator [] (const Coordinate & pos) const;


template <typename DT>
DT Matrix_CUDA<DT>::operator [] (std::initializer_list<lint> list) const
{
    Coordinate pos{ list, this->m_shape };
    return this->at(pos);
}
template
int Matrix_CUDA<int>::operator [] (std::initializer_list<lint> list) const;
template
float Matrix_CUDA<float>::operator [] (std::initializer_list<lint> list) const;


template <typename DT>
Matrix_CUDA<DT> & Matrix_CUDA<DT>::operator += (const Matrix_CUDA<DT> & other)
{
    lint big_size = this->m_shape.m_size;
    lint small_size = other.m_shape.m_size;

    if (small_size == 0 || big_size < small_size || big_size % small_size != 0)
    {
        throw std::invalid_argument(invalid_shape + std::string(__FUNCTION__));
    }

    lint n_blocks = std::min(big_size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    __broadcast_add_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(this->m_data, other.m_data, big_size, small_size);

    return *this;
}
template
Matrix_CUDA<int> & Matrix_CUDA<int>::operator += (const Matrix_CUDA<int> & other);
template
Matrix_CUDA<float> & Matrix_CUDA<float>::operator += (const Matrix_CUDA<float> & other);


template <typename DT>
Matrix_CUDA<DT> & Matrix_CUDA<DT>::operator -= (const Matrix_CUDA<DT> & other)
{
    lint big_size = this->m_shape.m_size;
    lint small_size = other.m_shape.m_size;

    if (small_size == 0 || big_size < small_size || big_size % small_size != 0)
    {
        throw std::invalid_argument(invalid_shape + std::string(__FUNCTION__));
    }

    lint n_blocks = std::min(big_size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    __broadcast_subtract_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(this->m_data, other.m_data, big_size, small_size);

    return *this;
}
template
Matrix_CUDA<int> & Matrix_CUDA<int>::operator -= (const Matrix_CUDA<int> & other);
template
Matrix_CUDA<float> & Matrix_CUDA<float>::operator -= (const Matrix_CUDA<float> & other);


template <typename DT>
Matrix_CUDA<DT> & Matrix_CUDA<DT>::operator *= (const Matrix_CUDA<DT> & other)
{
    lint big_size = this->m_shape.m_size;
    lint small_size = other.m_shape.m_size;

    if (small_size == 0 || big_size < small_size || big_size % small_size != 0)
    {
        throw std::invalid_argument(invalid_shape + std::string(__FUNCTION__));
    }

    lint n_blocks = std::min(big_size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    __broadcast_multiply_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(this->m_data, other.m_data, big_size, small_size);

    return *this;
}
template
Matrix_CUDA<int> & Matrix_CUDA<int>::operator *= (const Matrix_CUDA<int> & other);
template
Matrix_CUDA<float> & Matrix_CUDA<float>::operator *= (const Matrix_CUDA<float> & other);


template <typename DT>
Matrix_CUDA<DT> & Matrix_CUDA<DT>::operator += (DT scalar)
{
    lint size = this->m_shape.m_size;
    lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    __add_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(this->m_data, scalar, size);

    return *this;
}
template
Matrix_CUDA<int> & Matrix_CUDA<int>::operator += (int scalar);
template
Matrix_CUDA<float> & Matrix_CUDA<float>::operator += (float scalar);


template <typename DT>
Matrix_CUDA<DT> & Matrix_CUDA<DT>::operator -= (DT scalar)
{
    lint size = this->m_shape.m_size;
    lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    __subtract_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(this->m_data, scalar, size);

    return *this;
}
template
Matrix_CUDA<int> & Matrix_CUDA<int>::operator -= (int scalar);
template
Matrix_CUDA<float> & Matrix_CUDA<float>::operator -= (float scalar);


template <typename DT>
Matrix_CUDA<DT> & Matrix_CUDA<DT>::operator *= (DT scalar)
{
    lint size = this->m_shape.m_size;
    lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    __multiply_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(this->m_data, scalar, size);

    return *this;
}
template
Matrix_CUDA<int> & Matrix_CUDA<int>::operator *= (int scalar);
template
Matrix_CUDA<float> & Matrix_CUDA<float>::operator *= (float scalar);


template <typename DT>
Matrix_CUDA<DT> & Matrix_CUDA<DT>::operator /= (DT scalar)
{
    lint size = this->m_shape.m_size;
    lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    __divide_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(this->m_data, scalar, size);

    return *this;
}
template
Matrix_CUDA<int> & Matrix_CUDA<int>::operator /= (int scalar);
template
Matrix_CUDA<float> & Matrix_CUDA<float>::operator /= (float scalar);


template <typename DT>
Matrix_CUDA<DT> & Matrix_CUDA<DT>::gaussian_random(DT mu, DT sigma)
{
    lint size = this->m_shape.m_size;

    if (size > 0)
    {
        julie::la::cpu::Matrix_CPU<DT> cpu_mat {*this};
        cpu_mat.gaussian_random(mu, sigma);

        lint cu_bytes = size * sizeof(DT);
        CUDA_CHECK( cudaMemcpy (this->m_data, cpu_mat.m_data, cu_bytes, cudaMemcpyHostToDevice) );
    }

    return *this;
}
template
Matrix_CUDA<float> & Matrix_CUDA<float>::gaussian_random(float mu, float sigma);


template <typename DT>
Matrix_CUDA<DT> & Matrix_CUDA<DT>::uniform_random(DT min, DT max)
{
    lint size = this->m_shape.m_size;

    if (size > 0)
    {
        julie::la::cpu::Matrix_CPU<DT> cpu_mat {*this};
        cpu_mat.uniform_random(min, max);

        lint cu_bytes = size * sizeof(DT);
        CUDA_CHECK( cudaMemcpy (this->m_data, cpu_mat.m_data, cu_bytes, cudaMemcpyHostToDevice) );
    }

    return *this;
}
template
Matrix_CUDA<int> & Matrix_CUDA<int>::uniform_random(int min, int max);
template
Matrix_CUDA<float> & Matrix_CUDA<float>::uniform_random(float min, float max);


template <typename DT>
Matrix_CUDA<DT> & Matrix_CUDA<DT>::normalize(DT min, DT max)
{
    if (min >= max)
    {
        throw std::invalid_argument(std::string("Matrix_CUDA::normalize: min should be smaller than max"));
    }

    lint size = m_shape.m_size;
    DT range = max - min;

    lint n_blocks_1 = std::min(size / BLOCK_WIDTH_1D / KERNEL_SLICE_LEN + 1, MAX_N_BLOCK_1D);
    lint n_blocks_2 = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    DT l_max = __max(this->m_data, n_blocks_1, BLOCK_WIDTH_1D, size);
    DT l_min = __min(this->m_data, n_blocks_1, BLOCK_WIDTH_1D, size);

    DT l_range = l_max - l_min;
    if (0 == l_range)
    {
        this->uniform_random(min, max);
    }
    else
    {
        __normalize_1d<<<n_blocks_2, BLOCK_WIDTH_1D>>>(this->m_data, l_min, l_range, min, range, size);
    }

    return *this;
}
template
Matrix_CUDA<float> & Matrix_CUDA<float>::normalize(float min, float max);


template <typename DT>
Matrix_CUDA<DT> & Matrix_CUDA<DT>::normalize()
{
    if (this->m_shape.m_size < 1)
    {
        throw std::string("Empty Matrix_CUDA cannot be normalized in ") + std::string(__FUNCTION__);
    }
    
    DT mean = this->mean();
    
    lint size = this->m_shape.m_size;

    lint n_blocks_1 = std::min(size / BLOCK_WIDTH_1D / KERNEL_SLICE_LEN + 1, MAX_N_BLOCK_1D);
    lint n_blocks_2 = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    // Apply memory for cuda
    DT *buffer;
    lint cu_bytes = size * sizeof(DT);
    CUDA_CHECK( cudaMalloc ((void**) &buffer, cu_bytes) );

    // x = x - mean
    __subtract_1d<<<n_blocks_2, BLOCK_WIDTH_1D>>>(this->m_data, mean, size);
    
    // copy x - mean
    CUDA_CHECK( cudaMemcpy (buffer, this->m_data, cu_bytes, cudaMemcpyDeviceToDevice) );
    
    // a = a * a
    __square_1d<<<n_blocks_2, BLOCK_WIDTH_1D>>>(buffer, size);
    
    // sum(a)
    DT square_sum = __sum(buffer, n_blocks_1, BLOCK_WIDTH_1D, size);

    CUDA_CHECK( cudaFree (buffer) );

    if (0 == square_sum)
    {
        this->gaussian_random(0, 1);
    }
    else
    {
        DT scale = sqrt(square_sum / size);

        // (x - mean) / sigma
        __divide_1d<<<n_blocks_2, BLOCK_WIDTH_1D>>>(this->m_data, scale, size);
    }

    return *this;
}
template
Matrix_CUDA<float> & Matrix_CUDA<float>::normalize();


template <typename DT>
Matrix_CUDA<DT> & Matrix_CUDA<DT>::reshape(const Shape & shape)
{
    if (shape.m_size != this->m_shape.m_size)
    {
        throw std::invalid_argument(
            std::string("Matrix_CUDA::reshape: the new shape should be compatible with number of elements in this matrix"));
    }

    this->m_shape = shape;

    return *this;
}
template
Matrix_CUDA<int> & Matrix_CUDA<int>::reshape(const Shape & shape);
template
Matrix_CUDA<float> & Matrix_CUDA<float>::reshape(const Shape & shape);


template <typename DT>
Matrix_CUDA<DT> & Matrix_CUDA<DT>::left_extend_shape()
{
    this->m_shape.left_extend();
    return *this;
}
template
Matrix_CUDA<int> & Matrix_CUDA<int>::left_extend_shape();
template
Matrix_CUDA<float> & Matrix_CUDA<float>::left_extend_shape();


template <typename DT>
Matrix_CUDA<DT> & Matrix_CUDA<DT>::right_extend_shape()
{
    this->m_shape.right_extend();
    return *this;
}
template
Matrix_CUDA<int> & Matrix_CUDA<int>::right_extend_shape();
template
Matrix_CUDA<float> & Matrix_CUDA<float>::right_extend_shape();


template <typename DT>
void Matrix_CUDA<DT>::get_left_extended(Matrix_CUDA<DT> &output, lint duplicate) const
{
    if (duplicate < 1)
    {
        throw std::invalid_argument(std::string("Matrix_CUDA::get_left_extended: duplicate should be a positive value"));
    }

    if (this->m_shape.m_size < 1)
    {
        throw std::string("Empty Matrix_CUDA cannot be left extended in ") + std::string(__FUNCTION__);
    }

    // Matrix_CUDA mat{ Shape{ duplicate } + this->m_shape };
    renew_if_shape_not_match( output, Shape {duplicate} + this->m_shape );
    
    lint size = this->m_shape.m_size;
    DT *output_start = output.m_data;

    for (lint j = 0; j < duplicate; ++j)
    {
        CUDA_CHECK( cudaMemcpy (output_start, this->m_data, size * sizeof(DT), cudaMemcpyDeviceToDevice) );
        output_start += size;
    }
}
template
void Matrix_CUDA<int>::get_left_extended(Matrix_CUDA<int> &output, lint duplicate) const;
template
void Matrix_CUDA<float>::get_left_extended(Matrix_CUDA<float> &output, lint duplicate) const;


template <typename DT>
void Matrix_CUDA<DT>::get_right_extended(Matrix_CUDA<DT> &output, lint duplicate) const
{
    if (duplicate < 1)
    {
        throw std::invalid_argument(std::string("Matrix_CUDA::get_right_extended: duplicate should be a positive value"));
    }

    if (this->m_shape.m_size < 1)
    {
        throw std::string("Empty Matrix_CUDA cannot be right extended in ") + std::string(__FUNCTION__);
    }

    // Matrix_CUDA mat{ this->m_shape + Shape{ duplicate } };
    renew_if_shape_not_match( output, this->m_shape + Shape {duplicate} );

    lint size = output.m_shape.size();

    lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);
    __copy_one2n<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, this->m_data, size, duplicate);
}
template
void Matrix_CUDA<int>::get_right_extended(Matrix_CUDA<int> &output, lint duplicate) const;
template
void Matrix_CUDA<float>::get_right_extended(Matrix_CUDA<float> &output, lint duplicate) const;


template <typename DT>
julie::la::Coordinate Matrix_CUDA<DT>::argmax() const
{
    if (this->m_shape.m_size < 1)
    {
        throw std::string("Empty Matrix_CUDA cannot do argmax in ") + std::string(__FUNCTION__);
    }

    lint size = this->m_shape.m_size;

    lint n_blocks = std::min(size / BLOCK_WIDTH_1D / KERNEL_SLICE_LEN + 1, MAX_N_BLOCK_1D);
    lint argmax_index = __argmax(this->m_data, n_blocks, BLOCK_WIDTH_1D, size);
    
    return julie::la::Coordinate {argmax_index, this->m_shape};
}
template
julie::la::Coordinate Matrix_CUDA<int>::argmax() const;
template
julie::la::Coordinate Matrix_CUDA<float>::argmax() const;


template <typename DT>
std::vector<julie::la::Coordinate> Matrix_CUDA<DT>::argmax(lint dim) const
{
    if (this->m_shape.m_size < 1)
    {
        throw std::invalid_argument{std::string{"Null matrix cannot do argmax operation."}};
    }

    Shape left_sub_sh = this->m_shape.sub_shape(0, dim - 1);
    Shape right_sub_sh = this->m_shape.sub_shape(dim + 1, this->m_shape.m_dim - 1);

    lint left_size = std::max(left_sub_sh.m_size, 1L);
    lint current_size = this->m_shape[dim];
    lint right_size = std::max(right_sub_sh.m_size, 1L);

    DT max = std::numeric_limits<DT>::max() * (-1);

    lint output_size = left_size * right_size;
    lint n_blocks = std::min(output_size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    int64_t grid_x = std::max(nsqrt(n_blocks * left_size / right_size), 1L);
    int64_t grid_y = std::max(n_blocks / grid_x, 1L);

    dim3 dimGrid(grid_x, grid_y, 1);
    dim3 dimBlock(BLOCK_WIDTH_1D / 16, 16, 1);

    lint *indexes_device;
    lint cu_bytes = output_size * sizeof(lint);
    CUDA_CHECK( cudaMalloc ((void**) &indexes_device, cu_bytes) );
    lint *indexes_host = new lint[output_size];

    __argmax_2d<<<dimGrid, dimBlock>>> (indexes_device, this->m_data, max, left_size, current_size, right_size);
    CUDA_CHECK( cudaMemcpy (indexes_host, indexes_device, cu_bytes, cudaMemcpyDeviceToHost) );

    std::vector<Coordinate> output;
    Coordinate left_coord {0, left_sub_sh};
    Coordinate right_coord {0, right_sub_sh};

    lint output_idx = 0;
    for (lint l_i = 0; l_i < left_size; ++l_i)
    {
        for (lint r_i = 0; r_i < right_size; ++r_i)
        {
            Coordinate current_coord {indexes_host[output_idx], Shape{current_size}};
            output.push_back(left_coord + current_coord + right_coord);

            ++right_coord;
            ++output_idx;
        }

        ++left_coord;
        right_coord = 0;
    }

    CUDA_CHECK( cudaFree(indexes_device) );
    delete[] indexes_host;

    return output;
}
template
std::vector<julie::la::Coordinate> Matrix_CUDA<int>::argmax(lint dim) const;
template
std::vector<julie::la::Coordinate> Matrix_CUDA<float>::argmax(lint dim) const;


template <typename DT>
DT Matrix_CUDA<DT>::max() const
{
    if (this->m_shape.m_size < 1)
    {
        throw std::string("Empty Matrix_CUDA cannot do max in ") + std::string(__FUNCTION__);
    }

    lint size = this->m_shape.m_size;

    lint n_blocks = std::min(size / BLOCK_WIDTH_1D / KERNEL_SLICE_LEN + 1, MAX_N_BLOCK_1D);
    return __max(this->m_data, n_blocks, BLOCK_WIDTH_1D, size);
}
template
int Matrix_CUDA<int>::max() const;
template
float Matrix_CUDA<float>::max() const;


template <typename DT>
julie::la::Coordinate Matrix_CUDA<DT>::argmin() const
{
    if (this->m_shape.m_size < 1)
    {
        throw std::string("Empty Matrix_CUDA cannot do argmin in ") + std::string(__FUNCTION__);
    }

    lint size = this->m_shape.m_size;

    lint n_blocks = std::min(size / BLOCK_WIDTH_1D / KERNEL_SLICE_LEN + 1, MAX_N_BLOCK_1D);
    lint argmin_index = __argmin(this->m_data, n_blocks, BLOCK_WIDTH_1D, size);
    
    return julie::la::Coordinate {argmin_index, this->m_shape};
}
template
julie::la::Coordinate Matrix_CUDA<int>::argmin() const;
template
julie::la::Coordinate Matrix_CUDA<float>::argmin() const;


template <typename DT>
std::vector<julie::la::Coordinate> Matrix_CUDA<DT>::argmin(lint dim) const
{
    if (this->m_shape.m_size < 1)
    {
        throw std::invalid_argument{std::string{"Null matrix cannot do argmax operation."}};
    }

    Shape left_sub_sh = this->m_shape.sub_shape(0, dim - 1);
    Shape right_sub_sh = this->m_shape.sub_shape(dim + 1, this->m_shape.m_dim - 1);

    lint left_size = std::max(left_sub_sh.m_size, 1L);
    lint current_size = this->m_shape[dim];
    lint right_size = std::max(right_sub_sh.m_size, 1L);

    DT min = std::numeric_limits<DT>::max();

    lint output_size = left_size * right_size;
    lint n_blocks = std::min(output_size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    int64_t grid_x = std::max(nsqrt(n_blocks * left_size / right_size), 1L);
    int64_t grid_y = std::max(n_blocks / grid_x, 1L);

    dim3 dimGrid(grid_x, grid_y, 1);
    dim3 dimBlock(BLOCK_WIDTH_1D / 16, 16, 1);

    lint *indexes_device;
    lint cu_bytes = output_size * sizeof(lint);
    CUDA_CHECK( cudaMalloc ((void**) &indexes_device, cu_bytes) );
    lint *indexes_host = new lint[output_size];

    __argmin_2d<<<dimGrid, dimBlock>>> (indexes_device, this->m_data, min, left_size, current_size, right_size);
    CUDA_CHECK( cudaMemcpy (indexes_host, indexes_device, cu_bytes, cudaMemcpyDeviceToHost) );

    std::vector<Coordinate> output;
    Coordinate left_coord {0, left_sub_sh};
    Coordinate right_coord {0, right_sub_sh};

    lint output_idx = 0;
    for (lint l_i = 0; l_i < left_size; ++l_i)
    {
        for (lint r_i = 0; r_i < right_size; ++r_i)
        {
            Coordinate current_coord {indexes_host[output_idx], Shape{current_size}};
            output.push_back(left_coord + current_coord + right_coord);

            ++right_coord;
            ++output_idx;
        }

        ++left_coord;
        right_coord = 0;
    }

    CUDA_CHECK( cudaFree(indexes_device) );
    delete[] indexes_host;

    return output;
}
template
std::vector<julie::la::Coordinate> Matrix_CUDA<int>::argmin(lint dim) const;
template
std::vector<julie::la::Coordinate> Matrix_CUDA<float>::argmin(lint dim) const;


template <typename DT>
DT Matrix_CUDA<DT>::min() const
{
    if (this->m_shape.m_size < 1)
    {
        throw std::string("Empty Matrix_CUDA cannot do min in ") + std::string(__FUNCTION__);
    }

    lint size = this->m_shape.m_size;

    lint n_blocks = std::min(size / BLOCK_WIDTH_1D / KERNEL_SLICE_LEN + 1, MAX_N_BLOCK_1D);
    return __min(this->m_data, n_blocks, BLOCK_WIDTH_1D, size);
}
template
int Matrix_CUDA<int>::min() const;
template
float Matrix_CUDA<float>::min() const;


template <typename DT>
DT Matrix_CUDA<DT>::sum() const
{
    if (this->m_shape.m_size < 1)
    {
        throw std::string("Empty Matrix_CUDA cannot do sum in ") + std::string(__FUNCTION__);
    }

    lint size = this->m_shape.m_size;
    lint n_blocks = std::min(size / BLOCK_WIDTH_1D / KERNEL_SLICE_LEN + 1, MAX_N_BLOCK_1D);

    return __sum(this->m_data, n_blocks, BLOCK_WIDTH_1D, size);
}
template
int Matrix_CUDA<int>::sum() const;
template
float Matrix_CUDA<float>::sum() const;


template <typename DT>
DT Matrix_CUDA<DT>::mean() const
{
    return this->sum() / this->m_shape.m_size;
}
template
int Matrix_CUDA<int>::mean() const;
template
float Matrix_CUDA<float>::mean() const;


template <typename DT>
DT Matrix_CUDA<DT>::variance() const
{
    if (this->m_shape.m_size < 1)
    {
        throw std::string("Empty Matrix_CUDA cannot do variance in ") + std::string(__FUNCTION__);
    }

    DT mean = this->mean();

    lint size = this->m_shape.m_size;

    lint n_blocks_1 = std::min(size / BLOCK_WIDTH_1D / KERNEL_SLICE_LEN + 1, MAX_N_BLOCK_1D);
    lint n_blocks_2 = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    // Apply memory for cuda
    DT *buffer;
    lint cu_bytes = size * sizeof(DT);
    CUDA_CHECK( cudaMalloc ((void**) &buffer, cu_bytes) );

    CUDA_CHECK( cudaMemcpy (buffer, this->m_data, cu_bytes, cudaMemcpyDeviceToDevice) );

    __subtract_1d<<<n_blocks_2, BLOCK_WIDTH_1D>>>(buffer, mean, size);
    __square_1d<<<n_blocks_2, BLOCK_WIDTH_1D>>>(buffer, size);
    
    DT square_sum = __sum(buffer, n_blocks_1, BLOCK_WIDTH_1D, size);

    CUDA_CHECK( cudaFree (buffer) );
    

    return square_sum / size;
}
template
int Matrix_CUDA<int>::variance() const;
template
float Matrix_CUDA<float>::variance() const;


template <typename DT>
std::vector<Matrix_CUDA<DT>> Matrix_CUDA<DT>::get_collapsed(lint dim) const
{
    if (dim < 0 || dim >= this->m_shape.m_dim)
    {
        throw std::invalid_argument("Matrix_CUDA::scale_one_dimension: dimension index out of range.");
    }

    Shape sh_before = this->m_shape.sub_shape(0, dim - 1);
    Shape sh_after = this->m_shape.sub_shape(dim + 1, this->m_shape.m_dim - 1);
    Shape sh_collapsed = sh_before + sh_after;

    lint size_dims_before = sh_before.m_size;
    lint size_dims_after = sh_after.m_size;
    lint size_dims_and_after = this->m_shape.sub_shape(dim, this->m_shape.m_dim - 1).m_size;

    if (0 == size_dims_before)
    {
        size_dims_before = 1;
    }

    if (0 == size_dims_after)
    {
        size_dims_after = 1;
    }

    std::vector<Matrix_CUDA<DT>> all_collapsed;
    DT *start = this->m_data;
    lint cu_bytes = size_dims_after * sizeof(DT);

    for (lint i = 0; i < this->m_shape.m_data[dim]; ++i)
    {
        Matrix_CUDA<DT> collapsed{ sh_collapsed };
        DT *l_start = start;
        DT *clps_ele = collapsed.m_data;

        for (lint j = 0; j < size_dims_before; ++j)
        {
            CUDA_CHECK( cudaMemcpy (clps_ele, l_start, cu_bytes, cudaMemcpyDeviceToDevice) );

            clps_ele += size_dims_after;
            l_start += size_dims_and_after;
        }

        all_collapsed.push_back(collapsed);
        start += size_dims_after;
    }

    return all_collapsed;
}
template
std::vector<Matrix_CUDA<int>> Matrix_CUDA<int>::get_collapsed(lint dim) const;
template
std::vector<Matrix_CUDA<float>> Matrix_CUDA<float>::get_collapsed(lint dim) const;


template <typename DT>
void Matrix_CUDA<DT>::get_reduce_sum(Matrix_CUDA<DT> &output, lint dim) const
{
    if (dim < 0 || dim >= this->m_shape.m_dim)
    {
        throw std::invalid_argument("Matrix_CUDA::fuse: dimension index out of range.");
    }

    Shape sh_before = this->m_shape.sub_shape(0, dim - 1);
    Shape sh_after = this->m_shape.sub_shape(dim + 1, this->m_shape.m_dim - 1);
    Shape sh_collapsed = sh_before + sh_after;

    renew_if_shape_not_match(output, sh_collapsed);
    output = 0;

    lint size_dims_before = sh_before.m_size;
    lint size_dims_after = sh_after.m_size;
    lint size_dims_and_after = this->m_shape.sub_shape(dim, this->m_shape.m_dim - 1).m_size;

    if (0 == size_dims_before)
    {
        size_dims_before = 1;
    }

    if (0 == size_dims_after)
    {
        size_dims_after = 1;
    }

    DT *start = this->m_data;
    DT *output_start = output.m_data;

    for (lint i = 0; i < this->m_shape.m_data[dim]; ++i)
    {
        DT *l_start = start;
        DT *clps_ele = output_start;

        for (lint j = 0; j < size_dims_before; ++j)
        {   
            lint n_blocks = std::min(size_dims_after / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);
            __add_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(clps_ele, l_start, size_dims_after);

            clps_ele += size_dims_after;
            l_start += size_dims_and_after;
        }

        start += size_dims_after;
    }
}
template
void Matrix_CUDA<int>::get_reduce_sum(Matrix_CUDA<int> &output, lint dim) const;
template
void Matrix_CUDA<float>::get_reduce_sum(Matrix_CUDA<float> &output, lint dim) const;


template <typename DT>
void Matrix_CUDA<DT>::get_reduce_mean(Matrix_CUDA<DT> &output, lint dim) const
{
    if (dim < 0 || dim >= this->m_shape.m_dim)
    {
        throw std::invalid_argument("Matrix_CUDA::fuse: dimension index out of range.");
    }

    this->get_reduce_sum(output, dim);
    lint dim_size = this->m_shape.m_data[dim];
    output /= dim_size;
}
template
void Matrix_CUDA<int>::get_reduce_mean(Matrix_CUDA<int> &output, lint dim) const;
template
void Matrix_CUDA<float>::get_reduce_mean(Matrix_CUDA<float> &output, lint dim) const;


template <typename DT>
DT Matrix_CUDA<DT>::euclidean_norm() const
{
    if (this->m_shape.m_size < 1)
    {
        throw std::string("Empty Matrix_CUDA cannot do euclidean_norm in ") + std::string(__FUNCTION__);
    }

    lint size = this->m_shape.m_size;

    lint n_blocks_1 = std::min(size / BLOCK_WIDTH_1D / KERNEL_SLICE_LEN + 1, MAX_N_BLOCK_1D);
    lint n_blocks_2 = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);

    // Apply memory for cuda
    DT *buffer;
    lint cu_bytes = size * sizeof(DT);
    CUDA_CHECK( cudaMalloc ((void**) &buffer, cu_bytes) );

    __square_1d<<<n_blocks_2, BLOCK_WIDTH_1D>>>(buffer, this->m_data, size);
    
    DT square_sum = __sum(buffer, n_blocks_1, BLOCK_WIDTH_1D, size);

    CUDA_CHECK( cudaFree (buffer) );


    return sqrt(square_sum);
}
template
int Matrix_CUDA<int>::euclidean_norm() const;
template
float Matrix_CUDA<float>::euclidean_norm() const;


template <typename DT>
julie::la::Shape Matrix_CUDA<DT>::shape() const
{
    return m_shape;
}
template
julie::la::Shape Matrix_CUDA<int>::shape() const;
template
julie::la::Shape Matrix_CUDA<float>::shape() const;


template <typename DT>
std::string Matrix_CUDA<DT>::to_string() const
{
    return julie::la::cpu::Matrix_CPU<DT> {*this}.to_string();
}
template
std::string Matrix_CUDA<int>::to_string() const;
template
std::string Matrix_CUDA<float>::to_string() const;

/* Overloading of output stream << operator
*/
template <typename DT>
std::ostream & operator << (std::ostream & os, const Matrix_CUDA<DT> & m)
{
    return os << m.to_string();
}
template
std::ostream & operator << (std::ostream & os, const Matrix_CUDA<int> & m);
template
std::ostream & operator << (std::ostream & os, const Matrix_CUDA<float> & m);

} // namespace cuda
} // namespace la
} // namespace julie