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

#include "Matrix_CPU.hpp"
#include "Matrix_CPU_func.hpp"
#include "utilities.hpp"

#include <iostream>
#include <fstream>
#include <tuple>
#include <algorithm>
#include <stdexcept>
#include <iterator>
#include <sstream>
#include <cstring>
#include <functional>
#include <type_traits>


namespace julie
{
namespace la
{
namespace cpu
{

template <typename DT>
Matrix_CPU<DT>::Matrix_CPU()
    : m_shape{}, m_data{ nullptr }
{}
template
Matrix_CPU<int>::Matrix_CPU();
template
Matrix_CPU<float>::Matrix_CPU();


template <typename DT>
Matrix_CPU<DT>::Matrix_CPU(const Shape & shape)
    : m_shape{ shape }, m_data{ nullptr }
{
    if (shape.m_size < 1)
    {
        return;
    }
    
    m_data = new DT[this->m_shape.m_size];
}
template
Matrix_CPU<int>::Matrix_CPU(const Shape & shape);
template
Matrix_CPU<float>::Matrix_CPU(const Shape & shape);


template <typename DT>
Matrix_CPU<DT>::Matrix_CPU(DT value, const Shape & shape)
    : Matrix_CPU{ shape }
{
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] = value;
    }
}
template
Matrix_CPU<int>::Matrix_CPU(int value, const Shape & shape);
template
Matrix_CPU<float>::Matrix_CPU(float value, const Shape & shape);


template <typename DT>
Matrix_CPU<DT>::Matrix_CPU(const std::vector<Matrix_CPU>& matrices)
    : Matrix_CPU{}
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
                        std::string("Matrix_CPU::Matrix_CPU: invalid matrix array because of different shapes of matrices"));
                }
            }

            this->m_shape = Shape{ array_size } +mat_sh;
            this->m_data = new DT[this->m_shape.m_size];
            DT *this_pos = this->m_data;
            DT *that_pos;

            for (lint i = 0; i < array_size; ++i)
            {
                that_pos = matrices[i].m_data;
                for (lint j = 0; j < mat_size; ++j)
                {
                    *this_pos = *that_pos;
                    ++this_pos;
                    ++that_pos;
                }
            }
        }
    }
}
template
Matrix_CPU<int>::Matrix_CPU(const std::vector<Matrix_CPU>& matrices);
template
Matrix_CPU<float>::Matrix_CPU(const std::vector<Matrix_CPU>& matrices);


template <typename DT>
Matrix_CPU<DT>::Matrix_CPU(const std::vector<std::shared_ptr<Matrix_CPU>>& matrices)
    : Matrix_CPU{}
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
                        std::string("Matrix_CPU::Matrix_CPU: invalid matrix array because of different shapes of matrices"));
                }
            }

            this->m_shape = Shape{ array_size } +mat_sh;
            this->m_data = new DT[this->m_shape.m_size];
            DT *this_pos = this->m_data;
            DT *that_pos;

            for (lint i = 0; i < array_size; ++i)
            {
                that_pos = matrices[i]->m_data;
                for (lint j = 0; j < mat_size; ++j)
                {
                    *this_pos = *that_pos;
                    ++this_pos;
                    ++that_pos;
                }
            }
        }
    }
}
template
Matrix_CPU<int>::Matrix_CPU(const std::vector<std::shared_ptr<Matrix_CPU>> &matrices);
template
Matrix_CPU<float>::Matrix_CPU(const std::vector<std::shared_ptr<Matrix_CPU>> &matrices);


template <typename DT>
Matrix_CPU<DT>::Matrix_CPU(const Matrix_CPU & other)
    : m_shape{ other.m_shape }, m_data{ nullptr }
{
    lint size = m_shape.m_size;

    if (size > 0)
    {
        m_data = new DT[size];
        std::memcpy(m_data, other.m_data, size * sizeof(DT));
    }
}
template
Matrix_CPU<int>::Matrix_CPU(const Matrix_CPU & other);
template
Matrix_CPU<float>::Matrix_CPU(const Matrix_CPU & other);


template <typename DT>
Matrix_CPU<DT>::Matrix_CPU(Matrix_CPU && other)
    : m_shape{ std::move(other.m_shape) }, m_data{ other.m_data }
{
    other.m_data = nullptr;
}
template
Matrix_CPU<int>::Matrix_CPU(Matrix_CPU && other);
template
Matrix_CPU<float>::Matrix_CPU(Matrix_CPU && other);


template <typename DT>
Matrix_CPU<DT>::Matrix_CPU(const std::vector<DT> & vec, bool horizontal)
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
        m_data = new DT[size];

        for (lint i = 0; i < size; ++i)
        {
            this->m_data[i] = vec[i];
        }
    }
}
template
Matrix_CPU<int>::Matrix_CPU(const std::vector<int> & vec, bool horizontal);
template
Matrix_CPU<float>::Matrix_CPU(const std::vector<float> & vec, bool horizontal);


template <typename DT>
Matrix_CPU<DT>::Matrix_CPU(const std::vector<DT> & vec, const Shape &shape)
    : m_shape{ static_cast<lint>(vec.size()) }, m_data{ nullptr }
{
    if (shape.m_size != this->m_shape.m_size)
    {
        throw std::invalid_argument(std::string("Matrix_CPU::Matrix_CPU: Size of shape argument does not match vector length"));
    }

    this->m_shape = shape;
    lint size = m_shape.m_size;

    if (size > 0)
    {
        m_data = new DT[size];

        for (lint i = 0; i < size; ++i)
        {
            this->m_data[i] = vec[i];
        }
    }
}
template
Matrix_CPU<int>::Matrix_CPU(const std::vector<int> & vec, const Shape &shape);
template
Matrix_CPU<float>::Matrix_CPU(const std::vector<float> & vec, const Shape &shape);



template <typename DT>
Matrix_CPU<DT>::Matrix_CPU(const std::vector<std::vector<DT>> & array)
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
        this->m_data = new DT[size];

        DT *pos = this->m_data;
        for (auto itr = array.begin(); itr != array.end(); ++itr)
        {
            DT *pos1 = pos;
            for (auto itr1 = itr->begin(); itr1 != itr->end(); ++itr1, ++pos1)
            {
                *pos1 = *itr1;
            }
            pos += this->m_shape.m_data[1];
        }
    }
}
template
Matrix_CPU<int>::Matrix_CPU(const std::vector<std::vector<int>> & array);
template
Matrix_CPU<float>::Matrix_CPU(const std::vector<std::vector<float>> & array);


template <typename DT>
Matrix_CPU<DT>::~Matrix_CPU()
{
    // std::cout << "Matrix_CPU destroyed, shape: " << this->m_shape << std::endl;
    delete[]this->m_data;
}
template 
Matrix_CPU<int>::~Matrix_CPU();
template 
Matrix_CPU<float>::~Matrix_CPU();


template <typename DT>
inline typename Matrix_CPU<DT>::diterator Matrix_CPU<DT>::begin() const
{
    diterator itr{ this->m_shape.m_size, this->m_data, 0 };
    return itr;
}
template
typename Matrix_CPU<int>::diterator Matrix_CPU<int>::begin() const;
template
typename Matrix_CPU<float>::diterator Matrix_CPU<float>::begin() const;


template<typename DT>
inline typename Matrix_CPU<DT>::diterator Matrix_CPU<DT>::end() const
{
    diterator itr{ this->m_shape.m_size, this->m_data, this->m_shape.m_size };
    return itr;
}
template
typename Matrix_CPU<int>::diterator Matrix_CPU<int>::end() const;
template
typename Matrix_CPU<float>::diterator Matrix_CPU<float>::end() const;


template<typename DT>
void Matrix_CPU<DT>::get_full_transpose(Matrix_CPU<DT> & out) const
{
    full_transpose(out, *this);
}
template
void Matrix_CPU<int>::get_full_transpose(Matrix_CPU<int> & out) const;
template
void Matrix_CPU<float>::get_full_transpose(Matrix_CPU<float> & out) const;


template<typename DT>
void Matrix_CPU<DT>::get_transpose(Matrix_CPU<DT> & out, lint left_dims) const
{
    transpose(out, *this, left_dims);
}
template
void Matrix_CPU<int>::get_transpose(Matrix_CPU<int> & out, lint left_dims) const;
template
void Matrix_CPU<float>::get_transpose(Matrix_CPU<float> & out, lint left_dims) const;


template <typename DT>
Matrix_CPU<DT> & Matrix_CPU<DT>::operator = (const Matrix_CPU & other)
{
    bool renew = false;
    if (this->m_shape.m_size != other.m_shape.m_size)
    {
        renew = true;
    }

    if (renew)
    {
        delete[]this->m_data;
        this->m_data = nullptr;
    }

    this->m_shape = other.m_shape;
    lint size = this->m_shape.m_size;
    if (size < 1)
    {
        return *this;
    }

    if (renew)
    {
        this->m_data = new DT[size];
    }

    std::memcpy(this->m_data, other.m_data, size * sizeof(DT));

    return *this;
}
template
Matrix_CPU<int> & Matrix_CPU<int>::operator = (const Matrix_CPU & other);
template
Matrix_CPU<float> & Matrix_CPU<float>::operator = (const Matrix_CPU & other);


template <typename DT>
Matrix_CPU<DT> & Matrix_CPU<DT>::operator = (Matrix_CPU && other)
{
    delete[]m_data;
    this->m_shape = std::move(other.m_shape);

    this->m_data = other.m_data;
    other.m_data = nullptr;

    return *this;
}
template
Matrix_CPU<int> & Matrix_CPU<int>::operator = (Matrix_CPU && other);
template
Matrix_CPU<float> & Matrix_CPU<float>::operator = (Matrix_CPU && other);


template <typename DT>
Matrix_CPU<DT> & Matrix_CPU<DT>::operator = (DT scalar)
{
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] = scalar;
    }

    return *this;
}
template
Matrix_CPU<int> & Matrix_CPU<int>::operator = (int scalar);
template
Matrix_CPU<float> & Matrix_CPU<float>::operator = (float scalar);


template <typename DT>
DT Matrix_CPU<DT>::at(const Coordinate & pos) const
{
    if (this->m_shape != pos.m_shape)
    {
        throw std::invalid_argument(invalid_coordinate);
    }

    return this->m_data[pos.index()];
}
template
int Matrix_CPU<int>::at(const Coordinate & pos) const;
template
float Matrix_CPU<float>::at(const Coordinate & pos) const;


template <typename DT>
DT & Matrix_CPU<DT>::at(const Coordinate & pos)
{
    if (this->m_shape != pos.m_shape)
    {
        throw std::invalid_argument(invalid_coordinate);
    }

    return this->m_data[pos.index()];
}
template
int & Matrix_CPU<int>::at(const Coordinate & pos);
template
float & Matrix_CPU<float>::at(const Coordinate & pos);


template <typename DT>
DT Matrix_CPU<DT>::operator [] (const Coordinate & pos) const
{
    if (this->m_shape != pos.m_shape)
    {
        throw std::invalid_argument(invalid_coordinate);
    }

    return this->m_data[pos.index()];
}
template
int Matrix_CPU<int>::operator [] (const Coordinate & pos) const;
template
float Matrix_CPU<float>::operator [] (const Coordinate & pos) const;


template <typename DT>
DT & Matrix_CPU<DT>::operator [] (const Coordinate & pos)
{
    if (this->m_shape != pos.m_shape)
    {
        throw std::invalid_argument(invalid_coordinate);
    }

    return this->m_data[pos.index()];
}
template
int & Matrix_CPU<int>::operator [] (const Coordinate & pos);
template
float & Matrix_CPU<float>::operator [] (const Coordinate & pos);


template <typename DT>
DT Matrix_CPU<DT>::operator [] (std::initializer_list<lint> list) const
{
    Coordinate pos{ list, this->m_shape };
    return this->m_data[pos.index()];
}
template
int Matrix_CPU<int>::operator [] (std::initializer_list<lint> list) const;
template
float Matrix_CPU<float>::operator [] (std::initializer_list<lint> list) const;


template <typename DT>
DT & Matrix_CPU<DT>::operator [] (std::initializer_list<lint> list)
{
    Coordinate pos{ list, this->m_shape };
    return this->m_data[pos.index()];
}
template
int & Matrix_CPU<int>::operator [] (std::initializer_list<lint> list);
template
float & Matrix_CPU<float>::operator [] (std::initializer_list<lint> list);


template <typename DT>
Matrix_CPU<DT> & Matrix_CPU<DT>::operator += (const Matrix_CPU & other)
{
    lint big_size = m_shape.m_size;
    lint small_size = other.m_shape.m_size;

    if (big_size < small_size || big_size % small_size != 0)
    {
        throw std::invalid_argument(invalid_shape + std::string(__FUNCTION__));
    }

    for (lint i = 0; i < big_size; ++i)
    {
        m_data[i] += other.m_data[i % small_size];
    }

    return *this;
}
template
Matrix_CPU<int> & Matrix_CPU<int>::operator += (const Matrix_CPU & other);
template
Matrix_CPU<float> & Matrix_CPU<float>::operator += (const Matrix_CPU & other);


template <typename DT>
Matrix_CPU<DT> & Matrix_CPU<DT>::operator -= (const Matrix_CPU & other)
{
    lint big_size = m_shape.m_size;
    lint small_size = other.m_shape.m_size;

    if (big_size < small_size || big_size % small_size != 0)
    {
        throw std::invalid_argument(invalid_shape + std::string(__FUNCTION__));
    }

    for (lint i = 0; i < big_size; ++i)
    {
        m_data[i] -= other.m_data[i % small_size];
    }

    return *this;
}
template
Matrix_CPU<int> & Matrix_CPU<int>::operator -= (const Matrix_CPU & other);
template
Matrix_CPU<float> & Matrix_CPU<float>::operator -= (const Matrix_CPU & other);


template <typename DT>
Matrix_CPU<DT> & Matrix_CPU<DT>::operator *= (const Matrix_CPU & other)
{
    lint big_size = m_shape.m_size;
    lint small_size = other.m_shape.m_size;

    if (big_size < small_size || big_size % small_size != 0)
    {
        throw std::invalid_argument(invalid_shape + std::string(__FUNCTION__));
    }

    for (lint i = 0; i < big_size; ++i)
    {
        m_data[i] *= other.m_data[i % small_size];
    }

    return *this;
}
template
Matrix_CPU<int> & Matrix_CPU<int>::operator *= (const Matrix_CPU & other);
template
Matrix_CPU<float> & Matrix_CPU<float>::operator *= (const Matrix_CPU & other);


template <typename DT>
Matrix_CPU<DT> & Matrix_CPU<DT>::operator += (DT scalar)
{
    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] += scalar;
    }

    return *this;
}
template
Matrix_CPU<int> & Matrix_CPU<int>::operator += (int scalar);
template
Matrix_CPU<float> & Matrix_CPU<float>::operator += (float scalar);


template <typename DT>
Matrix_CPU<DT> & Matrix_CPU<DT>::operator -= (DT scalar)
{
    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] -= scalar;
    }

    return *this;
}
template
Matrix_CPU<int> & Matrix_CPU<int>::operator -= (int scalar);
template
Matrix_CPU<float> & Matrix_CPU<float>::operator -= (float scalar);


template <typename DT>
Matrix_CPU<DT> & Matrix_CPU<DT>::operator *= (DT scalar)
{
    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] *= scalar;
    }

    return *this;
}
template
Matrix_CPU<int> & Matrix_CPU<int>::operator *= (int scalar);
template
Matrix_CPU<float> & Matrix_CPU<float>::operator *= (float scalar);


template <typename DT>
Matrix_CPU<DT> & Matrix_CPU<DT>::operator /= (DT scalar)
{
    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] /= scalar;
    }

    return *this;
}
template
Matrix_CPU<int> & Matrix_CPU<int>::operator /= (int scalar);
template
Matrix_CPU<float> & Matrix_CPU<float>::operator /= (float scalar);


template <typename DT>
Matrix_CPU<DT> & Matrix_CPU<DT>::gaussian_random(DT mu, DT sigma)
{
    std::normal_distribution<DT> distribution{ mu, sigma };

    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        this->m_data[i] = distribution(global_rand_engine);
    }

    return *this;
}
// template
// Matrix_CPU<int> & Matrix_CPU<int>::gaussian_random(int mu, int sigma);
template
Matrix_CPU<float> & Matrix_CPU<float>::gaussian_random(float mu, float sigma);


template <>
Matrix_CPU<int> & Matrix_CPU<int>::uniform_random(int min, int max)
{
    if (min >= max)
    {
        throw std::invalid_argument(std::string("Matrix_CPU::uniform_random: min should be smaller than max"));
    }

    std::uniform_int_distribution<int> distribution{ min, max };

    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        this->m_data[i] = distribution(global_rand_engine);
    }

    return *this;
}
template <>
Matrix_CPU<float> & Matrix_CPU<float>::uniform_random(float min, float max)
{
    if (min >= max)
    {
        throw std::invalid_argument(std::string("Matrix_CPU::uniform_random: min should be smaller than max"));
    }

    std::uniform_real_distribution<float> distribution{ min, max };

    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        this->m_data[i] = distribution(global_rand_engine);
    }

    return *this;
}
template
Matrix_CPU<int> & Matrix_CPU<int>::uniform_random(int min, int max);
template
Matrix_CPU<float> & Matrix_CPU<float>::uniform_random(float min, float max);


template <typename DT>
Matrix_CPU<DT> & Matrix_CPU<DT>::normalize(DT min, DT max)
{
    if (min >= max)
    {
        throw std::invalid_argument(std::string("Matrix_CPU::normalize: min should be smaller than max"));
    }

    lint size = m_shape.m_size;
    DT range = max - min;

    DT l_max = std::numeric_limits<DT>::max() * (-1);
    DT l_min = std::numeric_limits<DT>::max();

    for (lint i = 0; i < size; ++i)
    {
        if (this->m_data[i] > l_max)
        {
            l_max = this->m_data[i];
        }

        if (this->m_data[i] < l_min)
        {
            l_min = this->m_data[i];
        }
    }

    DT l_range = l_max - l_min;
    if (0 == l_range)
    {
        this->uniform_random(min, max);
    }
    else
    {
        for (lint i = 0; i < size; ++i)
        {
            this->m_data[i] = min + ((this->m_data[i] - l_min) / l_range) * range;
        }
    }

    return *this;
}
// template
// Matrix_CPU<int> & Matrix_CPU<int>::normalize(int min, int max);
template
Matrix_CPU<float> & Matrix_CPU<float>::normalize(float min, float max);


template <typename DT>
Matrix_CPU<DT> & Matrix_CPU<DT>::normalize()
{
    lint size = this->m_shape.m_size;

    if (0 == size)
    {
        throw std::string("Normalize cannot be done in an empty Matrix_CPU in ") + std::string(__FUNCTION__);
    }

    DT mean = 0;
    for (lint i = 0; i < size; ++i)
    {
        mean += this->m_data[i];
    }
    mean /= size;

    DT var = 0;
    DT sub;
    for (lint i = 0; i < size; ++i)
    {
        sub = this->m_data[i] - mean;
        var += sub * sub;
    }
    var /= size;
    var = sqrt(var);

    if (0 == var)
    {
        this->gaussian_random(0, 1);
    }
    else
    {
        for (lint i = 0; i < size; ++i)
        {
            this->m_data[i] = (this->m_data[i] - mean) / var;
        }
    }

    return *this;
}
// template
// Matrix_CPU<int> & Matrix_CPU<int>::normalize();
template
Matrix_CPU<float> & Matrix_CPU<float>::normalize();


template <typename DT>
Matrix_CPU<DT> & Matrix_CPU<DT>::reshape(const Shape & shape)
{
    if (shape.m_size != this->m_shape.m_size)
    {
        throw std::invalid_argument(
            std::string("Matrix_CPU::reshape: the new shape should be compatible with number of elements in this matrix"));
    }

    this->m_shape = shape;

    return *this;
}
template
Matrix_CPU<int> & Matrix_CPU<int>::reshape(const Shape & shape);
template
Matrix_CPU<float> & Matrix_CPU<float>::reshape(const Shape & shape);


template <typename DT>
Matrix_CPU<DT> & Matrix_CPU<DT>::left_extend_shape()
{
    this->m_shape.left_extend();
    return *this;
}
template
Matrix_CPU<int> & Matrix_CPU<int>::left_extend_shape();
template
Matrix_CPU<float> & Matrix_CPU<float>::left_extend_shape();


template <typename DT>
Matrix_CPU<DT> & Matrix_CPU<DT>::right_extend_shape()
{
    this->m_shape.right_extend();
    return *this;
}
template
Matrix_CPU<int> & Matrix_CPU<int>::right_extend_shape();
template
Matrix_CPU<float> & Matrix_CPU<float>::right_extend_shape();


template <typename DT>
void Matrix_CPU<DT>::get_left_extended(Matrix_CPU<DT> &output, lint duplicate) const
{
    if (duplicate < 1)
    {
        throw std::invalid_argument(std::string("Matrix_CPU::get_left_extended: duplicate should be a positive value"));
    }

    if (this->m_shape.m_size < 1)
    {
        throw std::string("Empty Matrix_CPU cannot be left extended in ") + std::string(__FUNCTION__);
    }

    renew_if_shape_not_match( output, Shape {duplicate} + this->m_shape );

    lint size = this->m_shape.m_size;
    DT *output_start = output.m_data;

    for (lint j = 0; j < duplicate; ++j)
    {
        std::memcpy(output_start, this->m_data, size * sizeof(DT));
        output_start += size;
    }
}
template
void Matrix_CPU<int>::get_left_extended(Matrix_CPU<int> &output, lint duplicate) const;
template
void Matrix_CPU<float>::get_left_extended(Matrix_CPU<float> &output, lint duplicate) const;


template <typename DT>
void Matrix_CPU<DT>::get_right_extended(Matrix_CPU<DT> &output, lint duplicate) const
{
    if (duplicate < 1)
    {
        throw std::invalid_argument(std::string("Matrix_CPU::get_left_extended: duplicate should be a positive value"));
    }

    if (this->m_shape.m_size < 1)
    {
        throw std::string("Empty Matrix_CPU cannot be right extended in ") + std::string(__FUNCTION__);
    }

    renew_if_shape_not_match( output, this->m_shape + Shape {duplicate} );

    lint size = this->m_shape.m_size;
    DT *output_start = output.m_data;

    for (lint i = 0; i < size; ++i)
    {
        for (lint j = 0; j < duplicate; ++j)
        {
            *output_start = this->m_data[i];
            ++output_start;
        }
    }
}
template
void Matrix_CPU<int>::get_right_extended(Matrix_CPU<int> &output, lint duplicate) const;
template
void Matrix_CPU<float>::get_right_extended(Matrix_CPU<float> &output, lint duplicate) const;


template <typename DT>
Matrix_CPU<DT> & Matrix_CPU<DT>::scale_one_dimension(lint dim, const std::vector<DT> & scales)
{
    if (dim < 0 || dim >= this->m_shape.m_dim)
    {
        throw std::invalid_argument("Matrix_CPU::scale_one_dimension: dimension index out of range.");
    }

    if (this->m_shape.m_data[dim] != scales.size())
    {
        throw std::invalid_argument(
            "Matrix_CPU::scale_one_dimension: size of the vector is not compatible with size of this matrix dimension.");
    }

    lint size_dims_before = this->m_shape.sub_shape(0, dim - 1).m_size;
    lint size_dims_after = this->m_shape.sub_shape(dim + 1, this->m_shape.m_dim - 1).m_size;
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

    for (lint i = 0; i < scales.size(); ++i)
    {
        DT scale = scales[i];
        DT *l_start = start;

        for (lint j = 0; j < size_dims_before; ++j)
        {
            DT *ele = l_start;
            for (lint k = 0; k < size_dims_after; ++k)
            {
                *ele *= scale;
                ++ele;
            }
            l_start += size_dims_and_after;
        }

        start += size_dims_after;
    }

    return *this;
}
template
Matrix_CPU<int> & Matrix_CPU<int>::scale_one_dimension(lint dim, const std::vector<int> & scales);
template
Matrix_CPU<float> & Matrix_CPU<float>::scale_one_dimension(lint dim, const std::vector<float> & scales);


template <typename DT>
julie::la::Coordinate Matrix_CPU<DT>::argmax() const
{
    DT max = std::numeric_limits<DT>::max() * (-1);
    lint argmax_index = 0;
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        if (this->m_data[i] > max)
        {
            max = this->m_data[i];
            argmax_index = i;
        }
    }

    return Coordinate {argmax_index, this->m_shape};
}
template
julie::la::Coordinate Matrix_CPU<int>::argmax() const;
template
julie::la::Coordinate Matrix_CPU<float>::argmax() const;


template <typename DT>
std::vector<julie::la::Coordinate> Matrix_CPU<DT>::argmax(lint dim) const
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

    lint current_and_right_size = current_size * right_size;

    DT *left_begin = this->m_data;
    DT max = std::numeric_limits<DT>::max() * (-1);

    std::vector<Coordinate> output;
    Coordinate left_coord {0, left_sub_sh};
    Coordinate right_coord {0, right_sub_sh};

    for (lint l_i = 0; l_i < left_size; ++l_i)
    {
        DT *current_begin = left_begin;
        
        for (lint r_i = 0; r_i < right_size; ++r_i)
        {
            DT *pos = current_begin;
            DT m = max;
            lint max_c = 0;

            for (lint c_i = 0; c_i < current_size; ++c_i)
            {
                // this->m_data[ right_size * (current_size * l_i + c_i) + r_i ];
                if (*pos > m)
                {
                    m = *pos;
                    max_c = c_i;
                }

                pos += right_size;
            }


            Coordinate current_coord {max_c, Shape{current_size}};
            output.push_back(left_coord + current_coord + right_coord);

            ++right_coord;
            ++current_begin;
        }

        ++left_coord;
        right_coord = 0;
        left_begin += current_and_right_size;
    }

    return output;
}
template
std::vector<julie::la::Coordinate> Matrix_CPU<int>::argmax(lint dim) const;
template
std::vector<julie::la::Coordinate> Matrix_CPU<float>::argmax(lint dim) const;


template <typename DT>
DT Matrix_CPU<DT>::max() const
{
    DT max = std::numeric_limits<DT>::max() * (-1);
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        if (this->m_data[i] > max)
        {
            max = this->m_data[i];
        }
    }

    return max;
}
template
int Matrix_CPU<int>::max() const;
template
float Matrix_CPU<float>::max() const;


template <typename DT>
julie::la::Coordinate Matrix_CPU<DT>::argmin() const
{
    DT min = std::numeric_limits<DT>::max();
    lint argmin_index = 0;
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        if (this->m_data[i] < min)
        {
            min = this->m_data[i];
            argmin_index = i;
        }
    }

    return julie::la::Coordinate {argmin_index, this->m_shape};
}
template
julie::la::Coordinate Matrix_CPU<int>::argmin() const;
template
julie::la::Coordinate Matrix_CPU<float>::argmin() const;


template <typename DT>
std::vector<julie::la::Coordinate> Matrix_CPU<DT>::argmin(lint dim) const
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

    lint current_and_right_size = current_size * right_size;

    DT *left_begin = this->m_data;
    DT min = std::numeric_limits<DT>::max();

    std::vector<Coordinate> output;
    Coordinate left_coord {0, left_sub_sh};
    Coordinate right_coord {0, right_sub_sh};

    for (lint l_i = 0; l_i < left_size; ++l_i)
    {
        DT *current_begin = left_begin;
        
        for (lint r_i = 0; r_i < right_size; ++r_i)
        {
            DT *pos = current_begin;
            DT m = min;
            lint min_c = 0;

            for (lint c_i = 0; c_i < current_size; ++c_i)
            {
                // this->m_data[ right_size * (current_size * l_i + c_i) + r_i ];
                if (*pos < m)
                {
                    m = *pos;
                    min_c = c_i;
                }

                pos += right_size;
            }


            Coordinate current_coord {min_c, Shape{current_size}};
            output.push_back(left_coord + current_coord + right_coord);

            ++right_coord;
            ++current_begin;
        }

        ++left_coord;
        right_coord = 0;
        left_begin += current_and_right_size;
    }

    return output;
}
template
std::vector<julie::la::Coordinate> Matrix_CPU<int>::argmin(lint dim) const;
template
std::vector<julie::la::Coordinate> Matrix_CPU<float>::argmin(lint dim) const;


template <typename DT>
DT Matrix_CPU<DT>::min() const
{
    DT min = std::numeric_limits<DT>::max();
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        if (this->m_data[i] < min)
        {
            min = this->m_data[i];
        }
    }

    return min;
}
template
int Matrix_CPU<int>::min() const;
template
float Matrix_CPU<float>::min() const;


template <typename DT>
DT Matrix_CPU<DT>::sum() const
{
    lint size = this->m_shape.m_size;
    DT sum = 0;
    for (lint i = 0; i < size; ++i)
    {
        sum += this->m_data[i];
    }

    return sum;
}
template
int Matrix_CPU<int>::sum() const;
template
float Matrix_CPU<float>::sum() const;


template <typename DT>
DT Matrix_CPU<DT>::mean() const
{
    return this->sum() / this->m_shape.m_size;
}
template
int Matrix_CPU<int>::mean() const;
template
float Matrix_CPU<float>::mean() const;


template <typename DT>
DT Matrix_CPU<DT>::variance() const
{
    DT mean = this->mean();

    lint size = this->m_shape.m_size;
    DT sum = 0;
    DT sub;
    for (lint i = 0; i < size; ++i)
    {
        sub = this->m_data[i] - mean;
        sum += sub * sub;
    }

    return sum / size;
}
template
int Matrix_CPU<int>::variance() const;
template
float Matrix_CPU<float>::variance() const;


template <typename DT>
std::vector<Matrix_CPU<DT>> Matrix_CPU<DT>::get_collapsed(lint dim) const
{
    if (dim < 0 || dim >= this->m_shape.m_dim)
    {
        throw std::invalid_argument("Matrix_CPU::get_collapsed: dimension index out of range.");
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

    std::vector<Matrix_CPU> all_collapsed;
    DT *start = this->m_data;

    for (lint i = 0; i < this->m_shape.m_data[dim]; ++i)
    {
        Matrix_CPU collapsed{ sh_collapsed };
        DT *l_start = start;
        DT *clps_ele = collapsed.m_data;

        for (lint j = 0; j < size_dims_before; ++j)
        {
            DT *ele = l_start;
            for (lint k = 0; k < size_dims_after; ++k)
            {
                *clps_ele = *ele;
                ++clps_ele;
                ++ele;
            }
            l_start += size_dims_and_after;
        }

        all_collapsed.push_back(collapsed);
        start += size_dims_after;
    }

    return all_collapsed;
}
template
std::vector<Matrix_CPU<int>> Matrix_CPU<int>::get_collapsed(lint dim) const;
template
std::vector<Matrix_CPU<float>> Matrix_CPU<float>::get_collapsed(lint dim) const;


template <typename DT>
void Matrix_CPU<DT>::get_reduce_sum(Matrix_CPU<DT> &output, lint dim) const
{
    if (dim < 0 || dim >= this->m_shape.m_dim)
    {
        throw std::invalid_argument("Matrix_CPU::fuse: dimension index out of range.");
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
            DT *ele = l_start;
            for (lint k = 0; k < size_dims_after; ++k)
            {
                *clps_ele += *ele;
                ++clps_ele;
                ++ele;
            }
            l_start += size_dims_and_after;
        }

        start += size_dims_after;
    }
}
template
void Matrix_CPU<int>::get_reduce_sum(Matrix_CPU<int> &output, lint dim) const;
template
void Matrix_CPU<float>::get_reduce_sum(Matrix_CPU<float> &output, lint dim) const;


template <typename DT>
void Matrix_CPU<DT>::get_reduce_mean(Matrix_CPU<DT> &output, lint dim) const
{
    if (dim < 0 || dim >= this->m_shape.m_dim)
    {
        throw std::invalid_argument("Matrix_CPU::fuse: dimension index out of range.");
    }

    this->get_reduce_sum(output, dim);
    lint dim_size = this->m_shape.m_data[dim];
    output /= dim_size;
}
template
void Matrix_CPU<int>::get_reduce_mean(Matrix_CPU<int> &output, lint dim) const;
template
void Matrix_CPU<float>::get_reduce_mean(Matrix_CPU<float> &output, lint dim) const;


template <typename DT>
DT Matrix_CPU<DT>::euclidean_norm() const
{
    // Calculate the norm
    DT norm = 0;
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        norm += this->m_data[i] * this->m_data[i];
    }

    return sqrt(norm);
}
template
int Matrix_CPU<int>::euclidean_norm() const;
template
float Matrix_CPU<float>::euclidean_norm() const;


template <typename DT>
julie::la::Shape Matrix_CPU<DT>::shape() const
{
    return m_shape;
}
template
julie::la::Shape Matrix_CPU<int>::shape() const;
template
julie::la::Shape Matrix_CPU<float>::shape() const;


template <typename DT>
void Matrix_CPU<DT>::print(std::ostream & os, lint dim_index, const DT *start, int integral_len, int dot_len, int frag_len) const
{
    if (dim_index >= this->m_shape.m_dim)
    {
        return;
    }

    for (lint i = 0; i < dim_index; ++i)
    {
        os << "    ";
    }

    if (dim_index == this->m_shape.dim() - 1)
    {
        os << "[";
    }
    else
    {
        os << "[\n";
    }

    lint dim_size = this->m_shape[dim_index];
    lint jump_dist = 1;

    for (lint i = this->m_shape.dim() - 1; i > dim_index; --i)
    {
        jump_dist *= this->m_shape[i];
    }

    for (lint i = 0; i < dim_size; i++)
    {
        if (dim_index == this->m_shape.dim() - 1)
        {
            DT val = *(start + i);

            std::tuple<int, int, int> tuple = this->str_len_of_a_number(val);
            int i_len = std::get<0>(tuple);
            int d_len = std::get<1>(tuple);
            int f_len = std::get<2>(tuple);

            for (int i = 0; i < integral_len - i_len + 1; ++i)
            {
                os << " ";
            }

            os << val;

            for (int i = 0; i < dot_len + frag_len - f_len - d_len + 1; ++i)
            {
                os << " ";
            }
        }
        else
        {
            this->print(os, dim_index + 1, start + i * jump_dist, integral_len, dot_len, frag_len);
        }
    }

    if (dim_index == this->m_shape.dim() - 1)
    {
        os << "]\n";
    }
    else
    {
        for (lint i = 0; i < dim_index; ++i)
        {
            os << "    ";
        }
        os << "]\n";
    }
}
template
void Matrix_CPU<int>::print(std::ostream & os, lint dim_index, const int *start, int integral_len, int dot_len, int frag_len) const;
template
void Matrix_CPU<float>::print(std::ostream & os, lint dim_index, const float *start, int integral_len, int dot_len, int frag_len) const;


template <typename DT>
std::tuple<int, int, int> Matrix_CPU<DT>::str_len_of_a_number(DT val) const
{
    //std::tuple<int, int> size_pair
    std::ostringstream o_stream;
    o_stream << std::fixed;
    o_stream.precision(Matrix_CPU<DT>::PRINT_PRECISION);

    o_stream << val;

    std::string val_str = o_stream.str();

    int integral_len = 0;
    int dot_len = 0;
    int frag_len = 0;

    bool integral = true;

    for (char & ch : val_str)
    {
        if (ch == '.')
        {
            integral = false;
            dot_len = 1;
        }
        else
        {
            if (integral)
            {
                ++integral_len;
            }
            else
            {
                ++frag_len;
            }
        }
    }

    return std::make_tuple<int, int, int>(integral_len++, dot_len++, frag_len++);
}
template
std::tuple<int, int, int> Matrix_CPU<int>::str_len_of_a_number(int val) const;
template
std::tuple<int, int, int> Matrix_CPU<float>::str_len_of_a_number(float val) const;


template <typename DT>
inline std::string Matrix_CPU<DT>::to_string() const
{
    lint size = this->m_shape.size();

    int integral_len = 0;
    int dot_len = 0;
    int frag_len = 0;

    for (lint i = 0; i < size; ++i)
    {
        DT val = this->m_data[i];

        std::tuple<int, int, int> tuple = this->str_len_of_a_number(val);
        int i_len = std::get<0>(tuple);
        int d_len = std::get<1>(tuple);
        int f_len = std::get<2>(tuple);

        if (i_len > integral_len)
        {
            integral_len = i_len;
        }

        if (f_len > frag_len)
        {
            frag_len = f_len;
        }

        if (d_len > dot_len)
        {
            dot_len = d_len;
        }
    }

    std::ostringstream o_stream;
    o_stream << std::fixed;
    o_stream.precision(Matrix_CPU<DT>::PRINT_PRECISION);

    this->print(o_stream, 0, this->m_data, integral_len, dot_len, frag_len);

    return o_stream.str();
}
template
std::string Matrix_CPU<int>::to_string() const;
template
std::string Matrix_CPU<float>::to_string() const;

/* Overloading of output stream << operator
*/
template <typename DT>
std::ostream & operator << (std::ostream & os, const Matrix_CPU<DT> & m)
{
    return os << m.to_string();
}
template
std::ostream & operator << (std::ostream & os, const Matrix_CPU<int> & m);
template
std::ostream & operator << (std::ostream & os, const Matrix_CPU<float> & m);

} // namespace cpu
} // namespace la
} // namespace julie