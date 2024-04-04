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

#include <iterator>
#include <cassert>

namespace julie
{
namespace la
{
namespace cpu
{

template <typename DT> class Matrix_CPU;

template <typename DT = float>
class Matrix_CPU_Iterator : public std::iterator<std::bidirectional_iterator_tag, DT>
{
protected:
    lint m_mat_size;
    DT * m_mat_data;
    lint m_ele_index;

public:
    Matrix_CPU_Iterator(lint mat_size, DT * mat_data, lint ele_index);

    Matrix_CPU_Iterator(const Matrix_CPU_Iterator & other);

    Matrix_CPU_Iterator(Matrix_CPU_Iterator && other);

    Matrix_CPU_Iterator & operator = (const Matrix_CPU_Iterator & other);

    Matrix_CPU_Iterator & operator = (Matrix_CPU_Iterator && other);

public:

    Matrix_CPU_Iterator & operator ++ ();

    Matrix_CPU_Iterator operator ++ (int);

    Matrix_CPU_Iterator & operator -- ();

    Matrix_CPU_Iterator operator -- (int);

    DT & operator * () const;

    DT * operator -> () const;

    bool operator == (const Matrix_CPU_Iterator & right) const;

    bool operator != (const Matrix_CPU_Iterator & right) const;

};

template<typename DT>
inline Matrix_CPU_Iterator<DT>::Matrix_CPU_Iterator(lint mat_size, DT * mat_data, lint ele_index)
    : m_mat_size{ mat_size }, m_mat_data{ mat_data }, m_ele_index{ ele_index }
{}

template<typename DT>
inline Matrix_CPU_Iterator<DT>::Matrix_CPU_Iterator(const Matrix_CPU_Iterator & other)
    : m_mat_size{ other.m_mat_size }, m_mat_data{ other.m_mat_data }, m_ele_index{ other.m_ele_index }
{}

template<typename DT>
inline Matrix_CPU_Iterator<DT>::Matrix_CPU_Iterator(Matrix_CPU_Iterator && other)
    : m_mat_size{ other.m_mat_size }, m_mat_data{ other.m_mat_data }, m_ele_index{ other.m_ele_index }
{
    other.m_mat_size = 0;
    other.m_mat_data = nullptr;
    other.m_ele_index = 0;
}

template<typename DT>
inline Matrix_CPU_Iterator<DT> & Matrix_CPU_Iterator<DT>::operator=(const Matrix_CPU_Iterator & other)
{
    this->m_mat_size = other.mat_size;
    this->m_mat_data = other.m_mat_data;
    this->m_ele_index = other.m_ele_index;
}

template<typename DT>
inline Matrix_CPU_Iterator<DT> & Matrix_CPU_Iterator<DT>::operator=(Matrix_CPU_Iterator && other)
{
    this->m_mat_size = other.mat_size;
    this->m_mat_data = other.m_mat_data;
    this->m_ele_index = other.m_ele_index;

    other.m_mat_data = nullptr;
}

template<typename DT>
inline Matrix_CPU_Iterator<DT> & Matrix_CPU_Iterator<DT>::operator++()
{
    assert(this->m_ele_index < this->m_mat_size);

    ++this->m_ele_index;

    return *this;
}

template<typename DT>
inline Matrix_CPU_Iterator<DT> Matrix_CPU_Iterator<DT>::operator++(int)
{
    assert(this->m_ele_index < this->m_mat_size);

    Matrix_CPU_Iterator old{ *this };

    ++this->m_ele_index;

    return old;
}

template<typename DT>
inline Matrix_CPU_Iterator<DT> & Matrix_CPU_Iterator<DT>::operator--()
{
    assert(this->m_ele_index > 0);

    --this->m_ele_index;

    return *this;
}

template<typename DT>
inline Matrix_CPU_Iterator<DT> Matrix_CPU_Iterator<DT>::operator--(int)
{
    assert(this->m_ele_index > 0);

    Matrix_CPU_Iterator old{ *this };

    --this->m_ele_index;

    return old;
}

template<typename DT>
inline DT & Matrix_CPU_Iterator<DT>::operator*() const
{
    return this->m_mat_data[this->m_ele_index];
}

template<typename DT>
inline DT * Matrix_CPU_Iterator<DT>::operator->() const
{
    return this->m_mat_data + this->m_ele_index;
}

template<typename DT>
inline bool Matrix_CPU_Iterator<DT>::operator==(const Matrix_CPU_Iterator & right) const
{
    return (this->m_ele_index == right.m_ele_index) &&
        (this->m_mat_data == right.m_mat_data) &&
        (this->m_mat_size == right.m_mat_size);
}

template<typename DT>
inline bool Matrix_CPU_Iterator<DT>::operator!=(const Matrix_CPU_Iterator & right) const
{
    return !(*this == right);
}

} // namespace cpu
} // namespace la
} // namespace julie