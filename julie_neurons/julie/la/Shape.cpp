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

#include "Shape.hpp"
#include "utilities.hpp"
#include <functional>
#include <cstring>
#include <algorithm>

namespace julie
{
namespace la
{

Shape::Shape()
    :
    m_dim {0},
    m_size {0},
    m_data {nullptr}
{}

Shape::Shape(std::initializer_list<lint> list)
    : 
    m_dim {list.end() - list.begin()},
    m_size {0},
    m_data {nullptr}
{
    if (this->m_dim < 1)
    {
        this->m_dim = 0;
    }
    else
    {
        this->m_data = new lint[this->m_dim];
        lint *p = this->m_data;
        for (auto itr = list.begin(); itr != list.end(); ++itr, ++p)
        {
            if (*itr <= 0)
            {
                throw std::invalid_argument(invalid_shape_val + std::string{__FUNCTION__});
            }
            *p = *itr;
        }

        this->m_size = 1;
        for (lint i = 0; i < this->m_dim; ++i)
        {
            this->m_size *= this->m_data[i];
        }
    }
}

Shape::Shape(std::vector<lint> &list)
    :
    m_dim {list.end() - list.begin()},
    m_size {0},
    m_data {nullptr}
{
    if (this->m_dim < 1)
    {
        this->m_dim = 0;
    }
    else
    {
        this->m_data = new lint[this->m_dim];
        lint *p = this->m_data;
        for (auto itr = list.begin(); itr != list.end(); ++itr, ++p)
        {
            if (*itr <= 0)
            {
                throw std::invalid_argument(invalid_shape_val + std::string{__FUNCTION__});
            }
            *p = *itr;
        }

        this->m_size = 1;
        for (lint i = 0; i < this->m_dim; ++i)
        {
            this->m_size *= this->m_data[i];
        }
    }
}

Shape::Shape(const Shape & other)
    :
    m_dim {other.m_dim},
    m_size {other.m_size},
    m_data {nullptr}
{
    if (this->m_dim < 1)
    {
        return;
    }

    this->m_data = new lint[this->m_dim];

    std::memcpy(this->m_data, other.m_data, this->m_dim * sizeof(lint));
}

Shape::Shape(Shape && other)
    :
    m_dim {other.m_dim},
    m_size {other.m_size},
    m_data {other.m_data}
{
    other.m_dim = 0;
    other.m_size = 0;
    other.m_data = nullptr;
}

Shape::~Shape()
{
    delete[]this->m_data;
}

Shape & Shape::operator = (const Shape &other)
{
    delete[]m_data;
    this->m_dim = other.m_dim;
    this->m_size = other.m_size;
    this->m_data = nullptr;

    if (other.m_dim < 1)
    {
        return *this;
    }

    this->m_data = new lint[m_dim];
    std::memcpy(this->m_data, other.m_data, m_dim * sizeof(lint));

    return *this;
}

Shape & Shape::operator = (Shape && other)
{
    delete[]m_data;
    this->m_dim = other.m_dim;
    this->m_size = other.m_size;
    this->m_data = other.m_data;

    other.m_dim = 0;
    other.m_size = 0;
    other.m_data = nullptr;

    return *this;
}

lint Shape::operator [] (lint index) const
{
    return this->m_data[index];
}

lint Shape::size() const
{
    return this->m_size;
}

lint Shape::dim() const
{
    return this->m_dim;
}

Shape & Shape::left_extend()
{
    if (this->m_dim < 1)
    {
        throw std::string("NULL shape cannot be left extended in ") + std::string(__FUNCTION__);
    }

    ++this->m_dim;
    lint *new_data = new lint[this->m_dim];
    *new_data = 1;

    for (lint i = 1; i < this->m_dim; ++i)
    {
        new_data[i] = this->m_data[i - 1];
    }
    delete[]this->m_data;
    this->m_data = new_data;

    return *this;
}

Shape & Shape::right_extend()
{
    if (this->m_dim < 1)
    {
        throw std::string("NULL shape cannot be right extended in ") + std::string(__FUNCTION__);
    }

    ++this->m_dim;
    lint *new_data = new lint[this->m_dim];

    for (lint i = 0; i < this->m_dim - 1; ++i)
    {
        new_data[i] = this->m_data[i];
    }
    new_data[this->m_dim - 1] = 1;
    delete[]this->m_data;
    this->m_data = new_data;

    return *this;
}

Shape Shape::get_reversed() const
{
    Shape reversed(*this);

    for (lint i = 0; i < reversed.m_dim / 2; ++i)
    {
        lint k = reversed.m_data[i];
        reversed.m_data[i] = reversed.m_data[reversed.m_dim - 1 - i];
        reversed.m_data[reversed.m_dim - 1 - i] = k;
    }

    return reversed;
}

Shape Shape::sub_shape(lint dim_first, lint dim_last) const
{
    Shape sub_shape;

    if (dim_first > dim_last)
    {
        return sub_shape;
    }

    if (dim_first < 0 || dim_last > this->m_dim - 1)
    {
        throw std::invalid_argument(
            std::string("Shape::sub_shape: index of dimension out of range in ")
            + std::string{__FUNCTION__}
        );
    }

    sub_shape.m_dim = dim_last - dim_first + 1;
    sub_shape.m_data = new lint[sub_shape.m_dim];
    sub_shape.m_size = 1;

    for (lint i = dim_first; i <= dim_last; ++i)
    {
        sub_shape.m_data[i - dim_first] = this->m_data[i];
    }

    for (lint i = 0; i < sub_shape.m_dim; ++i)
    {
        sub_shape.m_size *= sub_shape.m_data[i];
    }

    return sub_shape;
}

// Overloading of a == b
bool operator == (const Shape &left, const Shape &right)
{
    if (left.m_dim != right.m_dim)
    {
        return false;
    }

    if (left.m_size != right.m_size)
    {
        return false;
    }

    for (lint i = 0; i < left.m_dim; ++i)
    {
        if (left.m_data[i] != right.m_data[i])
        {
            return false;
        }
    }

    return true;
}

// Overloading of a != b
bool operator != (const Shape &left, const Shape &right)
{
    return !(left == right);
}

Shape operator + (const Shape & left, const Shape & right)
{
    Shape merged;

    if (left.m_dim < 1 && right.m_dim < 1 )
    {
        return merged;
    }

    merged.m_dim = left.m_dim + right.m_dim;
    merged.m_size = std::max<lint>(left.m_size, 1) * std::max<lint>(right.m_size, 1);
    merged.m_data = new lint[merged.m_dim];

    for (lint i = 0; i < left.m_dim; ++i)
    {
        merged.m_data[i] = left.m_data[i];
    }

    for (lint i = left.m_dim; i < merged.m_dim; ++i)
    {
        merged.m_data[i] = right.m_data[i - left.m_dim];
    }

    return merged;
}

Shape reverse(const Shape & sh)
{
    return sh.get_reversed();
}

std::ostream & operator<<(std::ostream & os, const Shape & sh)
{
    os << '(';
    for (lint i = 0; i < sh.m_dim; ++i)
    {
        os << sh.m_data[i];
        if (i < sh.m_dim - 1)
        {
            os << ", ";
        }
    }
    os << ')';

    return os;
}

bool Shape::CanDoMatMul(const Shape & left_sh, const Shape & right_sh)
{
    if (left_sh.m_dim < 2 || right_sh.m_dim < 2)
    {
        throw std::invalid_argument(std::string(
            "Matrix_CPU multiplication does not allow matrices of less than 2 dimensions.\n Please extend dimensions of matrices first."));
    }

    // Check shape compatibilities between these 2 matrices
    Shape left_cols_sh;
    Shape right_rows_sh;

    bool can_multiply = false;

    for (lint l = left_sh.dim() - 1, r = 0; l >= 1 && r < right_sh.dim() - 1;)
    {
        left_cols_sh = left_sh.sub_shape(l, left_sh.dim() - 1);
        right_rows_sh = right_sh.sub_shape(0, r);

        if (left_cols_sh.size() == right_rows_sh.size())
        {
            can_multiply = true;
            break;
        }
        else if (left_cols_sh.size() > right_rows_sh.size())
        {
            ++r;
        }
        else
        {
            --l;
        }
    }

    return can_multiply;
}


} // namespace la
} // namespace julie
