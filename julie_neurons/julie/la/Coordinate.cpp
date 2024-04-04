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

#include "Coordinate.hpp"
#include "utilities.hpp"
#include <functional>
#include <cstring>


namespace julie
{
namespace la
{

Coordinate::Coordinate(std::initializer_list<lint> list, const Shape & shape)
    :
    m_dim {list.end() - list.begin()},
    m_data {nullptr},
    m_shape {shape},
    m_index {0}
{
    if (shape.m_dim != this->m_dim)
    {
        throw std::invalid_argument(std::string("Shape and coordinate incompatible"));
    }

    if (this->m_dim < 1)
    {
        this->m_dim = 0;
        return;
    }

    this->m_data = new lint[this->m_dim];

    lint *p = this->m_data;
    for (auto itr = list.begin(); itr != list.end(); ++itr, ++p)
    {
        if (*itr < 0)
        {
            throw std::invalid_argument(invalid_coord_val);
        }
        *p = *itr;
    }

    for (lint i = 0; i < this->m_dim; ++i)
    {
        if (this->m_data[i] >= shape.m_data[i])
        {
            delete[]this->m_data;
            throw std::invalid_argument(std::string("Shape and coordinate incompatible"));
        }
    }

    lint index = 0;
    for (lint i = 0; i < this->m_dim; ++i)
    {
        index *= this->m_shape.m_data[i];
        index += this->m_data[i];
    }

    this->m_index = index;
}


Coordinate::Coordinate(const Shape & shape)
    :
    m_dim {shape.m_dim},
    m_data {nullptr},
    m_shape {shape},
    m_index {0}
{
    if (this->m_dim < 1)
    {
        return;
    }

    this->m_data = new lint[this->m_dim];
    for (lint i = 0; i < this->m_dim; ++i)
    {
        this->m_data[i] = 0;
    }
}


Coordinate::Coordinate(lint index, const Shape & shape)
    :
    m_dim {shape.m_dim},
    m_data {nullptr},
    m_shape {shape},
    m_index {0}
{
    if (this->m_dim < 1)
    {
        return;
    }

    this->m_data = new lint[this->m_dim];

    index %= this->m_shape.m_size;
    if (index < 0)
    {
        index += this->m_shape.m_size;
    }

    this->m_index = index;

    for (lint i = this->m_dim - 1; i >= 0; --i)
    {
        this->m_data[i] = index % this->m_shape.m_data[i];
        index /= this->m_shape.m_data[i];
    }
}


Coordinate::Coordinate()
    :
    m_dim {0},
    m_data {nullptr},
    m_shape {},
    m_index {0}
{}


Coordinate::Coordinate(const Coordinate & other)
    :
    m_dim {other.m_dim},
    m_data {nullptr},
    m_shape {other.m_shape},
    m_index {other.m_index}
{
    if (other.m_dim < 1)
    {
        return;
    }

    this->m_data = new lint[this->m_dim];

    std::memcpy(this->m_data, other.m_data, this->m_dim * sizeof(lint));
}


Coordinate::Coordinate(Coordinate && other)
    :
    m_dim {other.m_dim},
    m_data {other.m_data},
    m_shape {std::move(other.m_shape)},
    m_index {other.m_index}
{
    other.m_dim = 0;
    other.m_data = nullptr;
    other.m_index = 0;
}

Coordinate::~Coordinate()
{
    delete[]this->m_data;
}

Coordinate & Coordinate::operator = (const Coordinate & other)
{
    delete[]m_data;
    this->m_dim = other.m_dim;
    this->m_data = nullptr;
    this->m_shape = other.m_shape;
    this->m_index = other.m_index;

    if (this->m_dim < 1)
    {
        return *this;
    }

    this->m_data = new lint[m_dim];
    std::memcpy(this->m_data, other.m_data, m_dim * sizeof(lint));

    return *this;
}

Coordinate & Coordinate::operator = (Coordinate && other)
{
    delete[]m_data;
    this->m_dim = other.m_dim;
    this->m_data = other.m_data;
    this->m_shape = std::move(other.m_shape);
    this->m_index = other.m_index;

    other.m_dim = 0;
    other.m_data = nullptr;

    return *this;
}

Coordinate & Coordinate::operator = (lint index)
{
    if (this->m_dim < 1)
    {
        return *this;
    }

    index %= this->m_shape.m_size;
    if (index < 0)
    {
        index += this->m_shape.m_size;
    }

    this->m_index = index;

    for (lint i = this->m_dim - 1; i >= 0; --i)
    {
        this->m_data[i] = index % this->m_shape.m_data[i];
        index /= this->m_shape.m_data[i];
    }

    return *this;
}

lint Coordinate::operator [] (lint index) const
{
    return this->m_data[index];
}

lint & Coordinate::operator [] (lint index)
{
    return this->m_data[index];
}

Coordinate & Coordinate::operator++()
{
    if (this->m_dim < 1)
    {
        return *this;
    }

    lint plus_pos = this->m_dim - 1;
    while (plus_pos >= 0)
    {
        lint increased = this->m_data[plus_pos] + 1;
        if (increased < this->m_shape.m_data[plus_pos])
        {
            this->m_data[plus_pos] = increased;
            break;
        }
        else
        {
            this->m_data[plus_pos] = 0;
            --plus_pos;
        }
    }

    ++this->m_index;
    if (this->m_index >= this->m_shape.m_size)
    {
        this->m_index -= this->m_shape.m_size;
    }

    return *this;
}

Coordinate Coordinate::operator++(int)
{
    Coordinate copy{ *this };
    ++(*this);
    return copy;
}

Coordinate & Coordinate::operator--()
{
    if (this->m_dim < 1)
    {
        return *this;
    }

    lint plus_pos = this->m_dim - 1;
    while (plus_pos >= 0)
    {
        lint decreased = this->m_data[plus_pos] - 1;
        if (decreased >= 0)
        {
            this->m_data[plus_pos] = decreased;
            break;
        }
        else
        {
            this->m_data[plus_pos] = this->m_shape.m_data[plus_pos] - 1;
            --plus_pos;
        }
    }
    
    --this->m_index;
    if (this->m_index < 0)
    {
        this->m_index += this->m_shape.m_size;
    }

    return *this;
}

Coordinate Coordinate::operator--(int)
{
    Coordinate copy{ *this };
    --(*this);
    return copy;
}

Coordinate Coordinate::sub_coordinate(lint dim_first, lint dim_last) const
{
    Coordinate sub_co;

    if (dim_first > dim_last)
    {
        return sub_co;
    }

    if (dim_first < 0 || dim_last > this->m_dim - 1)
    {
        throw std::invalid_argument(
            std::string("index of dimension out of range")
        );
    }

    sub_co.m_dim = dim_last - dim_first + 1;
    sub_co.m_data = new lint[sub_co.m_dim];

    for (lint i = dim_first; i <= dim_last; ++i)
    {
        sub_co.m_data[i - dim_first] = this->m_data[i];
    }

    sub_co.m_shape = this->m_shape.sub_shape(dim_first, dim_last);

    lint index = 0;
    for (lint i = 0; i < sub_co.m_dim; ++i)
    {
        index *= sub_co.m_shape.m_data[i];
        index += sub_co.m_data[i];
    }
    sub_co.m_index = index;

    return sub_co;
}

lint Coordinate::index() const
{
    return this->m_index;
}

lint Coordinate::dim() const
{
    return this->m_dim;
}

Coordinate Coordinate::get_reversed() const
{
    Coordinate reversed{*this};
    reversed.m_shape = this->m_shape.get_reversed();

    for (lint i = 0; i < reversed.m_dim / 2; ++i)
    {
        lint k = reversed.m_data[i];
        reversed.m_data[i] = reversed.m_data[this->m_dim - 1 - i];
        reversed.m_data[this->m_dim - 1 - i] = k;
    }

    lint index = 0;
    for (lint i = 0; i < reversed.m_dim; ++i)
    {
        index *= reversed.m_shape.m_data[i];
        index += reversed.m_data[i];
    }
    reversed.m_index = index;

    return reversed;
}

Shape Coordinate::get_shape() const
{
    return this->m_shape;
}

Coordinate operator + (const Coordinate &left, const Coordinate &right)
{
    Coordinate merged;
    merged.m_dim = left.m_dim + right.m_dim;
    merged.m_data = new lint[merged.m_dim];
    merged.m_shape = left.m_shape + right.m_shape;  

    for (lint i = 0; i < left.m_dim; ++i)
    {
        merged.m_data[i] = left.m_data[i];
    }

    for (lint i = left.m_dim; i < merged.m_dim; ++i)
    {
        merged.m_data[i] = right.m_data[i - left.m_dim];
    }

    lint index = 0;
    for (lint i = 0; i < merged.m_dim; ++i)
    {
        index *= merged.m_shape[i];
        index += merged.m_data[i];
    }
    merged.m_index = index;

    return merged;
}

bool operator == (const Coordinate & left, const Coordinate & right)
{
    if (left.m_dim != right.m_dim)
    {
        return false;
    }

    if (left.m_shape != right.m_shape)
    {
        return false;
    }

    if (left.m_index != right.m_index)
    {
        return false;
    }

    return true;
}

bool operator != (const Coordinate & left, const Coordinate & right)
{
    return !(left == right);
}

bool operator < (const Coordinate &left, const Coordinate &right)
{
    if (left.m_dim != right.m_dim)
    {
        return false;
    }

    if (left.m_shape != right.m_shape)
    {
        return false;
    }

    return left.m_index < right.m_index;
}

bool operator <= (const Coordinate &left, const Coordinate &right)
{
    if (left < right || left == right)
    {
        return true;
    }

    return false;
}

bool operator > (const Coordinate &left, const Coordinate &right)
{
    if (left.m_dim != right.m_dim)
    {
        return false;
    }

    if (left.m_shape != right.m_shape)
    {
        return false;
    }

    return left.m_index > right.m_index;
}

bool operator >= (const Coordinate &left, const Coordinate &right)
{
    if (left > right || left == right)
    {
        return true;
    }

    return false;
}

std::ostream & operator<<(std::ostream & os, const Coordinate & co)
{
    os << '[';
    for (lint i = 0; i < co.m_dim; ++i)
    {
        os << co.m_data[i];
        if (i < co.m_dim - 1)
        {
            os << ", ";
        }
    }
    os << ']';

    return os;
}

Coordinate reverse(const Coordinate & sh)
{
    return sh.get_reversed();
}


} // namespace la
} // namespace julie