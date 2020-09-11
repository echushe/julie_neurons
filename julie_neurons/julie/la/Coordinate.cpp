#include "Coordinate.hpp"
#include "Exceptions.hpp"
#include <functional>
#include <cstring>


julie::la::Coordinate::Coordinate(std::initializer_list<lint> list, const Shape & shape)
    : m_dim(list.end() - list.begin()), m_shape{ shape }
{
    if (shape.m_dim != this->m_dim)
    {
        throw std::invalid_argument(std::string("Shape and coordinate incompatible"));
    }

    if (this->m_dim < 1)
    {
        this->m_dim = 0;
        this->m_data = nullptr;
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
}


julie::la::Coordinate::Coordinate(const Shape & shape)
    : m_dim{ shape.m_dim }, m_shape{ shape }
{
    this->m_data = new lint[this->m_dim];
    for (lint i = 0; i < this->m_dim; ++i)
    {
        this->m_data[i] = 0;
    }
}


julie::la::Coordinate::Coordinate(lint index, const Shape & shape)
    : m_dim{ shape.m_dim }, m_shape{ shape }
{
    this->m_data = new lint[this->m_dim];
    
    index %= this->m_shape.m_size;

    for (lint i = this->m_dim - 1; i >= 0; --i)
    {
        this->m_data[i] = index % this->m_shape.m_data[i];
        index /= this->m_shape.m_data[i];
    }
}


julie::la::Coordinate::Coordinate()
    : m_dim{ 0 }, m_shape{}, m_data {nullptr}
{}


julie::la::Coordinate::Coordinate(const Coordinate & other)
    : m_dim{ other.m_dim }, m_shape{ other.m_shape }
{
    this->m_data = new lint[this->m_dim];

    std::memcpy(this->m_data, other.m_data, this->m_dim * sizeof(lint));
}


julie::la::Coordinate::Coordinate(Coordinate && other)
    : m_dim{ other.m_dim }, m_data{ other.m_data }, m_shape{ std::move(other.m_shape) }
{
    other.m_dim = 0;
    other.m_data = nullptr;
}

julie::la::Coordinate::~Coordinate()
{
    delete[]this->m_data;
}

julie::la::Coordinate & julie::la::Coordinate::operator = (const Coordinate & other)
{
    delete[]m_data;
    this->m_dim = other.m_dim;
    this->m_shape = other.m_shape;

    this->m_data = new lint[m_dim];
    std::memcpy(this->m_data, other.m_data, m_dim * sizeof(lint));

    return *this;
}

julie::la::Coordinate & julie::la::Coordinate::operator = (Coordinate && other)
{
    delete[]m_data;
    this->m_dim = other.m_dim;
    this->m_shape = std::move(other.m_shape);
    this->m_data = other.m_data;

    other.m_dim = 0;
    other.m_data = nullptr;

    return *this;
}

julie::la::Coordinate & julie::la::Coordinate::operator = (lint index)
{
    index %= this->m_shape.m_size;

    for (lint i = this->m_dim - 1; i >= 0; --i)
    {
        this->m_data[i] = index % this->m_shape.m_data[i];
        index /= this->m_shape.m_data[i];
    }

    return *this;
}

lint julie::la::Coordinate::operator [] (lint index) const
{
    return this->m_data[index];
}

lint & julie::la::Coordinate::operator [] (lint index)
{
    return this->m_data[index];
}

julie::la::Coordinate & julie::la::Coordinate::operator++()
{
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

    return *this;
}

julie::la::Coordinate julie::la::Coordinate::operator++(int)
{
    Coordinate copy{ *this };
    ++(*this);
    return copy;
}

julie::la::Coordinate & julie::la::Coordinate::operator--()
{
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

    return *this;
}

julie::la::Coordinate julie::la::Coordinate::operator--(int)
{
    Coordinate copy{ *this };
    --(*this);
    return copy;
}

julie::la::Coordinate & julie::la::Coordinate::transposed_plus()
{
    lint plus_pos = 0;
    while (plus_pos < this->m_dim)
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
            ++plus_pos;
        }
    }

    return *this;
}

julie::la::Coordinate julie::la::Coordinate::sub_coordinate(lint dim_first, lint dim_last) const
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

    return sub_co;
}

lint julie::la::Coordinate::index() const
{
    lint index = 0;
    for (lint i = 0; i < this->m_dim; ++i)
    {
        index *= this->m_shape.m_data[i];
        index += this->m_data[i];
    }

    return index;
}

lint julie::la::Coordinate::dim() const
{
    return this->m_dim;
}

julie::la::Coordinate julie::la::Coordinate::get_reversed() const
{
    Coordinate reversed{*this};
    reversed.m_shape = this->m_shape.get_reversed();

    for (lint i = 0; i < reversed.m_dim / 2; ++i)
    {
        lint k = reversed.m_data[i];
        reversed.m_data[i] = reversed.m_data[this->m_dim - 1 - i];
        reversed.m_data[this->m_dim - 1 - i] = k;
    }

    return reversed;
}

julie::la::Shape julie::la::Coordinate::get_shape() const
{
    return this->m_shape;
}

julie::la::Coordinate julie::la::operator + (const Coordinate &left, const Coordinate &right)
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

    return merged;
}

bool julie::la::operator == (const Coordinate & left, const Coordinate & right)
{
    if (left.m_shape != right.m_shape)
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

bool julie::la::operator != (const Coordinate & left, const Coordinate & right)
{
    return !(left == right);
}

bool julie::la::operator < (const Coordinate &left, const Coordinate &right)
{
    if (left.m_shape != right.m_shape)
    {
        return false;
    }

    for (lint i = 0; i < left.m_dim; ++i)
    {
        if (left.m_data[i] < right.m_data[i])
        {
            return true;
        }
        else if (left.m_data[i] > right.m_data[i])
        {
            return false;
        }
    }

    return false;
}

bool julie::la::operator <= (const Coordinate &left, const Coordinate &right)
{
    if (left < right || left == right)
    {
        return true;
    }

    return false;
}

bool julie::la::operator > (const Coordinate &left, const Coordinate &right)
{
    if (left.m_shape != right.m_shape)
    {
        return false;
    }

    for (lint i = 0; i < left.m_dim; ++i)
    {
        if (left.m_data[i] > right.m_data[i])
        {
            return true;
        }
        else if (left.m_data[i] < right.m_data[i])
        {
            return false;
        }
    }

    return false;
}

bool julie::la::operator >= (const Coordinate &left, const Coordinate &right)
{
    if (left > right || left == right)
    {
        return true;
    }

    return false;
}

std::ostream & julie::la::operator<<(std::ostream & os, const Coordinate & co)
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

julie::la::Coordinate julie::la::reverse(const Coordinate & sh)
{
    return sh.get_reversed();
}
