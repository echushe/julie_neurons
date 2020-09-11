#include "Shape.hpp"
#include "Exceptions.hpp"
#include <functional>
#include <cstring>
#include <algorithm>

julie::la::Shape::Shape()
    : m_dim{ 0 }, m_size{ 0 }, m_data{ nullptr }
{}

julie::la::Shape::Shape(std::initializer_list<lint> list)
    : m_dim{ list.end() - list.begin() }, m_size{ 1 }
{
    if (this->m_dim < 1)
    {
        this->m_dim = 0;
        this->m_size = 0;
        this->m_data = nullptr;
    }
    else
    {
        this->m_data = new lint[this->m_dim];
        lint *p = this->m_data;
        for (auto itr = list.begin(); itr != list.end(); ++itr, ++p)
        {
            if (*itr <= 0)
            {
                throw std::invalid_argument(invalid_shape_val);
            }
            *p = *itr;
        }

        for (lint i = 0; i < this->m_dim; ++i)
        {
            this->m_size *= this->m_data[i];
        }
    }
}

julie::la::Shape::Shape(std::vector<lint>& list)
    : m_dim{ list.end() - list.begin() }, m_size{ 1 }
{
    if (this->m_dim < 1)
    {
        this->m_dim = 0;
        this->m_size = 0;
        this->m_data = nullptr;
    }
    else
    {
        this->m_data = new lint[this->m_dim];
        lint *p = this->m_data;
        for (auto itr = list.begin(); itr != list.end(); ++itr, ++p)
        {
            if (*itr <= 0)
            {
                throw std::invalid_argument(invalid_shape_val);
            }
            *p = *itr;
        }

        for (lint i = 0; i < this->m_dim; ++i)
        {
            this->m_size *= this->m_data[i];
        }
    }
}

julie::la::Shape::Shape(const Shape & other)
    : m_dim{ other.m_dim }, m_size(other.m_size)
{
    this->m_data = new lint[this->m_dim];

    std::memcpy(this->m_data, other.m_data, this->m_dim * sizeof(lint));
}

julie::la::Shape::Shape(Shape && other)
    : m_dim{ other.m_dim }, m_size(other.m_size), m_data{ other.m_data }
{
    other.m_dim = 0;
    other.m_size = 0;
    other.m_data = nullptr;
}

julie::la::Shape::~Shape()
{
    delete[]this->m_data;
}

julie::la::Shape & julie::la::Shape::operator = (const Shape & other)
{
    delete[]m_data;
    this->m_dim = other.m_dim;
    this->m_size = other.m_size;

    this->m_data = new lint[m_dim];
    std::memcpy(this->m_data, other.m_data, m_dim * sizeof(lint));

    return *this;
}

julie::la::Shape & julie::la::Shape::operator = (Shape && other)
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

lint julie::la::Shape::operator [] (lint index) const
{
    return this->m_data[index];
}

/*
lint & julie::la::Shape::operator [] (lint index)
{
    return this->m_data[index];
}
*/

lint julie::la::Shape::size() const
{
    return this->m_size;
}

lint julie::la::Shape::dim() const
{
    return this->m_dim;
}

julie::la::Shape & julie::la::Shape::left_extend()
{
    if (0 == this->m_dim)
    {
        throw std::bad_function_call();
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

julie::la::Shape & julie::la::Shape::right_extend()
{
    if (0 == this->m_dim)
    {
        throw std::bad_function_call();
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

julie::la::Shape julie::la::Shape::get_reversed() const
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

julie::la::Shape julie::la::Shape::sub_shape(lint dim_first, lint dim_last) const
{
    Shape sub_shape;

    if (dim_first > dim_last)
    {
        return sub_shape;
    }

    if (dim_first < 0 || dim_last > this->m_dim - 1)
    {
        throw std::invalid_argument(
            std::string("Shape::sub_shape: index of dimension out of range")
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
bool julie::la::operator == (const Shape &left, const Shape &right)
{
    if (left.m_dim != right.m_dim)
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
bool julie::la::operator != (const Shape &left, const Shape &right)
{
    return !(left == right);
}

julie::la::Shape julie::la::operator + (const Shape & left, const Shape & right)
{
    Shape merged;
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

julie::la::Shape julie::la::reverse(const Shape & sh)
{
    return sh.get_reversed();
}

std::ostream & julie::la::operator<<(std::ostream & os, const Shape & sh)
{
    os << '[';
    for (lint i = 0; i < sh.m_dim; ++i)
    {
        os << sh.m_data[i];
        if (i < sh.m_dim - 1)
        {
            os << ", ";
        }
    }
    os << ']';

    return os;
}

bool julie::la::Shape::CanDoMatMul(const Shape & left_sh, const Shape & right_sh)
{
    if (left_sh.m_dim < 2 || right_sh.m_dim < 2)
    {
        throw std::invalid_argument(std::string(
            "DMatrix multiplication does not allow matrices of less than 2 dimensions.\n Please extend dimensions of matrices first."));
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
