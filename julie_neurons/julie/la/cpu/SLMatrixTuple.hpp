
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
#include "Shape.hpp"
#include "utilities.hpp"

#include <ostream>

namespace julie
{
namespace la
{
namespace cpu
{

template <typename DT = float>
struct SLMatrixTuple
{
    lint m_row;
    lint m_col;
    DT m_val;
    SLMatrixTuple * m_right;
    SLMatrixTuple * m_down;

    static lint REF_COUNT;

    SLMatrixTuple(lint row, lint col, DT value, const SLMatrixTuple *right, const SLMatrixTuple *down);

    SLMatrixTuple(lint row, lint col, DT value);

    SLMatrixTuple(lint row, lint col);

    SLMatrixTuple();

    SLMatrixTuple(const SLMatrixTuple & other);

    SLMatrixTuple(SLMatrixTuple && other);

    ~SLMatrixTuple()
    {
        // --REF_COUNT;
    }

    SLMatrixTuple & operator = (const SLMatrixTuple & other);

    SLMatrixTuple & operator = (SLMatrixTuple && other);

    SLMatrixTuple & operator += (DT scalar);

    SLMatrixTuple & operator -= (DT scalar);

    SLMatrixTuple & operator *= (DT scalar);

    SLMatrixTuple & operator /= (DT scalar);
};

template <typename DT>
bool operator < (const SLMatrixTuple<DT> & left, const SLMatrixTuple<DT> & right);

template <typename DT>
bool operator > (const SLMatrixTuple<DT> & left, const SLMatrixTuple<DT> & right);

template <typename DT>
bool operator <= (const SLMatrixTuple<DT> & left, const SLMatrixTuple<DT> & right);

template <typename DT>
bool operator >= (const SLMatrixTuple<DT> & left, const SLMatrixTuple<DT> & right);

template <typename DT>
bool operator == (const SLMatrixTuple<DT> & left, const SLMatrixTuple<DT> & right);


template <typename DT>
SLMatrixTuple<DT> operator + (const SLMatrixTuple<DT> & tp, DT scalar);

template <typename DT>
SLMatrixTuple<DT> operator - (const SLMatrixTuple<DT> & tp, DT scalar);

template <typename DT>
SLMatrixTuple<DT> operator * (const SLMatrixTuple<DT> & tp, DT scalar);

template <typename DT>
SLMatrixTuple<DT> operator / (const SLMatrixTuple<DT> & tp, DT scalar);

template <typename DT>
SLMatrixTuple<DT> operator + (DT scalar, const SLMatrixTuple<DT> & tp);

template <typename DT>
SLMatrixTuple<DT> operator - (DT scalar, const SLMatrixTuple<DT> & tp);

template <typename DT>
SLMatrixTuple<DT> operator * (DT scalar, const SLMatrixTuple<DT> & tp);

template <typename DT>
SLMatrixTuple<DT> operator / (DT scalar, const SLMatrixTuple<DT> & tp);

template <typename DT>
std::ostream & operator << (std::ostream & os, const SLMatrixTuple<DT> & mt);

} // namespace cpu
} // namespace la
} // namespace julie

template <typename DT>
lint julie::la::cpu::SLMatrixTuple<DT>::REF_COUNT = 0;

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT>::SLMatrixTuple(
    lint row, lint col, DT value,
    const SLMatrixTuple *right, const SLMatrixTuple *down)
    : m_row {row}, m_col {col}, m_val {value},
      m_right {right}, m_down {down}
{
    ++REF_COUNT;
}

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT>::SLMatrixTuple(lint row, lint col, DT value)
    : m_row {row}, m_col {col}, m_val {value},
      m_right {nullptr}, m_down {nullptr}
{
    //++REF_COUNT;
}

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT>::SLMatrixTuple(lint row, lint col)
    : m_row {row}, m_col {col}, m_val {0},
      m_right {nullptr}, m_down {nullptr}
{
    //++REF_COUNT;
}

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT>::SLMatrixTuple()
    : m_row {-1}, m_col {-1}, m_val {0},
      m_right {nullptr}, m_down {nullptr}
{
    //++REF_COUNT;
}

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT>::SLMatrixTuple(const SLMatrixTuple & other)
    : m_row {other.m_row}, m_col {other.m_col}, m_val {other.m_val},
      m_right {other.m_right}, m_down {other.m_down}
{
    //++REF_COUNT;
}

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT>::SLMatrixTuple(SLMatrixTuple && other)
    : m_row {other.m_row}, m_col {other.m_col}, m_val {other.m_val},
      m_right {other.m_right}, m_down {other.m_down}
{
    other.m_row = 0;
    other.m_col = 0;
    other.m_val = 0;
    other.m_right = nullptr;
    other.m_down = nullptr;

    //++REF_COUNT;
}

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT> & julie::la::cpu::SLMatrixTuple<DT>::operator = (const SLMatrixTuple & other)
{
    this->m_row = other.m_row;
    this->m_col = other.m_col;
    this->m_val = other.m_val;
    this->m_right = other.m_right;
    this->m_down = other.m_down;

    return *this;
}

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT> & julie::la::cpu::SLMatrixTuple<DT>::operator = (SLMatrixTuple && other)
{
    this->m_row = other.m_row;
    this->m_col = other.m_col;
    this->m_val = other.m_val;
    this->m_right = other.m_right;
    this->m_down = other.m_down;

    other.m_row = 0;
    other.m_col = 0;
    other.m_val = 0;
    other.m_right = nullptr;
    other.m_down = nullptr;

    return *this;
}

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT> & julie::la::cpu::SLMatrixTuple<DT>::operator += (DT scalar)
{
    this->m_val += scalar;

    return *this;
}

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT> & julie::la::cpu::SLMatrixTuple<DT>::operator -= (DT scalar)
{
    this->m_val -= scalar;

    return *this;
}

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT> & julie::la::cpu::SLMatrixTuple<DT>::operator *= (DT scalar)
{
    this->m_val *= scalar;

    return *this;
}

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT> & julie::la::cpu::SLMatrixTuple<DT>::operator /= (DT scalar)
{
    this->m_val /= scalar;

    return *this;
}

template <typename DT>
bool julie::la::cpu::operator < (const julie::la::cpu::SLMatrixTuple<DT> & left, const julie::la::cpu::SLMatrixTuple<DT> & right)
{
    if (left.m_row < right.m_row)
    {
        return true;
    }
    else if (left.m_row > right.m_row)
    {
        return false;
    }
    else
    {
        if (left.m_col < right.m_col)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

template <typename DT>
bool julie::la::cpu::operator > (const julie::la::cpu::SLMatrixTuple<DT> & left, const julie::la::cpu::SLMatrixTuple<DT> & right)
{
    if (left.m_row > right.m_row)
    {
        return true;
    }
    else if (left.m_row < right.m_row)
    {
        return false;
    }
    else
    {
        if (left.m_col > right.m_col)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

template <typename DT>
bool julie::la::cpu::operator <= (const julie::la::cpu::SLMatrixTuple<DT> & left, const julie::la::cpu::SLMatrixTuple<DT> & right)
{
    return (left < right || left == right);
}

template <typename DT>
bool julie::la::cpu::operator >= (const julie::la::cpu::SLMatrixTuple<DT> & left, const julie::la::cpu::SLMatrixTuple<DT> & right)
{
    return (left > right || left == right);
}

template <typename DT>
bool julie::la::cpu::operator == (const julie::la::cpu::SLMatrixTuple<DT> & left, const julie::la::cpu::SLMatrixTuple<DT> & right)
{
    if (left.m_row == right.m_row && left.m_col == right.m_col)
    {
        return true;
    }
    else
    {
        return false;
    }
}

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT> julie::la::cpu::operator + (const julie::la::cpu::SLMatrixTuple<DT> & tp, DT scalar)
{
    julie::la::cpu::SLMatrixTuple<DT> out {tp.m_row, tp.m_col};
    out.m_val = tp.m_val + scalar;

    return out;
}

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT> julie::la::cpu::operator - (const julie::la::cpu::SLMatrixTuple<DT> & tp, DT scalar)
{
    julie::la::cpu::SLMatrixTuple<DT> out {tp.m_row, tp.m_col};
    out.m_val = tp.m_val - scalar;

    return out;
}

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT> julie::la::cpu::operator * (const julie::la::cpu::SLMatrixTuple<DT> & tp, DT scalar)
{
    julie::la::cpu::SLMatrixTuple<DT> out {tp.m_row, tp.m_col};
    out.m_val = tp.m_val * scalar;

    return out;
}

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT> julie::la::cpu::operator / (const julie::la::cpu::SLMatrixTuple<DT> & tp, DT scalar)
{
    julie::la::cpu::SLMatrixTuple<DT> out {tp.m_row, tp.m_col};
    out.m_val = tp.m_val / scalar;

    return out;
}

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT> julie::la::cpu::operator + (DT scalar, const SLMatrixTuple<DT> & tp)
{
    julie::la::cpu::SLMatrixTuple<DT> out {tp.m_row, tp.m_cols};
    out.m_val = scalar + tp.m_val;

    return out;
}

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT> julie::la::cpu::operator - (DT scalar, const SLMatrixTuple<DT> & tp)
{
    julie::la::cpu::SLMatrixTuple<DT> out {tp.m_row, tp.m_col};
    out.m_val = scalar - tp.m_val;

    return out;
}

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT> julie::la::cpu::operator * (DT scalar, const SLMatrixTuple<DT> & tp)
{
    julie::la::cpu::SLMatrixTuple<DT> out {tp.m_row, tp.m_col};
    out.m_val = scalar * tp.m_val;

    return out;
}

template <typename DT>
julie::la::cpu::SLMatrixTuple<DT> julie::la::cpu::operator / (DT scalar, const SLMatrixTuple<DT> & tp)
{
    julie::la::cpu::SLMatrixTuple<DT> out {tp.m_row, tp.m_col};
    out.m_val = scalar / tp.m_val;

    return out;
}

template <typename DT>
std::ostream & julie::la::cpu::operator << (std::ostream & os, const julie::la::cpu::SLMatrixTuple<DT> & mt)
{
    os << "[" << mt.m_row << ", " << mt.m_col << "]" << "\t" << mt.m_val;

    return os;
}


