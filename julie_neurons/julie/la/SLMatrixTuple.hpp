
#pragma once
#include "Shape.hpp"
#include "Exceptions.hpp"

#include <ostream>

namespace julie
{

namespace la
{
template <typename DTYPE = double>
struct SLMatrixTuple
{
    lint m_row;
    lint m_col;
    DTYPE m_val;
    SLMatrixTuple * m_right;
    SLMatrixTuple * m_down;

    static lint REF_COUNT;

    SLMatrixTuple(lint row, lint col, DTYPE value, const SLMatrixTuple *right, const SLMatrixTuple *down);

    SLMatrixTuple(lint row, lint col, DTYPE value);

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

    SLMatrixTuple & operator += (DTYPE scalar);

    SLMatrixTuple & operator -= (DTYPE scalar);

    SLMatrixTuple & operator *= (DTYPE scalar);

    SLMatrixTuple & operator /= (DTYPE scalar);
};

template <typename DTYPE>
bool operator < (const SLMatrixTuple<DTYPE> & left, const SLMatrixTuple<DTYPE> & right);

template <typename DTYPE>
bool operator > (const SLMatrixTuple<DTYPE> & left, const SLMatrixTuple<DTYPE> & right);

template <typename DTYPE>
bool operator <= (const SLMatrixTuple<DTYPE> & left, const SLMatrixTuple<DTYPE> & right);

template <typename DTYPE>
bool operator >= (const SLMatrixTuple<DTYPE> & left, const SLMatrixTuple<DTYPE> & right);

template <typename DTYPE>
bool operator == (const SLMatrixTuple<DTYPE> & left, const SLMatrixTuple<DTYPE> & right);


template <typename DTYPE>
SLMatrixTuple<DTYPE> operator + (const SLMatrixTuple<DTYPE> & tp, DTYPE scalar);

template <typename DTYPE>
SLMatrixTuple<DTYPE> operator - (const SLMatrixTuple<DTYPE> & tp, DTYPE scalar);

template <typename DTYPE>
SLMatrixTuple<DTYPE> operator * (const SLMatrixTuple<DTYPE> & tp, DTYPE scalar);

template <typename DTYPE>
SLMatrixTuple<DTYPE> operator / (const SLMatrixTuple<DTYPE> & tp, DTYPE scalar);

template <typename DTYPE>
SLMatrixTuple<DTYPE> operator + (DTYPE scalar, const SLMatrixTuple<DTYPE> & tp);

template <typename DTYPE>
SLMatrixTuple<DTYPE> operator - (DTYPE scalar, const SLMatrixTuple<DTYPE> & tp);

template <typename DTYPE>
SLMatrixTuple<DTYPE> operator * (DTYPE scalar, const SLMatrixTuple<DTYPE> & tp);

template <typename DTYPE>
SLMatrixTuple<DTYPE> operator / (DTYPE scalar, const SLMatrixTuple<DTYPE> & tp);

template <typename DTYPE>
std::ostream & operator << (std::ostream & os, const SLMatrixTuple<DTYPE> & mt);

} // namespace la
} // namespace julie

template <typename DTYPE>
lint julie::la::SLMatrixTuple<DTYPE>::REF_COUNT = 0;

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE>::SLMatrixTuple(
    lint row, lint col, DTYPE value,
    const SLMatrixTuple *right, const SLMatrixTuple *down)
    : m_row {row}, m_col {col}, m_val {value},
      m_right {right}, m_down {down}
{
    ++REF_COUNT;
}

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE>::SLMatrixTuple(lint row, lint col, DTYPE value)
    : m_row {row}, m_col {col}, m_val {value},
      m_right {nullptr}, m_down {nullptr}
{
    //++REF_COUNT;
}

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE>::SLMatrixTuple(lint row, lint col)
    : m_row {row}, m_col {col}, m_val {0},
      m_right {nullptr}, m_down {nullptr}
{
    //++REF_COUNT;
}

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE>::SLMatrixTuple()
    : m_row {-1}, m_col {-1}, m_val {0},
      m_right {nullptr}, m_down {nullptr}
{
    //++REF_COUNT;
}

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE>::SLMatrixTuple(const SLMatrixTuple & other)
    : m_row {other.m_row}, m_col {other.m_col}, m_val {other.m_val},
      m_right {other.m_right}, m_down {other.m_down}
{
    //++REF_COUNT;
}

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE>::SLMatrixTuple(SLMatrixTuple && other)
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

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE> & julie::la::SLMatrixTuple<DTYPE>::operator = (const SLMatrixTuple & other)
{
    this->m_row = other.m_row;
    this->m_col = other.m_col;
    this->m_val = other.m_val;
    this->m_right = other.m_right;
    this->m_down = other.m_down;

    return *this;
}

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE> & julie::la::SLMatrixTuple<DTYPE>::operator = (SLMatrixTuple && other)
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

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE> & julie::la::SLMatrixTuple<DTYPE>::operator += (DTYPE scalar)
{
    this->m_val += scalar;

    return *this;
}

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE> & julie::la::SLMatrixTuple<DTYPE>::operator -= (DTYPE scalar)
{
    this->m_val -= scalar;

    return *this;
}

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE> & julie::la::SLMatrixTuple<DTYPE>::operator *= (DTYPE scalar)
{
    this->m_val *= scalar;

    return *this;
}

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE> & julie::la::SLMatrixTuple<DTYPE>::operator /= (DTYPE scalar)
{
    this->m_val /= scalar;

    return *this;
}

template <typename DTYPE>
bool julie::la::operator < (const julie::la::SLMatrixTuple<DTYPE> & left, const julie::la::SLMatrixTuple<DTYPE> & right)
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

template <typename DTYPE>
bool julie::la::operator > (const julie::la::SLMatrixTuple<DTYPE> & left, const julie::la::SLMatrixTuple<DTYPE> & right)
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

template <typename DTYPE>
bool julie::la::operator <= (const julie::la::SLMatrixTuple<DTYPE> & left, const julie::la::SLMatrixTuple<DTYPE> & right)
{
    return (left < right || left == right);
}

template <typename DTYPE>
bool julie::la::operator >= (const julie::la::SLMatrixTuple<DTYPE> & left, const julie::la::SLMatrixTuple<DTYPE> & right)
{
    return (left > right || left == right);
}

template <typename DTYPE>
bool julie::la::operator == (const julie::la::SLMatrixTuple<DTYPE> & left, const julie::la::SLMatrixTuple<DTYPE> & right)
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

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE> julie::la::operator + (const julie::la::SLMatrixTuple<DTYPE> & tp, DTYPE scalar)
{
    julie::la::SLMatrixTuple<DTYPE> out {tp.m_row, tp.m_col};
    out.m_val = tp.m_val + scalar;

    return out;
}

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE> julie::la::operator - (const julie::la::SLMatrixTuple<DTYPE> & tp, DTYPE scalar)
{
    julie::la::SLMatrixTuple<DTYPE> out {tp.m_row, tp.m_col};
    out.m_val = tp.m_val - scalar;

    return out;
}

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE> julie::la::operator * (const julie::la::SLMatrixTuple<DTYPE> & tp, DTYPE scalar)
{
    julie::la::SLMatrixTuple<DTYPE> out {tp.m_row, tp.m_col};
    out.m_val = tp.m_val * scalar;

    return out;
}

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE> julie::la::operator / (const julie::la::SLMatrixTuple<DTYPE> & tp, DTYPE scalar)
{
    julie::la::SLMatrixTuple<DTYPE> out {tp.m_row, tp.m_col};
    out.m_val = tp.m_val / scalar;

    return out;
}

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE> julie::la::operator + (DTYPE scalar, const SLMatrixTuple<DTYPE> & tp)
{
    julie::la::SLMatrixTuple<DTYPE> out {tp.m_row, tp.m_cols};
    out.m_val = scalar + tp.m_val;

    return out;
}

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE> julie::la::operator - (DTYPE scalar, const SLMatrixTuple<DTYPE> & tp)
{
    julie::la::SLMatrixTuple<DTYPE> out {tp.m_row, tp.m_col};
    out.m_val = scalar - tp.m_val;

    return out;
}

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE> julie::la::operator * (DTYPE scalar, const SLMatrixTuple<DTYPE> & tp)
{
    julie::la::SLMatrixTuple<DTYPE> out {tp.m_row, tp.m_col};
    out.m_val = scalar * tp.m_val;

    return out;
}

template <typename DTYPE>
julie::la::SLMatrixTuple<DTYPE> julie::la::operator / (DTYPE scalar, const SLMatrixTuple<DTYPE> & tp)
{
    julie::la::SLMatrixTuple<DTYPE> out {tp.m_row, tp.m_col};
    out.m_val = scalar / tp.m_val;

    return out;
}

template <typename DTYPE>
std::ostream & julie::la::operator << (std::ostream & os, const julie::la::SLMatrixTuple<DTYPE> & mt)
{
    os << "[" << mt.m_row << ", " << mt.m_col << "]" << "\t" << mt.m_val;

    return os;
}


