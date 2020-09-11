#pragma once

#include <iterator>
#include <cassert>

namespace julie
{

namespace la
{

template <typename DTYPE> class DMatrix;

template <typename DTYPE = double>
class DMatrix_Iterator : public std::iterator<std::bidirectional_iterator_tag, DTYPE>
{
protected:
    lint m_mat_size;
    DTYPE * m_mat_data;
    lint m_ele_index;

public:
    DMatrix_Iterator(lint mat_size, DTYPE * mat_data, lint ele_index);

    DMatrix_Iterator(const DMatrix_Iterator & other);

    DMatrix_Iterator(DMatrix_Iterator && other);

    DMatrix_Iterator & operator = (const DMatrix_Iterator & other);

    DMatrix_Iterator & operator = (DMatrix_Iterator && other);

public:

    DMatrix_Iterator & operator ++ ();

    DMatrix_Iterator operator ++ (int);

    DMatrix_Iterator & operator -- ();

    DMatrix_Iterator operator -- (int);

    DTYPE & operator * () const;

    DTYPE * operator -> () const;

    bool operator == (const DMatrix_Iterator & right) const;

    bool operator != (const DMatrix_Iterator & right) const;

};

template<typename DTYPE>
inline DMatrix_Iterator<DTYPE>::DMatrix_Iterator(lint mat_size, DTYPE * mat_data, lint ele_index)
    : m_mat_size{ mat_size }, m_mat_data{ mat_data }, m_ele_index{ ele_index }
{}

template<typename DTYPE>
inline DMatrix_Iterator<DTYPE>::DMatrix_Iterator(const DMatrix_Iterator & other)
    : m_mat_size{ other.m_mat_size }, m_mat_data{ other.m_mat_data }, m_ele_index{ other.m_ele_index }
{}

template<typename DTYPE>
inline DMatrix_Iterator<DTYPE>::DMatrix_Iterator(DMatrix_Iterator && other)
    : m_mat_size{ other.m_mat_size }, m_mat_data{ other.m_mat_data }, m_ele_index{ other.m_ele_index }
{
    other.m_mat_size = 0;
    other.m_mat_data = nullptr;
    other.m_ele_index = 0;
}

template<typename DTYPE>
inline DMatrix_Iterator<DTYPE> & DMatrix_Iterator<DTYPE>::operator=(const DMatrix_Iterator & other)
{
    this->m_mat_size = other.mat_size;
    this->m_mat_data = other.m_mat_data;
    this->m_ele_index = other.m_ele_index;
}

template<typename DTYPE>
inline DMatrix_Iterator<DTYPE> & DMatrix_Iterator<DTYPE>::operator=(DMatrix_Iterator && other)
{
    this->m_mat_size = other.mat_size;
    this->m_mat_data = other.m_mat_data;
    this->m_ele_index = other.m_ele_index;

    other.m_mat_data = nullptr;
}

template<typename DTYPE>
inline DMatrix_Iterator<DTYPE> & DMatrix_Iterator<DTYPE>::operator++()
{
    assert(this->m_ele_index < this->m_mat_size);

    ++this->m_ele_index;

    return *this;
}

template<typename DTYPE>
inline DMatrix_Iterator<DTYPE> DMatrix_Iterator<DTYPE>::operator++(int)
{
    assert(this->m_ele_index < this->m_mat_size);

    DMatrix_Iterator old{ *this };

    ++this->m_ele_index;

    return old;
}

template<typename DTYPE>
inline DMatrix_Iterator<DTYPE> & DMatrix_Iterator<DTYPE>::operator--()
{
    assert(this->m_ele_index > 0);

    --this->m_ele_index;

    return *this;
}

template<typename DTYPE>
inline DMatrix_Iterator<DTYPE> DMatrix_Iterator<DTYPE>::operator--(int)
{
    assert(this->m_ele_index > 0);

    DMatrix_Iterator old{ *this };

    --this->m_ele_index;

    return old;
}

template<typename DTYPE>
inline DTYPE & DMatrix_Iterator<DTYPE>::operator*() const
{
    return this->m_mat_data[this->m_ele_index];
}

template<typename DTYPE>
inline DTYPE * DMatrix_Iterator<DTYPE>::operator->() const
{
    return this->m_mat_data + this->m_ele_index;
}

template<typename DTYPE>
inline bool DMatrix_Iterator<DTYPE>::operator==(const DMatrix_Iterator & right) const
{
    return (this->m_ele_index == right.m_ele_index) &&
        (this->m_mat_data == right.m_mat_data) &&
        (this->m_mat_size == right.m_mat_size);
}

template<typename DTYPE>
inline bool DMatrix_Iterator<DTYPE>::operator!=(const DMatrix_Iterator & right) const
{
    return !(*this == right);
}


} // namespace la
} // namespace julie