#pragma once
#include "Vector.hpp"
#include "Shape.hpp"
#include "SLMatrixTuple.hpp"
#include "DMatrix.hpp"


#include <iostream>
#include <random>
#include <fstream>
#include <vector>
#include <chrono>
#include <math.h>

namespace julie
{

namespace la
{
    /*
    static lint get_time_in_milliseconds()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>
                (std::chrono::system_clock::now().time_since_epoch()).count();
    }
    */


template <typename DTYPE = double>
class SLMatrix
{
    template <typename DT>
    friend SLMatrix<DT> operator + (const SLMatrix<DT> & left, const SLMatrix<DT> & right);

    template <typename DT>
    friend SLMatrix<DT> operator - (const SLMatrix<DT> & left, const SLMatrix<DT> & right);

    template <typename DT>
    friend SLMatrix<DT> operator * (const SLMatrix<DT> & left, const SLMatrix<DT> & right);

    template <typename DT>
    friend SLMatrix<DT> multiply (const SLMatrix<DT> & left, const SLMatrix<DT> & right);

    template <typename DT>
    friend SLMatrix<DT> matmul (const SLMatrix<DT> & left, const SLMatrix<DT> & right);

    template <typename DT>
    friend std::ostream & operator << (std::ostream & os, const SLMatrix<DT> & mat);

public:

    // SLMatrix created by default constructor is not usable
    // until it is assigned by a usable matrix
    SLMatrix();

    SLMatrix(const Shape & sh, DTYPE zero_threshold = 0);

    SLMatrix(const DMatrix<DTYPE> & dmat, DTYPE zero_threshold = 0);

    SLMatrix(const SLMatrix & smat);

    SLMatrix(SLMatrix && smat);

    /*
    SLMatrix(
        SLMatrixTuple<DTYPE>** row_headers,
        SLMatrixTuple<DTYPE>** col_headers,
        const Shape & shape, DTYPE zero_threshold = 0);
    */

    ~SLMatrix();

    DMatrix<DTYPE> to_DMatrix() const;

    SLMatrix & operator = (const SLMatrix & other);

    SLMatrix & operator = (SLMatrix && other);

    SLMatrix get_transpose() const;

public:

    SLMatrix & operator += (const SLMatrix & other);

    SLMatrix & operator -= (const SLMatrix & other);

    std::string to_string() const;

    Shape shape() const;

    lint rows() const;

    lint columns() const;

    lint non_zero_items() const;

    static inline void new_item(
        SLMatrix<DTYPE> & mat,
        SLMatrixTuple<DTYPE> ** row_ref,
        SLMatrixTuple<DTYPE> ** col_ref,
        SLMatrixTuple<DTYPE> *t_ptr);

    inline bool is_not_zero(DTYPE val) const
    {
        if (abs(val) > this->m_zero_threshold)
        {
            return true;
        }

        return false;
    }

private:

    void release();     

private:

    lint m_rows;
    lint m_cols;

    lint m_non_zero_items;
    
    SLMatrixTuple<DTYPE> ** m_row_headers;
    SLMatrixTuple<DTYPE> ** m_col_headers;
    
    DTYPE m_zero_threshold;
};

template <typename DT>
SLMatrix<DT> operator + (const SLMatrix<DT> & left, const SLMatrix<DT> & right)
{
    if (left.m_rows != right.m_rows || left.m_cols != right.m_cols)
    {
        throw std::invalid_argument(std::string("Inconsistent shapes while adding sparse matrices."));
    }

    SLMatrix<DT> output {Shape{left.m_rows, left.m_cols}, std::min(left.m_zero_threshold, right.m_zero_threshold)};
    
    SLMatrixTuple<DT> ** row_ref = new SLMatrixTuple<DT>*[left.m_rows] {nullptr};
    SLMatrixTuple<DT> ** col_ref = new SLMatrixTuple<DT>*[left.m_cols] {nullptr};

    for (lint i = 0; i < left.m_rows; ++i)
    {
        SLMatrixTuple<DT> *l_ptr = left.m_row_headers[i];
        SLMatrixTuple<DT> *r_ptr = right.m_row_headers[i];
        
        while (l_ptr && r_ptr)
        {
            if (l_ptr->m_col < r_ptr->m_col)
            {
                SLMatrixTuple<DT> *t_ptr_sum = new SLMatrixTuple<DT>{i, l_ptr->m_col, l_ptr->m_val};
                SLMatrix<DT>::new_item(output, row_ref, col_ref, t_ptr_sum);

                l_ptr = l_ptr->m_right;
            }
            else if (l_ptr->m_col == r_ptr->m_col)
            {
                DT val_sum = l_ptr->m_val + r_ptr->m_val;
                if (output.is_not_zero(val_sum))
                {
                    SLMatrixTuple<DT> *t_ptr_sum = new SLMatrixTuple<DT>{i, l_ptr->m_col, val_sum};
                    SLMatrix<DT>::new_item(output, row_ref, col_ref, t_ptr_sum);
                }

                l_ptr = l_ptr->m_right;
                r_ptr = r_ptr->m_right;
            }
            else
            {
                SLMatrixTuple<DT> *t_ptr_sum = new SLMatrixTuple<DT>{i, r_ptr->m_col, r_ptr->m_val};
                SLMatrix<DT>::new_item(output, row_ref, col_ref, t_ptr_sum);

                r_ptr = r_ptr->m_right;
            }      
        }

        if (l_ptr)
        {
            while (l_ptr)
            {
                SLMatrixTuple<DT> *t_ptr_sum = new SLMatrixTuple<DT>{i, l_ptr->m_col, l_ptr->m_val};
                SLMatrix<DT>::new_item(output, row_ref, col_ref, t_ptr_sum);

                l_ptr = l_ptr->m_right;
            }
        }
        else if (r_ptr)
        {
            while (r_ptr)
            {
                SLMatrixTuple<DT> *t_ptr_sum = new SLMatrixTuple<DT>{i, r_ptr->m_col, r_ptr->m_val};
                SLMatrix<DT>::new_item(output, row_ref, col_ref, t_ptr_sum);

                r_ptr = r_ptr->m_right;
            }
        }
    }

    delete []row_ref;
    delete []col_ref;

    return output;
}

template <typename DT>
SLMatrix<DT> operator - (const SLMatrix<DT> & left, const SLMatrix<DT> & right)
{
    if (left.m_rows != right.m_rows || left.m_cols != right.m_cols)
    {
        throw std::invalid_argument(std::string("Inconsistent shapes while subtracting sparse matrices."));
    }

    SLMatrix<DT> output {Shape{left.m_rows, left.m_cols}, std::min(left.m_zero_threshold, right.m_zero_threshold)};
    
    SLMatrixTuple<DT> ** row_ref = new SLMatrixTuple<DT>*[left.m_rows] {nullptr};
    SLMatrixTuple<DT> ** col_ref = new SLMatrixTuple<DT>*[left.m_cols] {nullptr};

    for (lint i = 0; i < left.m_rows; ++i)
    {
        SLMatrixTuple<DT> *l_ptr = left.m_row_headers[i];
        SLMatrixTuple<DT> *r_ptr = right.m_row_headers[i];
        
        while (l_ptr && r_ptr)
        {
            if (l_ptr->m_col < r_ptr->m_col)
            {
                SLMatrixTuple<DT> *t_ptr_sub = new SLMatrixTuple<DT>{i, l_ptr->m_col, l_ptr->m_val};
                SLMatrix<DT>::new_item(output, row_ref, col_ref, t_ptr_sub);

                l_ptr = l_ptr->m_right;
            }
            else if (l_ptr->m_col == r_ptr->m_col)
            {
                DT val_sub = l_ptr->m_val - r_ptr->m_val;
                if (output.is_not_zero(val_sub))
                {
                    SLMatrixTuple<DT> *t_ptr_sub = new SLMatrixTuple<DT>{i, l_ptr->m_col, val_sub};
                    SLMatrix<DT>::new_item(output, row_ref, col_ref, t_ptr_sub);
                }

                l_ptr = l_ptr->m_right;
                r_ptr = r_ptr->m_right;
            }
            else
            {
                SLMatrixTuple<DT> *t_ptr_sub = new SLMatrixTuple<DT>{i, r_ptr->m_col, r_ptr->m_val * (-1)};
                SLMatrix<DT>::new_item(output, row_ref, col_ref, t_ptr_sub);

                r_ptr = r_ptr->m_right;
            }      
        }

        if (l_ptr)
        {
            while (l_ptr)
            {
                SLMatrixTuple<DT> *t_ptr_sub = new SLMatrixTuple<DT>{i, l_ptr->m_col, l_ptr->m_val};
                SLMatrix<DT>::new_item(output, row_ref, col_ref, t_ptr_sub);

                l_ptr = l_ptr->m_right;
            }
        }
        else if (r_ptr)
        {
            while (r_ptr)
            {
                SLMatrixTuple<DT> *t_ptr_sub = new SLMatrixTuple<DT>{i, r_ptr->m_col, r_ptr->m_val * (-1)};
                SLMatrix<DT>::new_item(output, row_ref, col_ref, t_ptr_sub);

                r_ptr = r_ptr->m_right;
            }
        }
    }

    delete []row_ref;
    delete []col_ref;

    return output;
}

template <typename DT>
SLMatrix<DT> operator * (const SLMatrix<DT> & left, const SLMatrix<DT> & right)
{
    if (left.m_rows != right.m_rows || left.m_cols != right.m_cols)
    {
        throw std::invalid_argument(std::string("Inconsistent shapes while multiplying sparse matrices."));
    }

    SLMatrix<DT> output {Shape{left.m_rows, left.m_cols}, std::min(left.m_zero_threshold, right.m_zero_threshold)};
    
    SLMatrixTuple<DT> ** row_ref = new SLMatrixTuple<DT>*[left.m_rows] {nullptr};
    SLMatrixTuple<DT> ** col_ref = new SLMatrixTuple<DT>*[left.m_cols] {nullptr};

    for (lint i = 0; i < left.m_rows; ++i)
    {
        SLMatrixTuple<DT> *l_ptr = left.m_row_headers[i];
        SLMatrixTuple<DT> *r_ptr = right.m_row_headers[i];
        
        while (l_ptr && r_ptr)
        {
            if (l_ptr->m_col < r_ptr->m_col)
            {
                l_ptr = l_ptr->m_right;
            }
            else if (l_ptr->m_col == r_ptr->m_col)
            {
                DT val_mul = l_ptr->m_val * r_ptr->m_val;
                if (output.is_not_zero(val_mul))
                {
                    SLMatrixTuple<DT> *t_ptr_mul = new SLMatrixTuple<DT>{i, l_ptr->m_col, val_mul};
                    SLMatrix<DT>::new_item(output, row_ref, col_ref, t_ptr_mul);
                }

                l_ptr = l_ptr->m_right;
                r_ptr = r_ptr->m_right;
            }
            else
            {
                r_ptr = r_ptr->m_right;
            }      
        }
    }

    delete []row_ref;
    delete []col_ref;

    return output;
}

template <typename DT>
SLMatrix<DT> multiply (const SLMatrix<DT> & left, const SLMatrix<DT> & right)
{
    return left * right;
}

template <typename DT>
SLMatrix<DT> matmul (const SLMatrix<DT> & left, const SLMatrix<DT> & right)
{
    if (left.m_cols != right.m_rows)
    {
        // std::cout << "----------------------------------------------------\n";
        throw std::invalid_argument(incompatible_shape);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////

    SLMatrix<DT> output {Shape{left.m_rows, right.m_cols}, std::min(left.m_zero_threshold, right.m_zero_threshold)};

    SLMatrixTuple<DT> ** row_ref = new SLMatrixTuple<DT>*[left.m_rows] {nullptr};
    SLMatrixTuple<DT> ** col_ref = new SLMatrixTuple<DT>*[right.m_cols] {nullptr};

    for (lint i = 0; i < left.m_rows; ++i)
    {
        for (lint j = 0; j < right.m_cols; ++j)
        {
            SLMatrixTuple<DT> *l_ptr = left.m_row_headers[i];
            SLMatrixTuple<DT> *r_ptr = right.m_col_headers[j];

            DT dot = 0;

            while (l_ptr && r_ptr)
            {
                // std::cout << "MatMul debug " << *r_ptr << std::endl;
                if (l_ptr->m_col < r_ptr->m_row)
                {
                    l_ptr = l_ptr->m_right;
                }
                else if (l_ptr->m_col == r_ptr->m_row)
                {
                    dot += l_ptr->m_val * r_ptr->m_val;

                    l_ptr = l_ptr->m_right;
                    r_ptr = r_ptr->m_down;
                }
                else
                {
                    r_ptr = r_ptr->m_down;
                }
            }

            if (output.is_not_zero(dot))
            {
                SLMatrixTuple<DT> *t_ptr_mul = new SLMatrixTuple<DT>{i, j, dot};
                SLMatrix<DT>::new_item(output, row_ref, col_ref, t_ptr_mul);
            }
        }
    }

    delete []row_ref;
    delete []col_ref;      

    return output;
}

template <typename DT>
std::ostream & operator << (std::ostream & os, const SLMatrix<DT> & mat)
{
    os << mat.to_string();

    return os;
}

} // namespace la
} // namespace julie


template <typename DTYPE>
julie::la::SLMatrix<DTYPE>::SLMatrix()
    :
    m_rows {0}, m_cols {0},
    m_non_zero_items {0},
    m_row_headers {nullptr},
    m_col_headers {nullptr},
    m_zero_threshold {0}
{}

template <typename DTYPE>
julie::la::SLMatrix<DTYPE>::SLMatrix(const Shape & shape, DTYPE zero_threshold)
    :
    m_rows {shape[0]},
    m_cols {shape[1]},
    m_non_zero_items {0},
    m_row_headers {new SLMatrixTuple<DTYPE>*[m_rows]{nullptr}},
    m_col_headers {new SLMatrixTuple<DTYPE>*[m_cols]{nullptr}},
    m_zero_threshold {zero_threshold}
{
    if (shape.dim() != 2)
    {
        this->release();
        throw std::invalid_argument("Shape of lower or higher dimensions than 2-dimension is not allowed");
    }
}

/*
template <typename DTYPE>
julie::la::SLMatrix<DTYPE>::SLMatrix(
    SLMatrixTuple<DTYPE>** row_headers,
    SLMatrixTuple<DTYPE>** col_headers,
    const julie::la::Shape & shape,
    DTYPE zero_threshold)
    :
    m_rows {shape[0]},
    m_cols {shape[1]},
    m_row_headers {row_headers},
    m_col_headers {col_headers},
    m_zero_threshold {zero_threshold}
{
    if (shape.dim() != 2)
    {
        this->release();
        throw std::invalid_argument("Shape of lower or higher dimensions than 2-dimension is not allowed");
    }
}
*/

template <typename DTYPE>
julie::la::SLMatrix<DTYPE>::SLMatrix(const DMatrix<DTYPE> & dmat, DTYPE zero_threshold)
    :
    m_rows {dmat.shape()[0]},
    m_cols {dmat.shape()[1]},
    m_non_zero_items {0},
    m_row_headers {new SLMatrixTuple<DTYPE>*[m_rows] {nullptr}},
    m_col_headers {new SLMatrixTuple<DTYPE>*[m_cols] {nullptr}},
    m_zero_threshold {zero_threshold}
{
    if (dmat.shape().dim() != 2)
    {
        this->release();
        throw std::invalid_argument("Shape of lower or higher dimensions than 2-dimension is not allowed");
    }

    Coordinate location {dmat.shape()};
    SLMatrixTuple<DTYPE> ** row_ref = new SLMatrixTuple<DTYPE>*[this->m_rows] {nullptr};
    SLMatrixTuple<DTYPE> ** col_ref = new SLMatrixTuple<DTYPE>*[this->m_cols] {nullptr};
    
    for (auto itr = dmat.begin(); itr != dmat.end(); ++itr)
    {
        DTYPE value = *itr;

        // std::cout << abs(value) << std::endl;
        // std::cout << m_zero_threshold << std::endl;

        if (this->is_not_zero(value))
        {
            SLMatrixTuple<DTYPE> * t_ptr = new SLMatrixTuple<DTYPE>{location[0], location[1], value};
            new_item(*this, row_ref, col_ref, t_ptr);
        }
        
        ++location;
    }

    delete []row_ref;
    delete []col_ref;
}

template <typename DTYPE>
julie::la::SLMatrix<DTYPE>::SLMatrix(const SLMatrix<DTYPE> & smat)
    :
    m_rows {smat.m_rows},
    m_cols {smat.m_cols},
    m_non_zero_items {0},
    m_row_headers {new SLMatrixTuple<DTYPE>*[m_rows] {nullptr}},
    m_col_headers {new SLMatrixTuple<DTYPE>*[m_cols] {nullptr}},
    m_zero_threshold {smat.m_zero_threshold}
{
    SLMatrixTuple<DTYPE> ** row_ref = new SLMatrixTuple<DTYPE>*[this->m_rows] {nullptr};
    SLMatrixTuple<DTYPE> ** col_ref = new SLMatrixTuple<DTYPE>*[this->m_cols] {nullptr};

    for (lint i = 0; i < this->m_rows; ++i)
    {
        SLMatrixTuple<DTYPE> *t_ptr = smat.m_row_headers[i];
        
        while (t_ptr)
        {
            SLMatrixTuple<DTYPE> *t_ptr_cpy = new SLMatrixTuple<DTYPE>{*t_ptr};
            new_item(*this, row_ref, col_ref, t_ptr_cpy);

            t_ptr = t_ptr->m_right;
        }
    }

    delete []row_ref;
    delete []col_ref;
}

template <typename DTYPE>
julie::la::SLMatrix<DTYPE>::SLMatrix(SLMatrix<DTYPE> && smat)
    :
    m_rows {smat.m_rows},
    m_cols {smat.m_cols},
    m_non_zero_items {smat.m_non_zero_items},
    m_row_headers {smat.m_row_headers},
    m_col_headers {smat.m_col_headers},
    m_zero_threshold {smat.m_zero_threshold}
{
    smat.m_rows = 0;
    smat.m_cols = 0;
    smat.m_row_headers = nullptr;
    smat.m_col_headers = nullptr;
}

template <typename DTYPE>
julie::la::SLMatrix<DTYPE>::~SLMatrix()
{
    this->release();
}

template <typename DTYPE>
void julie::la::SLMatrix<DTYPE>::release()
{
    std::vector<SLMatrixTuple<DTYPE> *> t_ptr_list;

    for (lint i = 0; i < this->m_rows; ++i)
    {
        SLMatrixTuple<DTYPE> *t_ptr = this->m_row_headers[i];
        
        while (t_ptr)
        {
            t_ptr_list.push_back(t_ptr);
            t_ptr = t_ptr->m_right;
        }
    }

    for (SLMatrixTuple<DTYPE> * t_ptr : t_ptr_list)
    {
        delete t_ptr;
    }

    delete []this->m_row_headers;
    delete []this->m_col_headers;
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE> julie::la::SLMatrix<DTYPE>::to_DMatrix () const
{
    Shape d_shape {this->m_rows, this->m_cols};
    // std::cout << d_shape << std::endl;
    DMatrix<DTYPE> dmat {d_shape};

    Coordinate d_location {d_shape};
    auto d_itr = dmat.begin();

    for (lint i = 0; i < this->m_rows; ++i)
    {
        SLMatrixTuple<DTYPE> *t_ptr = this->m_row_headers[i];
        
        while (t_ptr)
        {
            auto s_pos = Coordinate{{t_ptr->m_row, t_ptr->m_col}, d_shape};

            while (d_location < s_pos)
            {
                *d_itr = 0;
                ++d_location;
                ++d_itr;
            }

            // Fill in non zero place
            *d_itr = t_ptr->m_val;
            
            ++d_location;
            ++d_itr;
            t_ptr = t_ptr->m_right;
        }
    }

    while (d_itr != dmat.end())
    {
        *d_itr = 0;
        ++d_itr;
    }

    return dmat;
}

template <typename DTYPE>
julie::la::SLMatrix<DTYPE> & julie::la::SLMatrix<DTYPE>::operator = (const SLMatrix<DTYPE> & other)
{
    this->release();

    this->m_rows = other.m_rows;
    this->m_cols = other.m_cols;
    this->m_row_headers = new SLMatrixTuple<DTYPE>*[this->m_rows] {nullptr};
    this->m_col_headers = new SLMatrixTuple<DTYPE>*[this->m_cols] {nullptr};
    this->m_zero_threshold = other.m_zero_threshold;

    SLMatrixTuple<DTYPE> ** row_ref = new SLMatrixTuple<DTYPE>*[this->m_rows] {nullptr};
    SLMatrixTuple<DTYPE> ** col_ref = new SLMatrixTuple<DTYPE>*[this->m_cols] {nullptr};

    for (lint i = 0; i < this->m_rows; ++i)
    {
        SLMatrixTuple<DTYPE> *t_ptr = other.m_row_headers[i];
        
        while (t_ptr)
        {
            SLMatrixTuple<DTYPE> *t_ptr_cpy = new SLMatrixTuple<DTYPE>{*t_ptr};
            new_item(*this, row_ref, col_ref, t_ptr_cpy);

            t_ptr = t_ptr->m_right;
        }
    }

    delete []row_ref;
    delete []col_ref;

    return *this;
}

template <typename DTYPE>
julie::la::SLMatrix<DTYPE> & julie::la::SLMatrix<DTYPE>::operator = (SLMatrix<DTYPE> && other)
{
    this->release();

    this->m_rows = other.m_rows;
    this->m_cols = other.m_cols;
    this->m_row_headers = other.m_row_headers;
    this->m_col_headers = other.m_col_headers;
    this->m_zero_threshold = other.m_zero_threshold;

    other.m_rows = 0;
    other.m_cols = 0;
    other.m_row_headers = nullptr;
    other.m_col_headers = nullptr;

    return *this;
}

template <typename DTYPE>
julie::la::SLMatrix<DTYPE> julie::la::SLMatrix<DTYPE>::get_transpose() const
{
    Shape reversed_shape{ this->m_cols, this->m_rows };
    SLMatrix<DTYPE> transposed{ reversed_shape, this->m_zero_threshold };

    SLMatrixTuple<DTYPE> ** row_ref = new SLMatrixTuple<DTYPE>*[transposed.m_rows] {nullptr};
    SLMatrixTuple<DTYPE> ** col_ref = new SLMatrixTuple<DTYPE>*[transposed.m_cols] {nullptr};

    for (lint i = 0; i < this->m_cols; ++i)
    {
        SLMatrixTuple<DTYPE> *t_ptr = this->m_col_headers[i];
        
        while (t_ptr)
        {
            SLMatrixTuple<DTYPE> *t_ptr_t = new SLMatrixTuple<DTYPE>{t_ptr->m_col, t_ptr->m_row, t_ptr->m_val};
            new_item(transposed, row_ref, col_ref, t_ptr_t);

            t_ptr = t_ptr->m_down;
        }
    }

    delete []row_ref;
    delete []col_ref;

    return transposed;
}

template <typename DTYPE>
julie::la::SLMatrix<DTYPE> & julie::la::SLMatrix<DTYPE>::operator += (const SLMatrix<DTYPE> & other)
{
    SLMatrix<DTYPE> sum = *this + other;
    *this = std::move(sum);

    return *this;
}

template <typename DTYPE>
julie::la::SLMatrix<DTYPE> & julie::la::SLMatrix<DTYPE>::operator -= (const SLMatrix<DTYPE> & other)
{
    SLMatrix<DTYPE> sub = *this - other;
    *this = std::move(sub);

    return *this;
}

template <typename DTYPE>
std::string julie::la::SLMatrix<DTYPE>::to_string() const
{
    std::ostringstream stream;

    /*
    for (lint i = 0; i < this->m_rows; ++i)
    {
        SLMatrixTuple<DTYPE> *t_ptr = this->m_row_headers[i];
        
        while (t_ptr)
        {
            stream << "[" << t_ptr->m_row << " " << t_ptr->m_col <<"]" << "\t" << t_ptr->m_val << "\n";
            t_ptr = t_ptr->m_right;
        }
    }
    */
    
    for (lint i = 0; i < this->m_cols; ++i)
    {
        SLMatrixTuple<DTYPE> *t_ptr = this->m_col_headers[i];
        
        while (t_ptr)
        {
            stream << "[" << t_ptr->m_row << " " << t_ptr->m_col <<"]" << "\t" << t_ptr->m_val << "\n";
            t_ptr = t_ptr->m_down;
        }
    }


    return stream.str();
}

template <typename DTYPE>
julie::la::Shape julie::la::SLMatrix<DTYPE>::shape() const
{
    return Shape{this->m_rows, this->m_cols};
}

template <typename DTYPE>
lint julie::la::SLMatrix<DTYPE>::rows() const
{
    return this->m_rows;
}

template <typename DTYPE>
lint julie::la::SLMatrix<DTYPE>::columns() const
{
    return this->m_cols;
}

template <typename DTYPE>
lint julie::la::SLMatrix<DTYPE>::non_zero_items() const
{
    return this->m_non_zero_items;
}

template <typename DTYPE>
inline void julie::la::SLMatrix<DTYPE>::new_item(
    SLMatrix<DTYPE> & mat,
    SLMatrixTuple<DTYPE> ** row_ref,
    SLMatrixTuple<DTYPE> ** col_ref,
    SLMatrixTuple<DTYPE> *t_ptr)
{
    lint row = t_ptr->m_row;
    lint col = t_ptr->m_col;

    if (!row_ref[row])
    {
        mat.m_row_headers[row] = t_ptr;
    }
    else
    {
        row_ref[row]->m_right = t_ptr;
    }

    if (!col_ref[col])
    {
        mat.m_col_headers[col] = t_ptr;
    }
    else
    {
        col_ref[col]->m_down = t_ptr;
    }

    row_ref[row] = t_ptr;
    col_ref[col] = t_ptr;

    ++mat.m_non_zero_items;
}
