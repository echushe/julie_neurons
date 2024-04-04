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
#include "SLMatrixTuple.hpp"
#include "Matrix_CPU.hpp"


#include <iostream>
#include <random>
#include <fstream>
#include <vector>
#include <chrono>
#include <math.h>
#include <sstream> 

namespace julie
{
namespace la
{
namespace cpu
{

/***********************************************************************************
 * A SLMatrix is a sparse matrix implemented via linked lists
 ***********************************************************************************/
template <typename DT = float>
class SLMatrix
{

/*
 * Declarations friend functions are listed here
 * */ 

    template <typename T>
    friend SLMatrix<T> operator + (const SLMatrix<T> & left, const SLMatrix<T> & right);

    template <typename T>
    friend SLMatrix<T> operator - (const SLMatrix<T> & left, const SLMatrix<T> & right);

    template <typename T>
    friend SLMatrix<T> operator * (const SLMatrix<T> & left, const SLMatrix<T> & right);

    template <typename T>
    friend SLMatrix<T> multiply (const SLMatrix<T> & left, const SLMatrix<T> & right);

    template <typename T>
    friend SLMatrix<T> matmul (const SLMatrix<T> & left, const SLMatrix<T> & right);

    template <typename T>
    friend std::ostream & operator << (std::ostream & os, const SLMatrix<T> & mat);

public:

    // SLMatrix created by default constructor is not usable
    // until it is assigned by a usable matrix
    SLMatrix();

    // This constructor is to create a sparse matrix of a specified shape.
    // Arguments:
    //     sh:             Shape of this sparse matrix
    //     zero_threshold: This argument is usually a small positive value working as a threshold
    //                     that if value of one element stays between -zero_threshold and +zero_threshold,
    //                     this element will be set to zero
    SLMatrix(const Shape & sh, DT zero_threshold = 0);

    // This is to construct a aparse matrix from a CPU mode matrix.
    // Arguments:
    //     dmat:           The normal matrix that is CPU mode
    //     zero_threshold: This argument is usually a small positive value working as a threshold
    //                     that if value of one element stays between -zero_threshold and +zero_threshold,
    //                     this element will be set to zero
    SLMatrix(const Matrix_CPU<DT> & dmat, DT zero_threshold = 0);

    // Copy constructor
    SLMatrix(const SLMatrix & smat);

    // Move constructor
    SLMatrix(SLMatrix && smat);

    /*
    SLMatrix(
        SLMatrixTuple<DT>** row_headers,
        SLMatrixTuple<DT>** col_headers,
        const Shape & shape, DT zero_threshold = 0);
    */

    // Destructor
    ~SLMatrix();

    // Get a normal matrix instance from the sparse matrix
    Matrix_CPU<DT> to_Matrix_CPU() const;

    // Copy assignment
    SLMatrix & operator = (const SLMatrix & other);

    // Move assignment
    SLMatrix & operator = (SLMatrix && other);

    // Get a transposed instance of the sparse matrix
    SLMatrix get_transpose() const;

public:

    // Element wise addition
    SLMatrix & operator += (const SLMatrix & other);

    // Element wise subtraction
    SLMatrix & operator -= (const SLMatrix & other);

    // String plot of this sparse matrix
    std::string to_string() const;

    // Get shape of this sparse matrix.
    // Shape of this sparse matrix is always 2 dimensional
    Shape shape() const;

    // Get number of rows of this sparse matrix
    lint rows() const;

    // Get number of columns of this sparse matrix
    lint columns() const;

    // Get number of non zero element of this sparse matrix
    lint non_zero_items() const;

    // This is a global helper function to add a new non zero item into this sparse matrix.
    // Row index or column index of the new element should be larger than any existing non
    // zero elements's row index or column index.
    // Arguments:
    //     mat:     The sparse matrix where the new item is added
    //     row_ref: An array of SLMatrixTuple pointers to remember pointer of the last non zero
    //              element of each row
    //     col_ref: An array of SLMatrixTuple pointers to remember pointer of the last non zero
    //              element of each column
    //     t_ptr:   Pointer to the new non zero element (SLMatrixTuple)
    // Returns: void
    static inline void new_item(
        SLMatrix<DT> & mat,
        SLMatrixTuple<DT> ** row_ref,
        SLMatrixTuple<DT> ** col_ref,
        SLMatrixTuple<DT> *t_ptr);

    // This method is to verify that if a number's absolute value is larger than a threshold,
    // it will be considered as a non zero value.
    inline bool is_not_zero(DT val) const
    {
        if (std::abs(val) > this->m_zero_threshold)
        {
            return true;
        }

        return false;
    }

private:

    // Free all memory
    void release();     

private:

    // Number of rows of this sparse matrix
    lint m_rows;

    // Number of columns of this sparse matrix
    lint m_cols;

    // Number of non zero items
    lint m_non_zero_items;
    
    // An array of pointers pointing to the first non zero element of each row.
    // The pointer will be NULL pointer if its row is empty.
    SLMatrixTuple<DT> ** m_row_headers;

    // An array of pointers pointing to the first non zero element of each column.
    // The pointer will be NULL pointer if its column is empty.
    SLMatrixTuple<DT> ** m_col_headers;
    
    // A threshold showing that a value will be considered as zero if its absolute value
    // is lower than this threshold
    DT m_zero_threshold;
};

// addition: a + b
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

// subtraction a - b
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

// Element wise multiplication
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

// Element wise multiplication
template <typename DT>
SLMatrix<DT> multiply (const SLMatrix<DT> & left, const SLMatrix<DT> & right)
{
    return left * right;
}

// Matrix multiplication
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

// Plot this sparse matrix as a stream outout
template <typename DT>
std::ostream & operator << (std::ostream & os, const SLMatrix<DT> & mat)
{
    os << mat.to_string();

    return os;
}

} // namespace cpu
} // namespace la
} // namespace julie


template <typename DT>
julie::la::cpu::SLMatrix<DT>::SLMatrix()
    :
    m_rows {0}, m_cols {0},
    m_non_zero_items {0},
    m_row_headers {nullptr},
    m_col_headers {nullptr},
    m_zero_threshold {0}
{}

template <typename DT>
julie::la::cpu::SLMatrix<DT>::SLMatrix(const Shape & shape, DT zero_threshold)
    :
    m_rows {shape[0]},
    m_cols {shape[1]},
    m_non_zero_items {0},
    m_row_headers {new SLMatrixTuple<DT>*[m_rows]{nullptr}},
    m_col_headers {new SLMatrixTuple<DT>*[m_cols]{nullptr}},
    m_zero_threshold {zero_threshold}
{
    if (shape.dim() != 2)
    {
        this->release();
        throw std::invalid_argument("Shape of lower or higher dimensions than 2-dimension is not allowed");
    }
}

/*
template <typename DT>
julie::la::cpu::SLMatrix<DT>::SLMatrix(
    SLMatrixTuple<DT>** row_headers,
    SLMatrixTuple<DT>** col_headers,
    const julie::la::Shape & shape,
    DT zero_threshold)
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

template <typename DT>
julie::la::cpu::SLMatrix<DT>::SLMatrix(const Matrix_CPU<DT> & dmat, DT zero_threshold)
    :
    m_rows {dmat.shape()[0]},
    m_cols {dmat.shape()[1]},
    m_non_zero_items {0},
    m_row_headers {new SLMatrixTuple<DT>*[m_rows] {nullptr}},
    m_col_headers {new SLMatrixTuple<DT>*[m_cols] {nullptr}},
    m_zero_threshold {zero_threshold}
{
    if (dmat.shape().dim() != 2)
    {
        this->release();
        throw std::invalid_argument("Shape of lower or higher dimensions than 2-dimension is not allowed");
    }

    Coordinate location {dmat.shape()};
    SLMatrixTuple<DT> ** row_ref = new SLMatrixTuple<DT>*[this->m_rows] {nullptr};
    SLMatrixTuple<DT> ** col_ref = new SLMatrixTuple<DT>*[this->m_cols] {nullptr};
    
    for (auto itr = dmat.begin(); itr != dmat.end(); ++itr)
    {
        DT value = *itr;

        // std::cout << abs(value) << std::endl;
        // std::cout << m_zero_threshold << std::endl;

        if (this->is_not_zero(value))
        {
            SLMatrixTuple<DT> * t_ptr = new SLMatrixTuple<DT>{location[0], location[1], value};
            new_item(*this, row_ref, col_ref, t_ptr);
        }
        
        ++location;
    }

    delete []row_ref;
    delete []col_ref;
}

template <typename DT>
julie::la::cpu::SLMatrix<DT>::SLMatrix(const SLMatrix<DT> & smat)
    :
    m_rows {smat.m_rows},
    m_cols {smat.m_cols},
    m_non_zero_items {0},
    m_row_headers {new SLMatrixTuple<DT>*[m_rows] {nullptr}},
    m_col_headers {new SLMatrixTuple<DT>*[m_cols] {nullptr}},
    m_zero_threshold {smat.m_zero_threshold}
{
    SLMatrixTuple<DT> ** row_ref = new SLMatrixTuple<DT>*[this->m_rows] {nullptr};
    SLMatrixTuple<DT> ** col_ref = new SLMatrixTuple<DT>*[this->m_cols] {nullptr};

    for (lint i = 0; i < this->m_rows; ++i)
    {
        SLMatrixTuple<DT> *t_ptr = smat.m_row_headers[i];
        
        while (t_ptr)
        {
            SLMatrixTuple<DT> *t_ptr_cpy = new SLMatrixTuple<DT>{*t_ptr};
            new_item(*this, row_ref, col_ref, t_ptr_cpy);

            t_ptr = t_ptr->m_right;
        }
    }

    delete []row_ref;
    delete []col_ref;
}

template <typename DT>
julie::la::cpu::SLMatrix<DT>::SLMatrix(SLMatrix<DT> && smat)
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

template <typename DT>
julie::la::cpu::SLMatrix<DT>::~SLMatrix()
{
    this->release();
}

template <typename DT>
void julie::la::cpu::SLMatrix<DT>::release()
{
    std::vector<SLMatrixTuple<DT> *> t_ptr_list;

    for (lint i = 0; i < this->m_rows; ++i)
    {
        SLMatrixTuple<DT> *t_ptr = this->m_row_headers[i];
        
        while (t_ptr)
        {
            t_ptr_list.push_back(t_ptr);
            t_ptr = t_ptr->m_right;
        }
    }

    for (SLMatrixTuple<DT> * t_ptr : t_ptr_list)
    {
        delete t_ptr;
    }

    delete []this->m_row_headers;
    delete []this->m_col_headers;
}

template <typename DT>
julie::la::cpu::Matrix_CPU<DT> julie::la::cpu::SLMatrix<DT>::to_Matrix_CPU () const
{
    Shape d_shape {this->m_rows, this->m_cols};
    // std::cout << d_shape << std::endl;
    Matrix_CPU<DT> dmat {d_shape};

    Coordinate d_location {d_shape};
    auto d_itr = dmat.begin();

    for (lint i = 0; i < this->m_rows; ++i)
    {
        SLMatrixTuple<DT> *t_ptr = this->m_row_headers[i];
        
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

template <typename DT>
julie::la::cpu::SLMatrix<DT> & julie::la::cpu::SLMatrix<DT>::operator = (const SLMatrix<DT> & other)
{
    this->release();

    this->m_rows = other.m_rows;
    this->m_cols = other.m_cols;
    this->m_row_headers = new SLMatrixTuple<DT>*[this->m_rows] {nullptr};
    this->m_col_headers = new SLMatrixTuple<DT>*[this->m_cols] {nullptr};
    this->m_zero_threshold = other.m_zero_threshold;

    SLMatrixTuple<DT> ** row_ref = new SLMatrixTuple<DT>*[this->m_rows] {nullptr};
    SLMatrixTuple<DT> ** col_ref = new SLMatrixTuple<DT>*[this->m_cols] {nullptr};

    for (lint i = 0; i < this->m_rows; ++i)
    {
        SLMatrixTuple<DT> *t_ptr = other.m_row_headers[i];
        
        while (t_ptr)
        {
            SLMatrixTuple<DT> *t_ptr_cpy = new SLMatrixTuple<DT>{*t_ptr};
            new_item(*this, row_ref, col_ref, t_ptr_cpy);

            t_ptr = t_ptr->m_right;
        }
    }

    delete []row_ref;
    delete []col_ref;

    return *this;
}

template <typename DT>
julie::la::cpu::SLMatrix<DT> & julie::la::cpu::SLMatrix<DT>::operator = (SLMatrix<DT> && other)
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

template <typename DT>
julie::la::cpu::SLMatrix<DT> julie::la::cpu::SLMatrix<DT>::get_transpose() const
{
    Shape reversed_shape{ this->m_cols, this->m_rows };
    SLMatrix<DT> transposed{ reversed_shape, this->m_zero_threshold };

    SLMatrixTuple<DT> ** row_ref = new SLMatrixTuple<DT>*[transposed.m_rows] {nullptr};
    SLMatrixTuple<DT> ** col_ref = new SLMatrixTuple<DT>*[transposed.m_cols] {nullptr};

    for (lint i = 0; i < this->m_cols; ++i)
    {
        SLMatrixTuple<DT> *t_ptr = this->m_col_headers[i];
        
        while (t_ptr)
        {
            SLMatrixTuple<DT> *t_ptr_t = new SLMatrixTuple<DT>{t_ptr->m_col, t_ptr->m_row, t_ptr->m_val};
            new_item(transposed, row_ref, col_ref, t_ptr_t);

            t_ptr = t_ptr->m_down;
        }
    }

    delete []row_ref;
    delete []col_ref;

    return transposed;
}

template <typename DT>
julie::la::cpu::SLMatrix<DT> & julie::la::cpu::SLMatrix<DT>::operator += (const SLMatrix<DT> & other)
{
    SLMatrix<DT> sum = *this + other;
    *this = std::move(sum);

    return *this;
}

template <typename DT>
julie::la::cpu::SLMatrix<DT> & julie::la::cpu::SLMatrix<DT>::operator -= (const SLMatrix<DT> & other)
{
    SLMatrix<DT> sub = *this - other;
    *this = std::move(sub);

    return *this;
}

template <typename DT>
std::string julie::la::cpu::SLMatrix<DT>::to_string() const
{
    std::ostringstream stream;

    /*
    for (lint i = 0; i < this->m_rows; ++i)
    {
        SLMatrixTuple<DT> *t_ptr = this->m_row_headers[i];
        
        while (t_ptr)
        {
            stream << "[" << t_ptr->m_row << " " << t_ptr->m_col <<"]" << "\t" << t_ptr->m_val << "\n";
            t_ptr = t_ptr->m_right;
        }
    }
    */
    
    for (lint i = 0; i < this->m_cols; ++i)
    {
        SLMatrixTuple<DT> *t_ptr = this->m_col_headers[i];
        
        while (t_ptr)
        {
            stream << "[" << t_ptr->m_row << " " << t_ptr->m_col <<"]" << "\t" << t_ptr->m_val << "\n";
            t_ptr = t_ptr->m_down;
        }
    }


    return stream.str();
}

template <typename DT>
julie::la::Shape julie::la::cpu::SLMatrix<DT>::shape() const
{
    return Shape{this->m_rows, this->m_cols};
}

template <typename DT>
lint julie::la::cpu::SLMatrix<DT>::rows() const
{
    return this->m_rows;
}

template <typename DT>
lint julie::la::cpu::SLMatrix<DT>::columns() const
{
    return this->m_cols;
}

template <typename DT>
lint julie::la::cpu::SLMatrix<DT>::non_zero_items() const
{
    return this->m_non_zero_items;
}

template <typename DT>
inline void julie::la::cpu::SLMatrix<DT>::new_item(
    SLMatrix<DT> & mat,
    SLMatrixTuple<DT> ** row_ref,
    SLMatrixTuple<DT> ** col_ref,
    SLMatrixTuple<DT> *t_ptr)
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
