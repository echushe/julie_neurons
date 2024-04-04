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

#include "Matrix_CPU_func.hpp"
#include "utilities.hpp"
#include <type_traits>
#ifdef WITH_OPENBLAS
#include <cblas.h>
#endif

namespace julie
{
namespace la
{
namespace cpu
{

template <typename DT>
bool renew_if_shape_not_match(Matrix_CPU<DT> &cache, const Shape &sh)
{
    if (cache.m_shape.size() != sh.size())
    {
        cache = Matrix_CPU<DT> { sh };
        return true;
    }
    
    if (cache.m_shape != sh)
    {
        cache.reshape(sh);
    }

    return false;
}
template
bool renew_if_shape_not_match(Matrix_CPU<int> &cache, const Shape &sh);
template
bool renew_if_shape_not_match(Matrix_CPU<float> &cache, const Shape &sh);


// Overloading of a == b
template <typename DT>
bool operator == (const Matrix_CPU<DT> &left, const Matrix_CPU<DT> &right)
{
    if (left.m_shape != right.m_shape)
    {
        return false;
    }

    lint size = left.m_shape.size();

    for (lint i = 0; i < size; ++i)
    {
        if (left.m_data[i] != right.m_data[i])
        {
            return false;
        }
    }

    return true;
}
template
bool operator == (const Matrix_CPU<int> &left, const Matrix_CPU<int> &right);
template
bool operator == (const Matrix_CPU<float> &left, const Matrix_CPU<float> &right);

// Overloading of a != b
template <typename DT>
bool operator != (const Matrix_CPU<DT> &left, const Matrix_CPU<DT> &right)
{
    return !(left == right);
}
template
bool operator != (const Matrix_CPU<int> &left, const Matrix_CPU<int> &right);
template
bool operator != (const Matrix_CPU<float> &left, const Matrix_CPU<float> &right);


// operation of a + b
template <typename DT>
void add (Matrix_CPU<DT> & output, const Matrix_CPU<DT> &left, const Matrix_CPU<DT> &right)
{
    if (left.m_shape.size() != right.m_shape.size())
    {
        throw std::invalid_argument(invalid_shape + std::string(__FUNCTION__));
    }

    renew_if_shape_not_match(output, left.m_shape);

    lint size = left.m_shape.size();
    
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = left.m_data[i] + right.m_data[i];
    }
}
template
void add (Matrix_CPU<int> & output, const Matrix_CPU<int> &left, const Matrix_CPU<int> &right);
template
void add (Matrix_CPU<float> & output, const Matrix_CPU<float> &left, const Matrix_CPU<float> &right);


// Overloading of a - b
template <typename DT>
void subtract (Matrix_CPU<DT> &output, const Matrix_CPU<DT> &left, const Matrix_CPU<DT> &right)
{
    if (left.m_shape.size() != right.m_shape.size())
    {
        throw std::invalid_argument(invalid_shape + std::string(__FUNCTION__));
    }

    renew_if_shape_not_match(output, left.m_shape);

    lint size = left.m_shape.size();

    Matrix_CPU<DT> mat{ left.m_shape };
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = left.m_data[i] - right.m_data[i];
    }
}
template
void subtract (Matrix_CPU<int> &output, const Matrix_CPU<int> &left, const Matrix_CPU<int> &right);
template
void subtract (Matrix_CPU<float> &output, const Matrix_CPU<float> &left, const Matrix_CPU<float> &right);


// Broadcast mode of a + b
template <typename DT>
void broadcast_add (Matrix_CPU<DT> &output, const Matrix_CPU<DT> &left, const Matrix_CPU<DT> &right)
{
    lint big_size = left.m_shape.size();
    lint small_size = right.m_shape.size();

    if (big_size < small_size || big_size % small_size != 0)
    {
        throw std::invalid_argument(invalid_shape + std::string(__FUNCTION__));
    }

    renew_if_shape_not_match(output, left.m_shape);

    for (lint i = 0; i < big_size; ++i)
    {
        output.m_data[i] = left.m_data[i] + right.m_data[i % small_size];
    }
}
template
void broadcast_add (Matrix_CPU<int> &output, const Matrix_CPU<int> &left, const Matrix_CPU<int> &right);
template
void broadcast_add (Matrix_CPU<float> &output, const Matrix_CPU<float> &left, const Matrix_CPU<float> &right);

// Matrix_CPU multiplication
// Which dimensions should be merged together is manually defined here.
// For example, two matrices: [7, 3, 2, 5, 6] and [6, 5, 6, 4], if l_dims_merge == 2 and r_dims_merge == 2,
// the shape of result will be [7, 3, 2, 6, 4]. However, if l_dims_merge == 4 and r_dims_merge == 3,
// the shape of output will be [7, 4].
// Note: Total sizes (number of elements) of dimensions to merge should be equal betweem the left and right.
template <typename DT>
void matmul(Matrix_CPU<DT> &output, const Matrix_CPU<DT> &left, const Matrix_CPU<DT> &right, lint l_dims_merge, lint r_dims_merge)
{
    //
    if (left.m_shape.dim() < 2 || right.m_shape.dim() < 2)
    {
        throw std::invalid_argument(std::string(
            "Matrix_CPU multiplication does not allow matrices of less than 2 dimensions.\n Please extend dimensions of matrices first."));
    }

    if (l_dims_merge >= left.m_shape.dim() || r_dims_merge >= right.m_shape.dim())
    {
        throw std::invalid_argument(std::string(
            "Matrix_CPU multiplication does not allow dims to merge covering the entire matrix shape."));
    }

    // Check shape compatibilities between these 2 matrices
    Shape left_sh = left.m_shape;
    Shape right_sh = right.m_shape;

    Shape left_rows_sh = left_sh.sub_shape(0, left_sh.dim() - l_dims_merge - 1);
    Shape left_cols_sh = left_sh.sub_shape(left_sh.dim() - l_dims_merge, left_sh.dim() - 1);

    Shape right_rows_sh = right_sh.sub_shape(0, r_dims_merge - 1);
    Shape right_cols_sh = right_sh.sub_shape(r_dims_merge, right_sh.dim() - 1);

    lint left_columns = left_cols_sh.size();
    lint right_rows = right_rows_sh.size();

    if (left_columns != right_rows)
    {
        throw std::invalid_argument(incompatible_shape);
    }

    lint left_rows = left_rows_sh.size();
    lint right_columns = right_cols_sh.size();

    // Calculation
    renew_if_shape_not_match(output, left_rows_sh + right_cols_sh);

#ifdef WITH_OPENBLAS
    if (std::is_same<DT, double>::value)
    {
        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans, CblasNoTrans, 
            left_rows, right_columns, left_columns, 1.0, 
            reinterpret_cast<double*>(left.m_data), left_columns,
            reinterpret_cast<double*>(right.m_data), right_columns,
            0.0,
            reinterpret_cast<double*>(output.m_data), right_columns);
    }
    else if (std::is_same<DT, float>::value)
    {
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans, CblasNoTrans, 
            left_rows, right_columns, left_columns, 1.0, 
            reinterpret_cast<float*>(left.m_data), left_columns,
            reinterpret_cast<float*>(right.m_data), right_columns,
            0.0,
            reinterpret_cast<float*>(output.m_data), right_columns);
    }
    else
#endif
    {
        DT *left_start = left.m_data;
        DT *right_start = right.m_data;
        DT *output_p = output.m_data;

        for (lint i = 0; i < left_rows; ++i)
        {
            for (lint j = 0; j < right_columns; ++j)
            {
                *output_p = 0.0;
                DT * left_p = left_start;
                DT * right_p = right_start;

                for (lint k = 0; k < left_columns; ++k)
                {
                    *output_p += *left_p * *right_p;
                    ++left_p;
                    right_p += right_columns;
                }

                ++right_start;
                ++output_p;
            }

            left_start += left_columns;
            right_start = right.m_data;
        }
    }
}
template
void matmul(Matrix_CPU<int> &output, const Matrix_CPU<int> &left, const Matrix_CPU<int> &right, lint l_dims_merge, lint r_dims_merge);
template
void matmul(Matrix_CPU<float> &output, const Matrix_CPU<float> & left, const Matrix_CPU<float> & right, lint l_dims_merge, lint r_dims_merge);


// Matrix_CPU multiplication
// This function can automatically figure out which dimensions should be merged together.
// For example, two matrices: [7, 3, 3, 5, 4] and [10, 6, 6, 4]
// [3, 5, 4] of the left matrix and [10, 6] of the right matrix will be merged.
// then, the shape of result will be [7, 3, 6, 4].
// This function will throw out an exception if no appropriate dimensions to merge can be found.
template <typename DT>
void matmul(Matrix_CPU<DT> &output, const Matrix_CPU<DT> & left, const Matrix_CPU<DT> & right)
{
    if (left.m_shape.dim() < 2 || right.m_shape.dim() < 2)
    {
        throw std::invalid_argument(std::string(
            "Matrix_CPU multiplication does not allow matrices of less than 2 dimensions.\n Please extend dimensions of matrices first."));
    }

    Shape left_sh = left.m_shape;
    Shape right_sh = right.m_shape;

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

    if (!can_multiply)
    {
        throw std::invalid_argument(incompatible_shape);
    }

    matmul(output, left, right, left_cols_sh.dim(), right_rows_sh.dim());
}
template
void matmul(Matrix_CPU<int> &output, const Matrix_CPU<int> & left, const Matrix_CPU<int> & right);
template
void matmul(Matrix_CPU<float> &output, const Matrix_CPU<float> & left, const Matrix_CPU<float> & right);


// Multiplication element by element.
// The two matrices should have the same shape.
template <typename DT>
void multiply(Matrix_CPU<DT> &output, const Matrix_CPU<DT> & left, const Matrix_CPU<DT> & right)
{
    if (left.m_shape.size() != right.m_shape.size())
    {
        throw std::invalid_argument(invalid_shape + std::string(__FUNCTION__));
    }

    renew_if_shape_not_match(output, left.m_shape);

    lint size = left.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = left.m_data[i] * right.m_data[i];
    }
}
template
void multiply(Matrix_CPU<int> &output, const Matrix_CPU<int> & left, const Matrix_CPU<int> & right);
template
void multiply(Matrix_CPU<float> &output, const Matrix_CPU<float> & left, const Matrix_CPU<float> & right);


// Broadcast mode of a * b
template <typename DT>
void broadcast_multiply (Matrix_CPU<DT> &output, const Matrix_CPU<DT> &left, const Matrix_CPU<DT> &right)
{
    lint big_size = left.m_shape.size();
    lint small_size = right.m_shape.size();

    if (small_size == 0 || big_size < small_size || big_size % small_size != 0)
    {
        throw std::invalid_argument(invalid_shape + std::string(__FUNCTION__));
    }

    renew_if_shape_not_match(output, left.m_shape);

    for (lint i = 0; i < big_size; ++i)
    {
        output.m_data[i] = left.m_data[i] * right.m_data[i % small_size];
    }
}
template
void broadcast_multiply (Matrix_CPU<int> &output, const Matrix_CPU<int> &left, const Matrix_CPU<int> &right);
template
void broadcast_multiply (Matrix_CPU<float> &output, const Matrix_CPU<float> &left, const Matrix_CPU<float> &right);


// Dot product of two matrices
// The two matrices should have the same amount of elements
template <typename DT>
DT dot_product(Matrix_CPU<DT> &multiply_cache, const Matrix_CPU<DT> & left, const Matrix_CPU<DT> & right)
{
    if (left.m_shape.size() != right.m_shape.size())
    {
        throw std::invalid_argument(incompatible_size);
    }

    multiply(multiply_cache, left, right);

    return multiply_cache.sum();
}
template
int dot_product(Matrix_CPU<int> &multiply_cache, const Matrix_CPU<int> & left, const Matrix_CPU<int> & right);
template
float dot_product(Matrix_CPU<float> &multiply_cache, const Matrix_CPU<float> & left, const Matrix_CPU<float> & right);

// Overloading of a * b, b is a scalar
template <typename DT>
void multiply (Matrix_CPU<DT> &output, const Matrix_CPU<DT> &left, DT scalar)
{
    renew_if_shape_not_match(output, left.m_shape);

    lint size = left.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = left.m_data[i] * scalar;
    }
}
template
void multiply (Matrix_CPU<int> &output, const Matrix_CPU<int> &left, int scalar);
template
void multiply (Matrix_CPU<float> &output, const Matrix_CPU<float> &left, float scalar);

// Overloading of a / b, b is a scalar
template <typename DT>
void divide (Matrix_CPU<DT> &output, const Matrix_CPU<DT> &left, DT scalar)
{
    renew_if_shape_not_match(output, left.m_shape);

    lint size = left.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = left.m_data[i] / scalar;
    }
}
template
void divide (Matrix_CPU<int> &output, const Matrix_CPU<int> &left, int scalar);
template
void divide (Matrix_CPU<float> &output, const Matrix_CPU<float> &left, float scalar);

// Calculate power of a matrix
template <typename DT>
void matrix_pow(Matrix_CPU<DT> &output, const Matrix_CPU<DT> & mat, int n)
{
    if (n == 1)
    {
        output = mat;
    }
    else if (n > 1)
    {
        Matrix_CPU<DT> child;
        matrix_pow(child, mat, n / 2);

        if (n % 2 == 0)
        {
            multiply(output, child, child);
        }
        else
        {
            Matrix_CPU<DT> tmp;
            multiply(tmp, child, mat);
            multiply(output, child, tmp);
        }
    }
    else
    {
        output = Matrix_CPU<DT>{};
    }
}
template
void matrix_pow(Matrix_CPU<int> &output, const Matrix_CPU<int> & mat, int n);
template
void matrix_pow(Matrix_CPU<float> &output, const Matrix_CPU<float> & mat, int n);


// Get a fully transposed matrix of a matrix
template <typename DT>
void full_transpose(Matrix_CPU<DT> & transposed, const Matrix_CPU<DT> & in)
{
    Shape reversed_shape{ reverse(in.m_shape) };
    // Renew the transposed matrix cache if the shape does not match
    renew_if_shape_not_match(transposed, reversed_shape);

    lint size = in.m_shape.size();

    lint plus_pos;
    lint dim_size = reversed_shape.dim();
    lint *coord_cache = new lint[dim_size];
    lint *jump_forward_cache = new lint[dim_size];

    for (lint i = dim_size - 1; i >= 0; --i)
    {
        coord_cache[i] = 0;
        if (i < dim_size - 1)
        {
            jump_forward_cache[i] = jump_forward_cache[i + 1] * reversed_shape[i + 1];
        }
        else
        {
            jump_forward_cache[i] = 1;
        }
    }

    DT *ele_pos = transposed.m_data;

    for (lint i = 0; i < size; ++i)
    {
        *ele_pos = in.m_data[i];

        plus_pos = 0;
        while (plus_pos < dim_size)
        {
            lint increased = coord_cache[plus_pos] + 1;
            if (increased < reversed_shape[plus_pos])
            {
                coord_cache[plus_pos] = increased;
                ele_pos += jump_forward_cache[plus_pos];
                // std::cout << "forward: " << jump_forward_cache[plus_pos] << '\n';
                break;
            }
            else
            {
                coord_cache[plus_pos] = 0;
                ele_pos -= jump_forward_cache[plus_pos] * (reversed_shape[plus_pos] - 1);
                // std::cout << "backward: " << jump_forward_cache[plus_pos] * (reversed_shape[plus_pos] - 1) << '\n';
                ++plus_pos;
            }
        }
    }

    delete[]coord_cache;
    delete[]jump_forward_cache;
}
template
void full_transpose(Matrix_CPU<int> & transposed, const Matrix_CPU<int> & in);
template
void full_transpose(Matrix_CPU<float> & transposed, const Matrix_CPU<float> & in);


template <typename DT>
void transpose(Matrix_CPU<DT> & transposed, const Matrix_CPU<DT> & in, lint left_dims)
{
    if (left_dims < 0 || left_dims > in.m_shape.dim())
    {
        throw std::invalid_argument(std::string("invalid size of left dimensions for matrix transpose."));
    }

    if (left_dims == 0 || left_dims == in.m_shape.dim())
    {
        transposed = in;
        return;
    }

    Shape left_sh = in.m_shape.sub_shape(0, left_dims - 1);
    Shape right_sh = in.m_shape.sub_shape(left_dims, in.m_shape.dim() - 1);

    lint left_size = left_sh.size();
    lint right_size = right_sh.size();
    
    // Renew the transposed matrix cache if the shape does not match
    renew_if_shape_not_match(transposed, right_sh + left_sh);

    DT *in_pos = in.m_data;
    DT *t_first_row_col = transposed.m_data;

    for (lint i = 0; i < left_size; ++i)
    {
        DT *t_pos = t_first_row_col;

        for (lint j = 0; j < right_size; ++j)
        {
            *t_pos = *in_pos;

            ++in_pos;
            t_pos += left_size;
        }

        ++t_first_row_col;
    }
}
template
void transpose(Matrix_CPU<int> & transposed, const Matrix_CPU<int> & in, lint left_dims);
template
void transpose(Matrix_CPU<float> & transposed, const Matrix_CPU<float> & in, lint left_dims);


template <typename DT>
void transpose_neighboring_dims(Matrix_CPU<DT> & transposed, const Matrix_CPU<DT> & in,
                                    lint l_dim_idx1, lint l_dim_idx2, lint r_dim_idx1, lint r_dim_idx2)
{
    if (l_dim_idx1 > l_dim_idx2 || r_dim_idx1 > r_dim_idx2 || r_dim_idx1 - l_dim_idx2 != 1 ||
        l_dim_idx1 < 0 || l_dim_idx1 >= in.m_shape.dim())
    {
        throw std::invalid_argument(std::string("dim index should be within valid range in matrix transpose_neighboring_dims()"));
    }

    Shape left_sh = in.m_shape.sub_shape(0, l_dim_idx1 - 1);
    Shape l_sh = in.m_shape.sub_shape(l_dim_idx1, l_dim_idx2);
    Shape r_sh = in.m_shape.sub_shape(r_dim_idx1, r_dim_idx2);
    Shape right_sh = in.m_shape.sub_shape(r_dim_idx2 + 1, in.m_shape.dim() - 1);

    lint left_size = std::max<lint>(left_sh.size(), 1);
    lint l_size = l_sh.size();
    lint r_size = r_sh.size();
    lint right_size = std::max<lint>(right_sh.size(), 1);

    // Renew the transposed matrix cache if the shape does not match
    renew_if_shape_not_match(transposed, left_sh + r_sh + l_sh + right_sh);

    lint left_sh_pos = 0;
    lint left_sh_jump = l_size * r_size * right_size;
    
    lint l_sh_jump = r_size * right_size;
    lint r_trans_sh_jump = l_size * right_size;

    for (lint i = 0; i < left_size; ++i)
    {
        lint  l_sh_pos = left_sh_pos;

        for (lint j = 0; j < l_size; ++j)
        {
            lint r_sh_pos = l_sh_pos;

            for (lint k = 0; k < r_size; ++k)
            {
                lint pos = r_sh_pos;

                for (lint m = 0; m < right_size; ++m)
                {
                    transposed.m_data[left_sh_pos + k * r_trans_sh_jump + j * right_size + m] = in.m_data[pos];
                    ++pos;
                }

                r_sh_pos += right_size;
            }

            l_sh_pos += l_sh_jump;
        }

        left_sh_pos += left_sh_jump;
    }
}
template
void transpose_neighboring_dims(Matrix_CPU<int> & transposed, const Matrix_CPU<int> & in,
                                    lint l_dim_idx1, lint l_dim_idx2, lint r_dim_idx1, lint r_dim_idx2);
template
void transpose_neighboring_dims(Matrix_CPU<float> & transposed, const Matrix_CPU<float> & in,
                                    lint l_dim_idx1, lint l_dim_idx2, lint r_dim_idx1, lint r_dim_idx2);


// This method is to get a matrix in which all members are absolute values of the input matrix
// Arguments:
//     output: The matrix whose memebers are absolute values of the input matrix
//     input:  The input matrix
template <typename DT>
void abs(Matrix_CPU<DT> &output, const Matrix_CPU<DT> &input)
{
    renew_if_shape_not_match(output, input.m_shape);

    lint size = input.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        if (input.m_data[i] < 0)
        {
            output.m_data[i] = input.m_data[i] * (-1);
        }
        else
        {
            output.m_data[i] = input.m_data[i];
        }
    }
}
template
void abs(Matrix_CPU<int> &output, const Matrix_CPU<int> &input);
template
void abs(Matrix_CPU<float> &output, const Matrix_CPU<float> &input);


template <typename DT>
void threshold(Matrix_CPU<DT> &mask, const Matrix_CPU<DT> &in, DT th)
{
    // Renew the mask if shape does not match
    renew_if_shape_not_match(mask, in.m_shape);

    lint size = in.m_shape.size();

    for (lint i = 0; i < size; ++i)
    {
        if (in.m_data[i] < th)
        {
            mask.m_data[i] = 0;
        }
        else
        {
            mask.m_data[i] = 1;
        }
    }
}
template
void threshold(Matrix_CPU<int> &mask, const Matrix_CPU<int> &in, int th);
template
void threshold(Matrix_CPU<float> &mask, const Matrix_CPU<float> &in, float th);


template <typename DT>
void abs_threshold(Matrix_CPU<DT> &mask, const Matrix_CPU<DT> &in, DT th)
{
    if (th < 0)
    {
        th *= -1;
    }
    // Renew the mask if shape does not match
    renew_if_shape_not_match(mask, in.m_shape);

    lint size = in.m_shape.size();

    for (lint i = 0; i < size; ++i)
    {
        if (in.m_data[i] < th && in.m_data[i] > -th)
        {
            mask.m_data[i] = 0;
        }
        else
        {
            mask.m_data[i] = 1;
        }
    }
}
template
void abs_threshold(Matrix_CPU<int> &mask, const Matrix_CPU<int> &in, int th);
template
void abs_threshold(Matrix_CPU<float> &mask, const Matrix_CPU<float> &in, float th);


template <typename DT>
void concatenate(Matrix_CPU<DT> &output, const Matrix_CPU<DT> &mat1, const Matrix_CPU<DT> &mat2, lint dim)
{
    if (mat1.m_shape.dim() != mat2.m_shape.dim())
    {
        throw std::invalid_argument(std::string("Number of dimensions for matrices to concatenate are not the same but they should be the same."));
    }

    if (dim < 0 || dim >= mat1.m_shape.dim())
    {
        throw std::invalid_argument(std::string("The dimension index for concatenation is out of range."));
    }

    for (lint i = 0; i < mat1.m_shape.dim(); ++i)
    {
        if (i != dim && mat1.m_shape[i] != mat2.m_shape[i])
        {
            throw std::invalid_argument(std::string("Dimension sizes should be same except the dimension to concatenate for concatenation."));
        }
    }

    Shape left_sh = mat1.m_shape.sub_shape(0, dim - 1);
    Shape right_sh = mat1.m_shape.sub_shape(dim + 1, mat1.m_shape.dim() - 1);

    lint left_size = std::max<lint>(left_sh.size(), 1);
    lint right_size = std::max<lint>(right_sh.size(), 1);

    lint cat1_size = mat1.m_shape[dim];
    lint cat2_size = mat2.m_shape[dim];

    lint cat1_right_size = cat1_size * right_size;
    lint cat2_right_size = cat2_size * right_size;
    lint cat1_cat2_right_size = cat1_right_size + cat2_right_size;

    //std::cout << left_sh + Shape{cat1_size + cat2_size} + right_sh << std::endl;

    renew_if_shape_not_match(output, left_sh + Shape{cat1_size + cat2_size} + right_sh);

    DT *mat1_cat_pos = mat1.m_data;
    DT *mat2_cat_pos = mat2.m_data;
    DT *output_cat_pos = output.m_data;

    for (lint l_idx = 0; l_idx < left_size; ++l_idx)
    {
        DT *mat1_pos = mat1_cat_pos;
        DT *mat2_pos = mat2_cat_pos;
        DT *output_pos = output_cat_pos;

        for (lint cat1_idx = 0; cat1_idx < cat1_right_size; ++cat1_idx)
        {
            output_pos[cat1_idx] = mat1_pos[cat1_idx];
        }

        output_pos += cat1_right_size;

        for (lint cat2_idx = 0; cat2_idx < cat2_right_size; ++cat2_idx)
        {
            output_pos[cat2_idx] = mat2_pos[cat2_idx];
        }
        
        mat1_cat_pos += cat1_right_size;
        mat2_cat_pos += cat2_right_size;
        output_cat_pos += cat1_cat2_right_size;
    }

    //std::cout << output << std::endl;
}
template
void concatenate(Matrix_CPU<int> &output, const Matrix_CPU<int> &left, const Matrix_CPU<int> &right, lint dim);
template
void concatenate(Matrix_CPU<float> &output, const Matrix_CPU<float> &left, const Matrix_CPU<float> &right, lint dim);


template <typename DT>
void slice(Matrix_CPU<DT> &output, const Matrix_CPU<DT> &input, lint dim, lint idx, lint slice_size)
{
    if (dim < 0 || dim >= input.m_shape.dim())
    {
        throw std::invalid_argument(std::string("The dimension index for slice is out of range."));
    }

    if (idx < 0 || idx >= input.m_shape[dim])
    {
        throw std::invalid_argument(std::string("The start index for slice is out of range."));
    }

    if (idx + slice_size > input.m_shape[dim])
    {
        throw std::invalid_argument(std::string("The end point for slice is out of range."));
    }

    Shape left_sh = input.m_shape.sub_shape(0, dim - 1);
    Shape right_sh = input.m_shape.sub_shape(dim + 1, input.m_shape.dim() - 1);

    lint left_size = std::max<lint>(left_sh.size(), 1);
    lint right_size = std::max<lint>(right_sh.size(), 1);

    lint input_right_size = input.m_shape[dim] * right_size;
    lint output_right_size = slice_size * right_size;
    lint shift = idx * right_size;

    renew_if_shape_not_match(output, left_sh + Shape{slice_size} + right_sh);

    DT *input_slice_pos = input.m_data;
    DT *output_slice_pos = output.m_data;

    for (lint l_idx = 0; l_idx < left_size; ++l_idx)
    {
        DT *input_pos = input_slice_pos;
        DT *output_pos = output_slice_pos;

        for (lint slice_idx = 0; slice_idx < output_right_size; ++slice_idx)
        {
            output_pos[slice_idx] = input_pos[shift + slice_idx];
        }
        
        input_slice_pos += input_right_size;
        output_slice_pos += output_right_size;
    }
}
template
void slice(Matrix_CPU<int> &output, const Matrix_CPU<int> &input, lint dim, lint idx, lint slice_size);
template
void slice(Matrix_CPU<float> &output, const Matrix_CPU<float> &input, lint dim, lint idx, lint slice_size);


template <typename DT>
void repeat(Matrix_CPU<DT> &output, const Matrix_CPU<DT> &input, lint dim, lint n_duplicate)
{
    if (dim < 0 || dim >= input.m_shape.dim())
    {
        throw std::invalid_argument(std::string("The dimension index is out of range in " + std::string(__FUNCTION__)));
    }

    if (n_duplicate < 1)
    {
        throw std::invalid_argument(std::string("Number of duplicates should be a positive value in " + std::string(__FUNCTION__)));
    }

    Shape left_sh = input.m_shape.sub_shape(0, dim - 1);
    Shape right_sh = input.m_shape.sub_shape(dim + 1, input.m_shape.dim() - 1);

    lint left_size = std::max<lint>(left_sh.size(), 1);
    lint right_size = std::max<lint>(right_sh.size(), 1);

    lint input_re_size = input.m_shape[dim] * right_size;
    lint output_re_size = input.m_shape[dim] * n_duplicate * right_size;


    renew_if_shape_not_match(output, left_sh + Shape{input.m_shape[dim] * n_duplicate} + right_sh);

    DT *input_re_pos = input.m_data;
    DT *output_re_pos = output.m_data;

    for (lint l_idx = 0; l_idx < left_size; ++l_idx)
    {
        DT *input_pos = input_re_pos;
        DT *output_pos = output_re_pos;

        for (lint re_idx = 0; re_idx < n_duplicate; ++re_idx)
        {
            for (lint idx = 0; idx < input_re_size; ++idx)
            {
                output_pos[idx] = input_pos[idx];
            }

            input_pos = input_re_pos;
            output_pos += input_re_size;
        }

        input_re_pos += input_re_size;
        output_re_pos += output_re_size;
    }
}
template
void repeat(Matrix_CPU<int> &output, const Matrix_CPU<int> &input, lint dim, lint n_duplicate);
template
void repeat(Matrix_CPU<float> &output, const Matrix_CPU<float> &input, lint dim, lint n_duplicate);


/* Scale one dimension of the matrix with a vector of scales
For example, if a two dimensional matrix A is like this:

| a11, a12, a13 |
| a21, a22, a23 |
| a31, a32, a33 |
| a41, a42, a43 |

And vector B is [ b1, b2, b3 ], vector C is [ c1, c2, c3, c4 ]
Then the result of A.scale_one_dimension(1, B) is:

| a11 * b1, a12 * b2, a13 * b3 |
| a21 * b1, a22 * b2, a23 * b3 |
| a31 * b1, a32 * b2, a33 * b3 |
| a41 * b1, a42 * b2, a43 * b3 |

Then the result of A.scale_one_dimension(0, C) is:

| a11 * c1, a12 * c1, a13 * c1 |
| a21 * c2, a22 * c2, a23 * c2 |
| a31 * c3, a32 * c3, a33 * c3 |
| a41 * c4, a42 * c4, a43 * c4 |

It seems this kind of calculation is needed in back-propagation.
Perhaps there is a better way to name this function.
*/
template <typename DT>
Matrix_CPU<DT> scale_one_dimension(const Matrix_CPU<DT> & in, lint in_dim, const std::vector<DT> & scales)
{
    Matrix_CPU<DT> mat{ in };
    mat.scale_one_dimension(in_dim, scales);
    return mat;
}
template
Matrix_CPU<int> scale_one_dimension(const Matrix_CPU<int> & in, lint in_dim, const std::vector<int> & scales);
template
Matrix_CPU<float> scale_one_dimension(const Matrix_CPU<float> & in, lint in_dim, const std::vector<float> & scales);

} // namespace cpu
} // namespace la
} // namespace julie