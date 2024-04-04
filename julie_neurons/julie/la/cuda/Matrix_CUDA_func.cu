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

#include "Matrix_CUDA_func.hpp"
#include "Matrix_CPU.hpp"
#include "Matrix_CPU_func.hpp"
#include "utilities.hpp"
#include "add.cuh"
#include "subtract.cuh"
#include "multiply.cuh"
#include "divide.cuh"
#include "transpose.cuh"
#include "matmul.cuh"
#include "concat.cuh"
#include "act.cuh"
#include "utilities.hpp"
// #include <cuda_runtime.h>
#include "cublas_v2.h"

namespace julie
{
namespace la
{
namespace cuda
{

static cublasHandle_t CUBLAS_WORKSPACE = nullptr;

template <typename DT>
bool renew_if_shape_not_match(Matrix_CUDA<DT> &cache, const Shape &sh)
{
    if (cache.m_shape.size() != sh.size())
    {
        cache = Matrix_CUDA<DT> { sh };
        return true;
    }
    
    if (cache.m_shape != sh)
    {
        cache.reshape(sh);
    }

    return false;
}
template
bool renew_if_shape_not_match(Matrix_CUDA<int> &cache, const Shape &sh);
template
bool renew_if_shape_not_match(Matrix_CUDA<float> &cache, const Shape &sh);


// Overloading of a == b
template <typename DT>
bool operator == (const Matrix_CUDA<DT> &left, const Matrix_CUDA<DT> &right)
{
    if (left.m_shape != right.m_shape)
    {
        return false;
    }

    // Convert to cpu matrix and then compare them
    return julie::la::cpu::Matrix_CPU<DT> {left} == julie::la::cpu::Matrix_CPU<DT> {right};
}
template
bool operator == (const Matrix_CUDA<int> &left, const Matrix_CUDA<int> &right);
template
bool operator == (const Matrix_CUDA<float> &left, const Matrix_CUDA<float> &right);


// Overloading of a != b
template <typename DT>
bool operator != (const Matrix_CUDA<DT> &left, const Matrix_CUDA<DT> &right)
{
    return !(left == right);
}
template
bool operator != (const Matrix_CUDA<int> &left, const Matrix_CUDA<int> &right);
template
bool operator != (const Matrix_CUDA<float> &left, const Matrix_CUDA<float> &right);


// operation of a + b
template <typename DT>
void add (Matrix_CUDA<DT> & output, const Matrix_CUDA<DT> &left, const Matrix_CUDA<DT> &right)
{
    if (left.m_shape.size() != right.m_shape.size())
    {
        throw std::invalid_argument(invalid_shape + std::string(__FUNCTION__));
    }

    renew_if_shape_not_match(output, left.m_shape);

    lint size = left.m_shape.size();
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);
        // TODO: CUDA kernel
        __add_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, left.m_data, right.m_data, size);
    }
}
template
void add (Matrix_CUDA<int> &output, const Matrix_CUDA<int> &left, const Matrix_CUDA<int> &right);
template
void add (Matrix_CUDA<float> &output, const Matrix_CUDA<float> &left, const Matrix_CUDA<float> &right);


// Overloading of a - b
template <typename DT>
void subtract (Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &left, const Matrix_CUDA<DT> &right)
{
    if (left.m_shape.size() != right.m_shape.size())
    {
        throw std::invalid_argument(invalid_shape + std::string(__FUNCTION__));
    }

    renew_if_shape_not_match(output, left.m_shape);

    lint size = left.m_shape.size();
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);
        // TODO: CUDA kernel
        __subtract_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, left.m_data, right.m_data, size);
    }
}
template
void subtract (Matrix_CUDA<int> &output, const Matrix_CUDA<int> &left, const Matrix_CUDA<int> &right);
template
void subtract (Matrix_CUDA<float> &output, const Matrix_CUDA<float> &left, const Matrix_CUDA<float> &right);


// Broadcast mode of a + b
template <typename DT>
void broadcast_add (Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &left, const Matrix_CUDA<DT> &right)
{
    lint big_size = left.m_shape.size();
    lint small_size = right.m_shape.size();

    if (small_size == 0 || big_size < small_size || big_size % small_size != 0)
    {
        throw std::invalid_argument(invalid_shape + std::string(__FUNCTION__));
    }

    renew_if_shape_not_match(output, left.m_shape);

    if (big_size > 0)
    {
        lint n_blocks = std::min(big_size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);
        // TODO: CUDA kernel
        __broadcast_add_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, left.m_data, right.m_data, big_size, small_size);
    }
}
template
void broadcast_add (Matrix_CUDA<int> &output, const Matrix_CUDA<int> &left, const Matrix_CUDA<int> &right);
template
void broadcast_add (Matrix_CUDA<float> &output, const Matrix_CUDA<float> &left, const Matrix_CUDA<float> &right);


// Matrix_CUDA multiplication
// Which dimensions should be merged together is manually defined here.
// For example, two matrices: [7, 3, 2, 5, 6] and [6, 5, 6, 4], if l_dims_merge == 2 and r_dims_merge == 2,
// the shape of result will be [7, 3, 2, 6, 4]. However, if l_dims_merge == 4 and r_dims_merge == 3,
// the shape of output will be [7, 4].
// Note: Total sizes (number of elements) of dimensions to merge should be equal betweem the left and right.
template <typename DT>
void matmul(Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &left, const Matrix_CUDA<DT> &right, lint l_dims_merge, lint r_dims_merge)
{
    //
    if (left.m_shape.dim() < 2 || right.m_shape.dim() < 2)
    {
        throw std::invalid_argument(std::string(
            "Matrix_CUDA multiplication does not allow matrices of less than 2 dimensions.\n Please extend dimensions of matrices first."));
    }

    if (l_dims_merge >= left.m_shape.dim() || r_dims_merge >= right.m_shape.dim())
    {
        throw std::invalid_argument(std::string(
            "Matrix_CUDA multiplication does not allow dims to merge covering the entire matrix shape."));
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

    if (left_rows * left_columns * right_columns <= 0)
    {
        return;
    }

    if (std::is_same<DT, double>::value)
    {
        double alpha = 1.0;
        double beta = 0.0;

        if (!CUBLAS_WORKSPACE)
        {
            cublasCreate(&CUBLAS_WORKSPACE);
        }

        cublasDgemm(
            CUBLAS_WORKSPACE,
            CUBLAS_OP_N, CUBLAS_OP_N,
            right_columns, left_rows, left_columns,
            &alpha,
            reinterpret_cast<double*>(right.m_data), right_columns,
            reinterpret_cast<double*>(left.m_data), left_columns,
            &beta,
            reinterpret_cast<double*>(output.m_data), right_columns);
        
        //cublasDestroy(handle);
    }
    else if (std::is_same<DT, float>::value)
    {
        float alpha = 1.0;
        float beta = 0.0;

        if (!CUBLAS_WORKSPACE)
        {
            cublasCreate(&CUBLAS_WORKSPACE);
        }

        cublasSgemm(
            CUBLAS_WORKSPACE,
            CUBLAS_OP_N, CUBLAS_OP_N,
            right_columns, left_rows, left_columns,
            &alpha,
            reinterpret_cast<float*>(right.m_data), right_columns,
            reinterpret_cast<float*>(left.m_data), left_columns,
            &beta,
            reinterpret_cast<float*>(output.m_data), right_columns);

        //cublasDestroy(handle);
    }
    else
    {
        lint n_blocks = std::min(left_rows * right_columns / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);
        // TODO: CUDA kernel
        __matmul_2d(output.m_data, left.m_data, right.m_data, n_blocks, BLOCK_WIDTH_1D, left_rows, left_columns, right_columns);
    }
}
template
void matmul(
    Matrix_CUDA<int> &output, const Matrix_CUDA<int> &left, const Matrix_CUDA<int> &right, lint l_dims_merge, lint r_dims_merge);
template
void matmul(
    Matrix_CUDA<float> &output, const Matrix_CUDA<float> &left, const Matrix_CUDA<float> &right, lint l_dims_merge, lint r_dims_merge);



// Matrix_CUDA multiplication
// This function can automatically figure out which dimensions should be merged together.
// For example, two matrices: [7, 3, 3, 5, 4] and [10, 6, 6, 4]
// [3, 5, 4] of the left matrix and [10, 6] of the right matrix will be merged.
// then, the shape of result will be [7, 3, 6, 4].
// This function will throw out an exception if no appropriate dimensions to merge can be found.
template <typename DT>
void matmul(Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &left, const Matrix_CUDA<DT> &right)
{

    if (left.m_shape.dim() < 2 || right.m_shape.dim() < 2)
    {
        throw std::invalid_argument(std::string(
            "Matrix_CUDA multiplication does not allow matrices of less than 2 dimensions.\n Please extend dimensions of matrices first."));
    }

    // Check shape compatibilities between these 2 matrices

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
void matmul(Matrix_CUDA<int> &output, const Matrix_CUDA<int> &left, const Matrix_CUDA<int> &right);
template
void matmul(Matrix_CUDA<float> &output, const Matrix_CUDA<float> &left, const Matrix_CUDA<float> &right);


// Multiplication element by element.
// The two matrices should have the same shape.
template <typename DT>
void multiply(Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> & left, const Matrix_CUDA<DT> & right)
{
    if (left.m_shape.size() != right.m_shape.size())
    {
        throw std::invalid_argument(invalid_shape + std::string(__FUNCTION__));
    }

    renew_if_shape_not_match(output, left.m_shape);

    lint size = left.m_shape.size();
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);
        // TODO: CUDA kernel
        __multiply_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, left.m_data, right.m_data, size);
    }
}
template
void multiply(Matrix_CUDA<int> &output, const Matrix_CUDA<int> & left, const Matrix_CUDA<int> & right);
template
void multiply(Matrix_CUDA<float> &output, const Matrix_CUDA<float> & left, const Matrix_CUDA<float> & right);


// Broadcast mode of a * b
template <typename DT>
void broadcast_multiply (Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &left, const Matrix_CUDA<DT> &right)
{
    lint big_size = left.m_shape.size();
    lint small_size = right.m_shape.size();

    if (small_size == 0 || big_size < small_size || big_size % small_size != 0)
    {
        throw std::invalid_argument(invalid_shape + std::string(__FUNCTION__));
    }

    renew_if_shape_not_match(output, left.m_shape);

    if (big_size > 0)
    {
        lint n_blocks = std::min(big_size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);
        // TODO: CUDA kernel
        __broadcast_multiply_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, left.m_data, right.m_data, big_size, small_size);
    }
}
template
void broadcast_multiply (Matrix_CUDA<int> &output, const Matrix_CUDA<int> &left, const Matrix_CUDA<int> &right);
template
void broadcast_multiply (Matrix_CUDA<float> &output, const Matrix_CUDA<float> &left, const Matrix_CUDA<float> &right);


// Dot product of two matrices
// The two matrices should have the same amount of elements
template <typename DT>
DT dot_product(Matrix_CUDA<DT> &multiply_cache, const Matrix_CUDA<DT> & left, const Matrix_CUDA<DT> & right)
{
    if (left.m_shape.size() != right.m_shape.size())
    {
        throw std::invalid_argument(incompatible_size);
    }

    multiply(multiply_cache, left, right);

    return multiply_cache.sum();
}
template
int dot_product(Matrix_CUDA<int> &multiply_cache, const Matrix_CUDA<int> & left, const Matrix_CUDA<int> & right);
template
float dot_product(Matrix_CUDA<float> &multiply_cache, const Matrix_CUDA<float> & left, const Matrix_CUDA<float> & right);


// Overloading of a * b, b is a scalar
template <typename DT>
void multiply (Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &left, DT scalar)
{
    renew_if_shape_not_match(output, left.m_shape);

    lint size = left.m_shape.size();

    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);
        // TODO: CUDA kernel
        __multiply_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, left.m_data, scalar, size);
    }
}
template
void multiply (Matrix_CUDA<int> &output, const Matrix_CUDA<int> &left, int scalar);
template
void multiply (Matrix_CUDA<float> &output, const Matrix_CUDA<float> &left, float scalar);


// Overloading of a / b, b is a scalar
template <typename DT>
void divide (Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &left, DT scalar)
{
    renew_if_shape_not_match(output, left.m_shape);

    lint size = left.m_shape.size();

    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);
        // TODO: CUDA kernel
        __divide_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, left.m_data, scalar, size);
    }
}
template
void divide (Matrix_CUDA<int> &output, const Matrix_CUDA<int> &left, int scalar);
template
void divide (Matrix_CUDA<float> &output, const Matrix_CUDA<float> &left, float scalar);


// Calculate power of a matrix
template <typename DT>
void matrix_pow(Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> & mat, int n)
{
    if (n == 1)
    {
        output = mat;
    }
    else if (n > 1)
    {
        Matrix_CUDA<DT> child;
        matrix_pow(child, mat, n / 2);

        if (n % 2 == 0)
        {
            multiply(output, child, child);
        }
        else
        {
            Matrix_CUDA<DT> tmp;
            multiply(tmp, child, mat);
            multiply(output, child, tmp);
        }
    }
    else
    {
        output = Matrix_CUDA<DT>{};
    }
}
template
void matrix_pow(Matrix_CUDA<int> &output, const Matrix_CUDA<int> & mat, int n);
template
void matrix_pow(Matrix_CUDA<float> &output, const Matrix_CUDA<float> & mat, int n);


// Get a fully transposed matrix of a matrix
template <typename DT>
void full_transpose(Matrix_CUDA<DT> & transposed, const Matrix_CUDA<DT> & in)
{
    julie::la::cpu::Matrix_CPU<DT> cpu_in {in};
    julie::la::cpu::Matrix_CPU<DT> cpu_trans;

    julie::la::cpu::full_transpose(cpu_trans, cpu_in);

    transposed = cpu_trans.get_CUDA();
}
template
void full_transpose(Matrix_CUDA<int> & transposed, const Matrix_CUDA<int> & in);
template
void full_transpose(Matrix_CUDA<float> & transposed, const Matrix_CUDA<float> & in);


template <typename DT>
void transpose(Matrix_CUDA<DT> & transposed, const Matrix_CUDA<DT> & in, lint left_dims)
{
    if (left_dims < 0 || left_dims > in.m_shape.dim())
    {
        throw std::invalid_argument(std::string("invalid size of left dimensions for matrix transpose."));
    }

    if (left_dims == 0 || left_dims == in.m_shape.dim())
    {
        transposed = in;
    }

    Shape left_sh = in.m_shape.sub_shape(0, left_dims - 1);
    Shape right_sh = in.m_shape.sub_shape(left_dims, in.m_shape.dim() - 1);

    lint left_size = left_sh.size();
    lint right_size = right_sh.size();

    // Renew the transposed matrix cache if the shape does not match
    renew_if_shape_not_match(transposed, right_sh + left_sh);

    lint size = in.m_shape.size();

    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);
        // TODO: CUDA kernel
        __transpose_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(transposed.m_data, in.m_data, left_size, right_size);
    }
}
template
void transpose(Matrix_CUDA<int> & transposed, const Matrix_CUDA<int> & in, lint left_dims);
template
void transpose(Matrix_CUDA<float> & transposed, const Matrix_CUDA<float> & in, lint left_dims);


template <typename DT>
void transpose_neighboring_dims(Matrix_CUDA<DT> & transposed, const Matrix_CUDA<DT> & in,
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

    lint size = in.m_shape.size();

    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);
        // TODO: CUDA kernel
        __transpose_neighboring_dim_pair_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(transposed.m_data, in.m_data, left_size, l_size, r_size, right_size);
    }
}
template
void transpose_neighboring_dims(Matrix_CUDA<int> & transposed, const Matrix_CUDA<int> & in,
                                            lint l_dim_idx1, lint l_dim_idx2, lint r_dim_idx1, lint r_dim_idx2);
template
void transpose_neighboring_dims(Matrix_CUDA<float> & transposed, const Matrix_CUDA<float> & in,
                                            lint l_dim_idx1, lint l_dim_idx2, lint r_dim_idx1, lint r_dim_idx2);


// This method is to get a matrix in which all members are absolute values of the input matrix
// Arguments:
//     output: The matrix whose memebers are absolute values of the input matrix
//     input:  The input matrix
template <typename DT>
void abs(Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &input)
{
    renew_if_shape_not_match(output, input.m_shape);

    lint size = input.m_shape.size();

    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);
        // TODO: CUDA kernel
        __act_abs_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, input.m_data, size);
    }
}
template
void abs(Matrix_CUDA<int> &output, const Matrix_CUDA<int> &input);
template
void abs(Matrix_CUDA<float> &output, const Matrix_CUDA<float> &input);


template <typename DT>
void threshold(Matrix_CUDA<DT> &mask, const Matrix_CUDA<DT> &in, DT th)
{
    // Renew the mask if shape does not match
    renew_if_shape_not_match(mask, in.m_shape);

    lint size = in.m_shape.size();
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);
        // TODO: CUDA kernel
        __threshold_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(mask.m_data, in.m_data, th, size);
    }
}
template
void threshold(Matrix_CUDA<int> &mask, const Matrix_CUDA<int> &in, int th);
template
void threshold(Matrix_CUDA<float> &mask, const Matrix_CUDA<float> &in, float th);


template <typename DT>
void abs_threshold(Matrix_CUDA<DT> &mask, const Matrix_CUDA<DT> &in, DT th)
{
    if (th < 0)
    {
        th *= -1;
    }
    // Renew the mask if shape does not match
    renew_if_shape_not_match(mask, in.m_shape);

    lint size = in.m_shape.size();
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);
        // TODO: CUDA kernel
        __abs_threshold_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(mask.m_data, in.m_data, th, size);
    }
}
template
void abs_threshold(Matrix_CUDA<int> &mask, const Matrix_CUDA<int> &in, int th);
template
void abs_threshold(Matrix_CUDA<float> &mask, const Matrix_CUDA<float> &in, float th);


template <typename DT>
void concatenate(Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &mat1, const Matrix_CUDA<DT> &mat2, lint dim)
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
    lint cat1_cat2_right_size = (cat1_size + cat2_size) * right_size;

    renew_if_shape_not_match(output, left_sh + Shape{cat1_size + cat2_size} + right_sh);

    lint size = output.m_shape.size();
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);
        // TODO: CUDA kernel
        __concat_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, mat1.m_data, mat2.m_data, 
                                                    cat1_right_size, cat2_right_size, cat1_cat2_right_size, size);
    }
}
template
void concatenate(Matrix_CUDA<int> &output, const Matrix_CUDA<int> &left, const Matrix_CUDA<int> &right, lint dim);
template
void concatenate(Matrix_CUDA<float> &output, const Matrix_CUDA<float> &left, const Matrix_CUDA<float> &right, lint dim);


template <typename DT>
void slice(Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &input, lint dim, lint idx, lint slice_size)
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

    lint size = output.m_shape.size();
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);
        // TODO: CUDA kernel
        __slice_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, input.m_data, shift, input_right_size, output_right_size, size);
    }
}
template
void slice(Matrix_CUDA<int> &output, const Matrix_CUDA<int> &input, lint dim, lint idx, lint slice_size);
template
void slice(Matrix_CUDA<float> &output, const Matrix_CUDA<float> &input, lint dim, lint idx, lint slice_size);


template <typename DT>
void repeat(Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &input, lint dim, lint n_duplicate)
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

    lint size = output.m_shape.size();
    if (size > 0)
    {
        lint n_blocks = std::min(size / BLOCK_WIDTH_1D + 1, MAX_N_BLOCK_1D);
        // TODO: CUDA kernel
        __repeat_1d<<<n_blocks, BLOCK_WIDTH_1D>>>(output.m_data, input.m_data, input_re_size, output_re_size, size);
    }
}
template
void repeat(Matrix_CUDA<int> &output, const Matrix_CUDA<int> &input, lint dim, lint n_duplicate);
template
void repeat(Matrix_CUDA<float> &output, const Matrix_CUDA<float> &input, lint dim, lint n_duplicate);


} // namsepace cuda
} // namespace la
} // namespace julie
