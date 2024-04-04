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
#include "Matrix_CUDA.hpp"


namespace julie
{
namespace la
{
namespace cuda
{

/*********************************************************************************
 * Here are some common calculations of matrices running on nvidia GPU
 *********************************************************************************/

// This method to check whether the matrix matches a specified shape.
// Nothing will change if shape matches; If shape does not match but size of
// shape matches, just do reshape to the matrix (memory stay untouched); if
// size of shape does not match, the matrix will get renewed (free and re-malloc
// memory) with the new shape specified.
// This method is often used in dealing with intermediate variables in calculations.
// Intermediate variables are usually cached to prevent frequent malloc and free
// operations.
// Arguments:
//     cache: The cached matrix that may be renewed.
//     sh:    The new shape specified.
template <typename DT>
bool renew_if_shape_not_match(Matrix_CUDA<DT> &cache, const Shape &sh);

// This method is to check whether one matrix is equal to another matrix.
// This method will return true if
//     1. These matrices have the same shape
// and
//     2. These matrices hold the same elements
template <typename DT>
bool operator == (const Matrix_CUDA<DT> &left, const Matrix_CUDA<DT> &right);

// This method is to check whether one matrix is NOT equal to another matrix.
// This method will return true if
//     1. Shapes of these matrices are different
// or
//     2. Difference(s) can be found comparing elements between these matrices. 
template <typename DT>
bool operator != (const Matrix_CUDA<DT> &left, const Matrix_CUDA<DT> &right);

// a + b operation of 2 matrices
// Arguments:
//     output: Output of a + b
//     left:   The left operand
//     right:  The right operand
// Returns: void
template <typename DT>
void add (Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &left, const Matrix_CUDA<DT> &right);

// a - b operation of 2 matrices
// Arguments:
//     output: Output of a - b
//     left:   The left operand
//     right:  The right operand
// Returns: void
template <typename DT>
void subtract (Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &left, const Matrix_CUDA<DT> &right);

// Broadcast mode of a + b of 2 matrices.
// For example, matrix_a == [[1, 2, 3], [3, 2, 1]], matrix_b == [4, 8, -1], then
// broadcast_add output of matrix_a and matrix_b is [[5, 10, 2], [7, 10, 0]].
// Arguments:
//     output: Output of this operation
//     left:   The left operand
//     right:  The right operand
// Returns: void
template <typename DT>
void broadcast_add (Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &left, const Matrix_CUDA<DT> &right);

// Matrix multiplication
// Which dimensions should be merged together is manually defined here.
// For example, two matrices: shape (7, 3, 2, 5, 6) and shape (6, 5, 6, 4),
// if l_dims_merge == 2 and r_dims_merge == 2, the shape of result will be (7, 3, 2, 6, 4).
// However, if l_dims_merge == 4 and r_dims_merge == 3, the shape of output will be (7, 4).
// Note: Total sizes (number of elements) of dimensions to merge should be equal betweem the left and right.
// Arguments:
//     output: Output of this operation
//     left:   The left operand
//     right:  The right operand
// Returns: void
template <typename DT>
void matmul(Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> & left, const Matrix_CUDA<DT> & right, lint l_dims_merge, lint r_dims_merge);

// Matrix multiplication
// This function can automatically figure out which dimensions should be merged together.
// For example, two matrices: shape (7, 3, 3, 5, 4) and shape (10, 6, 6, 4)
// (3, 5, 4) of the left matrix and (10, 6) of the right matrix will be merged.
// then, the shape of result will be (7, 3, 6, 4).
// This function will throw out an exception if no appropriate dimensions to merge can be found.
// Arguments:
//     output: Output of this operation
//     left:   The left operand
//     right:  The right operand
// Returns: void
template <typename DT>
void matmul(Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> & left, const Matrix_CUDA<DT> & right);

// Multiplication of 2 matrices element by element (element wise)
// The two matrices should have the same shape.
// Arguments:
//     output: Output of this operation
//     left:   The left operand
//     right:  The right operand
// Returns: void
template <typename DT>
void multiply(Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> & left, const Matrix_CUDA<DT> & right);

// Broadcast mode of element wise multiplication
// For example, matrix_a == [[1, 2, 3], [3, 2, 1]], matrix_b == [4, 8, -1], then
// broadcast_multiply output of matrix_a and matrix_b is [[4, 16, -3], [12, 16, -1]].
// Arguments:
//     output: Output of this operation
//     left:   The left operand
//     right:  The right operand
// Returns: void
template <typename DT>
void broadcast_multiply (Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &left, const Matrix_CUDA<DT> &right);

// Dot product of two matrices
// The two matrices should have the same amount of elements.
// Arguments:
//     multiply_cache: Intermediate variable that is element wise multiplication of left and right
//     left:           The left operand
//     right:          The right operand
// Returns: A scalar value which is dot product of left and right
template <typename DT>
DT dot_product(Matrix_CUDA<DT> &multiply_cache, const Matrix_CUDA<DT> & left, const Matrix_CUDA<DT> & right);

// Multiplication of a matrix and a scalar
// Arguments:
//     output: Output of this operation
//     left:   The left operand
//     scalar:  The right operand which is a scalar value
// Returns: void
template <typename DT>
void multiply(Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> & left, DT scalar);

// Division of a matrix by a scalar
// Arguments:
//     output: Output of this operation
//     left:   The left operand
//     scalar:  The right operand which is a scalar value
// Returns: void
template <typename DT>
void divide(Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &left, DT scalar);

// Calculate power (element wise multiplication) of a matrix.
// Arguments:
//     output: Output of this operation
//     mat:    The matrix to be powered
//     n:      The exponent
// Returns: void
template <typename DT>
void matrix_pow(Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> & mat, int n);

// Get a fully transposed matrix of a matrix.
// For example, if shape of the original matrix is (a, b, c, d, e, f),
// then the fully transposed one is (f, e, d, c, b, a)
// Arguments:
//     transposed: Output of this operation (the transposed matrix)
//     in:         The matrix input
// Returns: void
template <typename DT>
void full_transpose(Matrix_CUDA<DT> & transposed, const Matrix_CUDA<DT> & in);

// Get a transposed matrix of a matrix
// Arguments:
//     transposed: Output of this operation (the transposed matrix)
//     in:         The matrix input
//     left_dims:  Left part of the dimensions that will switch with the other
//                 part of the dimensions. For example, if left_dims == 2 and
//                 the matrix's shape is (a, b, c, d, e, f), then shape of the
//                 transposed matrix is (c, d, e, f, a, b).
// Returns: void
template <typename DT>
void transpose(Matrix_CUDA<DT> & transposed, const Matrix_CUDA<DT> & in, lint left_dims);

// This method does transpose operation on a part of the matrix dimensions.
// For example, a matrix of shape (a, b, c, d, e, f) is transposed to shape (a, b, e, c, d, f)
// Arguments:
//     transposed: The transposed matrix
//     in:         The matrix input
//     l_dim_idx1: Index of the first dimension the of left part of transpose
//     l_dim_idx2: Index of the last dimension of the left part of transpose
//     r_dim_idx1: Index of the first dimension of the right part of transpose
//     r_dim_idx2: Index of the last dimension of the right part of transpose
// Returns: void
template <typename DT>
void transpose_neighboring_dims(Matrix_CUDA<DT> & transposed, const Matrix_CUDA<DT> & in,
                                lint l_dim_idx1, lint l_dim_idx2, lint r_dim_idx1, lint r_dim_idx2);

// This method is to get a matrix in which all members are absolute values of the input matrix
// Arguments:
//     output: The matrix whose memebers are absolute values of the input matrix
//     input:  The input matrix
template <typename DT>
void abs(Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &input);

// This method is to get a mask matrix where element value is ZERO if the corresponding element
// value in the input matrix falls bellow the given threshold, otherwise ONE if the corresponding
// element value reaches or exceeds the given threshold.
// Arguments:
//     mask: The mask matrix which is output of this function
//     in:   The input matrix
//     th:   The threshold
// Returns: void
template <typename DT>
void threshold(Matrix_CUDA<DT> &mask, const Matrix_CUDA<DT> &in, DT th);

// This method is to get a mask matrix where element value is ZERO if the corresponding element's
// absolute value in the input matrix falls bellow the given threshold, otherwise ONE if the 
// corresponding element's absolute value reaches or exceeds the given threshold.
// Arguments:
//     mask: The mask matrix which is output of this function
//     in:   The input matrix
//     th:   The threshold
// Returns: void
template <typename DT>
void abs_threshold(Matrix_CUDA<DT> &mask, const Matrix_CUDA<DT> &in, DT th);

// This method is to concatenate 2 matrices along a certain dimension.
// For example, if one matrix of shape (a, b1, c, d) and another matrix of shape (a, b2, c, d) are concatenated
// together along dimension 1, then shape of the concatenated matrix will be (a, b1 + b2, c, d)
// Arguments:
//     output: the concatenated matrix
//     left:   the left operand
//     right:  the right operand
//     dim:    the dimension to be concatenated
template <typename DT>
void concatenate(Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &mat1, const Matrix_CUDA<DT> &mat2, lint dim);

// This method is to get a slice of a matrix along a certain dimension
// Arguments:
//     output:     the slice
//     input:      the matrix where this slice is from
//     dim:        the dimension where the slice is from
//     idx:        start point of the slice
//     slice_size: size of this slice
template <typename DT>
void slice(Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &input, lint dim, lint idx, lint slice_size);

// This method is to make duplicates along a dimension specified
// Arguments:
//     output:      the output with repeats along the specified dimension
//     input:       the input
//     dim:         the dimension along which repeats are applied
//     n_duplicate: number of repeats
template <typename DT>
void repeat(Matrix_CUDA<DT> &output, const Matrix_CUDA<DT> &input, lint dim, lint n_duplicate);

} // namespace cuda
} // namespace la
} // namespace julie