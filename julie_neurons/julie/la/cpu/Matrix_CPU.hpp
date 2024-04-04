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
#include "Coordinate.hpp"
#include "Matrix_CPU_Iterator.hpp"
#ifdef WITH_CUDA
#include "Matrix_CUDA.hpp"
#endif

#include <iostream>
#include <fstream>
#include <tuple>


namespace julie
{
namespace la
{
namespace cpu
{

/*********************************************************************************
 * Almost all calculations of deep learning depend on matrices.
 * 
 * The class Matrix_CPU here is a linear algebra interface holding a group of
 * numerical elements for matrix calculations in deep learning.
 * 
 * Matrix_CPU runs on CPU.
 * 
 * A matrix can be of one dimension, two dimensions, three dimensions or even more
 * dimensons.
 * 
 * A batch of data set can be represented by a matrix or a list of matrices;
 * weights of neural networks can also be represented as matrices.
**********************************************************************************/
template <typename DT = float>
class Matrix_CPU
{
public:
    static const int PRINT_PRECISION = 5;

public:
    // Shape of this matrix
    Shape m_shape;
    
    // All the elements of this matrix (CPU)
    DT *m_data;

public:
    // The default constructor.
    // The matrix created by default constructor is not usable until
    // it is assigned by a usable matrix
    Matrix_CPU();

    // Create a matrix of a certain shape.
    Matrix_CPU(const Shape & shape);

    // Create a matrix of a certain shape. All of its elements are set by a certain value
    // Arguments:
    //     value: value of all elements
    //     shape: Shape of this matrix, for example: (3), (64, 128) or (72, 9, 18, 44)
    Matrix_CPU(DT value, const Shape & shape);

    // This constructor is to convert a list of matrices into one matrix.
    // The first dimension of the new matrix is size of this list.
    explicit Matrix_CPU(const std::vector<Matrix_CPU> & matrices);

    // This constructor is to convert a list if shared pointers of matrices into one matrix.
    // The first dimension of the new matrix is size of this list.
    explicit Matrix_CPU(const std::vector<std::shared_ptr<Matrix_CPU>> & matrices);

    // Copy constructor
    Matrix_CPU(const Matrix_CPU & other);

    // Move constructor
    Matrix_CPU(Matrix_CPU && other);

    // Create a matrix from a list (std::vector) of numeric elements.
    // Arguments:
    //     vec:        The list holding numeric numbers
    //     horizontal: Shape of the new matrix will be (vec.size(), 1) if horizontal == true
    //                 Shape of the new matrix will be (1, vec.size()) if horizontal == false
    Matrix_CPU(const std::vector<DT> & vec, bool horizontal);

    // Create a matrix from a list (std::vector) of numeric elements.
    Matrix_CPU(const std::vector<DT> &vec, const Shape &shape);

    // Create a matrix from a 2-dimensional list (std::vector) of numerical numbers
    Matrix_CPU(const std::vector<std::vector<DT>> & array);

#ifdef WITH_CUDA
    // Construct a CPU mode matrix from a GPU matrix
    Matrix_CPU(const julie::la::cuda::Matrix_CUDA<DT> & gpu_mat);
#endif

    // Destructor
    ~Matrix_CPU();

public:

    // Element iterator of this matrix
    typedef Matrix_CPU_Iterator<DT> diterator;

    // Begin semantics of iterator
    diterator begin() const;

    // End semantics of iterator
    diterator end() const;

    // Get a fully transposed matrix.
    // For example, if this matrix is of shape (a, b, c, d, e, f).
    // Then shape of the fully transposed matrix would be (f, e, d, c, b, a).
    void get_full_transpose(Matrix_CPU & out) const;

    // Get a transposed matrix
    // Arguments:
    //     out:       The output which is the transposed matrix
    //     left_dims: Left part of the dimensions that will switch with the other
    //                part of the dimensions. For example, if left_dims == 2 and
    //                the matrix's shape is (a, b, c, d, e, f), then shape of the
    //                transposed matrix is (c, d, e, f, a, b).
    void get_transpose(Matrix_CPU & out, lint left_dims) const;

public:
    // Copy assignment
    Matrix_CPU & operator = (const Matrix_CPU & other);

    // Move assignment
    Matrix_CPU & operator = (Matrix_CPU && other);

    // Assign a scalar value to all matrix elements
    Matrix_CPU & operator = (DT scalar);

    // Get an element of a certain position
    DT at(const Coordinate & pos) const;

    // Get an element of a certain position. This element can be updated
    DT & at(const Coordinate & pos);

    // Get an element of a certain position
    DT operator [] (const Coordinate & pos) const;

    // Get an element of a certain position. This element can be updated
    DT & operator [] (const Coordinate & pos);

    // Get an element of a certain position
    DT operator [] (std::initializer_list<lint> list) const;

    // Get an element of a certain position. This element can be updated
    DT & operator [] (std::initializer_list<lint> list);

    // Elementwise add of matrix element values.
    Matrix_CPU & operator += (const Matrix_CPU & other);

    // Elementwise subtraction of matrix element values.
    Matrix_CPU & operator -= (const Matrix_CPU & other);

    // Elementwise multiplication of matrix element values.
    Matrix_CPU & operator *= (const Matrix_CPU & other);

    // Plus all elements of the matrix with a scalar.
    Matrix_CPU & operator += (DT scalar);

    // Subtract all elements of the matrix with a scalar.
    Matrix_CPU & operator -= (DT scalar);

    // Multiply all elements of the matrix with a scalar.
    Matrix_CPU & operator *= (DT scalar);

    // Divide all elements of the matrix with a scalar.
    Matrix_CPU & operator /= (DT scalar);

    // Randomize all elements of this matrix. Distribution of the elements complies with
    // Gaussian distribution (normal distribution).
    // Arguments:
    //     mu: mean of these elements
    //     sigma: sqrt(variance) of these elements
    // Returns:
    //     Reference of the matrix self
    Matrix_CPU & gaussian_random(DT mu, DT sigma);

    // Randomize all elements of this matrix. Distribution of the elements complies with
    // uniform distribution.
    // Arguments:
    //     min: Lower bound random range
    //     max: Upper bound random range
    // Returns:
    //     Reference of the matrix self
    Matrix_CPU & uniform_random(DT min, DT max);

    // Normalize all elements of this matrix to the range of [min, max]
    Matrix_CPU & normalize(DT min, DT max);

    // Normalize all elements of this matrix to mean == 0 and variance == 1
    Matrix_CPU & normalize();

    // Change shape of this matrix.
    // For example, we can change (20, 30) to (2, 10, 3, 5, 2), or change (4, 5, 6) to (3, 5, 8).
    // However, size of this matrix will not change, which means 20 * 30 == 2 * 10 * 3 * 5 * 2,
    // or 4 * 5 * 6 == 3 * 5 * 8.
    // Order of all elements in this matrix will not change either
    Matrix_CPU & reshape(const Shape & shape);

    // Extend one dimension of the matrix.
    // For example, (30, 50) is extended to (1, 30, 50).
    // This is a special case of reshape.
    Matrix_CPU & left_extend_shape();

    // Extend one dimension of the matrix.
    // For example, (30, 50) is extended to (30, 50, 1).
    // This is a special case of reshape.
    Matrix_CPU & right_extend_shape();

    // Get left extended version of the matrix with duplicates.
    // For example, (35, 45) can be extend to (12, 35, 45) in which
    // there are exactly 12 copies of (35, 45)
    void get_left_extended(Matrix_CPU &output, lint duplicate) const;

    // Get right extended version of the matrix with duplicates.
    // For example, (35, 45) can be extend to (35, 45, 16) in which
    // there are exactly 16 copies of (35, 45)
    void get_right_extended(Matrix_CPU &output, lint duplicate) const;

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
    Perhaps there is a better way to name this function
    */
    Matrix_CPU & scale_one_dimension(lint dim, const std::vector<DT> & scales);

    // Get coordinate of the largest element
    Coordinate argmax() const;

    // Get coordinates of largest elements along one dimension
    std::vector<Coordinate> argmax(lint dim) const;

    // Get value of the largest element
    DT max() const;

    // Get coordinates of largest elements along one dimension
    std::vector<Coordinate> argmin(lint dim) const;

    // Get coordinate of the lowest element
    Coordinate argmin() const;

    // Get value of the lowest element
    DT min() const;

    // Get mean of all the elements
    DT mean() const;

    // Get sum of all the elements
    DT sum() const;

    // Get variance of all the elements of this matrix
    DT variance() const;

    // Collapse a certain dimension of a matrix into a list of matrices (smart pointers)
    // For example, a matrix of shape (4, 6, 5, 7), if we collapse it with argument dim = 1,
    // it will be turned into 6 matrices of shape (4, 5, 7).
    std::vector<Matrix_CPU> get_collapsed(lint dim) const;

    // Collapse a certain dimension of a matrix, and summate all matrices into one matrix
    // For example, there is a matrix of shape (4, 6, 5, 7), if we fuse it with argument
    // dim = 1, it will be turned into a sum of 6 matrices of shape (4, 5, 7)
    void get_reduce_sum(Matrix_CPU &output, lint dim) const;

    // Collapse a certain dimension of a matrix, and get a mean of all the matrices.
    // For example, there is a matrix of shape (4, 6, 5, 7), if we get its reduce_mean
    // with argument dim = 1, it will be turned into a mean of 6 matrices of shape (4, 5, 7)
    void get_reduce_mean(Matrix_CPU &output, lint dim) const;

    // sqrt(sum(a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 + ...))
    DT euclidean_norm() const;

    // Get shape of the matrix
    Shape shape() const;

    // Get string plot of the matrix.
    // For example, there is a matrix of shape (2, 3, 4), plot of this matrix would be like:
    // [
    //     [
    //         [a111, a112, a113, a114]
    //         [a121, a122, a123, a124]
    //         [a131, a132, a133, a134]
    //     ]
    //     [
    //         [a211, a212, a213, a214]
    //         [a221, a222, a223, a224]
    //         [a231, a232, a233, a234]
    //     ]
    // ]
    std::string to_string() const;

#ifdef WITH_CUDA
    // Get a CUDA mode Matrix from this Matrix
    julie::la::cuda::Matrix_CUDA<DT> get_CUDA () const;
#endif

private:

    // This method is to convert a number into a string, and measure each part of the
    // number: The part before dot, the part of dot and the part after dot.
    std::tuple<int, int, int> str_len_of_a_number(DT val) const;

    // This is a recurive function to print elements of this Matrix on all dimensions.
    // Arguments:
    //     os:           The output stream
    //     dim_index:    Index of the dimension to print
    //     start:        Start point of this dimension
    //     integral_len: Largest length of digits before dot
    //     dot_len:      Largest length of dot (normally 1)
    //     frag_len:     Largest length of digits after dot
    // Returns: void
    void print(std::ostream & os, lint dim_index, const DT *start, int integral_len, int dot_len, int frag_len) const;

};

/* Overloading of output stream << operator
*/
template <typename DT>
std::ostream & operator << (std::ostream & os, const Matrix_CPU<DT> & m);

} // namespace cpu
} // namespace la
} // namespace julie
