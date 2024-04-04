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
#include "Matrix_CPU.hpp"
#include "utilities.hpp"

#ifdef WITH_CUDA
#include "Matrix_CUDA.hpp"
#endif

#include <iostream>
#include <fstream>
#include <tuple>


namespace julie
{

// Namespace la stands for linear algebra
namespace la
{

/*********************************************************************************
 * Almost all calculations of deep learning depend on matrices.
 * 
 * The class iMatrix here is a linear algebra interface holding a group of numerical
 * elements for matrix calculations in deep learning.
 * 
 * A matrix can be of one dimension, two dimensions, three dimensions or even more
 * dimensons.
 * 
 * A batch of data set can be represented by a matrix or a list of matrices;
 * weights of neural networks can also be represented as matrices.
**********************************************************************************/
template <typename DT = float>
class iMatrix
{

public:

    // The default constructor.
    // The matrix created by default constructor is not usable until it is assigned
    // by a usable matrix. Default type of iMatrix is CPU mode.
    iMatrix(MatrixType type = MatrixType::CPU);

    // This constructor is to create a matrix of a certain shape.
    // Arguments:
    //     shape: Shape of this matrix, for example: (3), (64, 128) or (72, 9, 18, 44).
    //     type:  Matrix type. It is CPU by default but you can specify it to other types.
    iMatrix(const Shape & shape, MatrixType type = MatrixType::CPU);

    // This constructor is to create a matrix of a certain shape with all of its elements
    // set by a certain value.
    // Arguments:
    //     value: value of all elements
    //     shape: Shape of this matrix, for example: (3), (64, 128) or (72, 9, 18, 44)
    //     type:    Matrix type. It is CPU by default but you can specify it to other types.
    iMatrix(DT value, const Shape & shape, MatrixType type = MatrixType::CPU);

    // This constructor is to convert a list if shared pointers of matrices into one matrix.
    // The first dimension of the new matrix is size of this list.
    explicit iMatrix(const std::vector<std::shared_ptr<iMatrix>> & matrices);
    
    // This constructor is to convert a list of matrices into one matrix.
    // The first dimension of the new matrix is size of this list.
    explicit iMatrix(const std::vector<iMatrix> & matrices);

    // Create a matrix from a list (std::vector) of numeric elements.
    // Arguments:
    //     vec:        The list holding numeric numbers
    //     horizontal: Shape of the new matrix will be (vec.size(), 1) if horizontal == true
    //                 Shape of the new matrix will be (1, vec.size()) if horizontal == false
    //     type:       Matrix type. It is CPU by default but you can specify it to other types.
    iMatrix(const std::vector<DT> & vec, bool horizontal, MatrixType type = MatrixType::CPU);

    // Create a matrix from a list (std::vector) of numeric elements.
    iMatrix(const std::vector<DT> & vec, const Shape &shape, MatrixType type = MatrixType::CPU);

    // Create a matrix from a 2-dimensional list (std::vector) of numerical numbers
    // Arguments:
    //     array: The 2-dimensional list holding numeric numbers
    //     type:  Matrix type. It is CPU by default but you can specify it to other types.
    iMatrix(const std::vector<std::vector<DT>> & array, MatrixType type = MatrixType::CPU);

    // Copy constructor. It will call deep copy of a matrix
    iMatrix(const iMatrix & other);

    // Move constructor.
    iMatrix(iMatrix && other);

    // Destructor
    ~iMatrix();

    // Create an iMatrix instance from a naked CPU type matrix via deep copy.
    iMatrix(const cpu::Matrix_CPU<DT> & mat_cpu);

    // Create an iMatrix instance from a naked CPU type matrix via move semantics.
    iMatrix(cpu::Matrix_CPU<DT> && mat_cpu);

    // Create an iMatrix instance from a naked CUDA type matrix via deep copy.
    iMatrix(const cuda::Matrix_CUDA<DT> & mat_cuda);

    // Create an iMatrix instance from a naked CUDA type matrix via move semantics.
    iMatrix(cuda::Matrix_CUDA<DT> && mat_cuda);

public:

    // Get a fully transposed matrix.
    // For example, if this matrix is of shape (a, b, c, d, e, f).
    // Then shape of the fully transposed matrix would be (f, e, d, c, b, a).
    void get_full_transpose(iMatrix & out) const;

    // Get a transposed matrix
    // Arguments:
    //     out:       The output which is the transposed matrix
    //     left_dims: Left part of the dimensions that will switch with the other
    //                part of the dimensions. For example, if left_dims == 2 and
    //                the matrix's shape is (a, b, c, d, e, f), then shape of the
    //                transposed matrix is (c, d, e, f, a, b).
    void get_transpose(iMatrix & out, lint left_dims) const; 

public:

    // Copy assignment
    iMatrix & operator = (const iMatrix & other);

    // Move assignment
    iMatrix & operator = (iMatrix && other);

    // Assign a scalar value to all matrix elements.
    iMatrix & operator = (DT scalar);

    // Elementwise add of matrix element values.
    iMatrix & operator += (const iMatrix & other);

    // Elementwise subtraction of matrix element values.
    iMatrix & operator -= (const iMatrix & other);

    // Elementwise multiplication of matrix element values.
    iMatrix & operator *= (const iMatrix & other);

    // Plus all elements of the matrix with a scalar.
    iMatrix & operator += (DT scalar);

    // Subtract all elements of the matrix with a scalar.
    iMatrix & operator -= (DT scalar);

    // Multiply all elements of the matrix with a scalar.
    iMatrix & operator *= (DT scalar);

    // Divide all elements of the matrix with a scalar.
    iMatrix & operator /= (DT scalar);

    // Get an element of a certain position.
    // This method is usually for debug work
    DT operator [] (const Coordinate & pos) const;

    // Randomize all elements of this matrix. Distribution of the elements complies with
    // Gaussian distribution (normal distribution).
    // Arguments:
    //     mu: mean of these elements
    //     sigma: sqrt(variance) of these elements
    // Returns:
    //     Reference of the matrix self
    iMatrix & gaussian_random(DT mu, DT sigma);

    // Randomize all elements of this matrix. Distribution of the elements complies with
    // uniform distribution.
    // Arguments:
    //     min: Lower bound random range
    //     max: Upper bound random range
    // Returns:
    //     Reference of the matrix self
    iMatrix & uniform_random(DT min, DT max);

    // Normalize all elements of this matrix to the range of [min, max]
    iMatrix & normalize(DT min, DT max);

    // Normalize all elements of this matrix to mean == 0 and variance == 1
    iMatrix & normalize();

    // Change shape of this matrix.
    // For example, we can change (20, 30) to (2, 10, 3, 5, 2), or change (4, 5, 6) to (3, 5, 8).
    // However, size of this matrix will not change, which means 20 * 30 == 2 * 10 * 3 * 5 * 2,
    // or 4 * 5 * 6 == 3 * 5 * 8.
    // Order of all elements in this matrix will not change either
    iMatrix & reshape(const Shape & shape);

    // Extend one dimension of the matrix.
    // For example, (30, 50) is extended to (1, 30, 50).
    // This is a special case of reshape.
    iMatrix & left_extend_shape();

    // Extend one dimension of the matrix.
    // For example, (30, 50) is extended to (30, 50, 1).
    // This is a special case of reshape.
    iMatrix & right_extend_shape();

    // Get left extended version of the matrix with duplicates.
    // For example, (35, 45) can be extend to (12, 35, 45) in which
    // there are exactly 12 copies of (35, 45)
    void get_left_extended(iMatrix &output, lint duplicate) const;

    // Get right extended version of the matrix with duplicates.
    // For example, (35, 45) can be extend to (35, 45, 16) in which
    // there are exactly 16 copies of (35, 45)
    void get_right_extended(iMatrix &output, lint duplicate) const;

    // Get coordinate of the largest element
    Coordinate argmax() const;

    // Get coordinates of the largest element along one dimension
    std::vector<Coordinate> argmax(lint dim) const;

    // Get value of the largest element
    DT max() const;

    // Get coordinate of the lowest element
    Coordinate argmin() const;

    // Get coordinates of the lowest element along one dimension
    std::vector<Coordinate> argmin(lint dim) const;

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
    std::vector<std::shared_ptr<iMatrix>> get_collapsed(lint dim) const;

    // Collapse a certain dimension of a matrix, and merge all matrices into one matrix
    // For example, there is a matrix of shape (4, 6, 5, 7), if we fuse it with argument
    // dim = 1, it will be turned into a sum of 6 matrices of shape (4, 5, 7)
    void get_reduce_sum(iMatrix &output, lint dim) const;

    // Collapse a certain dimension of a matrix, and get a mean of all the matrices.
    // For example, there is a matrix of shape (4, 6, 5, 7), if we get its reduce_mean
    // with argument dim = 1, it will be turned into a mean of 6 matrices of shape (4, 5, 7)
    void get_reduce_mean(iMatrix &output, lint dim) const;

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

    // Get matrix type of this matrix
    // Matrix types:
    //     CPU:  A matrix of this type is supposed to run on a CPU
    //     CUDA: A matrix of this type is supposed to run on an nvidia GPU via CUDA APIs
    //     CL:   A matrix of this type is supposed to run via openCL APIs (under development)
    MatrixType get_matrix_type() const { return this->m_type; }

    // Set matrix type of this matrix
    // Matrix types:
    //     CPU:  A matrix of this type is supposed to run on a CPU
    //     CUDA: A matrix of this type is supposed to run on an nvidia GPU via CUDA APIs
    //     CL:   A matrix of this type is supposed to run via openCL APIs (under development)
    iMatrix & set_matrix_type(MatrixType type);

    // Get reference of naked CPU type instance of this matrix self.
    // This function will return a NULL pointer of the matrix is not CPU type.
    std::shared_ptr<cpu::Matrix_CPU<DT>> get_cpu_instance() const { return this->m_mat_cpu; }

#ifdef WITH_CUDA
    // Get reference of naked CUDA type instance of this matrix self.
    // This function will return a NULL pointer of the matrix is not CUDA type.
    std::shared_ptr<cuda::Matrix_CUDA<DT>> get_cuda_instance() const { return this->m_mat_cuda; }
#endif

private:

    //Type of this matrix:
    // UNKNOWN = 0,
    // CPU = 1,
    // CUDA = 2,
    // CL = 3
    MatrixType m_type;

    // Reference of naked CPU type instance of this matrix self.
    // It will be a NULL pointer of this matrix is not CPU type.
    std::shared_ptr<cpu::Matrix_CPU<DT>> m_mat_cpu;

#ifdef WITH_CUDA
    // Reference of naked CUDA type instance of this matrix self.
    // It will be a NULL pointer of this matrix is not CUDA type.
    std::shared_ptr<cuda::Matrix_CUDA<DT>> m_mat_cuda;
#endif

};


// Overloading of output stream to plot a matrix
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
template <typename DT>
std::ostream & operator << (std::ostream & os, const iMatrix<DT> & m);


} // namespace la
} // namespace julie
