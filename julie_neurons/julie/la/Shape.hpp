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
#include <list>
#include <iostream>
#include <vector>

// All integers are 64bit wide here
typedef int64_t lint;

namespace julie
{

namespace la
{

template <typename DT>
class iMatrix;
namespace cpu
{
template <typename DT>
class Matrix_CPU;
}

namespace cuda
{
template <typename DT>
class Matrix_CUDA;
} // namespace cuda


/*********************************************************************************
 * The class Shape represents shape of a matrix.
 * For example, ( 4, 5, 6, 8 ) is shape of a four dimensional matrix, sizes of its
 * dimensions are 4, 5, 6, 8 respectively.
**********************************************************************************/
class Shape
{

/*********************************************************************************
 * Following are friend classes that can access protected or private members
 * of Shape class
**********************************************************************************/

    friend class Coordinate;

    friend class julie::la::iMatrix<float>;
    friend class julie::la::iMatrix<int>;

    friend class julie::la::cpu::Matrix_CPU<float>;
    friend class julie::la::cpu::Matrix_CPU<float>;
    friend class julie::la::cpu::Matrix_CPU<short>;
    friend class julie::la::cpu::Matrix_CPU<int>;
    friend class julie::la::cpu::Matrix_CPU<lint>;

    friend class julie::la::cuda::Matrix_CUDA<int>;
    friend class julie::la::cuda::Matrix_CUDA<float>;

    friend bool operator == (const Shape &left, const Shape &right);
    friend bool operator != (const Shape &left, const Shape &right);
    friend Shape operator + (const Shape &left, const Shape &right);
    friend std::ostream & operator << (std::ostream & os, const Shape & co);

private:
    
    // How many dimensions this shape has
    lint m_dim;
    
    // Amount of elements this shape has
    lint m_size;
    
    // Raw data of this shape (size of each dimension of this shape)
    lint *m_data;

public:
    
    // A shape can be intialized by a list of dimension sizes
    Shape(std::initializer_list<lint> list);
    
    // Create a shape from a vector array of dimension sizes
    Shape(std::vector<lint> & list);

    // Copy constructor
    Shape(const Shape & other);
    
    // Move constructor
    Shape(Shape && other);
    
    // Destructor
    ~Shape();
    
    // Copy assignment
    Shape & operator = (const Shape & other);
    
    // Move assignment
    Shape & operator = (Shape && other);
    
    Shape();

public:
    // [] operator is to access a dimension of shape via its index
    lint operator [] (lint index) const;

    // Get amount of elements of this shape
    lint size() const;

    // Get amount of dimensions of this shape
    lint dim() const;

    // Add one dimension to left side of this shape.
    // Size of this shape won't change after this operation.
    // For example, shape ( 100, 120 ) will be changed to ( 1, 100, 120 ).
    Shape & left_extend();

    // Add one dimension to right side of this shape
    // Size of this shape won't change after this operation.
    // For example, shape ( 70, 80 ) will be changed to ( 70, 80, 1 ).
    Shape & right_extend();

    // Get a reversed shape.
    // For example, ( 130, 180, 70, 6 ) will be changed to ( 6, 70, 180, 130 ).
    Shape get_reversed() const;

    // Get a part of this shape by indexes.
    // For example, given ( 30, 25, 8, 9, 17 ), its sub shape with index 0 and
    // index 2 is ( 30, 25, 8 ).
    // Arguments:
    //     dim_first: Index of the first dimension of the sub shape
    //     dim_last:  Index of the last dimension of the sub shape
    // Returns: The sub shape
    Shape sub_shape(lint dim_first, lint dim_last) const;

public:

    // This method is to check whether this pair of shapes are suitable for matrix multiplication.
    // For example, matrices of shape (8, 9, 1) and shape (3, 3, 7) can do MatMul and shape of the output is (8, 7).
    // Nevertheless, matrices of shape (8, 9, 1) and (4, 3, 5) cannot do MatMul.
    static bool CanDoMatMul(const Shape & left_sh, const Shape & right_sh);
};


// This method is to check if shapes are the same
bool operator == (const Shape &left, const Shape &right);

// This method is to check if shapes are different
bool operator != (const Shape &left, const Shape &right);

// This method is to concatenate 2 shapes.
// For example, concatenation of shape (128, 64) and (38, 54, 4) is (128, 64, 38, 54, 4)
Shape operator + (const Shape &left, const Shape &right);

// This function returns the reversed shape without changing the original shape.
Shape reverse(const Shape & sh);

// Output stream of the shape.
// The output will be like (a1, a2, a3, ...)
std::ostream & operator<<(std::ostream & os, const Shape & co);

} // namespace la

} // namespace julie