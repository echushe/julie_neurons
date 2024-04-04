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
#include <memory>

namespace julie
{

namespace la
{

/**********************************************************************************
 * This class helps locate a place in a matrix.
 * For example, coordinate [ 0, 2 ] is one of the places in a matrix of shape ( 3, 4 )
 **********************************************************************************/
class Coordinate
{
/*********************************************************************************
 * Following are friend classes that can access protected or private members
 * of Shape class
**********************************************************************************/

    friend class julie::la::iMatrix<float>;
    friend class julie::la::iMatrix<int>;

    friend class julie::la::cpu::Matrix_CPU<float>;
    friend class julie::la::cpu::Matrix_CPU<float>;
    friend class julie::la::cpu::Matrix_CPU<short>;
    friend class julie::la::cpu::Matrix_CPU<int>;
    friend class julie::la::cpu::Matrix_CPU<lint>;

    friend class julie::la::cuda::Matrix_CUDA<int>;
    friend class julie::la::cuda::Matrix_CUDA<float>;

    friend Coordinate operator + (const Coordinate &left, const Coordinate &right);
    friend bool operator == (const Coordinate &left, const Coordinate &right);
    friend bool operator != (const Coordinate &left, const Coordinate &right);
    friend bool operator < (const Coordinate &left, const Coordinate &right);
    friend bool operator <= (const Coordinate &left, const Coordinate &right);
    friend bool operator > (const Coordinate &left, const Coordinate &right);
    friend bool operator >= (const Coordinate &left, const Coordinate &right);
    friend std::ostream & operator << (std::ostream & os, const Coordinate & co);

private:
    
    // How many dimensions this coordinate has
    lint m_dim;
    
    // Raw data of this coordinate
    lint *m_data;

    // This shape is used to restrict behavior of the coordinate.
    // The coordinate can only change within the range defined by the shape.
    Shape m_shape;

    // Index of this coordinate in a matrix
    lint m_index;

public:

    // Create a coordinate by a list of numbers.
    // But these numbers should agree with the shape.
    // For example, [ 0, 90 ] is not a location in shape ( 30, 20, 75 )
    Coordinate(std::initializer_list<lint> list, const Shape & shape);

    // Create a coordinate with a shape
    // In this circumstance, the coordinate will be initialized by zeros
    Coordinate(const Shape & shape);

    // Create a coordinate with its index in matrix and its shape
    Coordinate(lint index, const Shape & shape);

    // Copy constructor
    Coordinate(const Coordinate & other);

    // Move constructor
    Coordinate(Coordinate && other);

    // Default constructor, it is not usable until it is assigned by a usable one
    Coordinate();

    // Destructor
    ~Coordinate();

    // Copy assignment
    Coordinate & operator = (const Coordinate & other);

    // Move assignment
    Coordinate & operator = (Coordinate && other);

    // Assigned by index
    Coordinate & operator = (lint index);

public:
    // Access a certain dimension of this coordinate
    lint operator [] (lint index) const;

    // Access a certain dimansion of this coordinate and update it
    lint & operator [] (lint index);

    // Plus the coordinate to traverse a matrix
    Coordinate & operator ++ ();

    // Plus the coordinate to 
    Coordinate operator ++ (int);

    // Subtract the coordinate to traverse a matrix
    Coordinate & operator -- ();

    // Subtract the coordinate to 
    Coordinate operator -- (int);

    lint index() const;

    // Get amount of dimensions of this coordinate
    lint dim() const;

    // Reverse the coordinate
    // For example, reverse of [ 7, 5, 3, 0, 44 ] is [ 44, 0, 3, 5, 7 ]
    Coordinate get_reversed() const;

    // Get a part of this coordinate by indexes.
    // For example, given [ 1, 13, 8, 0, 17 ], its sub coordinate with index 0 and
    // index 2 is [ 1, 13, 8 ].
    // Arguments:
    //     dim_first: Index of the first dimension of the sub coordinate
    //     dim_last:  Index of the last dimension of the sub coordinate
    // Returns: The sub coordinate
    Coordinate sub_coordinate(lint dim_first, lint dim_last) const;

    // Get the shape this coordinate depends on
    Shape get_shape() const;
};

// This method is to concatenate 2 coordinates.
Coordinate operator + (const Coordinate &left, const Coordinate &right);

// This method is to check if the coordinates are the same
bool operator == (const Coordinate &left, const Coordinate &right);

// This method is to check if the coordinates are different
bool operator != (const Coordinate &left, const Coordinate &right);

// This method is to check if one coordinate is smaller than the other one
bool operator < (const Coordinate &left, const Coordinate &right);

// This method is to check if one coordinate is the same or smaller than
// the other one.
bool operator <= (const Coordinate &left, const Coordinate &right);

// This method is to check if one coordinate is larger than the other one
bool operator > (const Coordinate &left, const Coordinate &right);

// This method is to check if one coordinate is the same or smaller than
// the other one.
bool operator >= (const Coordinate &left, const Coordinate &right);

// Output stream of the coordinate.
// The output will be like (b1, b2, b3, ...)
std::ostream & operator << (std::ostream & os, const Coordinate & co);

// Reverse the coordinate
// For example, reverse of [ 7, 5, 3, 0, 44 ] is [ 44, 0, 3, 5, 7 ]
Coordinate reverse(const Coordinate & sh);

} // namespace la
} // namespace julie
