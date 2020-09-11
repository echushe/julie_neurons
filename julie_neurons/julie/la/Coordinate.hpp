#pragma once
#include "Shape.hpp"
#include <memory>

namespace julie
{

namespace la
{

/*
This class helps locate a place in a matrix.
For example, coordinate [ 0, 2 ] is one of the places in a matrix of shape [ 3, 4 ]
*/
class Coordinate
{
    friend class DMatrix<double>;
    friend class DMatrix<float>;
    friend class DMatrix<short>;
    friend class DMatrix<int>;
    friend class DMatrix<lint>;

private:
    // Number of dimensions
    lint m_dim;
    // Detailed data of this coordinate
    lint *m_data;

    // The shape used to restrict behavior of the coordinate
    // The coordinate can only change within the range defined by the shape
    Shape m_shape;

public:

    // Create a coordinate by a list of numbers.
    // But these numbers should agree with the shape.
    // For example, [ 0, 90 ] is not a location in shape [ 30, 20, 75 ]
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

    Coordinate & transposed_plus();

    Coordinate sub_coordinate(lint dim_first, lint dim_last) const;

    Shape get_shape() const;

    friend Coordinate operator + (const Coordinate &left, const Coordinate &right);

    friend bool operator == (const Coordinate &left, const Coordinate &right);
    friend bool operator != (const Coordinate &left, const Coordinate &right);
    friend bool operator < (const Coordinate &left, const Coordinate &right);
    friend bool operator <= (const Coordinate &left, const Coordinate &right);
    friend bool operator > (const Coordinate &left, const Coordinate &right);
    friend bool operator >= (const Coordinate &left, const Coordinate &right);

    friend std::ostream & operator << (std::ostream & os, const Coordinate & co);
};

Coordinate operator + (const Coordinate &left, const Coordinate &right);

// coordinate_a == coordinate_b
bool operator == (const Coordinate &left, const Coordinate &right);

// coordinate_a != coordinate_b
bool operator != (const Coordinate &left, const Coordinate &right);

bool operator < (const Coordinate &left, const Coordinate &right);

bool operator <= (const Coordinate &left, const Coordinate &right);

bool operator > (const Coordinate &left, const Coordinate &right);

bool operator >= (const Coordinate &left, const Coordinate &right);

std::ostream & operator << (std::ostream & os, const Coordinate & co);

// Get reverse of the coordinate without changing the original one
Coordinate reverse(const Coordinate & sh);

} // namespace la
} // namespace julie
