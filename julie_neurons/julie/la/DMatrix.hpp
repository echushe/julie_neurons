#pragma once
#include "Vector.hpp"
#include "Shape.hpp"
#include "Coordinate.hpp"
#include "Exceptions.hpp"

#include "DMatrix_Iterator.hpp"
#include <iostream>
#include <random>
#include <fstream>
#include <tuple>


namespace julie
{

namespace la
{

//#ifndef GLOBAL_RAND_ENGINE
//    #define GLOBAL_RAND_ENGINE
class global
{
public:
    static std::default_random_engine global_rand_engine;
};
//#endif // GLOBAL_RAND_ENGINE
    
/*
Almost all calculations of deep learning depend on matrices.
A matrix can be of one dimension, two dimensions, three dimensions or even more dimensons.
A batch of training set or test set can be represented by a list or a vector of matrices.
The weights of a neural network layer is a matrix.
The bias of a neural network layer is a matrix.
The labels/targets of a network are also saved in a matrix.
*/
template <typename DTYPE = double>
class DMatrix
{
public:
    static int PRINT_PRECISION;

public:
    // Shape of this matrix
    Shape m_shape;
    // All the elements of this matrix
    DTYPE *m_data;

public:
    // The default constructor.
    // The matrix created by default constructor is not usable until
    // it is assigned by a usable matrix
    DMatrix();

    // Create a matrix of a certain shape. But none of its elements are initialized
    DMatrix(const Shape & shape);

    // Create a matrix of a certain shape. All of its elements are initialized by a value
    DMatrix(DTYPE value, const Shape & shape);

    // Combine an array if matrices into one matrix
    explicit DMatrix(const std::vector<DMatrix> & matrices);

    // Copy constructor
    DMatrix(const DMatrix & other);

    // Move constructor
    DMatrix(DMatrix && other);

    // Create a matrix from a Vector
    // DMatrix(const Vector & vec, bool horizontal);

    // Create a matrix from an std::vector
    DMatrix(const std::vector<DTYPE> & vec, bool horizontal);

    // Create a matrix (2 dimensional)
    DMatrix(const std::vector<std::vector<DTYPE>> & array);

    ~DMatrix();

public:
    typedef DMatrix_Iterator<DTYPE> diterator;

    diterator begin() const;
    diterator end() const;

    DMatrix get_full_transpose() const;

    DMatrix get_transpose(lint left_dims) const;

    // DMatrix 

public:
    // Copy assignment
    DMatrix & operator = (const DMatrix & other);

    // Move assignment
    DMatrix & operator = (DMatrix && other);

    // Assign a scalar value to all matrix elements
    DMatrix & operator = (DTYPE scalar);

    // Get an element of a certain position
    DTYPE at(const Coordinate & pos) const;

    // Get an element of a certain position. This element can be updated
    DTYPE & at(const Coordinate & pos);

    // Get an element of a certain position
    DTYPE operator [] (const Coordinate & pos) const;

    // Get an element of a certain position. This element can be updated
    DTYPE & operator [] (const Coordinate & pos);

    // Get an element of a certain position
    DTYPE operator [] (std::initializer_list<lint> list) const;

    // Get an element of a certain position. This element can be updated
    DTYPE & operator [] (std::initializer_list<lint> list);

    // Get a flatened matrix (vector)
    Vector flaten() const;

    DMatrix & operator += (const DMatrix & other);

    DMatrix & operator -= (const DMatrix & other);

    DMatrix & operator += (DTYPE scalar);

    DMatrix & operator -= (DTYPE scalar);

    DMatrix & operator *= (DTYPE scalar);

    DMatrix & operator /= (DTYPE scalar);

    // Randomize all elements of this matrix. Distribution of the elements complies with
    // Gaussian distribution (normal distribution).
    // DTYPE mu: mean of these elements
    // DTYPE sigma: sqrt(variance) of these elements
    DMatrix & gaussian_random(DTYPE mu, DTYPE sigma);

    // Randomize all elements of this matrix. Distribution of the elements complies with
    // uniform distribution.
    DMatrix & uniform_random(DTYPE min, DTYPE max);

    // Normalize all elements of this matrix to the range of [min, max]
    DMatrix & normalize(DTYPE min, DTYPE max);

    // Normalize all elements of this matrix to mean == 0 and variance == 1
    DMatrix & normalize();

    // Change shape of this matrix.
    // For example, we can change [20, 30] to [2, 10, 3, 5, 2], or
    // change [4, 5, 6] to [3, 5, 8].
    // However, size of this matrix will not change, which means 20 * 30 == 2 * 10 * 3 * 5 * 2,
    // or 4 * 5 * 6 == 3 * 5 * 8.
    // Order of all elements in this matrix will not change either
    DMatrix & reshape(const Shape & shape);

    // Extend one dimension of the matrix.
    // For example, [30, 50] is extended to [1, 30, 50].
    // This is a special case of reshape.
    DMatrix & left_extend_shape();

    // Extend one dimension of the matrix.
    // For example, [30, 50] is extended to [30, 50, 1].
    // This is a special case of reshape.
    DMatrix & right_extend_shape();

    // Get duplicates of left extended version of the matrix.
    // For example, [35, 45] can be extend to [12, 35, 45] in which
    // there are exactly 12 copies of [35, 45]
    DMatrix get_left_extended(lint duplicate) const;

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

    It seems this kind of calculation is needed in back propagation.
    Perhaps there is a better way to name this function
    */
    DMatrix & scale_one_dimension(lint dim, const Vector & scales);

    // Get coordinate of the largest element
    Coordinate argmax() const;

    // Get value of the largest element
    DTYPE max() const;

    // Get coordinate of the lowest element
    Coordinate argmin() const;

    // Get value of the lowest element
    DTYPE min() const;

    // Get mean of all the elements
    DTYPE mean() const;

    // Get sum of all the elements
    DTYPE sum() const;

    // Get variance of all the elements of this matrix
    DTYPE variance() const;

    // Collapse a certain dimension of a matrix into a vector of matrices
    // For example, a matrix of shape [4, 6, 5, 7], if we collapse it with argument dim = 1,
    // it will be turned into 6 matrices of shape [4, 5, 7]
    std::vector<DMatrix> get_collapsed(lint dim) const;

    // Collapse a certain dimension of a matrix, and merge all matrices into one matrix
    // For example, a matrix of shape [4, 6, 5, 7], if we fuse it with argument dim = 1,
    // it will be turned into a sum of 6 matrices of shape [4, 5, 7]
    DMatrix get_fused(lint dim) const;

    // Collapse a certain dimension of a matrix, and get a mean of all the matrices
    // For example, a matrix of shape [4, 6, 5, 7], if we get its reduce_mean with argument dim = 1,
    // it will be turned into a mean of 6 matrices of shape [4, 5, 7]
    DMatrix get_reduce_mean(lint dim) const;


    DTYPE euclidean_norm() const;


    // Get shape of the matrix
    Shape shape() const;

    std::string to_string() const;

private:

    std::tuple<int, int, int> integral_size(DTYPE val) const;
    void print(std::ostream & os, lint dim_index, const DTYPE *start, int integral_len, int dot_len, int frag_len) const;

};


// Overloading of a == b
template <typename DTYPE>
bool operator == (const DMatrix<DTYPE> &left, const DMatrix<DTYPE> &right)
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

// Overloading of a != b
template <typename DTYPE>
bool operator != (const DMatrix<DTYPE> &left, const DMatrix<DTYPE> &right)
{
    return !(left == right);
}

// Overloading of a + b
template <typename DTYPE>
DMatrix<DTYPE> operator + (const DMatrix<DTYPE> &left, const DMatrix<DTYPE> &right)
{
    if (left.m_shape.size() != right.m_shape.size())
    {
        throw std::invalid_argument(invalid_shape);
    }

    lint size = left.m_shape.size();

    DMatrix<DTYPE> mat{ left.m_shape };
    for (lint i = 0; i < size; ++i)
    {
        mat.m_data[i] = left.m_data[i] + right.m_data[i];
    }

    return mat;
}

// Overloading of a - b
template <typename DTYPE>
DMatrix<DTYPE> operator - (const DMatrix<DTYPE> &left, const DMatrix<DTYPE> &right)
{
    if (left.m_shape.size() != right.m_shape.size())
    {
        throw std::invalid_argument(invalid_shape);
    }

    lint size = left.m_shape.size();

    DMatrix<DTYPE> mat{ left.m_shape };
    for (lint i = 0; i < size; ++i)
    {
        mat.m_data[i] = left.m_data[i] - right.m_data[i];
    }

    return mat;
}

// Broadcast mode of a + b
template <typename DTYPE>
DMatrix<DTYPE> broadcast_add (const DMatrix<DTYPE> &left, const DMatrix<DTYPE> &right)
{
    lint big_size = left.m_shape.size();
    lint small_size = right.m_shape.size();

    if (big_size < small_size || big_size % small_size != 0)
    {
        throw std::invalid_argument(invalid_shape);
    }

    DMatrix<DTYPE> mat{ left.m_shape };
    for (lint i = 0; i < big_size; ++i)
    {
        mat.m_data[i] = left.m_data[i] + right.m_data[i % small_size];
    }

    return mat;
}

// DMatrix multiplication
// Which dimensions should be merged together is manually defined here.
// For example, two matrices: [7, 3, 2, 5, 6] and [6, 5, 6, 4], if l_dims_merge == 2 and r_dims_merge == 2,
// the shape of result will be [7, 3, 2, 6, 4]. However, if l_dims_merge == 4 and r_dims_merge == 3,
// the shape of output will be [7, 4].
// Note: Total sizes (number of elements) of dimensions to merge should be equal betweem the left and right.
template <typename DTYPE>
DMatrix<DTYPE> matmul(const DMatrix<DTYPE> & left, const DMatrix<DTYPE> & right, lint l_dims_merge, lint r_dims_merge)
{
    //
    if (left.m_shape.dim() < 2 || right.m_shape.dim() < 2)
    {
        throw std::invalid_argument(std::string(
            "DMatrix multiplication does not allow matrices of less than 2 dimensions.\n Please extend dimensions of matrices first."));
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
    DMatrix<DTYPE> mat{ left_rows_sh + right_cols_sh };
    DTYPE *left_start = left.m_data;
    DTYPE *right_start = right.m_data;
    DTYPE *mat_p = mat.m_data;

    for (lint i = 0; i < left_rows; ++i)
    {
        for (lint j = 0; j < right_columns; ++j)
        {
            *mat_p = 0.0;
            DTYPE * left_p = left_start;
            DTYPE * right_p = right_start;

            for (lint k = 0; k < left_columns; ++k)
            {
                *mat_p += *left_p * *right_p;
                ++left_p;
                right_p += right_columns;
            }

            ++right_start;
            ++mat_p;
        }

        left_start += left_columns;
        right_start = right.m_data;
    }

    return mat;
}

// DMatrix multiplication
// This function can automatically figure out which dimensions should be merged together.
// For example, two matrices: [7, 3, 3, 5, 4] and [10, 6, 6, 4]
// [3, 5, 4] of the left matrix and [10, 6] of the right matrix will be merged.
// then, the shape of result will be [7, 3, 6, 4].
// This function will throw out an exception if no appropriate dimensions to merge can be found.
template <typename DTYPE>
DMatrix<DTYPE> matmul(const DMatrix<DTYPE> & left, const DMatrix<DTYPE> & right)
{
    /*
    if (1 == left.m_shape.dim() || 1 == right.m_shape.dim())
    {
    if (1 == left.m_shape.dim())
    {
    DMatrix extended_left{ left };
    extended_left.left_extend_shape();
    return extended_left * right;
    }
    else
    {
    DMatrix extended_right{ right };
    extended_right.right_extend_shape();
    return left * extended_right;
    }
    }
    */

    if (left.m_shape.dim() < 2 || right.m_shape.dim() < 2)
    {
        throw std::invalid_argument(std::string(
            "DMatrix multiplication does not allow matrices of less than 2 dimensions.\n Please extend dimensions of matrices first."));
    }

    // Check shape compatibilities between these 2 matrices
    lint left_columns;
    lint left_rows;
    // lint right_rows;
    lint right_columns;

    Shape left_sh = left.m_shape;
    Shape right_sh = right.m_shape;

    Shape left_rows_sh;
    Shape left_cols_sh;

    Shape right_rows_sh;
    Shape right_cols_sh;

    bool can_multiply = false;

    for (lint l = left_sh.dim() - 1, r = 0; l >= 1 && r < right_sh.dim() - 1;)
    {
        left_cols_sh = left_sh.sub_shape(l, left_sh.dim() - 1);
        right_rows_sh = right_sh.sub_shape(0, r);

        if (left_cols_sh.size() == right_rows_sh.size())
        {
            left_rows_sh = left_sh.sub_shape(0, l - 1);
            left_rows = left_rows_sh.size();
            left_columns = left_cols_sh.size();

            // right_rows = left_columns;
            right_cols_sh = right_sh.sub_shape(r + 1, right_sh.dim() - 1);
            right_columns = right_cols_sh.size();

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

    // Calculation

    DMatrix<DTYPE> mat{ left_rows_sh + right_cols_sh };
    DTYPE *left_start = left.m_data;
    DTYPE *right_start = right.m_data;
    DTYPE *mat_p = mat.m_data;

    for (lint i = 0; i < left_rows; ++i)
    {
        for (lint j = 0; j < right_columns; ++j)
        {
            *mat_p = 0;
            DTYPE * left_p = left_start;
            DTYPE * right_p = right_start;

            for (lint k = 0; k < left_columns; ++k)
            {
                *mat_p += *left_p * *right_p;
                ++left_p;
                right_p += right_columns;
            }

            ++right_start;
            ++mat_p;
        }

        left_start += left_columns;
        right_start = right.m_data;
    }

    return mat;
}

// Multiplication element by element.
// The two matrices should have the same shape.
template <typename DTYPE>
DMatrix<DTYPE> multiply(const DMatrix<DTYPE> & left, const DMatrix<DTYPE> & right)
{
    if (left.m_shape != right.m_shape)
    {
        throw std::invalid_argument(invalid_shape);
    }

    lint size = left.m_shape.size();

    DMatrix<DTYPE> mat{ left.m_shape };
    for (lint i = 0; i < size; ++i)
    {
        mat.m_data[i] = left.m_data[i] * right.m_data[i];
    }

    return mat;
}

// Broadcast mode of a * b
template <typename DTYPE>
DMatrix<DTYPE> broadcast_multiply (const DMatrix<DTYPE> &left, const DMatrix<DTYPE> &right)
{
    lint big_size = left.m_shape.size();
    lint small_size = right.m_shape.size();

    if (big_size < small_size || big_size % small_size != 0)
    {
        throw std::invalid_argument(invalid_shape);
    }

    DMatrix<DTYPE> mat{ left.m_shape };
    for (lint i = 0; i < big_size; ++i)
    {
        mat.m_data[i] = left.m_data[i] * right.m_data[i % small_size];
    }

    return mat;
}

// Dot product of two matrices
// The two matrices should have the same amount of elements
template <typename DTYPE>
DTYPE dot_product(const DMatrix<DTYPE> & left, const DMatrix<DTYPE> & right)
{
    if (left.m_shape.size() != right.m_shape.size())
    {
        throw std::invalid_argument(incompatible_size);
    }

    lint size = left.m_shape.size();
    DTYPE sum = 0.0;

    for (lint i = 0; i < size; ++i)
    {
        sum += left.m_data[i] * right.m_data[i];
    }

    return sum;
}

// DMatrix multiplication
// This function can automatically figure out which dimensions should be merged together.
// For example, two matrices: [7, 3, 3, 5, 4] and [10, 6, 6, 4]
// [3, 5, 4] of the left matrix and [10, 6] of the right matrix will be merged.
// then, the shape of result will be [7, 3, 6, 4].
// This function will throw out an exception if no appropriate dimensions to merge can be found.
template <typename DTYPE>
DMatrix<DTYPE> operator * (const DMatrix<DTYPE> &left, const DMatrix<DTYPE> &right)
{
    return multiply(left, right);
}

// Overloading of a * b, b is a scalar
template <typename DTYPE>
DMatrix<DTYPE> operator * (const DMatrix<DTYPE> &left, DTYPE scalar)
{
    DMatrix<DTYPE> mat{ left.m_shape };

    lint size = left.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        mat.m_data[i] = left.m_data[i] * scalar;
    }

    return mat;
}

// Overloading of a * b, a is a scalar
template <typename DTYPE>
DMatrix<DTYPE> operator * (DTYPE scalar, const DMatrix<DTYPE> &right)
{
    DMatrix<DTYPE> mat{ right.m_shape };

    lint size = right.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        mat.m_data[i] = right.m_data[i] * scalar;
    }

    return mat;
}

// Overloading of a / b, b is a scalar
template <typename DTYPE>
DMatrix<DTYPE> operator / (const DMatrix<DTYPE> &left, DTYPE scalar)
{
    DMatrix<DTYPE> mat{ left.m_shape };

    lint size = left.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        mat.m_data[i] = left.m_data[i] / scalar;
    }

    return mat;
}

// Calculate power of a matrix
template <typename DTYPE>
DMatrix<DTYPE> matrix_pow(const DMatrix<DTYPE> & mat, int n)
{
    if (n == 1)
    {
        return mat;
    }
    else if (n > 1)
    {
        DMatrix<DTYPE> child = matrix_pow(mat, n / 2);

        if (n % 2 == 0)
        {
            return child * child;
        }
        else
        {
            return child * child * mat;
        }
    }
    else
    {
        return DMatrix<lint>{};
    }
}

// Get a fully transposed matrix of a matrix
template <typename DTYPE>
DMatrix<DTYPE> full_transpose(const DMatrix<DTYPE> & in)
{
    Shape reversed_shape{ reverse(in.m_shape) };
    DMatrix<DTYPE> transposed{ reversed_shape };

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

    DTYPE *ele_pos = transposed.m_data;

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

    return transposed;
}


template <typename DTYPE>
DMatrix<DTYPE> transpose(const DMatrix<DTYPE> & in, lint left_dims)
{
    if (left_dims < 0 || left_dims > in.m_shape.dim())
    {
        throw std::invalid_argument(std::string("invalid size of left dimensions for matrix transpose."));
    }

    if (left_dims == 0 || left_dims == in.m_shape.dim())
    {
        return DMatrix<DTYPE> {in};
    }

    Shape left_sh = in.m_shape.sub_shape(0, left_dims - 1);
    Shape right_sh = in.m_shape.sub_shape(left_dims, in.m_shape.dim() - 1);

    lint left_size = left_sh.size();
    lint right_size = right_sh.size();

    DMatrix<DTYPE> transposed {right_sh + left_sh};

    DTYPE *in_pos = in.m_data;
    DTYPE *t_first_row_col = transposed.m_data;

    for (lint i = 0; i < left_size; ++i)
    {
        DTYPE *t_pos = t_first_row_col;

        for (lint j = 0; j < right_size; ++j)
        {
            *t_pos = *in_pos;

            ++in_pos;
            t_pos += left_size;
        }

        ++t_first_row_col;
    }

    return transposed;
}

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

It seems this kind of calculation is needed in back propagation.
Perhaps there is a better way to name this function.
*/
template <typename DTYPE>
DMatrix<DTYPE> scale_one_dimension(const DMatrix<DTYPE> & in, lint in_dim, const Vector & scales)
{
    DMatrix<> mat{ in };
    mat.scale_one_dimension(in_dim, scales);
    return mat;
}

/* Overloading of output stream << operator
The following example is output of a matrix of shape [2, 4, 3, 5]:
[
[
[
[  -0.146382    0.13453      -1.87138     0.46065      -0.214253    ]
[  0.163712     -0.827944    0.298595     1.05547      0.0102154    ]
[  1.17457      -0.546841    -1.04944     0.660682     -0.625276    ]
]
[
[  1.48596      -0.829081    -2.55912     -0.888707    -0.539781    ]
[  1.01922      -0.628956    -0.482589    0.339587     -0.121306    ]
[  2.10886      -0.371003    -0.287389    -2.30144     -1.05935     ]
]
[
[  -0.0615274   1.45502      1.35433      0.925328     -0.243275    ]
[  1.51561      0.197497     1.00886      0.439499     0.438945     ]
[  0.645743     -0.128149    -1.68599     1.77643      -0.613857    ]
]
[
[  0.469861     -0.582398    0.668493     -0.103692    0.149386     ]
[  0.624049     1.53727      1.17067      1.07825      -2.05006     ]
[  1.17196      -1.45473     0.136395     -1.11552     -1.71463     ]
]
]
[
[
[  1.12422      -1.73985     -1.47975     -1.58694     1.48247      ]
[  -0.727862    0.754843     -0.1128      0.984235     0.326633     ]
[  -1.03745     -0.0764704   -2.08402     0.389231     0.243215     ]
]
[
[  0.455092     0.275194     2.91628      0.272422     -3.20464     ]
[  1.86225      -2.09501     1.05544      0.310367     -0.00122802  ]
[  0.404831     -1.08115     1.41863      -0.400148    0.926096     ]
]
[
[  -0.358203    0.126072     0.387892     -0.569566    -0.634654    ]
[  0.882249     -0.677104    0.204175     1.35715      -2.453       ]
[  -0.315325    -0.379922    -0.608541    1.35717      0.0195746    ]
]
[
[  1.32359      -0.0912438   -0.208138    -1.61209     0.281664     ]
[  0.785215     -0.316253    0.353801     -0.271609    -1.77443     ]
[  -0.0590157   -1.53723     -0.0539041   0.386642     0.129153     ]
]
]
]
*/
template <typename DTYPE>
std::ostream & operator << (std::ostream & os, const DMatrix<DTYPE> & m)
{
    return os << m.to_string();
}

} // namespace la
} // namespace julie

#include <algorithm>
#include <stdexcept>
#include <iterator>
#include <sstream>
#include <cstring>
#include <functional>

template <typename DTYPE>
julie::la::DMatrix<DTYPE>::DMatrix()
    : m_shape{}, m_data{ nullptr }
{}

template <typename DTYPE>
julie::la::DMatrix<DTYPE>::DMatrix(const Shape & shape)
    : m_shape{ shape }
{
    if (shape.m_size < 1)
    {
        this->m_data = nullptr;
    }
    else
    {
        m_data = new DTYPE[this->m_shape.m_size];
    }
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE>::DMatrix(DTYPE value, const Shape & shape)
    : DMatrix{ shape }
{
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] = value;
    }
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE>::DMatrix(const std::vector<DMatrix>& matrices)
    : DMatrix{}
{
    lint array_size = matrices.size();

    if (array_size > 0)
    {
        Shape mat_sh = matrices[0].m_shape;
        lint mat_size = mat_sh.m_size;

        if (mat_size > 0)
        {
            for (lint i = 1; i < array_size; ++i)
            {
                if (matrices[i].m_shape != mat_sh)
                {
                    throw std::invalid_argument(
                        std::string("DMatrix::DMatrix: invalid matrix array because of different shapes of matrices"));
                }
            }

            this->m_shape = Shape{ array_size } +mat_sh;
            this->m_data = new DTYPE[this->m_shape.m_size];
            DTYPE *this_pos = this->m_data;
            DTYPE *that_pos;

            for (lint i = 0; i < array_size; ++i)
            {
                that_pos = matrices[i].m_data;
                for (lint j = 0; j < mat_size; ++j)
                {
                    *this_pos = *that_pos;
                    ++this_pos;
                    ++that_pos;
                }
            }
        }
    }
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE>::DMatrix(const DMatrix & other)
    : m_shape{ other.m_shape }
{
    lint size = m_shape.m_size;
    m_data = new DTYPE[size];

    std::memcpy(m_data, other.m_data, size * sizeof(DTYPE));
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE>::DMatrix(DMatrix && other)
    : m_shape{ std::move(other.m_shape) }, m_data{ other.m_data }
{
    other.m_data = nullptr;
}

/*
template <typename DTYPE>
julie::la::DMatrix<DTYPE>::DMatrix(const Vector & vec, bool horizontal)
    : m_shape{ vec.m_dim, 1 }
{
    if (horizontal)
    {
        m_shape.m_data[0] = 1;
        m_shape.m_data[1] = vec.m_dim;
    }

    lint size = m_shape.m_size;
    m_data = new DTYPE[size];

    std::memcpy(this->m_data, vec.m_data, size * sizeof(DTYPE));
}
*/

template <typename DTYPE>
julie::la::DMatrix<DTYPE>::DMatrix(const std::vector<DTYPE> & vec, bool horizontal)
    : m_shape{ static_cast<lint>(vec.size()), 1 }
{
    if (horizontal)
    {
        m_shape.m_data[0] = 1;
        m_shape.m_data[1] = vec.size();
    }

    lint size = m_shape.m_size;
    m_data = new DTYPE[size];

    for (lint i = 0; i < size; ++i)
    {
        this->m_data[i] = vec[i];
    }
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE>::DMatrix(const std::vector<std::vector<DTYPE>> & array)
    : m_shape{ static_cast<lint>(array.size()),
               static_cast<lint>(array.size() > 0 ? array[0].size() : 0) }
{
    //Check validity of list
    lint n_cols = this->m_shape.m_data[1];
    for (auto itr = array.begin(); itr != array.end(); ++itr)
    {
        if (itr->size() != n_cols)
        {
            throw std::invalid_argument(std::string("Each row should have the same number of elements!"));
        }
    }

    lint size = this->m_shape.m_size;
    this->m_data = new DTYPE[size];

    DTYPE *pos = this->m_data;
    for (auto itr = array.begin(); itr != array.end(); ++itr)
    {
        DTYPE *pos1 = pos;
        for (auto itr1 = itr->begin(); itr1 != itr->end(); ++itr1, ++pos1)
        {
            *pos1 = *itr1;
        }
        pos += this->m_shape.m_data[1];
    }
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE>::~DMatrix()
{
    delete[]this->m_data;
}

template<typename DTYPE>
inline typename julie::la::DMatrix<DTYPE>::diterator julie::la::DMatrix<DTYPE>::begin() const
{
    diterator itr{ this->m_shape.m_size, this->m_data, 0 };
    return itr;
}

template<typename DTYPE>
inline typename julie::la::DMatrix<DTYPE>::diterator julie::la::DMatrix<DTYPE>::end() const
{
    diterator itr{ this->m_shape.m_size, this->m_data, this->m_shape.m_size };
    return itr;
}

template<typename DTYPE>
julie::la::DMatrix<DTYPE> julie::la::DMatrix<DTYPE>::get_full_transpose() const
{
    return full_transpose(*this);
}

template<typename DTYPE>
julie::la::DMatrix<DTYPE> julie::la::DMatrix<DTYPE>::get_transpose(lint left_dims) const
{
    return transpose(*this, left_dims);
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE> & julie::la::DMatrix<DTYPE>::operator = (const DMatrix & other)
{
    delete[]this->m_data;
    this->m_shape = other.m_shape;

    lint size = this->m_shape.m_size;
    this->m_data = new DTYPE[size];

    std::memcpy(this->m_data, other.m_data, size * sizeof(DTYPE));

    return *this;
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE> & julie::la::DMatrix<DTYPE>::operator = (DMatrix && other)
{
    delete[]m_data;
    this->m_shape = std::move(other.m_shape);

    this->m_data = other.m_data;
    other.m_data = nullptr;

    return *this;
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE> & julie::la::DMatrix<DTYPE>::operator = (DTYPE scalar)
{
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] = scalar;
    }

    return *this;
}

template <typename DTYPE>
DTYPE julie::la::DMatrix<DTYPE>::at(const Coordinate & pos) const
{
    if (this->m_shape != pos.m_shape)
    {
        throw std::invalid_argument(invalid_coordinate);
    }

    return this->m_data[pos.index()];
}

template <typename DTYPE>
DTYPE & julie::la::DMatrix<DTYPE>::at(const Coordinate & pos)
{
    if (this->m_shape != pos.m_shape)
    {
        throw std::invalid_argument(invalid_coordinate);
    }

    return this->m_data[pos.index()];
}

template <typename DTYPE>
DTYPE julie::la::DMatrix<DTYPE>::operator [] (const Coordinate & pos) const
{
    if (this->m_shape != pos.m_shape)
    {
        throw std::invalid_argument(invalid_coordinate);
    }

    return this->m_data[pos.index()];
}

template <typename DTYPE>
DTYPE & julie::la::DMatrix<DTYPE>::operator [] (const Coordinate & pos)
{
    if (this->m_shape != pos.m_shape)
    {
        throw std::invalid_argument(invalid_coordinate);
    }

    return this->m_data[pos.index()];
}

template <typename DTYPE>
DTYPE julie::la::DMatrix<DTYPE>::operator [] (std::initializer_list<lint> list) const
{
    Coordinate pos{ list, this->m_shape };
    return this->m_data[pos.index()];
}

template <typename DTYPE>
DTYPE & julie::la::DMatrix<DTYPE>::operator [] (std::initializer_list<lint> list)
{
    Coordinate pos{ list, this->m_shape };
    return this->m_data[pos.index()];
}

template <typename DTYPE>
julie::la::Vector julie::la::DMatrix<DTYPE>::flaten() const
{
    lint size = this->m_shape.m_size;
    Vector vec = Vector(size);
    for (lint i = 0; i < size; ++i)
    {
        vec[i] = this->m_data[i];
    }

    return vec;
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE> & julie::la::DMatrix<DTYPE>::operator += (const DMatrix & other)
{
    if (m_shape.m_size != other.m_shape.m_size)
    {
        throw std::invalid_argument(invalid_shape);
    }

    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] += other.m_data[i];
    }

    return *this;
}


template <typename DTYPE>
julie::la::DMatrix<DTYPE> & julie::la::DMatrix<DTYPE>::operator -= (const DMatrix & other)
{
    if (m_shape.m_size != other.m_shape.m_size)
    {
        throw std::invalid_argument(invalid_shape);
    }

    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] -= other.m_data[i];
    }

    return *this;
}


template <typename DTYPE>
julie::la::DMatrix<DTYPE> & julie::la::DMatrix<DTYPE>::operator += (DTYPE scalar)
{
    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] += scalar;
    }

    return *this;
}


template <typename DTYPE>
julie::la::DMatrix<DTYPE> & julie::la::DMatrix<DTYPE>::operator -= (DTYPE scalar)
{
    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] -= scalar;
    }

    return *this;
}


template <typename DTYPE>
julie::la::DMatrix<DTYPE> & julie::la::DMatrix<DTYPE>::operator *= (DTYPE scalar)
{
    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] *= scalar;
    }

    return *this;
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE> & julie::la::DMatrix<DTYPE>::operator /= (DTYPE scalar)
{
    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        m_data[i] /= scalar;
    }

    return *this;
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE> & julie::la::DMatrix<DTYPE>::gaussian_random(DTYPE mu, DTYPE sigma)
{
    std::normal_distribution<DTYPE> distribution{ mu, sigma };

    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        this->m_data[i] = distribution(global::global_rand_engine);
    }

    return *this;
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE> & julie::la::DMatrix<DTYPE>::uniform_random(DTYPE min, DTYPE max)
{
    if (min >= max)
    {
        throw std::invalid_argument(std::string("DMatrix::uniform_random: min should be smaller than max"));
    }

    std::uniform_real_distribution<DTYPE> distribution{ min, max };

    lint size = m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        this->m_data[i] = distribution(global::global_rand_engine);
    }

    return *this;
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE> & julie::la::DMatrix<DTYPE>::normalize(DTYPE min, DTYPE max)
{
    if (min >= max)
    {
        throw std::invalid_argument(std::string("DMatrix::normalize: min should be smaller than max"));
    }

    lint size = m_shape.m_size;
    DTYPE range = max - min;

    DTYPE l_max = std::numeric_limits<DTYPE>::max() * (-1);
    DTYPE l_min = std::numeric_limits<DTYPE>::max();

    for (lint i = 0; i < size; ++i)
    {
        if (this->m_data[i] > l_max)
        {
            l_max = this->m_data[i];
        }

        if (this->m_data[i] < l_min)
        {
            l_min = this->m_data[i];
        }
    }

    DTYPE l_range = l_max - l_min;
    if (0 == l_range)
    {
        this->uniform_random(min, max);
    }
    else
    {
        for (lint i = 0; i < size; ++i)
        {
            this->m_data[i] = min + ((this->m_data[i] - l_min) / l_range) * range;
        }
    }

    return *this;
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE> & julie::la::DMatrix<DTYPE>::normalize()
{
    lint size = this->m_shape.m_size;

    if (0 == size)
    {
        throw std::bad_function_call();
    }

    DTYPE mean = 0;
    for (lint i = 0; i < size; ++i)
    {
        mean += this->m_data[i];
    }
    mean /= size;

    DTYPE var = 0;
    DTYPE sub;
    for (lint i = 0; i < size; ++i)
    {
        sub = this->m_data[i] - mean;
        var += sub * sub;
    }
    var /= size;
    var = sqrt(var);

    if (0 == var)
    {
        this->gaussian_random(0, 1);
    }
    else
    {
        for (lint i = 0; i < size; ++i)
        {
            this->m_data[i] = (this->m_data[i] - mean) / var;
        }
    }

    return *this;
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE> & julie::la::DMatrix<DTYPE>::reshape(const Shape & shape)
{
    if (shape.m_size != this->m_shape.m_size)
    {
        throw std::invalid_argument(
            std::string("DMatrix::reshape: the new shape should be compatible with number of elements in this matrix"));
    }

    this->m_shape = shape;

    return *this;
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE> & julie::la::DMatrix<DTYPE>::left_extend_shape()
{
    this->m_shape.left_extend();
    return *this;
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE> & julie::la::DMatrix<DTYPE>::right_extend_shape()
{
    this->m_shape.right_extend();
    return *this;
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE> julie::la::DMatrix<DTYPE>::get_left_extended(lint duplicate) const
{
    if (duplicate < 1)
    {
        throw std::invalid_argument(std::string("julie::la::DMatrix::get_left_extended: duplicate should be a positive value"));
    }

    if (this->m_shape.m_size < 1)
    {
        throw std::bad_function_call();
    }

    DMatrix mat{ Shape{ duplicate } +this->m_shape };
    lint size = this->m_shape.m_size;
    DTYPE *mat_start = mat.m_data;

    for (lint j = 0; j < duplicate; ++j)
    {
        std::memcpy(mat_start, this->m_data, size * sizeof(DTYPE));
        mat_start += size;
    }

    return mat;
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE> & julie::la::DMatrix<DTYPE>::scale_one_dimension(lint dim, const Vector & scales)
{
    if (dim < 0 || dim >= this->m_shape.m_dim)
    {
        throw std::invalid_argument("julie::la::DMatrix::scale_one_dimension: dimension index out of range.");
    }

    if (this->m_shape.m_data[dim] != scales.dim())
    {
        throw std::invalid_argument(
            "julie::la::DMatrix::scale_one_dimension: size of the vector is not compatible with size of this matrix dimension.");
    }

    lint size_dims_before = this->m_shape.sub_shape(0, dim - 1).m_size;
    lint size_dims_after = this->m_shape.sub_shape(dim + 1, this->m_shape.m_dim - 1).m_size;
    lint size_dims_and_after = this->m_shape.sub_shape(dim, this->m_shape.m_dim - 1).m_size;

    if (0 == size_dims_before)
    {
        size_dims_before = 1;
    }

    if (0 == size_dims_after)
    {
        size_dims_after = 1;
    }

    DTYPE *start = this->m_data;

    for (lint i = 0; i < scales.dim(); ++i)
    {
        DTYPE scale = scales[i];
        DTYPE *l_start = start;

        for (lint j = 0; j < size_dims_before; ++j)
        {
            DTYPE *ele = l_start;
            for (lint k = 0; k < size_dims_after; ++k)
            {
                *ele *= scale;
                ++ele;
            }
            l_start += size_dims_and_after;
        }

        start += size_dims_after;
    }

    return *this;
}

template <typename DTYPE>
julie::la::Coordinate julie::la::DMatrix<DTYPE>::argmax() const
{
    DTYPE max = std::numeric_limits<DTYPE>::max() * (-1);
    lint argmax_index = 0;
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        if (this->m_data[i] > max)
        {
            max = this->m_data[i];
            argmax_index = i;
        }
    }

    Coordinate coord{ this->m_shape };
    lint marker = argmax_index;
    for (lint i = this->m_shape.m_dim - 1; i >= 0; --i)
    {
        coord.m_data[i] = marker % this->m_shape.m_data[i];
        marker /= this->m_shape.m_data[i];
    }

    return coord;
}

template <typename DTYPE>
DTYPE julie::la::DMatrix<DTYPE>::max() const
{
    DTYPE max = std::numeric_limits<DTYPE>::max() * (-1);
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        if (this->m_data[i] > max)
        {
            max = this->m_data[i];
        }
    }

    return max;
}

template <typename DTYPE>
julie::la::Coordinate julie::la::DMatrix<DTYPE>::argmin() const
{
    DTYPE min = std::numeric_limits<DTYPE>::max() * (-1);
    lint argmin_index = 0;
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        if (this->m_data[i] < min)
        {
            min = this->m_data[i];
            argmin_index = i;
        }
    }

    Coordinate coord{ this->m_shape };
    lint marker = argmin_index;
    for (lint i = this->m_shape.m_dim - 1; i >= 0; --i)
    {
        coord.m_data[i] = marker % this->m_shape.m_data[i];
        marker /= this->m_shape.m_data[i];
    }

    return coord;
}

template <typename DTYPE>
DTYPE julie::la::DMatrix<DTYPE>::min() const
{
    DTYPE min = std::numeric_limits<DTYPE>::max();
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        if (this->m_data[i] < min)
        {
            min = this->m_data[i];
        }
    }

    return min;
}

template <typename DTYPE>
DTYPE julie::la::DMatrix<DTYPE>::sum() const
{
    lint size = this->m_shape.m_size;
    DTYPE sum = 0;
    for (lint i = 0; i < size; ++i)
    {
        sum += this->m_data[i];
    }

    return sum;
}

template <typename DTYPE>
DTYPE julie::la::DMatrix<DTYPE>::mean() const
{
    return this->sum() / this->m_shape.m_size;
}

template <typename DTYPE>
DTYPE julie::la::DMatrix<DTYPE>::variance() const
{
    DTYPE mean = this->mean();

    lint size = this->m_shape.m_size;
    DTYPE sum = 0;
    DTYPE sub;
    for (lint i = 0; i < size; ++i)
    {
        sub = this->m_data[i] - mean;
        sum += sub * sub;
    }

    return sum / size;
}

template <typename DTYPE>
std::vector<julie::la::DMatrix<DTYPE>> julie::la::DMatrix<DTYPE>::get_collapsed(lint dim) const
{
    if (dim < 0 || dim >= this->m_shape.m_dim)
    {
        throw std::invalid_argument("julie::la::DMatrix::scale_one_dimension: dimension index out of range.");
    }

    Shape sh_before = this->m_shape.sub_shape(0, dim - 1);
    Shape sh_after = this->m_shape.sub_shape(dim + 1, this->m_shape.m_dim - 1);
    Shape sh_collapsed = sh_before + sh_after;

    lint size_dims_before = sh_before.m_size;
    lint size_dims_after = sh_after.m_size;
    lint size_dims_and_after = this->m_shape.sub_shape(dim, this->m_shape.m_dim - 1).m_size;

    if (0 == size_dims_before)
    {
        size_dims_before = 1;
    }

    if (0 == size_dims_after)
    {
        size_dims_after = 1;
    }

    std::vector<DMatrix> all_collapsed;
    DTYPE *start = this->m_data;

    for (lint i = 0; i < this->m_shape.m_data[dim]; ++i)
    {
        DMatrix collapsed{ sh_collapsed };
        DTYPE *l_start = start;
        DTYPE *clps_ele = collapsed.m_data;

        for (lint j = 0; j < size_dims_before; ++j)
        {
            DTYPE *ele = l_start;
            for (lint k = 0; k < size_dims_after; ++k)
            {
                *clps_ele = *ele;
                ++clps_ele;
                ++ele;
            }
            l_start += size_dims_and_after;
        }

        all_collapsed.push_back(collapsed);
        start += size_dims_after;
    }

    return all_collapsed;
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE> julie::la::DMatrix<DTYPE>::get_fused(lint dim) const
{
    if (dim < 0 || dim >= this->m_shape.m_dim)
    {
        throw std::invalid_argument("julie::la::DMatrix::fuse: dimension index out of range.");
    }

    Shape sh_before = this->m_shape.sub_shape(0, dim - 1);
    Shape sh_after = this->m_shape.sub_shape(dim + 1, this->m_shape.m_dim - 1);
    Shape sh_collapsed = sh_before + sh_after;

    lint size_dims_before = sh_before.m_size;
    lint size_dims_after = sh_after.m_size;
    lint size_dims_and_after = this->m_shape.sub_shape(dim, this->m_shape.m_dim - 1).m_size;

    if (0 == size_dims_before)
    {
        size_dims_before = 1;
    }

    if (0 == size_dims_after)
    {
        size_dims_after = 1;
    }

    DMatrix fused{ 0, sh_collapsed };
    DTYPE *start = this->m_data;

    for (lint i = 0; i < this->m_shape.m_data[dim]; ++i)
    {
        DMatrix collapsed{ sh_collapsed };
        DTYPE *l_start = start;
        DTYPE *clps_ele = collapsed.m_data;

        for (lint j = 0; j < size_dims_before; ++j)
        {
            DTYPE *ele = l_start;
            for (lint k = 0; k < size_dims_after; ++k)
            {
                *clps_ele = *ele;
                ++clps_ele;
                ++ele;
            }
            l_start += size_dims_and_after;
        }

        fused += collapsed;
        start += size_dims_after;
    }

    return fused;
}

template <typename DTYPE>
julie::la::DMatrix<DTYPE> julie::la::DMatrix<DTYPE>::get_reduce_mean(lint dim) const
{
    if (dim < 0 || dim >= this->m_shape.m_dim)
    {
        throw std::invalid_argument("julie::la::DMatrix::fuse: dimension index out of range.");
    }

    lint dim_size = this->m_shape.m_data[dim];
    return this->get_fused(dim) / static_cast<DTYPE>(dim_size);
}

template <typename DTYPE>
DTYPE julie::la::DMatrix<DTYPE>::euclidean_norm() const
{
    // Calculate the norm
    DTYPE norm = 0;
    lint size = this->m_shape.m_size;
    for (lint i = 0; i < size; ++i)
    {
        norm += this->m_data[i] * this->m_data[i];
    }

    return norm;
}

template <typename DTYPE>
julie::la::Shape julie::la::DMatrix<DTYPE>::shape() const
{
    return m_shape;
}

template <typename DTYPE>
int julie::la::DMatrix<DTYPE>::PRINT_PRECISION = 5;

template <typename DTYPE>
void julie::la::DMatrix<DTYPE>::print(std::ostream & os, lint dim_index, const DTYPE *start, int integral_len, int dot_len, int frag_len) const
{
    if (dim_index >= this->m_shape.m_dim)
    {
        return;
    }

    for (lint i = 0; i < dim_index; ++i)
    {
        os << "    ";
    }

    if (dim_index == this->m_shape.dim() - 1)
    {
        os << "[";
    }
    else
    {
        os << "[\n";
    }

    lint dim_size = this->m_shape[dim_index];
    lint jump_dist = 1;

    for (lint i = this->m_shape.dim() - 1; i > dim_index; --i)
    {
        jump_dist *= this->m_shape[i];
    }

    for (lint i = 0; i < dim_size; i++)
    {
        if (dim_index == this->m_shape.dim() - 1)
        {
            DTYPE val = *(start + i);

            std::tuple<int, int, int> tuple = this->integral_size(val);
            int i_len = std::get<0>(tuple);
            int d_len = std::get<1>(tuple);
            int f_len = std::get<2>(tuple);

            for (int i = 0; i < integral_len - i_len + 1; ++i)
            {
                os << " ";
            }

            os << val;

            for (int i = 0; i < dot_len + frag_len - f_len - d_len + 1; ++i)
            {
                os << " ";
            }
        }
        else
        {
            this->print(os, dim_index + 1, start + i * jump_dist, integral_len, dot_len, frag_len);
        }
    }

    if (dim_index == this->m_shape.dim() - 1)
    {
        os << "]\n";
    }
    else
    {
        for (lint i = 0; i < dim_index; ++i)
        {
            os << "    ";
        }
        os << "]\n";
    }
}

template <typename DTYPE>
std::tuple<int, int, int> julie::la::DMatrix<DTYPE>::integral_size(DTYPE val) const
{
    //std::tuple<int, int> size_pair
    std::ostringstream o_stream;
    o_stream << std::fixed;
    o_stream.precision(julie::la::DMatrix<DTYPE>::PRINT_PRECISION);

    o_stream << val;

    std::string val_str = o_stream.str();

    int integral_len = 0;
    int dot_len = 0;
    int frag_len = 0;

    bool integral = true;

    for (char & ch : val_str)
    {
        if (ch == '.')
        {
            integral = false;
            dot_len = 1;
        }
        else
        {
            if (integral)
            {
                ++integral_len;
            }
            else
            {
                ++frag_len;
            }
        }
    }

    return std::make_tuple<int, int, int>(integral_len++, dot_len++, frag_len++);
}

template <typename DTYPE>
inline std::string julie::la::DMatrix<DTYPE>::to_string() const
{
    lint size = this->m_shape.size();

    int integral_len = 0;
    int dot_len = 0;
    int frag_len = 0;

    for (lint i = 0; i < size; ++i)
    {
        DTYPE val = this->m_data[i];

        std::tuple<int, int, int> tuple = this->integral_size(val);
        int i_len = std::get<0>(tuple);
        int d_len = std::get<1>(tuple);
        int f_len = std::get<2>(tuple);

        if (i_len > integral_len)
        {
            integral_len = i_len;
        }

        if (f_len > frag_len)
        {
            frag_len = f_len;
        }

        if (d_len > dot_len)
        {
            dot_len = d_len;
        }
    }

    std::ostringstream o_stream;
    o_stream << std::fixed;
    o_stream.precision(julie::la::DMatrix<DTYPE>::PRINT_PRECISION);

    this->print(o_stream, 0, this->m_data, integral_len, dot_len, frag_len);

    return o_stream.str();
}

