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

#include "iMatrix_func.hpp"
#include "utilities.hpp"
#include "Matrix_CPU_func.hpp"
#ifdef WITH_CUDA
#include "Matrix_CUDA_func.hpp"
#endif


namespace julie
{
namespace la
{

template <typename DT>
bool renew_if_shape_not_match(iMatrix<DT> &cache, const Shape &sh)
{
    if (cache.get_matrix_type() == MatrixType::CPU)
    {
        return julie::la::cpu::renew_if_shape_not_match (*(cache.get_cpu_instance()), sh );
    }
    else if (cache.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        return julie::la::cuda::renew_if_shape_not_match (*(cache.get_cuda_instance()), sh );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (cache.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
bool renew_if_shape_not_match(iMatrix<int> &cache, const Shape &sh);
template
bool renew_if_shape_not_match(iMatrix<float> &cache, const Shape &sh);


// Overloading of a == b
template <typename DT>
bool operator == (const iMatrix<DT> &left, const iMatrix<DT> &right)
{
    if (left.get_matrix_type() != right.get_matrix_type())
    {
        throw std::invalid_argument {
            std::string {"Matrix type should be the same for all matrices in "} +
            std::string {__FUNCTION__} };
    }

    if (left.get_matrix_type() == MatrixType::CPU)
    {
        return *(left.get_cpu_instance()) == *(right.get_cpu_instance());
    }
    else if (left.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        return *(left.get_cuda_instance()) == *(right.get_cuda_instance());
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (left.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
bool operator == (const iMatrix<int> &left, const iMatrix<int> &right);
template
bool operator == (const iMatrix<float> &left, const iMatrix<float> &right);


// Overloading of a != b
template <typename DT>
bool operator != (const iMatrix<DT> &left, const iMatrix<DT> &right)
{
    return !(left == right);
}
template
bool operator != (const iMatrix<int> &left, const iMatrix<int> &right);
template
bool operator != (const iMatrix<float> &left, const iMatrix<float> &right);


// Overloading of a + b
template <typename DT>
void add (iMatrix<DT> &output, const iMatrix<DT> &left, const iMatrix<DT> &right)
{
    if (left.get_matrix_type() != right.get_matrix_type())
    {
        throw std::invalid_argument {
            std::string {"Matrix type should be the same for all matrices in "} +
            std::string {__FUNCTION__} };
    }

    output.set_matrix_type(left.get_matrix_type());

    if (left.get_matrix_type() == MatrixType::CPU)
    {
        cpu::add (*(output.get_cpu_instance()), *(left.get_cpu_instance()), *(right.get_cpu_instance()));
    }
    else if (left.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::add (*(output.get_cuda_instance()), *(left.get_cuda_instance()), *(right.get_cuda_instance()));
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (left.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void add (iMatrix<int> &output, const iMatrix<int> &left, const iMatrix<int> &right);
template
void add (iMatrix<float> &output, const iMatrix<float> &left, const iMatrix<float> &right);


// Overloading of a - b
template <typename DT>
void subtract (iMatrix<DT> &output, const iMatrix<DT> &left, const iMatrix<DT> &right)
{
    if (left.get_matrix_type() != right.get_matrix_type())
    {
        throw std::invalid_argument {
            std::string {"Matrix type should be the same for all matrices in "} +
            std::string {__FUNCTION__} };
    }

    output.set_matrix_type(left.get_matrix_type());

    if (left.get_matrix_type() == MatrixType::CPU)
    {
        cpu::subtract(*(output.get_cpu_instance()), *(left.get_cpu_instance()), *(right.get_cpu_instance()));
    }
    else if (left.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::subtract(*(output.get_cuda_instance()), *(left.get_cuda_instance()), *(right.get_cuda_instance()));
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (left.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void subtract (iMatrix<int> &output, const iMatrix<int> &left, const iMatrix<int> &right);
template
void subtract (iMatrix<float> &output, const iMatrix<float> &left, const iMatrix<float> &right);


// Broadcast mode of a + b
template <typename DT>
void broadcast_add (iMatrix<DT> &output, const iMatrix<DT> &left, const iMatrix<DT> &right)
{
    if (left.get_matrix_type() != right.get_matrix_type())
    {
        throw std::invalid_argument {
            std::string {"Matrix type should be the same for all matrices in "} +
            std::string {__FUNCTION__} };
    }

    output.set_matrix_type(left.get_matrix_type());

    if (left.get_matrix_type() == MatrixType::CPU)
    {
        cpu::broadcast_add(*(output.get_cpu_instance()), *(left.get_cpu_instance()), *(right.get_cpu_instance()));
    }
    else if (left.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::broadcast_add(*(output.get_cuda_instance()), *(left.get_cuda_instance()), *(right.get_cuda_instance()) );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (left.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void broadcast_add (iMatrix<int> &output, const iMatrix<int> &left, const iMatrix<int> &right);
template
void broadcast_add (iMatrix<float> &output, const iMatrix<float> &left, const iMatrix<float> &right);


// iMatrix multiplication
// Which dimensions should be merged together is manually defined here.
// For example, two matrices: [7, 3, 2, 5, 6] and [6, 5, 6, 4], if l_dims_merge == 2 and r_dims_merge == 2,
// the shape of result will be [7, 3, 2, 6, 4]. However, if l_dims_merge == 4 and r_dims_merge == 3,
// the shape of output will be [7, 4].
// Note: Total sizes (number of elements) of dimensions to merge should be equal betweem the left and right.
template <typename DT>
void matmul(iMatrix<DT> &output, const iMatrix<DT> &left, const iMatrix<DT> &right, lint l_dims_merge, lint r_dims_merge)
{
    if (left.get_matrix_type() != right.get_matrix_type())
    {
        throw std::invalid_argument {
            std::string {"Matrix type should be the same for all matrices in "} +
            std::string {__FUNCTION__} };
    }

    output.set_matrix_type(left.get_matrix_type());

    if (left.get_matrix_type() == MatrixType::CPU)
    {
        cpu::matmul( *(output.get_cpu_instance()), *(left.get_cpu_instance()), *(right.get_cpu_instance()), l_dims_merge, r_dims_merge );
    }
    else if (left.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::matmul( *(output.get_cuda_instance()), *(left.get_cuda_instance()), *(right.get_cuda_instance()), l_dims_merge, r_dims_merge );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (left.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void matmul(
    iMatrix<int> &output, const iMatrix<int> &left, const iMatrix<int> &right, lint l_dims_merge, lint r_dims_merge);
template
void matmul(
    iMatrix<float> &output, const iMatrix<float> &left, const iMatrix<float> &right, lint l_dims_merge, lint r_dims_merge);



// iMatrix multiplication
// This function can automatically figure out which dimensions should be merged together.
// For example, two matrices: [7, 3, 3, 5, 4] and [10, 6, 6, 4]
// [3, 5, 4] of the left matrix and [10, 6] of the right matrix will be merged.
// then, the shape of result will be [7, 3, 6, 4].
// This function will throw out an exception if no appropriate dimensions to merge can be found.
template <typename DT>
void matmul(iMatrix<DT> &output, const iMatrix<DT> &left, const iMatrix<DT> &right)
{
    if (left.get_matrix_type() != right.get_matrix_type())
    {
        throw std::invalid_argument {
            std::string {"Matrix type should be the same for all matrices in "} +
            std::string {__FUNCTION__} };
    }

    output.set_matrix_type(left.get_matrix_type());

    if (left.get_matrix_type() == MatrixType::CPU)
    {
        cpu::matmul( *(output.get_cpu_instance()), *(left.get_cpu_instance()), *(right.get_cpu_instance()) );
    }
    else if (left.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::matmul( *(output.get_cuda_instance()), *(left.get_cuda_instance()), *(right.get_cuda_instance()) );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (left.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void matmul(iMatrix<int> &output, const iMatrix<int> &left, const iMatrix<int> &right);
template
void matmul(iMatrix<float> &output, const iMatrix<float> &left, const iMatrix<float> &right);


// Multiplication element by element.
// The two matrices should have the same shape.
template <typename DT>
void multiply(iMatrix<DT> &output, const iMatrix<DT> & left, const iMatrix<DT> & right)
{
    if (left.get_matrix_type() != right.get_matrix_type())
    {
        throw std::invalid_argument {
            std::string {"Matrix type should be the same for all matrices in "} +
            std::string {__FUNCTION__} };
    }

    output.set_matrix_type(left.get_matrix_type());

    if (left.get_matrix_type() == MatrixType::CPU)
    {
        cpu::multiply( *(output.get_cpu_instance()), *(left.get_cpu_instance()), *(right.get_cpu_instance()) );
    }
    else if (left.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::multiply( *(output.get_cuda_instance()), *(left.get_cuda_instance()), *(right.get_cuda_instance()) );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (left.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void multiply(iMatrix<int> &output, const iMatrix<int> & left, const iMatrix<int> & right);
template
void multiply(iMatrix<float> &output, const iMatrix<float> & left, const iMatrix<float> & right);


// Broadcast mode of a * b
template <typename DT>
void broadcast_multiply (iMatrix<DT> &output, const iMatrix<DT> &left, const iMatrix<DT> &right)
{
    if (left.get_matrix_type() != right.get_matrix_type())
    {
        throw std::invalid_argument {
            std::string {"Matrix type should be the same for all matrices in "} +
            std::string {__FUNCTION__} };
    }

    output.set_matrix_type(left.get_matrix_type());

    if (left.get_matrix_type() == MatrixType::CPU)
    {
        cpu::broadcast_multiply( *(output.get_cpu_instance()), *(left.get_cpu_instance()), *(right.get_cpu_instance()) );
    }
    else if (left.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::broadcast_multiply( *(output.get_cuda_instance()), *(left.get_cuda_instance()), *(right.get_cuda_instance()) );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (left.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void broadcast_multiply (iMatrix<int> &output, const iMatrix<int> &left, const iMatrix<int> &right);
template
void broadcast_multiply (iMatrix<float> &output, const iMatrix<float> &left, const iMatrix<float> &right);


// Dot product of two matrices
// The two matrices should have the same amount of elements
template <typename DT>
DT dot_product(iMatrix<DT> &multiply_cache, const iMatrix<DT> & left, const iMatrix<DT> & right)
{
    if (left.get_matrix_type() != right.get_matrix_type())
    {
        throw std::invalid_argument {
            std::string {"Matrix type should be the same for all matrices in "} +
            std::string {__FUNCTION__} };
    }

    multiply_cache.set_matrix_type(left.get_matrix_type());

    if (left.get_matrix_type() == MatrixType::CPU)
    {
        return cpu::dot_product( *(multiply_cache.get_cpu_instance()), *(left.get_cpu_instance()), *(right.get_cpu_instance()) );
    }
    else if (left.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        return cuda::dot_product( *(multiply_cache.get_cuda_instance()), *(left.get_cuda_instance()), *(right.get_cuda_instance()) );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (left.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
int dot_product(iMatrix<int> &multiply_cache, const iMatrix<int> & left, const iMatrix<int> & right);
template
float dot_product(iMatrix<float> &multiply_cache, const iMatrix<float> & left, const iMatrix<float> & right);


// Overloading of a * b, b is a scalar
template <typename DT>
void multiply (iMatrix<DT> &output, const iMatrix<DT> &left, DT scalar)
{
    output.set_matrix_type(left.get_matrix_type());

    if (left.get_matrix_type() == MatrixType::CPU)
    {
        julie::la::cpu::multiply (*(output.get_cpu_instance()), *(left.get_cpu_instance()), scalar );
    }
    else if (left.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        julie::la::cuda::multiply (*(output.get_cuda_instance()), *(left.get_cuda_instance()), scalar );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (left.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void multiply (iMatrix<int> &output, const iMatrix<int> &left, int scalar);
template
void multiply (iMatrix<float> &output, const iMatrix<float> &left, float scalar);


// Overloading of a / b, b is a scalar
template <typename DT>
void divide (iMatrix<DT> &output, const iMatrix<DT> &left, DT scalar)
{
    output.set_matrix_type(left.get_matrix_type());

    if (left.get_matrix_type() == MatrixType::CPU)
    {
        julie::la::cpu::divide (*(output.get_cpu_instance()), *(left.get_cpu_instance()), scalar );
    }
    else if (left.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        julie::la::cuda::divide (*(output.get_cuda_instance()), *(left.get_cuda_instance()), scalar );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (left.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void divide (iMatrix<int> &output, const iMatrix<int> &left, int scalar);
template
void divide (iMatrix<float> &output, const iMatrix<float> &left, float scalar);


// Calculate power of a matrix
template <typename DT>
void matrix_pow(iMatrix<DT> &output, const iMatrix<DT> & mat, int n)
{
    output.set_matrix_type(mat.get_matrix_type());

    if (mat.get_matrix_type() == MatrixType::CPU)
    {
        cpu::matrix_pow( *(output.get_cpu_instance()), *(mat.get_cpu_instance()), n );
    }
    else if (mat.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::matrix_pow( *(output.get_cuda_instance()), *(mat.get_cuda_instance()), n );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (mat.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void matrix_pow(iMatrix<int> &output, const iMatrix<int> & mat, int n);
template
void matrix_pow(iMatrix<float> &output, const iMatrix<float> & mat, int n);


// Get a fully transposed matrix of a matrix
template <typename DT>
void full_transpose(iMatrix<DT> & transposed, const iMatrix<DT> & in)
{
    transposed.set_matrix_type(in.get_matrix_type());
    
    if (in.get_matrix_type() == MatrixType::CPU)
    {
        cpu::full_transpose( *(transposed.get_cpu_instance()), *(in.get_cpu_instance()) );
    }
    else if (in.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::full_transpose( *(transposed.get_cuda_instance()), *(in.get_cuda_instance()) );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (in.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void full_transpose(iMatrix<int> & transposed, const iMatrix<int> & in);
template
void full_transpose(iMatrix<float> & transposed, const iMatrix<float> & in);


template <typename DT>
void transpose(iMatrix<DT> & transposed, const iMatrix<DT> & in, lint left_dims)
{
    transposed.set_matrix_type(in.get_matrix_type());
    
    if (in.get_matrix_type() == MatrixType::CPU)
    {
        cpu::transpose( *(transposed.get_cpu_instance()), *(in.get_cpu_instance()), left_dims );
    }
    else if (in.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::transpose( *(transposed.get_cuda_instance()), *(in.get_cuda_instance()), left_dims );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (in.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void transpose(iMatrix<int> & transposed, const iMatrix<int> & in, lint left_dims);
template
void transpose(iMatrix<float> & transposed, const iMatrix<float> & in, lint right_dims);


template <typename DT>
void transpose_neighboring_dims(iMatrix<DT> & transposed, const iMatrix<DT> & in,
                lint l_dim_idx1, lint l_dim_idx2, lint r_dim_idx1, lint r_dim_idx2)
{
    transposed.set_matrix_type(in.get_matrix_type());
    
    if (in.get_matrix_type() == MatrixType::CPU)
    {
        cpu::transpose_neighboring_dims( *(transposed.get_cpu_instance()), *(in.get_cpu_instance()),
                                        l_dim_idx1, l_dim_idx2, r_dim_idx1, r_dim_idx2 );
    }
    else if (in.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::transpose_neighboring_dims( *(transposed.get_cuda_instance()), *(in.get_cuda_instance()),
                                        l_dim_idx1, l_dim_idx2, r_dim_idx1, r_dim_idx2 );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (in.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }   
}
template
void transpose_neighboring_dims(iMatrix<int> & transposed, const iMatrix<int> & in,
                    lint l_dim_idx1, lint l_dim_idx2, lint r_dim_idx1, lint r_dim_idx2);
template
void transpose_neighboring_dims(iMatrix<float> & transposed, const iMatrix<float> & in,
                    lint l_dim_idx1, lint l_dim_idx2, lint r_dim_idx1, lint r_dim_idx2);


template <typename DT>
void abs(iMatrix<DT> &output, const iMatrix<DT> &input)
{
    output.set_matrix_type(input.get_matrix_type());

    if (input.get_matrix_type() == MatrixType::CPU)
    {
        cpu::abs (*(output.get_cpu_instance()), *(input.get_cpu_instance()));
    }
    else if (input.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::abs (*(output.get_cuda_instance()), *(input.get_cuda_instance()));
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (input.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void abs(iMatrix<int> &output, const iMatrix<int> &input);
template
void abs(iMatrix<float> &output, const iMatrix<float> &input);


template <typename DT>
void threshold(iMatrix<DT> &mask, const iMatrix<DT> &in, DT th)
{
    mask.set_matrix_type(in.get_matrix_type());

    if (in.get_matrix_type() == MatrixType::CPU)
    {
        cpu::threshold( *(mask.get_cpu_instance()), *(in.get_cpu_instance()), th);
    }
    else if (in.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::threshold( *(mask.get_cuda_instance()), *(in.get_cuda_instance()), th);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (in.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }  
}
template
void threshold(iMatrix<int> &mask, const iMatrix<int> &in, int th);
template
void threshold(iMatrix<float> &mask, const iMatrix<float> &in, float th);


template <typename DT>
void abs_threshold(iMatrix<DT> &mask, const iMatrix<DT> &in, DT th)
{
    mask.set_matrix_type(in.get_matrix_type());

    if (in.get_matrix_type() == MatrixType::CPU)
    {
        cpu::abs_threshold( *(mask.get_cpu_instance()), *(in.get_cpu_instance()), th);
    }
    else if (in.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::abs_threshold( *(mask.get_cuda_instance()), *(in.get_cuda_instance()), th);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (in.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }  
}
template
void abs_threshold(iMatrix<int> &mask, const iMatrix<int> &in, int th);
template
void abs_threshold(iMatrix<float> &mask, const iMatrix<float> &in, float th);


template <typename DT>
void concatenate(iMatrix<DT> &output, const iMatrix<DT> &left, const iMatrix<DT> &right, lint dim)
{
    if (left.get_matrix_type() != right.get_matrix_type())
    {
        throw std::invalid_argument {
            std::string {"Matrix type should be the same for all matrices in "} +
            std::string {__FUNCTION__} };
    }

    output.set_matrix_type(left.get_matrix_type());

    if (left.get_matrix_type() == MatrixType::CPU)
    {
        cpu::concatenate( *(output.get_cpu_instance()), *(left.get_cpu_instance()), *(right.get_cpu_instance()), dim );
    }
    else if (left.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::concatenate( *(output.get_cuda_instance()), *(left.get_cuda_instance()), *(right.get_cuda_instance()), dim );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (left.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void concatenate(iMatrix<int> &output, const iMatrix<int> &left, const iMatrix<int> &right, lint dim);
template
void concatenate(iMatrix<float> &output, const iMatrix<float> &left, const iMatrix<float> &right, lint dim);


template <typename DT>
void slice(iMatrix<DT> &output, const iMatrix<DT> &input, lint dim, lint idx, lint slice_size)
{
    output.set_matrix_type(input.get_matrix_type());

    if (input.get_matrix_type() == MatrixType::CPU)
    {
        cpu::slice( *(output.get_cpu_instance()), *(input.get_cpu_instance()), dim, idx, slice_size );
    }
    else if (input.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::slice( *(output.get_cuda_instance()), *(input.get_cuda_instance()), dim, idx, slice_size );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (input.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void slice(iMatrix<int> &output, const iMatrix<int> &input, lint dim, lint idx, lint slice_size);
template
void slice(iMatrix<float> &output, const iMatrix<float> &input, lint dim, lint idx, lint slice_size);


template <typename DT>
void repeat(iMatrix<DT> &output, const iMatrix<DT> &input, lint dim, lint n_duplicate)
{
    output.set_matrix_type(input.get_matrix_type());

    if (input.get_matrix_type() == MatrixType::CPU)
    {
        cpu::repeat( *(output.get_cpu_instance()), *(input.get_cpu_instance()), dim, n_duplicate );
    }
    else if (input.get_matrix_type() == MatrixType::CUDA)
    {
#ifdef WITH_CUDA
        cuda::repeat( *(output.get_cuda_instance()), *(input.get_cuda_instance()), dim, n_duplicate );
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (input.get_matrix_type() == MatrixType::CL)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void repeat(iMatrix<int> &output, const iMatrix<int> &input, lint dim, lint n_duplicate);
template
void repeat(iMatrix<float> &output, const iMatrix<float> &input, lint dim, lint n_duplicate);


} // namespace la
} // namespace julie