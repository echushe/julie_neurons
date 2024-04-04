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

#include "iMatrix.hpp"
#include "iMatrix_func.hpp"
#include "utilities.hpp"

#include <stdexcept>
#include <sstream>
#include <functional>


namespace julie
{

namespace la
{

template <typename DT>
iMatrix<DT>::iMatrix(MatrixType type)
    :
    m_type{ type },
    m_mat_cpu{ nullptr }
#ifdef WITH_CUDA
    ,m_mat_cuda{ nullptr }
#endif
{
    if (MatrixType::CPU == type)
    {
        m_mat_cpu = std::make_shared<cpu::Matrix_CPU<DT>>();
    }
    else if (MatrixType::CUDA == type)
    {
#ifdef WITH_CUDA
        m_mat_cuda = std::make_shared<cuda::Matrix_CUDA<DT>>();
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
iMatrix<int>::iMatrix(MatrixType type);
template
iMatrix<float>::iMatrix(MatrixType type);


template <typename DT>
iMatrix<DT>::iMatrix(const Shape & shape, MatrixType type)
    :
    m_type{ type },
    m_mat_cpu{ nullptr }
#ifdef WITH_CUDA
    ,m_mat_cuda{ nullptr }
#endif
{
    if (MatrixType::CPU == type)
    {
        m_mat_cpu = std::make_shared<cpu::Matrix_CPU<DT>>(shape);
    }
    else if (MatrixType::CUDA == type)
    {
#ifdef WITH_CUDA
        m_mat_cuda = std::make_shared<cuda::Matrix_CUDA<DT>>(shape);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
iMatrix<int>::iMatrix(const Shape & shape, MatrixType type);
template
iMatrix<float>::iMatrix(const Shape & shape, MatrixType type);


template <typename DT>
iMatrix<DT>::iMatrix(DT value, const Shape & shape, MatrixType type)
    :
    m_type{ type },
    m_mat_cpu{ nullptr }
#ifdef WITH_CUDA
    ,m_mat_cuda{ nullptr }
#endif
{
    if (MatrixType::CPU == type)
    {
        m_mat_cpu = std::make_shared<cpu::Matrix_CPU<DT>>(value, shape);
    }
    else if (MatrixType::CUDA == type)
    {
#ifdef WITH_CUDA
        m_mat_cuda = std::make_shared<cuda::Matrix_CUDA<DT>>(value, shape);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
iMatrix<int>::iMatrix(int value, const Shape & shape, MatrixType type);
template
iMatrix<float>::iMatrix(float value, const Shape & shape, MatrixType type);


// Combine an array if matrices into one matrix
template <typename DT>
iMatrix<DT>::iMatrix(const std::vector<std::shared_ptr<iMatrix<DT>>> & matrices)
    :
    m_type{ MatrixType::UNKNOWN },
    m_mat_cpu{ nullptr }
#ifdef WITH_CUDA
    ,m_mat_cuda{ nullptr }
#endif
{
    if (matrices.empty())
    {
        return;
    }

    // We assume that all matrices in this array are of the same type

    if (MatrixType::CPU == matrices[0]->m_type)
    {
        std::vector<std::shared_ptr<cpu::Matrix_CPU<DT>>> mats;
        for (auto & imat : matrices)
        {
            mats.push_back(imat->m_mat_cpu);
        }
        this->m_type = MatrixType::CPU;
        m_mat_cpu = std::make_shared<cpu::Matrix_CPU<DT>>(mats);

    }
    else if (MatrixType::CUDA == matrices[0]->m_type)
    {
#ifdef WITH_CUDA
        std::vector<std::shared_ptr<cuda::Matrix_CUDA<DT>>> mats;
        for (auto & imat : matrices)
        {
            mats.push_back(imat->m_mat_cuda);
        }
        this->m_type = MatrixType::CUDA;
        m_mat_cuda = std::make_shared<cuda::Matrix_CUDA<DT>>(mats);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == matrices[0]->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
iMatrix<float>::iMatrix(const std::vector<std::shared_ptr<iMatrix<float>>> & matrices);
template
iMatrix<int>::iMatrix(const std::vector<std::shared_ptr<iMatrix<int>>> & matrices);


template <typename DT>
iMatrix<DT>::iMatrix(const std::vector<iMatrix<DT>> & matrices)
    :
    m_type{ MatrixType::UNKNOWN },
    m_mat_cpu{ nullptr }
#ifdef WITH_CUDA
    ,m_mat_cuda{ nullptr }
#endif
{
    if (matrices.empty())
    {
        return;
    }

    // We assume that all matrices in this array are of the same type

    if (MatrixType::CPU == matrices[0].m_type)
    {
        std::vector<std::shared_ptr<cpu::Matrix_CPU<DT>>> mats;
        for (auto & imat : matrices)
        {
            mats.push_back(imat.m_mat_cpu);
        }
        this->m_type = MatrixType::CPU;
        m_mat_cpu = std::make_shared<cpu::Matrix_CPU<DT>>(mats);
    }
    else if (MatrixType::CUDA == matrices[0].m_type)
    {
#ifdef WITH_CUDA
        std::vector<std::shared_ptr<cuda::Matrix_CUDA<DT>>> mats;
        for (auto & imat : matrices)
        {
            mats.push_back(imat.m_mat_cuda);
        }
        this->m_type = MatrixType::CUDA;
        m_mat_cuda = std::make_shared<cuda::Matrix_CUDA<DT>>(mats);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == matrices[0].m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
iMatrix<float>::iMatrix(const std::vector<iMatrix<float>> & matrices);
template
iMatrix<int>::iMatrix(const std::vector<iMatrix<int>> & matrices);


template <typename DT>
iMatrix<DT>::iMatrix(const std::vector<DT> & vec, bool horizontal, MatrixType type)
    :
    m_type{ type },
    m_mat_cpu{ nullptr }
#ifdef WITH_CUDA
    ,m_mat_cuda{ nullptr }
#endif
{
    if (MatrixType::CPU == type)
    {
        m_mat_cpu = std::make_shared<cpu::Matrix_CPU<DT>>(vec, horizontal);
    }
    else if (MatrixType::CUDA == type)
    {
#ifdef WITH_CUDA
        m_mat_cuda = std::make_shared<cuda::Matrix_CUDA<DT>>(vec, horizontal);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
iMatrix<int>::iMatrix(const std::vector<int> & vec, bool horizontal, MatrixType type);
template
iMatrix<float>::iMatrix(const std::vector<float> & vec, bool horizontal, MatrixType type);


template <typename DT>
iMatrix<DT>::iMatrix(const std::vector<DT> & vec, const Shape &shape, MatrixType type)
    :
    m_type{ type },
    m_mat_cpu{ nullptr }
#ifdef WITH_CUDA
    ,m_mat_cuda{ nullptr }
#endif
{
    if (MatrixType::CPU == type)
    {
        m_mat_cpu = std::make_shared<cpu::Matrix_CPU<DT>>(vec, shape);
    }
    else if (MatrixType::CUDA == type)
    {
#ifdef WITH_CUDA
        m_mat_cuda = std::make_shared<cuda::Matrix_CUDA<DT>>(vec, shape);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
iMatrix<int>::iMatrix(const std::vector<int> & vec, const Shape &shape, MatrixType type);
template
iMatrix<float>::iMatrix(const std::vector<float> & vec, const Shape &shape, MatrixType type);


template <typename DT>
iMatrix<DT>::iMatrix(const std::vector<std::vector<DT>> & array, MatrixType type)
    :
    m_type{ type },
    m_mat_cpu{ nullptr }
#ifdef WITH_CUDA
    ,m_mat_cuda{ nullptr }
#endif
{
    if (MatrixType::CPU == type)
    {
        m_mat_cpu = std::make_shared<cpu::Matrix_CPU<DT>>(array);
    }
    else if (MatrixType::CUDA == type)
    {
#ifdef WITH_CUDA
        m_mat_cuda = std::make_shared<cuda::Matrix_CUDA<DT>>(array);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
iMatrix<int>::iMatrix(const std::vector<std::vector<int>> & array, MatrixType type);
template
iMatrix<float>::iMatrix(const std::vector<std::vector<float>> & array, MatrixType type);


// Copy constructor
template <typename DT>
iMatrix<DT>::iMatrix(const iMatrix<DT> & other)
    :
    m_type{ other.m_type },
    m_mat_cpu{ nullptr }
#ifdef WITH_CUDA
    ,m_mat_cuda{ nullptr }
#endif
{
    if (MatrixType::CPU == this->m_type)
    {
        this->m_mat_cpu = std::make_shared<cpu::Matrix_CPU<DT>>(*(other.m_mat_cpu));
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        this->m_mat_cuda = std::make_shared<cuda::Matrix_CUDA<DT>>(*(other.m_mat_cuda));
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
iMatrix<int>::iMatrix(const iMatrix<int> & other);
template
iMatrix<float>::iMatrix(const iMatrix<float> & other);


// Move constructor
template <typename DT>
iMatrix<DT>::iMatrix(iMatrix<DT> && other)
    :
    m_type{ other.m_type },
    m_mat_cpu{ std::move(other.m_mat_cpu) }
#ifdef WITH_CUDA
    ,m_mat_cuda{ std::move(other.m_mat_cuda) }
#endif
{
    other.m_type = MatrixType::UNKNOWN;
    other.m_mat_cpu = nullptr;
#ifdef WITH_CUDA
    other.m_mat_cuda = nullptr;
#endif
}
template
iMatrix<int>::iMatrix(iMatrix<int> && other);
template
iMatrix<float>::iMatrix(iMatrix<float> && other);


template <typename DT>
iMatrix<DT>::~iMatrix()
{}
template
iMatrix<int>::~iMatrix();
template
iMatrix<float>::~iMatrix();


template <typename DT>
iMatrix<DT>::iMatrix(const cpu::Matrix_CPU<DT> & mat_cpu)
    :
    m_type{ MatrixType::CPU },
    m_mat_cpu{ std::make_shared<cpu::Matrix_CPU<DT>>(mat_cpu) }
#ifdef WITH_CUDA
    ,m_mat_cuda{ nullptr }
#endif
{}
template
iMatrix<int>::iMatrix(const cpu::Matrix_CPU<int> & mat_cpu);
template
iMatrix<float>::iMatrix(const cpu::Matrix_CPU<float> & mat_cpu);


template <typename DT>
iMatrix<DT>::iMatrix(cpu::Matrix_CPU<DT> && mat_cpu)
    :
    m_type{ MatrixType::CPU },
    m_mat_cpu{ std::make_shared<cpu::Matrix_CPU<DT>>(std::move(mat_cpu)) }
#ifdef WITH_CUDA
    ,m_mat_cuda{ nullptr }
#endif
{}
template
iMatrix<int>::iMatrix(cpu::Matrix_CPU<int> && mat_cpu);
template
iMatrix<float>::iMatrix(cpu::Matrix_CPU<float> && mat_cpu);


template <typename DT>
iMatrix<DT>::iMatrix(const cuda::Matrix_CUDA<DT> & mat_cuda)
    :
    m_type{ MatrixType::CUDA },
    m_mat_cpu{ nullptr }
#ifdef WITH_CUDA
    ,m_mat_cuda{ std::make_shared<cuda::Matrix_CUDA<DT>>(mat_cuda) }
#endif
{}
template
iMatrix<int>::iMatrix(const cuda::Matrix_CUDA<int> & mat_cuda);
template
iMatrix<float>::iMatrix(const cuda::Matrix_CUDA<float> & mat_cuda);


template <typename DT>
iMatrix<DT>::iMatrix(cuda::Matrix_CUDA<DT> && mat_cuda)
    :
    m_type{ MatrixType::CUDA },
    m_mat_cpu{ nullptr }
#ifdef WITH_CUDA
    ,m_mat_cuda{ std::make_shared<cuda::Matrix_CUDA<DT>>(std::move(mat_cuda)) }
#endif
{}
template
iMatrix<int>::iMatrix(cuda::Matrix_CUDA<int> && mat_cuda);
template
iMatrix<float>::iMatrix(cuda::Matrix_CUDA<float> && mat_cuda);


template <typename DT>
void iMatrix<DT>::get_full_transpose(iMatrix<DT> & out) const
{
    out.set_matrix_type(this->m_type);
    
    if (MatrixType::CPU == this->m_type)
    {
        this->m_mat_cpu->get_full_transpose(*(out.m_mat_cpu));   
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        this->m_mat_cuda->get_full_transpose(*(out.m_mat_cuda));
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void iMatrix<int>::get_full_transpose(iMatrix<int> & out) const;
template
void iMatrix<float>::get_full_transpose(iMatrix<float> & out) const;


template <typename DT>
void iMatrix<DT>::get_transpose(iMatrix<DT> & out, lint left_dims) const
{
    out.set_matrix_type(this->m_type);
    
    if (MatrixType::CPU == this->m_type)
    {
        this->m_mat_cpu->get_transpose(*(out.m_mat_cpu), left_dims);   
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        this->m_mat_cuda->get_transpose(*(out.m_mat_cuda), left_dims);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void iMatrix<int>::get_transpose(iMatrix<int> & out, lint left_dims) const;
template
void iMatrix<float>::get_transpose(iMatrix<float> & out, lint left_dims) const;


template <typename DT>
iMatrix<DT> & iMatrix<DT>::operator = (const iMatrix & other)
{
    this->m_type = other.m_type;

    if (MatrixType::CPU == this->m_type)
    {
        if (this->m_mat_cpu)
        {
            *(this->m_mat_cpu) = *(other.m_mat_cpu);
        }
        else
        {
            this->m_mat_cpu = std::make_shared<cpu::Matrix_CPU<DT>>(*(other.m_mat_cpu));
        }
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        if (this->m_mat_cuda)
        {
            *(this->m_mat_cuda) = *(other.m_mat_cuda);
        }
        else
        {
            this->m_mat_cuda = std::make_shared<cuda::Matrix_CUDA<DT>>(*(other.m_mat_cuda));
        }
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }

    return *this;
}
template
iMatrix<int> & iMatrix<int>::operator = (const iMatrix & other);
template
iMatrix<float> & iMatrix<float>::operator = (const iMatrix & other);


template <typename DT>
iMatrix<DT> & iMatrix<DT>::operator = (iMatrix && other)
{
    this->m_type = other.m_type;
    this->m_mat_cpu = std::move(other.m_mat_cpu);
#ifdef WITH_CUDA
    this->m_mat_cuda = std::move(other.m_mat_cuda);
#endif

    other.m_type = MatrixType::UNKNOWN;
    other.m_mat_cpu = nullptr;
#ifdef WITH_CUDA
    other.m_mat_cuda = nullptr;
#endif

    return *this;
}
template
iMatrix<int> & iMatrix<int>::operator = (iMatrix && other);
template
iMatrix<float> & iMatrix<float>::operator = (iMatrix && other);


template <typename DT>
iMatrix<DT> & iMatrix<DT>::operator = (DT scalar)
{
    if (MatrixType::CPU == this->m_type)
    {
        *(this->m_mat_cpu) = scalar;
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        *(this->m_mat_cuda) = scalar;
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }

    return *this;
}
template
iMatrix<int> & iMatrix<int>::operator = (int scalar);
template
iMatrix<float> & iMatrix<float>::operator = (float scalar);


template <typename DT>
iMatrix<DT> & iMatrix<DT>::operator += (const iMatrix & other)
{
    if (this->m_type != other.m_type)
    {
        throw std::invalid_argument {
            std::string {"Matrix type should be the same for all matrices in "} +
            std::string {__FUNCTION__} };
    }

    if (MatrixType::CPU == this->m_type)
    {
        *(this->m_mat_cpu) += *(other.m_mat_cpu);
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        *(this->m_mat_cuda) += *(other.m_mat_cuda);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }

    return *this;
}
template
iMatrix<int> & iMatrix<int>::operator += (const iMatrix & other);
template
iMatrix<float> & iMatrix<float>::operator += (const iMatrix & other);


template <typename DT>
iMatrix<DT> & iMatrix<DT>::operator -= (const iMatrix & other)
{
    if (this->m_type != other.m_type)
    {
        throw std::invalid_argument {
            std::string {"Matrix type should be the same for all matrices in "} +
            std::string {__FUNCTION__} };
    }

    if (MatrixType::CPU == this->m_type)
    {
        *(this->m_mat_cpu) -= *(other.m_mat_cpu);
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        *(this->m_mat_cuda) -= *(other.m_mat_cuda);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }

    return *this;
}
template
iMatrix<int> & iMatrix<int>::operator -= (const iMatrix & other);
template
iMatrix<float> & iMatrix<float>::operator -= (const iMatrix & other);


template <typename DT>
iMatrix<DT> & iMatrix<DT>::operator *= (const iMatrix & other)
{
    if (this->m_type != other.m_type)
    {
        throw std::invalid_argument {
            std::string {"Matrix type should be the same for all matrices in "} +
            std::string {__FUNCTION__} };
    }

    if (MatrixType::CPU == this->m_type)
    {
        *(this->m_mat_cpu) *= *(other.m_mat_cpu);
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        *(this->m_mat_cuda) *= *(other.m_mat_cuda);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }

    return *this;
}
template
iMatrix<int> & iMatrix<int>::operator *= (const iMatrix & other);
template
iMatrix<float> & iMatrix<float>::operator *= (const iMatrix & other);


template <typename DT>
iMatrix<DT> & iMatrix<DT>::operator += (DT scalar)
{
    if (MatrixType::CPU == this->m_type)
    {
        *(this->m_mat_cpu) += scalar;
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        *(this->m_mat_cuda) += scalar;
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }

    return *this;
}
template
iMatrix<int> & iMatrix<int>::operator += (int scalar);
template
iMatrix<float> & iMatrix<float>::operator += (float scalar);


template <typename DT>
iMatrix<DT> & iMatrix<DT>::operator -= (DT scalar)
{
    if (MatrixType::CPU == this->m_type)
    {
        *(this->m_mat_cpu) -= scalar;
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        *(this->m_mat_cuda) -= scalar;
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }

    return *this;
}
template
iMatrix<int> & iMatrix<int>::operator -= (int scalar);
template
iMatrix<float> & iMatrix<float>::operator -= (float scalar);


template <typename DT>
iMatrix<DT> & iMatrix<DT>::operator *= (DT scalar)
{
    if (MatrixType::CPU == this->m_type)
    {
        *(this->m_mat_cpu) *= scalar;
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        *(this->m_mat_cuda) *= scalar;
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }

    return *this;
}
template
iMatrix<int> & iMatrix<int>::operator *= (int scalar);
template
iMatrix<float> & iMatrix<float>::operator *= (float scalar);


template <typename DT>
iMatrix<DT> & iMatrix<DT>::operator /= (DT scalar)
{
    if (MatrixType::CPU == this->m_type)
    {
        *(this->m_mat_cpu) /= scalar;
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        *(this->m_mat_cuda) /= scalar;
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }

    return *this;
}
template
iMatrix<int> & iMatrix<int>::operator /= (int scalar);
template
iMatrix<float> & iMatrix<float>::operator /= (float scalar);


template <typename DT>
DT iMatrix<DT>::operator [] (const Coordinate & pos) const
{
    if (MatrixType::CPU == this->m_type)
    {
        return (*(this->m_mat_cpu)) [pos];
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        return julie::la::cpu::Matrix_CPU<DT> {*(this->m_mat_cuda)} [pos];
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
int iMatrix<int>::operator [] (const Coordinate & pos) const;
template
float iMatrix<float>::operator [] (const Coordinate & pos) const;


template <typename DT>
iMatrix<DT> & iMatrix<DT>::gaussian_random(DT mu, DT sigma)
{
    if (MatrixType::CPU == this->m_type)
    {
        this->m_mat_cpu->gaussian_random(mu, sigma);
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        this->m_mat_cuda->gaussian_random(mu, sigma);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }

    return *this;
}
template
iMatrix<float> & iMatrix<float>::gaussian_random(float mu, float sigma);


// Randomize all elements of this matrix. Distribution of the elements complies with
// uniform distribution.
template <typename DT>
iMatrix<DT> & iMatrix<DT>::uniform_random(DT min, DT max)
{
    if (MatrixType::CPU == this->m_type)
    {
        this->m_mat_cpu->uniform_random(min, max);
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        this->m_mat_cuda->uniform_random(min, max);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }

    return *this;
}
template
iMatrix<int> & iMatrix<int>::uniform_random(int min, int max);
template
iMatrix<float> & iMatrix<float>::uniform_random(float min, float max);


// Normalize all elements of this matrix to the range of [min, max]
template <typename DT>
iMatrix<DT> & iMatrix<DT>::normalize(DT min, DT max)
{
    if (MatrixType::CPU == this->m_type)
    {
        this->m_mat_cpu->normalize(min, max);
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        this->m_mat_cuda->normalize(min, max);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }

    return *this;
}
template
iMatrix<float> & iMatrix<float>::normalize(float min, float max);


// Normalize all elements of this matrix to mean == 0 and variance == 1
template <typename DT>
iMatrix<DT> & iMatrix<DT>::normalize()
{
    if (MatrixType::CPU == this->m_type)
    {
        this->m_mat_cpu->normalize();
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        this->m_mat_cuda->normalize();
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }

    return *this;
}
template
iMatrix<float> & iMatrix<float>::normalize();


// Change shape of this matrix.
// For example, we can change [20, 30] to [2, 10, 3, 5, 2], or
// change [4, 5, 6] to [3, 5, 8].
// However, size of this matrix will not change, which means 20 * 30 == 2 * 10 * 3 * 5 * 2,
// or 4 * 5 * 6 == 3 * 5 * 8.
// Order of all elements in this matrix will not change either
template <typename DT>
iMatrix<DT> & iMatrix<DT>::reshape(const Shape & shape)
{
    if (MatrixType::CPU == this->m_type)
    {
        this->m_mat_cpu->reshape(shape);
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        this->m_mat_cuda->reshape(shape);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }

    return *this;
}
template
iMatrix<int> & iMatrix<int>::reshape(const Shape & shape);
template
iMatrix<float> & iMatrix<float>::reshape(const Shape & shape);


// Extend one dimension of the matrix.
// For example, [30, 50] is extended to [1, 30, 50].
// This is a special case of reshape.
template <typename DT>
iMatrix<DT> & iMatrix<DT>::left_extend_shape()
{
    if (MatrixType::CPU == this->m_type)
    {
        this->m_mat_cpu->left_extend_shape();
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        this->m_mat_cuda->left_extend_shape();
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }

    return *this;
}
template
iMatrix<int> & iMatrix<int>::left_extend_shape();
template
iMatrix<float> & iMatrix<float>::left_extend_shape();


// Extend one dimension of the matrix.
// For example, [30, 50] is extended to [30, 50, 1].
// This is a special case of reshape.
template <typename DT>
iMatrix<DT> & iMatrix<DT>::right_extend_shape()
{
    if (MatrixType::CPU == this->m_type)
    {
        this->m_mat_cpu->right_extend_shape();
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        this->m_mat_cuda->right_extend_shape();
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }

    return *this;
}
template
iMatrix<int> & iMatrix<int>::right_extend_shape();
template
iMatrix<float> & iMatrix<float>::right_extend_shape();


// Get duplicates of left extended version of the matrix.
// For example, [35, 45] can be extend to [12, 35, 45] in which
// there are exactly 12 copies of [35, 45]
template <typename DT>
void iMatrix<DT>::get_left_extended(iMatrix<DT> &output, lint duplicate) const
{
    output.set_matrix_type(this->m_type);

    if (MatrixType::CPU == this->m_type)
    {
        this->m_mat_cpu->get_left_extended(*(output.get_cpu_instance()), duplicate);
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        this->m_mat_cuda->get_left_extended(*(output.get_cuda_instance()), duplicate);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void iMatrix<int>::get_left_extended(iMatrix<int> &output, lint duplicate) const;
template
void iMatrix<float>::get_left_extended(iMatrix<float> &output, lint duplicate) const;


// Get duplicates of left extended version of the matrix.
// For example, [35, 45] can be extend to [35, 45, 16] in which
// there are exactly 16 copies of [35, 45]
template <typename DT>
void iMatrix<DT>::get_right_extended(iMatrix<DT> &output, lint duplicate) const
{
    output.set_matrix_type(this->m_type);

    if (MatrixType::CPU == this->m_type)
    {
        this->m_mat_cpu->get_right_extended(*(output.get_cpu_instance()), duplicate);
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        this->m_mat_cuda->get_right_extended(*(output.get_cuda_instance()), duplicate);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void iMatrix<int>::get_right_extended(iMatrix<int> &output, lint duplicate) const;
template
void iMatrix<float>::get_right_extended(iMatrix<float> &output, lint duplicate) const;


// Get coordinate of the largest element
template <typename DT>
Coordinate iMatrix<DT>::argmax() const
{
    if (MatrixType::CPU == this->m_type)
    {
        return this->m_mat_cpu->argmax();
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        return this->m_mat_cuda->argmax();
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
Coordinate iMatrix<int>::argmax() const;
template
Coordinate iMatrix<float>::argmax() const;


template <typename DT>
std::vector<Coordinate> iMatrix<DT>::argmax(lint dim) const
{
    if (MatrixType::CPU == this->m_type)
    {
        return this->m_mat_cpu->argmax(dim);
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        return this->m_mat_cuda->argmax(dim);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
std::vector<Coordinate> iMatrix<int>::argmax(lint dim) const;
template
std::vector<Coordinate> iMatrix<float>::argmax(lint dim) const;


// Get value of the largest element
template <typename DT>
DT iMatrix<DT>::max() const
{
    if (MatrixType::CPU == this->m_type)
    {
        return this->m_mat_cpu->max();
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        return this->m_mat_cuda->max();
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
int iMatrix<int>::max() const;
template
float iMatrix<float>::max() const;


// Get coordinate of the lowest element
template <typename DT>
Coordinate iMatrix<DT>::argmin() const
{
    if (MatrixType::CPU == this->m_type)
    {
        return this->m_mat_cpu->argmin();
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        return this->m_mat_cuda->argmin();
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
Coordinate iMatrix<int>::argmin() const;
template
Coordinate iMatrix<float>::argmin() const;


template <typename DT>
std::vector<Coordinate> iMatrix<DT>::argmin(lint dim) const
{
    if (MatrixType::CPU == this->m_type)
    {
        return this->m_mat_cpu->argmin(dim);
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        return this->m_mat_cuda->argmin(dim);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
std::vector<Coordinate> iMatrix<int>::argmin(lint dim) const;
template
std::vector<Coordinate> iMatrix<float>::argmin(lint dim) const;


// Get value of the lowest element
template <typename DT>
DT iMatrix<DT>::min() const
{
    if (MatrixType::CPU == this->m_type)
    {
        return this->m_mat_cpu->min();
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        return this->m_mat_cuda->min();
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
int iMatrix<int>::min() const;
template
float iMatrix<float>::min() const;


// Get mean of all the elements
template <typename DT>
DT iMatrix<DT>::mean() const
{
    if (MatrixType::CPU == this->m_type)
    {
        return this->m_mat_cpu->mean();
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        return this->m_mat_cuda->mean();
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
int iMatrix<int>::mean() const;
template
float iMatrix<float>::mean() const;


// Get sum of all the elements
template <typename DT>
DT iMatrix<DT>::sum() const
{
    if (MatrixType::CPU == this->m_type)
    {
        return this->m_mat_cpu->sum();
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        return this->m_mat_cuda->sum();
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
int iMatrix<int>::sum() const;
template
float iMatrix<float>::sum() const;


// Get variance of all the elements of this matrix
template <typename DT>
DT iMatrix<DT>::variance() const
{
    if (MatrixType::CPU == this->m_type)
    {
        return this->m_mat_cpu->variance();
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        return this->m_mat_cuda->variance();
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
int iMatrix<int>::variance() const;
template
float iMatrix<float>::variance() const;


// Collapse a certain dimension of a matrix into a vector of matrices
// For example, a matrix of shape [4, 6, 5, 7], if we collapse it with argument dim = 1,
// it will be turned into 6 matrices of shape [4, 5, 7]
template <typename DT>
std::vector<std::shared_ptr<iMatrix<DT>>> iMatrix<DT>::get_collapsed(lint dim) const
{
    std::vector<std::shared_ptr<iMatrix<DT>>> out;

    if (MatrixType::CPU == this->m_type)
    {
        std::vector<cpu::Matrix_CPU<DT>> mats_cpu = this->m_mat_cpu->get_collapsed(dim);

        for (auto & mat_cpu : mats_cpu)
        {
            out.push_back( std::make_shared<iMatrix<DT>>(iMatrix<DT>{std::move(mat_cpu)}) );
        }
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        std::vector<cuda::Matrix_CUDA<DT>> mats_cuda = this->m_mat_cuda->get_collapsed(dim);

        for (auto & mat_cuda : mats_cuda)
        {
            out.push_back( std::make_shared<iMatrix<DT>>(iMatrix<DT>{std::move(mat_cuda)}) );
        }
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }

    return out;
}
template
std::vector<std::shared_ptr<iMatrix<int>>> iMatrix<int>::get_collapsed(lint dim) const;
template
std::vector<std::shared_ptr<iMatrix<float>>> iMatrix<float>::get_collapsed(lint dim) const;


// Collapse a certain dimension of a matrix, and merge all matrices into one matrix
// For example, a matrix of shape [4, 6, 5, 7], if we fuse it with argument dim = 1,
// it will be turned into a sum of 6 matrices of shape [4, 5, 7]
template <typename DT>
void iMatrix<DT>::get_reduce_sum(iMatrix<DT> &output, lint dim) const
{
    output.set_matrix_type(this->m_type);

    if (MatrixType::CPU == this->m_type)
    {
        this->m_mat_cpu->get_reduce_sum(*(output.get_cpu_instance()), dim);
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        this->m_mat_cuda->get_reduce_sum(*(output.get_cuda_instance()), dim);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void iMatrix<int>::get_reduce_sum(iMatrix<int> &output, lint dim) const;
template
void iMatrix<float>::get_reduce_sum(iMatrix<float> &output, lint dim) const;


// Collapse a certain dimension of a matrix, and get a mean of all the matrices
// For example, a matrix of shape [4, 6, 5, 7], if we get its reduce_mean with argument dim = 1,
// it will be turned into a mean of 6 matrices of shape [4, 5, 7]
template <typename DT>
void iMatrix<DT>::get_reduce_mean(iMatrix<DT> &output, lint dim) const
{
    output.set_matrix_type(this->m_type);

    if (MatrixType::CPU == this->m_type)
    {
        this->m_mat_cpu->get_reduce_mean(*(output.get_cpu_instance()), dim);
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        this->m_mat_cuda->get_reduce_mean(*(output.get_cuda_instance()), dim);
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
void iMatrix<int>::get_reduce_mean(iMatrix<int> &output, lint dim) const;
template
void iMatrix<float>::get_reduce_mean(iMatrix<float> &output, lint dim) const;


template <typename DT>
DT iMatrix<DT>::euclidean_norm() const
{
    if (MatrixType::CPU == this->m_type)
    {
        return this->m_mat_cpu->euclidean_norm();
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        return this->m_mat_cuda->euclidean_norm();
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
int iMatrix<int>::euclidean_norm() const;
template
float iMatrix<float>::euclidean_norm() const;


// Get shape of the matrix
template <typename DT>
Shape iMatrix<DT>::shape() const
{
    if (MatrixType::CPU == this->m_type)
    {
        return this->m_mat_cpu->shape();
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        return this->m_mat_cuda->shape();
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
Shape iMatrix<int>::shape() const;
template
Shape iMatrix<float>::shape() const;


template <typename DT>
std::string iMatrix<DT>::to_string() const
{
    if (MatrixType::CPU == this->m_type)
    {
        return this->m_mat_cpu->to_string();
    }
    else if (MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        return this->m_mat_cuda->to_string();
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == this->m_type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }
}
template
std::string iMatrix<int>::to_string() const;
template
std::string iMatrix<float>::to_string() const;


template <typename DT>
iMatrix<DT> & iMatrix<DT>::set_matrix_type(MatrixType type)
{
    if (this->m_type == type)
    {
        return *this;
    }

    if (MatrixType::CPU == type && MatrixType::CUDA == this->m_type)
    {
#ifdef WITH_CUDA
        this->m_type = type;
        this->m_mat_cpu = std::make_shared<cpu::Matrix_CPU<DT>>(*(this->m_mat_cuda));
        this->m_mat_cuda = nullptr;
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CUDA == type && MatrixType::CPU == this->m_type)
    {
#ifdef WITH_CUDA
        this->m_type = type;
        this->m_mat_cuda = std::make_shared<cuda::Matrix_CUDA<DT>>(
            std::move(this->m_mat_cpu->get_CUDA()));
        this->m_mat_cpu = nullptr;
#else
        throw std::invalid_argument { std::string{"CUDA Matrix type not supported in "} + std::string{__FUNCTION__} };
#endif
    }
    else if (MatrixType::CL == type)
    {
        throw std::invalid_argument { std::string{"CL Matrix type not supported in "} + std::string{__FUNCTION__} };
    }
    else
    {
        throw std::invalid_argument { std::string{"Unknown matrix type in "} + std::string{__FUNCTION__} };
    }

    return *this;
}
template
iMatrix<int> & iMatrix<int>::set_matrix_type(MatrixType type);
template
iMatrix<float> & iMatrix<float>::set_matrix_type(MatrixType type);

/* Overloading of output stream << operator
*/

template <typename DT>
std::ostream & operator << (std::ostream & os, const iMatrix<DT> & m)
{
    return os << m.to_string();
}
template
std::ostream & operator << (std::ostream & os, const iMatrix<int> & m);
template
std::ostream & operator << (std::ostream & os, const iMatrix<float> & m);

} // namespace la
} // namespace julie
