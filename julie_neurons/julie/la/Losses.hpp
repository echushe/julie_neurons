#pragma once
#include "DMatrix.hpp"
#include "Activations.hpp"

namespace julie
{
namespace la
{
    // # define M_PI          3.141592653589793238462643383279502884L /* pi */

    template <typename DT>
    class ErrorFunction
    {
    public:
        static const std::string HALF_SQUARE_ERROR;
        static const std::string SIGMOID_CROSS_ENTROPY;
        static const std::string SOFTMAX_CROSS_ENTROPY;
        static const std::string NULL_FUNC;

        static std::unique_ptr<ErrorFunction> get_function_by_name(std::string & func_name);

    public:
        ErrorFunction(lint axis) : m_axis {axis} {}

        virtual std::unique_ptr<ErrorFunction> clone() = 0;
        virtual DMatrix<DT> operator () (DMatrix<DT> & diff, const DMatrix<DT> & target, const DMatrix<DT> & input) = 0;
        virtual std::string to_string() const = 0;

    protected:
        lint m_axis;
    };


    template <typename DT>
    class HalfSquareError : public ErrorFunction<DT>
    {
    private:

    public:
        HalfSquareError(lint axis) : ErrorFunction<DT> {axis} {};

        virtual std::unique_ptr<ErrorFunction<DT>> clone();
        virtual DMatrix<DT> operator () (DMatrix<DT> & diff, const DMatrix<DT> & target, const DMatrix<DT> & input);
        virtual std::string to_string() const;

    private:
        DT elementary_HalfSquareError(DT *diff, const DT *target, const DT *input, lint size, lint r_sh_size);
    };


    template <typename DT>
    class Sigmoid_CrossEntropy : public ErrorFunction<DT>
    {
    private:
        // mutable DMatrix<DT> m_sigmoid;

    public:
        Sigmoid_CrossEntropy(lint axis) : ErrorFunction<DT> {axis} {};

        virtual std::unique_ptr<ErrorFunction<DT>> clone();

        virtual DMatrix<DT> operator () (DMatrix<DT> & diff, const DMatrix<DT> & target, const DMatrix<DT> & input);
        virtual std::string to_string() const;

    private:
        DT elementary_sigmoid_crossentropy(DT *output, DT *diff, const DT *target, const DT *input, lint size, lint r_sh_size);
    };


    template <typename DT>
    class Softmax_CrossEntropy : public ErrorFunction<DT>
    {
    private:
        // mutable DMatrix<DT> m_softmax;

    public:
        Softmax_CrossEntropy(lint axis) : ErrorFunction<DT> {axis} {};

        virtual std::unique_ptr<ErrorFunction<DT>> clone();

        virtual DMatrix<DT> operator () (DMatrix<DT> & diff, const DMatrix<DT> & target, const DMatrix<DT> & input);
        virtual std::string to_string() const;

    private:
        DT elementary_softmax_crossentropy(DT *output, DT *diff, const DT *target, const DT *input, lint size, lint r_sh_size);
    };

}
}


template <typename DT>
const std::string julie::la::ErrorFunction<DT>::HALF_SQUARE_ERROR{ "HalfSquareError" };

template <typename DT>
const std::string julie::la::ErrorFunction<DT>::SIGMOID_CROSS_ENTROPY{ "Sigmoid_CrossEntropy" };

template <typename DT>
const std::string julie::la::ErrorFunction<DT>::SOFTMAX_CROSS_ENTROPY{ "Softmax_CrossEntropy" };

template <typename DT>
const std::string julie::la::ErrorFunction<DT>::NULL_FUNC{ "NULL" };

template <typename DT>
std::unique_ptr<julie::la::ErrorFunction<DT>> julie::la::ErrorFunction<DT>::get_function_by_name(std::string & func_name)
{
    std::istringstream buf(func_name);
    std::istream_iterator<std::string> beg(buf), end;

    std::vector<std::string> tokens(beg, end);

    if (tokens[0] == SIGMOID_CROSS_ENTROPY)
    {
        return std::make_unique<Sigmoid_CrossEntropy<DT>>();
    }
    else if (tokens[0] == SOFTMAX_CROSS_ENTROPY)
    {
        return std::make_unique<Softmax_CrossEntropy<DT>>();
    }
    else if (tokens[0] == HALF_SQUARE_ERROR)
    {
        return std::make_unique<HalfSquareError>(Activation<DT>::get_function_by_name(tokens[1]));
    }
    else if (tokens[0] == NULL_FUNC)
    {
        return nullptr;
    }
    else
    {
        return std::make_unique<Softmax_CrossEntropy<DT>>();
    }
}

template <typename DT>
std::unique_ptr<julie::la::ErrorFunction<DT>> julie::la::HalfSquareError<DT>::clone()
{
    return std::make_unique<HalfSquareError>(this->m_axis);
}

template <typename DT>
DT julie::la::HalfSquareError<DT>::elementary_HalfSquareError(DT *diff, const DT *target, const DT *input, lint size, lint r_sh_size)
{
    const DT *input_p = input;
    const DT *target_p = target;
    DT *diff_p = diff;
    
    DT sum = 0;

    for (lint i = 0; i < size; ++i)
    {
        DT sub = *input_p - *target_p;
        sum += sub * sub;

        *diff_p = sub;

        input_p += r_sh_size;
        target_p += r_sh_size;
        diff_p += r_sh_size;
    }

    return sum / 2;
}

template <typename DT>
julie::la::DMatrix<DT> julie::la::HalfSquareError<DT>::operator()(DMatrix<DT> & diff, const DMatrix<DT> & target, const DMatrix<DT> & input)
{
    DMatrix<DT> l_diff { input.m_shape };

    Shape l_sh = input.m_shape.sub_shape(0, this->m_axis - 1);
    Shape r_sh = input.m_shape.sub_shape(this->m_axis + 1, input.m_shape.dim() - 1);

    lint n_ele = std::max<lint>(l_sh.size(), 1);
    lint my_axis_size = input.m_shape[this->m_axis];
    lint r_sh_size = std::max<lint>(r_sh.size(), 1);
    lint step = my_axis_size * r_sh_size;

    DT *diff_data = l_diff.m_data;
    DT *t_data = target.m_data;
    DT *in_data = input.m_data;

    DMatrix <DT> loss_mat {l_sh + r_sh};
    DT *loss_mat_data = loss_mat.m_data;

    for (lint i = 0; i < n_ele; ++i)
    {
        DT *r_diff_data = diff_data;
        DT *r_t_data = t_data;
        DT *r_in_data = in_data;

        for (lint j = 0; j < r_sh_size; ++j)
        {
            *loss_mat_data = this->elementary_HalfSquareError(r_diff_data, r_t_data, r_in_data, my_axis_size, r_sh_size);
            ++r_diff_data;
            ++r_t_data;
            ++r_in_data;

            ++loss_mat_data;
        }

        diff_data += step;
        t_data += step;
        in_data += step;
    }

    diff = std::move(l_diff);

    return loss_mat;
}

template <typename DT>
std::string julie::la::HalfSquareError<DT>::to_string() const
{
    return std::string(ErrorFunction<DT>::HALF_SQUARE_ERROR);
}

template <typename DT>
std::unique_ptr<julie::la::ErrorFunction<DT>> julie::la::Sigmoid_CrossEntropy<DT>::clone()
{
    return std::make_unique<Sigmoid_CrossEntropy<DT>>(this->m_axis);
}

template <typename DT>
DT julie::la::Sigmoid_CrossEntropy<DT>::elementary_sigmoid_crossentropy(
    DT *sigmoid, DT *diff, const DT *target, const DT *input, lint size, lint r_sh_size)
{
    const DT *input_p = input;
    DT *diff_p = diff;
    DT *sigmoid_p = sigmoid;

    for (lint i = 0; i < size; ++i)
    {
        *sigmoid_p = 1.0 / (1.0 + exp(*input_p * (-1)));

        input_p += r_sh_size;
        sigmoid_p += r_sh_size;        
    }

    DT centropy_sum = 0;

    const DT *target_p = target;
    sigmoid_p = sigmoid;
    diff_p = diff;
    for (lint i = 0; i < size; ++i)
    {
        centropy_sum += *target_p * log(*sigmoid_p);
        *diff_p = *sigmoid_p - *target_p;

        target_p += r_sh_size;
        sigmoid_p += r_sh_size;
        diff_p += r_sh_size;
    }

    centropy_sum *= -1;

    return centropy_sum;
}

template <typename DT>
julie::la::DMatrix<DT> julie::la::Sigmoid_CrossEntropy<DT>::operator()(DMatrix<DT> & diff, const DMatrix<DT> & target, const DMatrix<DT> & input)
{
    DMatrix<DT> l_sigmoid { input.m_shape };
    DMatrix<DT> l_diff { input.m_shape };

    Shape l_sh = input.m_shape.sub_shape(0, this->m_axis - 1);
    Shape r_sh = input.m_shape.sub_shape(this->m_axis + 1, input.m_shape.dim() - 1);

    lint n_ele = std::max<lint>(l_sh.size(), 1);
    lint my_axis_size = input.m_shape[this->m_axis];
    lint r_sh_size = std::max<lint>(r_sh.size(), 1);
    lint step = my_axis_size * r_sh_size;

    DT *sigmoid = l_sigmoid.m_data;
    DT *diff_data = l_diff.m_data;
    DT *t_data = target.m_data;
    DT *in_data = input.m_data;

    DMatrix <DT> loss_mat {l_sh + r_sh};
    DT *loss_mat_data = loss_mat.m_data;

    for (lint i = 0; i < n_ele; ++i)
    {
        DT *r_sigmoid = sigmoid;
        DT *r_diff_data = diff_data;
        DT *r_t_data = t_data;
        DT *r_in_data = in_data;

        for (lint j = 0; j < r_sh_size; ++j)
        {
            *loss_mat_data = this->elementary_sigmoid_crossentropy(r_sigmoid, r_diff_data, r_t_data, r_in_data, my_axis_size, r_sh_size);
            ++r_sigmoid;
            ++r_diff_data;
            ++r_t_data;
            ++r_in_data;

            ++loss_mat_data;
        }

        sigmoid += step;
        diff_data += step;
        t_data += step;
        in_data += step;
    }

    diff = std::move(l_diff);

    return loss_mat;
}

template <typename DT>
std::string julie::la::Sigmoid_CrossEntropy<DT>::to_string() const
{
    return ErrorFunction<DT>::SIGMOID_CROSS_ENTROPY;
}

template <typename DT>
std::unique_ptr<julie::la::ErrorFunction<DT>> julie::la::Softmax_CrossEntropy<DT>::clone()
{
    return std::make_unique<julie::la::Softmax_CrossEntropy<DT>>(this->m_axis);
}

template <typename DT>
DT julie::la::Softmax_CrossEntropy<DT>::elementary_softmax_crossentropy(
    DT *softmax, DT *diff, const DT *target, const DT *input, lint size, lint r_sh_size)
{
    const DT *input_p = input;
    DT *diff_p = diff;
    DT *softmax_p = softmax;

    DT sum = 0;
    for (lint i = 0; i < size; ++i)
    {
        *softmax_p = exp(*input_p);
        sum += *softmax_p;

        input_p += r_sh_size;
        softmax_p += r_sh_size;        
    }

    DT centropy_sum = 0;

    const DT *target_p = target;
    softmax_p = softmax;
    diff_p = diff;
    for (lint i = 0; i < size; ++i)
    {
        *softmax_p /= sum;

        centropy_sum += *target_p * log(*softmax_p);
        *diff_p = *softmax_p - *target_p;

        target_p += r_sh_size;
        softmax_p += r_sh_size;
        diff_p += r_sh_size;
    }

    centropy_sum *= -1;

    return centropy_sum;
}

template <typename DT>
julie::la::DMatrix<DT> julie::la::Softmax_CrossEntropy<DT>::operator()(DMatrix<DT> & diff, const DMatrix<DT> & target, const DMatrix<DT> & input)
{
    DMatrix<DT> l_softmax { input.m_shape };
    DMatrix<DT> l_diff { input.m_shape };

    Shape l_sh = input.m_shape.sub_shape(0, this->m_axis - 1);
    Shape r_sh = input.m_shape.sub_shape(this->m_axis + 1, input.m_shape.dim() - 1);

    lint n_ele = std::max<lint>(l_sh.size(), 1);
    lint my_axis_size = input.m_shape[this->m_axis];
    lint r_sh_size = std::max<lint>(r_sh.size(), 1);
    lint step = my_axis_size * r_sh_size;

    DT *softmax = l_softmax.m_data;
    DT *diff_data = l_diff.m_data;
    DT *t_data = target.m_data;
    DT *in_data = input.m_data;

    DMatrix <DT> loss_mat {l_sh + r_sh};
    DT *loss_mat_data = loss_mat.m_data;

    for (lint i = 0; i < n_ele; ++i)
    {
        DT *r_softmax = softmax;
        DT *r_diff_data = diff_data;
        DT *r_t_data = t_data;
        DT *r_in_data = in_data;

        for (lint j = 0; j < r_sh_size; ++j)
        {
            *loss_mat_data = this->elementary_softmax_crossentropy(r_softmax, r_diff_data, r_t_data, r_in_data, my_axis_size, r_sh_size);
            ++r_softmax;
            ++r_diff_data;
            ++r_t_data;
            ++r_in_data;

            ++loss_mat_data;
        }

        softmax += step;
        diff_data += step;
        t_data += step;
        in_data += step;
    }

    diff = std::move(l_diff);

    // std::cout << l_softmax << std::endl;

    return loss_mat;
}


template <typename DT>
std::string julie::la::Softmax_CrossEntropy<DT>::to_string() const
{
    return ErrorFunction<DT>::SOFTMAX_CROSS_ENTROPY;
}
