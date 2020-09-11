#pragma once
#include "DMatrix.hpp"
#include <algorithm>

namespace julie
{
namespace la
{

///////////////////////////////////////////////////////////////
//                     Activation Functions                  //
///////////////////////////////////////////////////////////////

// # define M_PI          3.141592653589793238462643383279502884L /* pi */

template <typename DT>
class Activation
{
public:
    static const std::string LINEAR;
    static const std::string SIGMOID;
    static const std::string TANH;
    static const std::string RELU;
    static const std::string LEAKYRELU;
    static const std::string PRELU;
    static const std::string ARCTAN;
    static const std::string SIN;
    static const std::string SOFTSIGN;
    static const std::string SOFTMAX;
    static const std::string NULL_FUNC;

    static std::unique_ptr<Activation> get_function_by_name(const std::string & func_name);

public:
    Activation() {}

    virtual std::unique_ptr<Activation> clone() = 0;

    // virtual void operator () (DMatrix<DT> & output, DMatrix<DT> & diff, const DMatrix<DT> & in) = 0;

    // virtual void operator () (DMatrix<DT> & output, const DMatrix<DT> & in) = 0;

    virtual std::string to_string() const = 0;
};

template <typename DT>
class Linear : public Activation<DT>
{
public:
    Linear() {};

    virtual std::unique_ptr<Activation<DT>> clone();

    virtual void operator () (DMatrix<DT> & output, DMatrix<DT> & diff, const DMatrix<DT> & in);

    virtual void operator () (DMatrix<DT> & output, const DMatrix<DT> & in);

    virtual std::string to_string() const;
};

template <typename DT>
class Sigmoid : public Activation<DT>
{
public:
    Sigmoid() {}

    virtual std::unique_ptr<Activation<DT>> clone();

    virtual void operator () (DMatrix<DT> & output, DMatrix<DT> & diff, const DMatrix<DT> & in);

    virtual void operator () (DMatrix<DT> & output, const DMatrix<DT> & in);

    virtual std::string to_string() const;
};

template <typename DT>
class TanH : public Activation<DT>
{
public:
    TanH() {}

    virtual std::unique_ptr<Activation<DT>> clone();

    virtual void operator () (DMatrix<DT> & output, DMatrix<DT> & diff, const DMatrix<DT> & in);

    virtual void operator () (DMatrix<DT> & output, const DMatrix<DT> & in);

    virtual std::string to_string() const;
};

template <typename DT>
class ReLU : public Activation<DT>
{
public:
    ReLU() {}

    virtual std::unique_ptr<Activation<DT>> clone();

    virtual void operator () (DMatrix<DT> & output, DMatrix<DT> & diff, const DMatrix<DT> & in);

    virtual void operator () (DMatrix<DT> & output, const DMatrix<DT> & in);

    virtual std::string to_string() const;
};

template <typename DT>
class PReLU : public Activation<DT>
{
public:
    PReLU() {}

    virtual std::unique_ptr<Activation<DT>> clone();

    virtual void operator () (DMatrix<DT> & output, DMatrix<DT> & diff, DMatrix<DT> & alpha_diff, const DMatrix<DT> & in, const DT & alpha);

    virtual void operator () (DMatrix<DT> & output, const DMatrix<DT> & in, const DT & alpha);

    virtual std::string to_string() const;
};

template <typename DT>
class LeakyReLU : public Activation<DT>
{
public:
    LeakyReLU() {}

    virtual std::unique_ptr<Activation<DT>> clone();

    virtual void operator () (DMatrix<DT> & output, DMatrix<DT> & diff, const DMatrix<DT> & in);

    virtual void operator () (DMatrix<DT> & output, const DMatrix<DT> & in);

    virtual std::string to_string() const;
};

template <typename DT>
class ArcTan : public Activation<DT>
{
public:
    ArcTan() {}

    virtual std::unique_ptr<Activation<DT>> clone();

    virtual void operator () (DMatrix<DT> & output, DMatrix<DT> & diff, const DMatrix<DT> & in);

    virtual void operator () (DMatrix<DT> & output, const DMatrix<DT> & in);

    virtual std::string to_string() const;
};

template <typename DT>
class Sin : public Activation<DT>
{
public:
    Sin() {}

    virtual std::unique_ptr<Activation<DT>> clone();

    virtual void operator () (DMatrix<DT> & output, DMatrix<DT> & diff, const DMatrix<DT> & in);

    virtual void operator () (DMatrix<DT> & output, const DMatrix<DT> & in);

    virtual std::string to_string() const;
};

template <typename DT>
class Softsign : public Activation<DT>
{
public:
    Softsign() {}

    virtual std::unique_ptr<Activation<DT>> clone();

    virtual void operator () (DMatrix<DT> & output, DMatrix<DT> & diff, const DMatrix<DT> & in);

    virtual void operator () (DMatrix<DT> & output, const DMatrix<DT> & in);

    virtual std::string to_string() const;
};

template <typename DT>
class Softmax : public Activation<DT>
{
public:
    Softmax(lint axis) : m_axis {axis} {};

    virtual std::unique_ptr<Activation<DT>> clone();

    virtual void operator () (DMatrix<DT> & output, DMatrix<DT> & diff, const DMatrix<DT> & in);

    virtual void operator () (DMatrix<DT> & output, const DMatrix<DT> & in);

    virtual std::string to_string() const;

private:
    void elementary_softmax(DT *output, const DT *input, lint size, lint r_sh_size);
    void elementary_softmax(DT *output, DT *diff, const DT *input, lint size, lint r_sh_size);

private:
    lint m_axis;
};

}
}

template <typename DT>
const std::string julie::la::Activation<DT>::LINEAR{ "Linear" };

template <typename DT>
const std::string julie::la::Activation<DT>::SIGMOID{ "Sigmoid" };

template <typename DT>
const std::string julie::la::Activation<DT>::TANH{ "TanH" };

template <typename DT>
const std::string julie::la::Activation<DT>::RELU{ "ReLU" };

template <typename DT>
const std::string julie::la::Activation<DT>::PRELU{ "PReLU" };

template <typename DT>
const std::string julie::la::Activation<DT>::LEAKYRELU{ "LeakyReLU" };

template <typename DT>
const std::string julie::la::Activation<DT>::ARCTAN{ "ArcTan" };

template <typename DT>
const std::string julie::la::Activation<DT>::SIN{ "Sin" };

template <typename DT>
const std::string julie::la::Activation<DT>::SOFTSIGN{ "Softsign" };

template <typename DT>
const std::string julie::la::Activation<DT>::SOFTMAX{ "Softmax" };

template <typename DT>
const std::string julie::la::Activation<DT>::NULL_FUNC{ "NULL" };

template <typename DT>
std::unique_ptr<julie::la::Activation<DT>> julie::la::Activation<DT>::get_function_by_name(const std::string & func_name)
{
    if (func_name == LINEAR)
    {
        return std::make_unique<Linear<DT>>();
    }
    else if (func_name == SIGMOID)
    {
        return std::make_unique<Sigmoid<DT>>();
    }
    else if (func_name == TANH)
    {
        return std::make_unique<TanH<DT>>();
    }
    else if (func_name == RELU)
    {
        return std::make_unique<ReLU<DT>>();
    }
    else if (func_name == PRELU)
    {
        return std::make_unique<PReLU<DT>>();
    }
    else if (func_name == LEAKYRELU)
    {
        return std::make_unique<LeakyReLU<DT>>();
    }
    else if (func_name == ARCTAN)
    {
        return std::make_unique<ArcTan<DT>>();
    }
    else if (func_name == SIN)
    {
        return std::make_unique<Sin<DT>>();
    }
    else if (func_name == SOFTSIGN)
    {
        return std::make_unique<Softsign<DT>>();
    }
    else if (func_name == SOFTMAX)
    {
        return std::make_unique<Softmax<DT>>();
    }
    else if (func_name == NULL_FUNC)
    {
        return nullptr;
    }
    else
    {
        return std::make_unique<Sigmoid<DT>>();
    }
}

template <typename DT>
std::unique_ptr<julie::la::Activation<DT>> julie::la::Linear<DT>::clone()
{
    return std::make_unique<Linear<DT>>();
}

template <typename DT>
void julie::la::Linear<DT>::operator () (DMatrix<DT> & output, DMatrix<DT> & diff, const DMatrix<DT> & in)
{
    output = DMatrix<DT>{ in.m_shape };
    DMatrix<DT> l_diff{ in.m_shape };
    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = in.m_data[i];
        l_diff.m_data[i] = 1;
    }

    diff = std::move(l_diff);
}

template <typename DT>
void julie::la::Linear<DT>::operator () (DMatrix<DT> & output, const DMatrix<DT> & in)
{
    output = DMatrix<DT>{ in.m_shape };

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = in.m_data[i];
    }
}

template <typename DT>
std::string julie::la::Linear<DT>::to_string() const
{
    return Activation<DT>::LINEAR;
}

template <typename DT>
std::unique_ptr<julie::la::Activation<DT>> julie::la::Sigmoid<DT>::clone()
{
    return std::make_unique<Sigmoid<DT>>();
}

template <typename DT>
void julie::la::Sigmoid<DT>::operator () (DMatrix<DT> & output, DMatrix<DT> & diff, const DMatrix<DT> & in)
{
    output = DMatrix<DT>{ in.m_shape };
    DMatrix<DT> l_diff{ in.m_shape };
    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        DT y = 1.0 / (1.0 + exp(-in.m_data[i]));
        output.m_data[i] = y;
        l_diff.m_data[i] = y * (1 - y);
    }

    diff = std::move(l_diff);
}

template <typename DT>
void julie::la::Sigmoid<DT>::operator () (DMatrix<DT> & output, const DMatrix<DT> & in)
{
    output = DMatrix<DT>{ in.m_shape };

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = 1.0 / (1.0 + exp(-in.m_data[i]));
    }
}

template <typename DT>
std::string julie::la::Sigmoid<DT>::to_string() const
{
    return Activation<DT>::SIGMOID;
}

template <typename DT>
std::unique_ptr<julie::la::Activation<DT>> julie::la::TanH<DT>::clone()
{
    return std::make_unique<TanH<DT>>();
}

template <typename DT>
void julie::la::TanH<DT>::operator () (DMatrix<DT> & output, DMatrix<DT> & diff, const DMatrix<DT> & in)
{
    output = DMatrix<DT>{ in.m_shape };
    DMatrix<DT> l_diff{ in.m_shape };
    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        DT y = tanh(in.m_data[i]);
        output.m_data[i] = y;
        l_diff.m_data[i] = 1 - y * y;
    }

    diff = std::move(l_diff);
}

template <typename DT>
void julie::la::TanH<DT>::operator () (DMatrix<DT> & output, const DMatrix<DT> & in)
{
    output = DMatrix<DT>{ in.m_shape };

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = tanh(in.m_data[i]);
    }
}

template <typename DT>
std::string julie::la::TanH<DT>::to_string() const
{
    return Activation<DT>::TANH;
}

template <typename DT>
std::unique_ptr<julie::la::Activation<DT>> julie::la::ReLU<DT>::clone()
{
    return std::make_unique<ReLU<DT>>();
}

template <typename DT>
void julie::la::ReLU<DT>::operator () (DMatrix<DT> & output, DMatrix<DT> & diff, const DMatrix<DT> & in)
{
    output = DMatrix<DT>{ in.m_shape };
    DMatrix<DT> l_diff{ in.m_shape };
    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        DT x = in.m_data[i];

        if (x >= 0)
        {
            output.m_data[i] = x;
            l_diff.m_data[i] = 1;
        }
        else
        {
            output.m_data[i] = 0;
            l_diff.m_data[i] = 0;
        }
    }

    diff = std::move(l_diff);
}

template <typename DT>
void julie::la::ReLU<DT>::operator () (DMatrix<DT> & output, const DMatrix<DT> & in)
{
    output = DMatrix<DT>{ in.m_shape };

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        DT x = in.m_data[i];

        if (x >= 0)
        {
            output.m_data[i] = x;
        }
        else
        {
            output.m_data[i] = 0;
        }
    }
}

template <typename DT>
std::string julie::la::ReLU<DT>::to_string() const
{
    return Activation<DT>::RELU;
}

template <typename DT>
std::unique_ptr<julie::la::Activation<DT>> julie::la::PReLU<DT>::clone()
{
    return std::make_unique<PReLU<DT>>();
}

template <typename DT>
void julie::la::PReLU<DT>::operator () (DMatrix<DT> & output, DMatrix<DT> & diff, DMatrix<DT> & alpha_diff, const DMatrix<DT> & in, const DT & alpha)
{
    output = DMatrix<DT> {in.m_shape};
    
    DMatrix<DT> l_diff {in.m_shape};
    DMatrix<DT> l_alpha_diff {in.m_shape};

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        DT x = in.m_data[i];

        if (x >= 0)
        {
            output.m_data[i] = x;
            l_diff.m_data[i] = 1;
            l_alpha_diff.m_data[i] = 0;
        }
        else
        {
            output.m_data[i] = alpha * x;
            l_diff.m_data[i] = alpha;
            l_alpha_diff.m_data[i] = x;
        }
    }

    alpha_diff = std::move(l_alpha_diff);
    diff = std::move(l_diff);
}

template <typename DT>
void julie::la::PReLU<DT>::operator () (DMatrix<DT> & output, const DMatrix<DT> & in, const DT & alpha)
{
    output = DMatrix<DT>{ in.m_shape };

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        DT x = in.m_data[i];

        if (x >= 0)
        {
            output.m_data[i] = x;
        }
        else
        {
            output.m_data[i] = alpha * x;
        }
    }
}

template <typename DT>
std::string julie::la::PReLU<DT>::to_string() const
{
    return Activation<DT>::PRELU;
}

template <typename DT>
std::unique_ptr<julie::la::Activation<DT>> julie::la::LeakyReLU<DT>::clone()
{
    return std::make_unique<LeakyReLU<DT>>();
}

template <typename DT>
void julie::la::LeakyReLU<DT>::operator () (DMatrix<DT> & output, DMatrix<DT> & diff, const DMatrix<DT> & in)
{
    output = DMatrix<DT>{ in.m_shape };
    DMatrix<DT> l_diff{ in.m_shape };
    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        DT x = in.m_data[i];

        if (x >= 0)
        {
            output.m_data[i] = x;
            l_diff.m_data[i] = 1;
        }
        else
        {
            output.m_data[i] = 0.01 * x;
            l_diff.m_data[i] = 0.01;
        }
    }

    diff = std::move(l_diff);
}

template <typename DT>
void julie::la::LeakyReLU<DT>::operator () (DMatrix<DT> & output, const DMatrix<DT> & in)
{
    output = DMatrix<DT>{ in.m_shape };

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        DT x = in.m_data[i];

        if (x >= 0)
        {
            output.m_data[i] = x;
        }
        else
        {
            output.m_data[i] = 0.01 * x;
        }
    }
}

template <typename DT>
std::string julie::la::LeakyReLU<DT>::to_string() const
{
    return Activation<DT>::LEAKYRELU;
}

template <typename DT>
std::unique_ptr<julie::la::Activation<DT>> julie::la::ArcTan<DT>::clone()
{
    return std::make_unique<julie::la::ArcTan<DT>>();
}

template <typename DT>
void julie::la::ArcTan<DT>::operator()(DMatrix<DT>& output, DMatrix<DT>& diff, const DMatrix<DT>& in)
{
    output = DMatrix<DT>{ in.m_shape };
    DMatrix<DT> l_diff{ in.m_shape };
    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = atan(in.m_data[i]);
        l_diff.m_data[i] = 1 / (1 + in.m_data[i] * in.m_data[i]);
    }

    diff = std::move(l_diff);
}

template <typename DT>
void julie::la::ArcTan<DT>::operator()(DMatrix<DT>& output, const DMatrix<DT>& in)
{
    output = DMatrix<DT>{ in.m_shape };

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = atan(in.m_data[i]);
    }
}

template <typename DT>
std::string julie::la::ArcTan<DT>::to_string() const
{
    return Activation<DT>::ARCTAN;
}

template <typename DT>
std::unique_ptr<julie::la::Activation<DT>> julie::la::Sin<DT>::clone()
{
    return std::make_unique<Sin<DT>>();
}

template <typename DT>
void julie::la::Sin<DT>::operator()(DMatrix<DT>& output, DMatrix<DT>& diff, const DMatrix<DT>& in)
{
    output = DMatrix<DT>{ in.m_shape };
    DMatrix<DT> l_diff{ in.m_shape };
    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = sin(in.m_data[i]);
        l_diff.m_data[i] = cos(in.m_data[i]);
    }

    diff = std::move(l_diff);
}

template <typename DT>
void julie::la::Sin<DT>::operator()(DMatrix<DT>& output, const DMatrix<DT>& in)
{
    output = DMatrix<DT>{ in.m_shape };

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = sin(in.m_data[i]);
    }
}

template <typename DT>
std::string julie::la::Sin<DT>::to_string() const
{
    return Activation<DT>::SIN;
}

template <typename DT>
std::unique_ptr<julie::la::Activation<DT>> julie::la::Softsign<DT>::clone()
{
    return std::make_unique<Softsign<DT>>();
}

template <typename DT>
void julie::la::Softsign<DT>::operator()(DMatrix<DT>& output, DMatrix<DT>& diff, const DMatrix<DT>& in)
{
    output = DMatrix<DT>{ in.m_shape };
    DMatrix<DT> l_diff{ in.m_shape };
    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        DT d = 1 + fabs(in.m_data[i]);
        output.m_data[i] = in.m_data[i] / d;
        l_diff.m_data[i] = 1 / (d * d);
    }

    diff = std::move(l_diff);
}

template <typename DT>
void julie::la::Softsign<DT>::operator()(DMatrix<DT>& output, const DMatrix<DT>& in)
{
    output = DMatrix<DT>{ in.m_shape };

    lint size = in.m_shape.size();
    for (lint i = 0; i < size; ++i)
    {
        output.m_data[i] = in.m_data[i] / (1 + fabs(in.m_data[i]));
    }
}

template <typename DT>
std::string julie::la::Softsign<DT>::to_string() const
{
    return Activation<DT>::SOFTSIGN;
}

template <typename DT>
std::unique_ptr<julie::la::Activation<DT>> julie::la::Softmax<DT>::clone()
{
    return std::make_unique<Softmax<DT>>(this->m_axis);
}

template <typename DT>
void julie::la::Softmax<DT>::elementary_softmax(DT *output, const DT *input, lint size, lint r_sh_size)
{
    const DT *input_p = input;
    DT *output_p = output;

    DT sum = 0;
    for (lint i = 0; i < size; ++i)
    {
        *output_p = exp(*input_p);
        sum += *output_p;

        input_p += r_sh_size;
        output_p += r_sh_size;
    }

    output_p = output;
    for (lint i = 0; i < size; ++i)
    {
        *output_p /= sum;

        output_p += r_sh_size;
    }
}

template <typename DT>
void julie::la::Softmax<DT>::elementary_softmax(DT *output, DT *diff, const DT *input, lint size, lint r_sh_size)
{
    const DT *input_p = input;
    DT *diff_p = diff;
    DT *output_p = output;

    DT sum = 0;
    for (lint i = 0; i < size; ++i)
    {
        *output_p = exp(*input_p);
        sum += *output_p;

        input_p += r_sh_size;
        output_p += r_sh_size;        
    }

    output_p = output;
    diff_p = diff;
    for (lint i = 0; i < size; ++i)
    {
        *output_p /= sum;
        *diff_p = *output_p * (1 - *output_p);

        output_p += r_sh_size;
        diff_p += r_sh_size;
    }
}

template <typename DT>
void julie::la::Softmax<DT>::operator () (DMatrix<DT> & output, DMatrix<DT> & diff, const DMatrix<DT> & in)
{
    output = DMatrix<DT>{ in.m_shape };
    DMatrix<DT> l_diff{ in.m_shape };

    Shape l_sh = in.m_shape.sub_shape(0, this->m_axis - 1);
    Shape r_sh = in.m_shape.sub_shape(this->m_axis + 1, in.m_shape.dim() - 1);

    lint n_ele = std::max<lint>(l_sh.size(), 1);
    lint my_axis_size = in.m_shape[this->m_axis];
    lint r_sh_size = std::max<lint>(r_sh.size(), 1);
    lint step = my_axis_size * r_sh_size;

    DT *in_data = in.m_data;
    DT *diff_data = l_diff.m_data;
    DT *out_data = output.m_data;

    for (lint i = 0; i < n_ele; ++i)
    {
        DT *r_in_data = in_data;
        DT *r_diff_data = diff_data;
        DT *r_out_data = out_data;

        for (lint j = 0; j < r_sh_size; ++j)
        {
            this->elementary_softmax(r_out_data, r_diff_data, r_in_data, my_axis_size, r_sh_size);
            ++r_in_data;
            ++r_diff_data;
            ++r_out_data;
        }

        in_data += step;
        diff_data += step;
        out_data += step;
    }

    diff = std::move(l_diff);
}

template <typename DT>
void julie::la::Softmax<DT>::operator () (DMatrix<DT> & output, const DMatrix<DT> & in)
{
    output = DMatrix<DT>{ in.m_shape };
    DMatrix<DT> l_diff{ in.m_shape };

    Shape l_sh = in.m_shape.sub_shape(0, this->m_axis - 1);
    Shape r_sh = in.m_shape.sub_shape(this->m_axis + 1, in.m_shape.dim() - 1);

    lint n_ele = std::max<lint>(l_sh.size(), 1);
    lint my_axis_size = in.m_shape[this->m_axis];
    lint r_sh_size = std::max<lint>(r_sh.size(), 1);
    lint step = my_axis_size * r_sh_size;

    DT *in_data = in.m_data;
    DT *out_data = output.m_data;

    for (lint i = 0; i < n_ele; ++i)
    {
        DT *r_in_data = in_data;
        DT *r_out_data = out_data;

        for (lint j = 0; j < r_sh_size; ++j)
        {
            this->elementary_softmax(r_out_data, r_in_data, my_axis_size, r_sh_size);
            ++r_in_data;
            ++r_out_data;
        }

        in_data += step;
        out_data += step;
    }
}

template <typename DT>
std::string julie::la::Softmax<DT>::to_string() const
{
    return Activation<DT>::SOFTMAX;
}


