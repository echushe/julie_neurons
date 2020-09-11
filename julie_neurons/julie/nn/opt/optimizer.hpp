#pragma once
#include "variable.hpp"

namespace julie
{
namespace nn
{
namespace opt
{

class Optimizer
{
public:
    Optimizer (const std::vector<std::shared_ptr<op::Variable>> & params);

    Optimizer (const Optimizer & other);

    Optimizer (Optimizer && other);

    Optimizer & operator = (const Optimizer & other);

    Optimizer & operator = (Optimizer && other);

    virtual void step() = 0;

protected:
    std::vector<std::shared_ptr<op::Variable>> m_params;
};

} // namespace opt
} // namespace nn
} // namespace julie
