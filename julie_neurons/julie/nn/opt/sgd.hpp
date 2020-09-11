#pragma once
#include "optimizer.hpp"
#include "DMatrix.hpp"
namespace julie
{
namespace nn
{
namespace opt
{

class SGD : public Optimizer
{
public:
    SGD (const std::vector<std::shared_ptr<op::Variable>> & params, double learning_rate, double momentum_rate);

    SGD (const SGD & other);

    SGD (SGD && other);

    SGD & operator = (const SGD & other);

    SGD & operator = (SGD && other);

    virtual void step();

private:

    void tensor_step(const la::DMatrix<double> & gradient, la::DMatrix<double> & momentum, la::DMatrix<double> & weight);

    void scalar_step(const double & gradient, double & momentum, double & weight);

private:
    double m_lr;
    double m_mr;
    std::vector<std::shared_ptr<op::Variable>> m_momentum;

};

}
}
}