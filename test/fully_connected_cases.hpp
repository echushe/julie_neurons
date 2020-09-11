#pragma once

#include "DMatrix.hpp"
#include "graph.hpp"
#include "add.hpp"
#include "matmul.hpp"
#include "relu.hpp"
#include "scale.hpp"
#include "softmax.hpp"
#include "sigmoid.hpp"
#include "tanh.hpp"
#include "arctan.hpp"
#include "prelu.hpp"

#include "softmax_crossentropy.hpp"
#include "sigmoid_crossentropy.hpp"

#include "sgd.hpp"

#include "Dataset.hpp"
#include "Mnist.hpp"

namespace test
{

void forward_fc_simple()
{
    std::cout << "====================== forward_fc_simple =====================" << std::endl;
    std::cout << "---------------------- 1 ---------------------" << std::endl;
    julie::la::DMatrix<double> w1_mat {
        {
            {4, 3},
            {2, 1},
            {1, 0}
        }
    };

    std::cout << "---------------------- 2 ---------------------" << std::endl;
    julie::la::DMatrix<double> b1_mat {
        {3, -4},
        true
    };

    auto x = std::make_shared<julie::nn::var::Tensor<double>> ();
    auto w1 = std::make_shared<julie::nn::var::Tensor<double>> (w1_mat);
    auto b1 = std::make_shared<julie::nn::var::Tensor<double>> (b1_mat);

    std::cout << "---------------------- 3 ---------------------" << std::endl;

    // x->needs_grad(false);
    // w->needs_grad(false);

    auto matmul_1 = std::make_shared<julie::nn::func::MatMul> (x, w1);

    std::cout << "---------------------- 31 ---------------------" << std::endl;
    
    auto add_1 = std::make_shared<julie::nn::func::Add> (matmul_1->get_output(), b1);

    std::cout << "---------------------- 32 ---------------------" << std::endl;

    auto act_1 = std::make_shared<julie::nn::func::ReLU> (add_1->get_output());

    std::cout << "---------------------- 4 ---------------------" << std::endl;

    julie::la::DMatrix<double> w2_mat {
        {
            {0, 1},
            {2, 3}
        }
    };

    julie::la::DMatrix<double> b2_mat {
        {-1, 2},
        true
    };

    std::cout << "---------------------- 5 ---------------------" << std::endl;

    auto w2 = std::make_shared<julie::nn::var::Tensor<double>> (w2_mat);
    auto b2 = std::make_shared<julie::nn::var::Tensor<double>> (b2_mat);

    auto matmul_2 = std::make_shared<julie::nn::func::MatMul> (act_1->get_output(), w2);

    auto add_2 = std::make_shared<julie::nn::func::Add> (matmul_2->get_output(), b2);

    auto act_2 = std::make_shared<julie::nn::func::SoftMax> (add_2->get_output(), 1);
    
    auto act_2_output = dynamic_cast<julie::nn::var::Tensor<double>*>(act_2->get_output().get());

    std::cout << "---------------------- 6 ---------------------" << std::endl;

    julie::op::Graph the_model_graph;
    the_model_graph.new_function(matmul_1);
    the_model_graph.new_function(add_1);
    the_model_graph.new_function(act_1);
    the_model_graph.new_function(matmul_2);
    the_model_graph.new_function(add_2);
    the_model_graph.new_function(act_2);

    std::cout << "---------------------- 7 ---------------------" << std::endl;

    dynamic_cast<julie::nn::var::Tensor<double>*>(x.get())->val(
        julie::la::DMatrix<double> {
        {
            { 1, 0, 1},
            { 2, 1, 0},
            {-1, 3, 4}
        }}
    );

    std::cout << "---------------------- 8 ---------------------" << std::endl;

    the_model_graph.func_forward(act_2);

    std::cout << "---------------------- 9 ---------------------" << std::endl;
    
    if(act_2_output->val())
    {
        std::cout << *(act_2_output->val()) << std::endl;
    }

    std::cout << "---------------------- 10 ---------------------" << std::endl;

    auto matmul_1_mat = julie::la::matmul(*(dynamic_cast<julie::nn::var::Tensor<double>*>(x.get())->val()), w1_mat);
    auto add_1_mat = julie::la::broadcast_add(matmul_1_mat, b1_mat);

    std::cout << "---------------------- 11 ---------------------" << std::endl;

    julie::la::ReLU<double> relu;
    julie::la::DMatrix<double> act_1_mat;
    relu(act_1_mat, add_1_mat);

    std::cout << "---------------------- 12 ---------------------" << std::endl;

    auto matmul_2_mat = julie::la::matmul(act_1_mat, w2_mat);
    auto add_2_mat = julie::la::broadcast_add(matmul_2_mat, b2_mat);

    std::cout << "---------------------- 13 ---------------------" << std::endl;

    julie::la::Softmax<double> softmax(1);
    julie::la::DMatrix<double> softmax_mat;
    softmax(softmax_mat, add_2_mat);

    std::cout << "---------------------- 14 ---------------------" << std::endl;

    std::cout << softmax_mat << std::endl;

    test::ASSERT(softmax_mat == *(act_2_output->val()));

    std::cout << "---------------------- 15 ---------------------" << std::endl;


}

void forward_and_backward_fc_simple()
{
    std::cout << "====================== forward_and_backward_fc_simple =====================" << std::endl;
    julie::la::DMatrix<double> w1_mat {
        {
            {4, 3},
            {2, 1},
            {1, 0}
        }
    };

    julie::la::DMatrix<double> b1_mat {
        {3, -4},
        true
    };

    auto x = std::make_shared<julie::nn::var::Tensor<double>> ();
    auto w1 = std::make_shared<julie::nn::var::Tensor<double>> (w1_mat);
    w1->trainable(true);
    auto b1 = std::make_shared<julie::nn::var::Tensor<double>> (b1_mat);
    b1->trainable(true);

    // x->needs_grad(false);
    // w->needs_grad(false);

    auto matmul_1 = std::make_shared<julie::nn::func::MatMul> (x, w1);   
    auto add_1 = std::make_shared<julie::nn::func::Add> (matmul_1->get_output(), b1);
    auto act_1 = std::make_shared<julie::nn::func::ReLU> (add_1->get_output());

    julie::la::DMatrix<double> w2_mat {
        {
            {0, 1},
            {2, 3}
        }
    };

    julie::la::DMatrix<double> b2_mat {
        {-1, 2},
        true
    };

    auto w2 = std::make_shared<julie::nn::var::Tensor<double>> (w2_mat);
    w2->trainable(true);
    auto b2 = std::make_shared<julie::nn::var::Tensor<double>> (b2_mat);
    b2->trainable(true);

    auto matmul_2 = std::make_shared<julie::nn::func::MatMul> (act_1->get_output(), w2);
    auto add_2 = std::make_shared<julie::nn::func::Add> (matmul_2->get_output(), b2);
    auto act_2 = std::make_shared<julie::nn::func::SoftMax> (add_2->get_output(), 1);

    auto target = std::make_shared<julie::nn::var::Tensor<double>> ();
    auto loss_func = std::make_shared<julie::nn::func::SoftMax_CrossEntropy> (add_2->get_output(), target, 1);

    julie::op::Graph the_model_graph;
    the_model_graph.new_function(matmul_1);
    the_model_graph.new_function(add_1);
    the_model_graph.new_function(act_1);
    the_model_graph.new_function(matmul_2);
    the_model_graph.new_function(add_2);
    the_model_graph.new_function(act_2);
    the_model_graph.new_function(loss_func);

    the_model_graph.pave_backward_route(w1);
    the_model_graph.pave_backward_route(b1);
    the_model_graph.pave_backward_route(w2);
    the_model_graph.pave_backward_route(b2);

    dynamic_cast<julie::nn::var::Tensor<double>*>(x.get())->val(
        julie::la::DMatrix<double> {
        {
            { 1, 0, 1},
            { 2, 1, 0},
            {-1, 3, 4}
        }}
    );

    dynamic_cast<julie::nn::var::Tensor<double>*>(target.get())->val(
        julie::la::DMatrix<double> {
        {
            {1, 0},
            {0, 1},
            {1, 0}
        }}
    );

    std::cout << "---------------------- 16 ---------------------" << std::endl;

    the_model_graph.func_forward(act_2);

    std::cout << "---------------------- 17 ---------------------" << std::endl;
    the_model_graph.func_forward(loss_func);

    std::cout << "---------------------- 18 ---------------------" << std::endl;
    the_model_graph.func_backward(matmul_1);

    std::cout << "---------------------- 19 ---------------------" << std::endl;

    
    auto act_2_output = dynamic_cast<julie::nn::var::Tensor<double>*>(act_2->get_output().get());
    if (act_2_output->val())
    {
        std::cout << "act_2 output data:" << std::endl;
        std::cout << *(act_2_output->val()) << std::endl;
    }

    auto add_2_output = dynamic_cast<julie::nn::var::Tensor<double>*>(add_2->get_output().get());
    if (add_2_output->val())
    {
        std::cout << "add_2 output data:" << std::endl;
        std::cout << *(add_2_output->val()) << std::endl;
        std::cout << *(add_2_output->grad()) << std::endl;
    }

    if (w1->has_grad())
    {
        std::cout << "grad of w1 and b1:" << std::endl;
        std::cout << *(w1->grad()) << std::endl;
        std::cout << *(b1->grad()) << std::endl;
    }

    if (w2->has_grad())
    {
        std::cout << "grad of w2 and b2:" << std::endl;
        std::cout << *(w2->grad()) << std::endl;
        std::cout << *(b2->grad()) << std::endl;
    }
}

void test_of_fc_model()
{
    forward_fc_simple();
    forward_and_backward_fc_simple();

    // fc_train_mnist();
    
}

} // namespace test