#pragma once

#include "iMatrix.hpp"
#include "iMatrix_func.hpp"
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
    julie::la::iMatrix<float> w1_mat {
        {
            {4, 3},
            {2, 1},
            {1, 0}
        }
    };

    std::cout << "---------------------- 2 ---------------------" << std::endl;
    julie::la::iMatrix<float> b1_mat {
        {3, -4},
        true
    };

    auto x = std::make_shared<julie::nn::var::Tensor<float>> ();
    auto w1 = std::make_shared<julie::nn::var::Tensor<float>> (w1_mat);
    auto b1 = std::make_shared<julie::nn::var::Tensor<float>> (b1_mat);

    std::cout << "---------------------- 3 ---------------------" << std::endl;

    // x->needs_grad(false);
    // w->needs_grad(false);

    auto matmul_1 = std::make_shared<julie::nn::func::MatMul> ();

    std::cout << "---------------------- 31 ---------------------" << std::endl;
    
    auto add_1 = std::make_shared<julie::nn::func::Add> ();

    std::cout << "---------------------- 32 ---------------------" << std::endl;

    auto act_1 = std::make_shared<julie::nn::func::ReLU> ();

    std::cout << "---------------------- 4 ---------------------" << std::endl;

    julie::la::iMatrix<float> w2_mat {
        std::vector<float>{
            0, 1,
            2, 3
        },
        julie::la::Shape{2, 2}
    };

    julie::la::iMatrix<float> b2_mat {
        {-1, 2},
        true
    };

    std::cout << "---------------------- 5 ---------------------" << std::endl;

    auto w2 = std::make_shared<julie::nn::var::Tensor<float>> (w2_mat);
    auto b2 = std::make_shared<julie::nn::var::Tensor<float>> (b2_mat);

    auto matmul_2 = std::make_shared<julie::nn::func::MatMul> ();

    auto add_2 = std::make_shared<julie::nn::func::Add> ();

    auto act_2 = std::make_shared<julie::nn::func::SoftMax> (1);
    
    auto act_2_output = dynamic_cast<julie::nn::var::Tensor<float>*>(act_2->get_output().get());

    std::cout << "---------------------- 6 ---------------------" << std::endl;

    julie::op::Graph the_model_graph;
    the_model_graph.add_node(matmul_1, {x, w1});
    the_model_graph.add_node(add_1, {matmul_1->get_output(), b1});
    the_model_graph.add_node(act_1, {add_1->get_output()});
    the_model_graph.add_node(matmul_2, {act_1->get_output(), w2});
    the_model_graph.add_node(add_2, {matmul_2->get_output(), b2});
    the_model_graph.add_node(act_2, {add_2->get_output()});

    std::cout << "---------------------- 7 ---------------------" << std::endl;

    dynamic_cast<julie::nn::var::Tensor<float>*>(x.get())->val(
        julie::la::iMatrix<float> {
        {
            { 1, 0, 1},
            { 2, 1, 0},
            {-1, 3, 4}
        }}
    );

    std::cout << "---------------------- 8 ---------------------" << std::endl;

    the_model_graph.forward(act_2->get_output());

    std::cout << "---------------------- 9 ---------------------" << std::endl;
    
    if(act_2_output->val())
    {
        std::cout << *(act_2_output->val()) << std::endl;
    }

    std::cout << "---------------------- 10 ---------------------" << std::endl;

    julie::la::iMatrix<float> matmul_1_mat;
    julie::la::matmul(matmul_1_mat, *(dynamic_cast<julie::nn::var::Tensor<float>*>(x.get())->val()), w1_mat);
    
    julie::la::iMatrix<float> add_1_mat;
    julie::la::broadcast_add(add_1_mat, matmul_1_mat, b1_mat);

    std::cout << "---------------------- 11 ---------------------" << std::endl;

    julie::la::ReLU<float> relu;
    julie::la::iMatrix<float> act_1_mat;
    relu(act_1_mat, add_1_mat);

    std::cout << "---------------------- 12 ---------------------" << std::endl;

    julie::la::iMatrix<float> matmul_2_mat;
    julie::la::matmul(matmul_2_mat, act_1_mat, w2_mat);

    julie::la::iMatrix<float> add_2_mat;
    julie::la::broadcast_add(add_2_mat, matmul_2_mat, b2_mat);

    std::cout << "---------------------- 13 ---------------------" << std::endl;

    julie::la::SoftMax<float> softmax(1);
    julie::la::iMatrix<float> softmax_mat;
    softmax(softmax_mat, add_2_mat);

    std::cout << "---------------------- 14 ---------------------" << std::endl;

    std::cout << softmax_mat << std::endl;

    test::ASSERT(softmax_mat == *(act_2_output->val()));

    std::cout << "---------------------- 15 ---------------------" << std::endl;


}

void forward_and_backward_fc_simple()
{
    std::cout << "====================== forward_and_backward_fc_simple =====================" << std::endl;
    julie::la::iMatrix<float> w1_mat {
        {
            {4, 3},
            {2, 1},
            {1, 0}
        }
    };

    julie::la::iMatrix<float> b1_mat {
        {3, -4},
        true
    };

    auto x = std::make_shared<julie::nn::var::Tensor<float>> ();
    auto w1 = std::make_shared<julie::nn::var::Tensor<float>> (w1_mat);
    w1->trainable(true);
    auto b1 = std::make_shared<julie::nn::var::Tensor<float>> (b1_mat);
    b1->trainable(true);

    // x->needs_grad(false);
    // w->needs_grad(false);

    auto matmul_1 = std::make_shared<julie::nn::func::MatMul> ();   
    auto add_1 = std::make_shared<julie::nn::func::Add> ();
    auto act_1 = std::make_shared<julie::nn::func::ReLU> ();

    julie::la::iMatrix<float> w2_mat {
        std::vector<float>{
            0, 1,
            2, 3
        },
        julie::la::Shape{2, 2}
    };

    julie::la::iMatrix<float> b2_mat {
        {-1, 2},
        true
    };

    auto w2 = std::make_shared<julie::nn::var::Tensor<float>> (w2_mat);
    w2->trainable(true);
    auto b2 = std::make_shared<julie::nn::var::Tensor<float>> (b2_mat);
    b2->trainable(true);

    auto matmul_2 = std::make_shared<julie::nn::func::MatMul> ();
    auto add_2 = std::make_shared<julie::nn::func::Add> ();
    auto act_2 = std::make_shared<julie::nn::func::SoftMax> (1);

    auto target = std::make_shared<julie::nn::var::Tensor<float>> ();
    auto loss_func = std::make_shared<julie::nn::func::SoftMax_CrossEntropy> (1);

    julie::op::Graph the_model_graph;
    the_model_graph.add_node(matmul_1, {x, w1});
    the_model_graph.add_node(add_1, {matmul_1->get_output(), b1});
    the_model_graph.add_node(act_1, {add_1->get_output()});
    the_model_graph.add_node(matmul_2, {act_1->get_output(), w2});
    the_model_graph.add_node(add_2, {matmul_2->get_output(), b2});
    the_model_graph.add_node(act_2, {add_2->get_output()});
    the_model_graph.add_node(loss_func, {add_2->get_output(), target});

    //the_model_graph.pave_backward_route(w1);
    //the_model_graph.pave_backward_route(b1);
    //the_model_graph.pave_backward_route(w2);
    //the_model_graph.pave_backward_route(b2);

    dynamic_cast<julie::nn::var::Tensor<float>*>(x.get())->val(
        julie::la::iMatrix<float> {
        {
            { 1, 0, 1},
            { 2, 1, 0},
            {-1, 3, 4}
        }}
    );

    dynamic_cast<julie::nn::var::Tensor<float>*>(target.get())->val(
        julie::la::iMatrix<float> {
        {
            {1, 0},
            {0, 1},
            {1, 0}
        }}
    );

    std::cout << "---------------------- 16 ---------------------" << std::endl;

    the_model_graph.forward(act_2->get_output());

    std::cout << "---------------------- 17 ---------------------" << std::endl;
    the_model_graph.forward(loss_func->get_output());

    std::cout << "---------------------- 18 ---------------------" << std::endl;
    the_model_graph.backward(w1);

    std::cout << "---------------------- 19 ---------------------" << std::endl;

    
    auto act_2_output = dynamic_cast<julie::nn::var::Tensor<float>*>(act_2->get_output().get());
    if (act_2_output->val())
    {
        std::cout << "act_2 output data:" << std::endl;
        std::cout << *(act_2_output->val()) << std::endl;
    }

    auto add_2_output = dynamic_cast<julie::nn::var::Tensor<float>*>(add_2->get_output().get());
    if (add_2_output->val())
    {
        std::cout << "add_2 output data:" << std::endl;
        std::cout << *(add_2_output->val()) << std::endl;
        std::cout << *(add_2_output->grad()) << std::endl;
    }

    if (w1->backward_visited())
    {
        std::cout << "grad of w1 and b1:" << std::endl;
        std::cout << *(w1->grad()) << std::endl;
        std::cout << *(b1->grad()) << std::endl;
    }

    if (w2->backward_visited())
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
}

} // namespace test