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

#include "Dataset.hpp"
#include "Mnist.hpp"

#include "softmax_crossentropy.hpp"
#include "sigmoid_crossentropy.hpp"

#include "sgd.hpp"

namespace demo
{

void fc_train_mnist()
{
    std::cout << "====================== fc_train_mnist =====================" << std::endl;
    // ----------------------------------------------------------------- //
    // --------------------- Load the MNIST data set ------------------- //
    // ----------------------------------------------------------------- //

    std::string dataset_dir = "../../../dataset/";

    std::vector<julie::la::DMatrix<double>> train_set;
    std::vector<julie::la::DMatrix<double>> train_label;

    std::vector<julie::la::DMatrix<double>> val_set;
    std::vector<julie::la::DMatrix<double>> val_label;

    std::shared_ptr<dataset::Dataset> train_set_getter = std::make_shared<dataset::Mnist>(
        dataset_dir + "mnist/train-images-idx3-ubyte",
        dataset_dir + "mnist/train-labels-idx1-ubyte");
    
    std::shared_ptr<dataset::Dataset> val_set_getter = std::make_shared<dataset::Mnist>(
        dataset_dir + "mnist/t10k-images-idx3-ubyte",
        dataset_dir + "mnist/t10k-labels-idx1-ubyte");

    train_set_getter->get_samples_and_labels(train_set, train_label);
    val_set_getter->get_samples_and_labels(val_set, val_label);

    // ----------------------------------------------------------------- //
    // ------------------ Buid the network structure ------------------- //
    // ----------------------------------------------------------------- //

    // Initialize weights

    julie::la::DMatrix<double> w1_mat { julie::la::Shape {28 * 28, 50} };
    julie::la::DMatrix<double> b1_mat { 0, julie::la::Shape {1, 50} };
    w1_mat.gaussian_random(0, 0.1);
    
    julie::la::DMatrix<double> w2_mat { julie::la::Shape {50, 50} };
    julie::la::DMatrix<double> b2_mat { 0, julie::la::Shape {1, 50} };
    w2_mat.gaussian_random(0, 0.1);

    julie::la::DMatrix<double> w3_mat { julie::la::Shape {50, 40} };
    julie::la::DMatrix<double> b3_mat { 0, julie::la::Shape {1, 40} };
    w3_mat.gaussian_random(0, 0.1);

    julie::la::DMatrix<double> w4_mat { julie::la::Shape {40, 10} };
    julie::la::DMatrix<double> b4_mat { 0, julie::la::Shape {1, 10} };
    w4_mat.gaussian_random(0, 0.1);

    // Build the network structure

    auto x = std::make_shared<julie::nn::var::Tensor<double>> ();
    auto w1 = std::make_shared<julie::nn::var::Tensor<double>> (w1_mat);
    w1->trainable(true);
    auto b1 = std::make_shared<julie::nn::var::Tensor<double>> (b1_mat);
    b1->trainable(true);
    auto alpha1 = std::make_shared<julie::nn::var::Scalar<double>> (-0.3);
    alpha1->trainable(true);

    auto matmul_1 = std::make_shared<julie::nn::func::MatMul> (x, w1);   
    auto add_1 = std::make_shared<julie::nn::func::Add> (matmul_1->get_output(), b1);
    // auto act_1 = std::make_shared<julie::nn::func::ArcTan> (add_1->get_output());
    auto act_1 = std::make_shared<julie::nn::func::ReLU> (add_1->get_output());

    auto w2 = std::make_shared<julie::nn::var::Tensor<double>> (w2_mat);
    w2->trainable(true);
    auto b2 = std::make_shared<julie::nn::var::Tensor<double>> (b2_mat);
    b2->trainable(true);
    auto alpha2 = std::make_shared<julie::nn::var::Scalar<double>> (-0.3);
    alpha2->trainable(true);

    auto matmul_2 = std::make_shared<julie::nn::func::MatMul> (act_1->get_output(), w2);
    auto add_2 = std::make_shared<julie::nn::func::Add> (matmul_2->get_output(), b2);
    // auto act_2 = std::make_shared<julie::nn::func::ArcTan> (add_2->get_output());
    auto act_2 = std::make_shared<julie::nn::func::ReLU> (add_2->get_output());

    auto w3 = std::make_shared<julie::nn::var::Tensor<double>> (w3_mat);
    w3->trainable(true);
    auto b3 = std::make_shared<julie::nn::var::Tensor<double>> (b3_mat);
    b3->trainable(true);
    auto alpha3 = std::make_shared<julie::nn::var::Scalar<double>> (-0.3);
    alpha3->trainable(true);

    auto matmul_3 = std::make_shared<julie::nn::func::MatMul> (act_2->get_output(), w3);
    auto add_3 = std::make_shared<julie::nn::func::Add> (matmul_3->get_output(), b3);
    // auto act_3 = std::make_shared<julie::nn::func::ArcTan> (add_3->get_output());
    auto act_3 = std::make_shared<julie::nn::func::ReLU> (add_3->get_output());

    auto w4 = std::make_shared<julie::nn::var::Tensor<double>> (w4_mat);
    w4->trainable(true);
    auto b4 = std::make_shared<julie::nn::var::Tensor<double>> (b4_mat);
    b4->trainable(true);

    auto matmul_4 = std::make_shared<julie::nn::func::MatMul> (act_3->get_output(), w4);
    auto add_4 = std::make_shared<julie::nn::func::Add> (matmul_4->get_output(), b4);
    auto act_4 = std::make_shared<julie::nn::func::SoftMax> (add_4->get_output(), 1);

    auto target = std::make_shared<julie::nn::var::Tensor<double>> ();
    auto loss_func = std::make_shared<julie::nn::func::SoftMax_CrossEntropy> (add_4->get_output(), target, 1);

    auto act_4_output = dynamic_cast<julie::nn::var::Tensor<double>*>(act_4->get_output().get());
    auto loss = dynamic_cast<julie::nn::var::Tensor<double>*>(loss_func->get_output().get());

    julie::op::Graph the_model_graph;
    the_model_graph.new_function(matmul_1);
    the_model_graph.new_function(add_1);
    the_model_graph.new_function(act_1);

    the_model_graph.new_function(matmul_2);
    the_model_graph.new_function(add_2);
    the_model_graph.new_function(act_2);

    the_model_graph.new_function(matmul_3);
    the_model_graph.new_function(add_3);
    the_model_graph.new_function(act_3);

    the_model_graph.new_function(matmul_4);
    the_model_graph.new_function(add_4);
    the_model_graph.new_function(act_4);

    the_model_graph.new_function(loss_func);

    // ----------------------------------------------------------------- //
    // ------------------ Prepair for the training --------------------- //
    // ----------------------------------------------------------------- //

    the_model_graph.pave_backward_route(w1);
    the_model_graph.pave_backward_route(b1);
    the_model_graph.pave_backward_route(alpha1);

    the_model_graph.pave_backward_route(w2);
    the_model_graph.pave_backward_route(b2);
    the_model_graph.pave_backward_route(alpha2);

    the_model_graph.pave_backward_route(w3);
    the_model_graph.pave_backward_route(b3);
    the_model_graph.pave_backward_route(alpha3);

    the_model_graph.pave_backward_route(w4);
    the_model_graph.pave_backward_route(b4);

    // ----------------------------------------------------------------- //
    // --------------------------- Training ---------------------------- //
    // ----------------------------------------------------------------- //

    std::vector<std::shared_ptr<julie::op::Variable>> params_to_train;
    params_to_train.push_back(w1);
    params_to_train.push_back(b1);
    params_to_train.push_back(alpha1);

    params_to_train.push_back(w2);
    params_to_train.push_back(b2);
    params_to_train.push_back(alpha2);

    params_to_train.push_back(w3);
    params_to_train.push_back(b3);
    params_to_train.push_back(alpha3);

    params_to_train.push_back(w4);
    params_to_train.push_back(b4);

    std::cout << "---------------------- 5 ---------------------" << std::endl;

    // Define the optimizer
    julie::nn::opt::SGD optimizer {params_to_train, 0.001, 0.5};

    // Define random generators
    std::uniform_int_distribution<lint> train_distribution{ 0, static_cast<lint>(train_set.size() - 1) };
    std::uniform_int_distribution<lint> val_distribution{ 0, static_cast<lint>(val_set.size() - 1) };

    std::cout << "---------------------- 6 ---------------------" << std::endl;

    lint train_batch_size = 256;
    lint val_batch_size = 1000;

    for (lint itr  = 0; itr < 200; ++itr)
    {
        the_model_graph.clear_forwards();

        std::vector<julie::la::DMatrix<double>> x_batch;
        std::vector<julie::la::DMatrix<double>> t_batch;

        for (lint itr = 0; itr < train_batch_size; ++itr)
        {
            lint index = train_distribution(julie::la::global::global_rand_engine);
            x_batch.push_back(train_set[index]);
            t_batch.push_back(train_label[index]);
        }
        
        julie::la::DMatrix<double> x_mat {x_batch};
        julie::la::DMatrix<double> t_mat {t_batch};

        x_mat.normalize().reshape(julie::la::Shape {train_batch_size, 28 * 28});
        t_mat.reshape(julie::la::Shape {train_batch_size, 10});

        // std::cout << "---------------------- 7 ---------------------" << std::endl;
        // std::cout << x_mat << std::endl;
        
        // std::cout << "The index is: " << index << std::endl; 

        dynamic_cast<julie::nn::var::Tensor<double>*>(x.get())->val( x_mat );
        dynamic_cast<julie::nn::var::Tensor<double>*>(target.get())->val( t_mat );

        // std::cout << "---------------------- 8 ---------------------" << std::endl;

        the_model_graph.func_forward(loss_func);

        the_model_graph.func_forward(act_4);
        
        // std::cout << "---------------------- 9 ---------------------" << std::endl;
        
        the_model_graph.func_backward(matmul_1);

        // std::cout << "---------------------- 11 ---------------------" << std::endl;



        auto pred_batch = act_4_output->val()->get_collapsed(0);
        double right = 0;

        for (size_t i = 0; i < pred_batch.size(); ++i)
        {
            if (pred_batch[i].argmax().index() == t_batch[i].argmax().index())
            {
                right += 1.0;
            }
        }

        std::cout << "Train Accuracy: " << right / train_batch_size << std::endl;



        // Update the weights
        optimizer.step();
        the_model_graph.clear_forwards();
        the_model_graph.clear_backwards();

        if ((itr + 1)% 100 == 0)
        {
            lint index = val_distribution(julie::la::global::global_rand_engine);

            x_batch.clear();
            t_batch.clear();

            for (lint itr = 0; itr < val_batch_size; ++itr)
            {
                lint index = val_distribution(julie::la::global::global_rand_engine);
                x_batch.push_back(val_set[index]);
                t_batch.push_back(val_label[index]);
            }

            julie::la::DMatrix<double> x_mat {x_batch};
            julie::la::DMatrix<double> t_mat {t_batch};

            x_mat.normalize().reshape(julie::la::Shape {val_batch_size, 28 * 28});
            t_mat.reshape(julie::la::Shape {val_batch_size, 10});

            dynamic_cast<julie::nn::var::Tensor<double>*>(x.get())->val( x_mat );
            dynamic_cast<julie::nn::var::Tensor<double>*>(target.get())->val( t_mat );
            
            the_model_graph.func_forward(act_4);

            auto pred_batch = act_4_output->val()->get_collapsed(0);
            double right = 0;

            for (size_t i = 0; i < pred_batch.size(); ++i)
            {
                if (pred_batch[i].argmax().index() == t_batch[i].argmax().index())
                {
                    right += 1.0;
                }
            }

            std::cout << "Itr: " << itr << " ============================== " << std::endl;
            // std::cout << "alpha1: " << *(alpha1->val()) << std::endl;
            // std::cout << "alpha2: " << *(alpha2->val()) << std::endl;
            // std::cout << "alpha3: " << *(alpha3->val()) << std::endl;
            std::cout << "Itr: " << itr << " ========== Test Accuracy: " << right / val_batch_size << std::endl;
        }
    }

    
}

} // namespace demo