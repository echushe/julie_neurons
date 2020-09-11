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
#include "conv2d_op.hpp"

#include "Dataset.hpp"
#include "Mnist.hpp"

#include "softmax_crossentropy.hpp"
#include "sigmoid_crossentropy.hpp"

#include "sgd.hpp"

namespace demo
{
    void conv2d_mnist()
    {
        std::cout << "====================== train mnist =====================" << std::endl;

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
        julie::la::DMatrix<double> w1_mat {julie::la::Shape{4, 1, 4, 4}};
        w1_mat.gaussian_random(0, 0.1);
        julie::la::DMatrix<double> b1_mat {0, julie::la::Shape{4}};

        julie::la::DMatrix<double> w2_mat {julie::la::Shape{8, 4, 3, 3}};
        w2_mat.gaussian_random(0, 0.1);
        julie::la::DMatrix<double> b2_mat {0, julie::la::Shape{8}};

        julie::la::DMatrix<double> w3_mat {julie::la::Shape{16, 8, 3, 3}};
        w3_mat.gaussian_random(0, 0.1);
        julie::la::DMatrix<double> b3_mat {0, julie::la::Shape{16}};

        julie::la::DMatrix<double> w4_mat {julie::la::Shape{32, 16, 3, 3}};
        w4_mat.gaussian_random(0, 0.1);
        julie::la::DMatrix<double> b4_mat {0, julie::la::Shape{32}};

        julie::la::DMatrix<double> w5_mat {julie::la::Shape{10, 32, 1, 1}};
        w5_mat.gaussian_random(0, 0.1);
        julie::la::DMatrix<double> b5_mat {0, julie::la::Shape{10}};


        
        // Build the network structure

        auto x = std::make_shared<julie::nn::var::Tensor<double>> ();
        auto w1 = std::make_shared<julie::nn::var::Tensor<double>> (w1_mat);
        w1->trainable(true);
        auto b1 = std::make_shared<julie::nn::var::Tensor<double>> (b1_mat);
        b1->trainable(true);

        auto conv_1 = std::make_shared<julie::nn::func::Conv2d> (x, w1, b1, 2, 2, 2, 2);
        auto act_1 = std::make_shared<julie::nn::func::ReLU> (conv_1->get_output());

        //////////////////////////////////////////////////////////////////////////////////////

        auto w2 = std::make_shared<julie::nn::var::Tensor<double>> (w2_mat);
        w2->trainable(true);
        auto b2 = std::make_shared<julie::nn::var::Tensor<double>> (b2_mat);
        b2->trainable(true);

        auto conv_2 = std::make_shared<julie::nn::func::Conv2d> (act_1->get_output(), w2, b2, 0, 0, 2, 2);
        auto act_2 = std::make_shared<julie::nn::func::ReLU> (conv_2->get_output());

        /////////////////////////////////////////////////////////////////////////////////////

        auto w3 = std::make_shared<julie::nn::var::Tensor<double>> (w3_mat);
        w3->trainable(true);
        auto b3 = std::make_shared<julie::nn::var::Tensor<double>> (b3_mat);
        b3->trainable(true);

        auto conv_3 = std::make_shared<julie::nn::func::Conv2d> (act_2->get_output(), w3, b3, 0, 0, 2, 2);
        auto act_3 = std::make_shared<julie::nn::func::ReLU> (conv_3->get_output());

        ////////////////////////////////////////////////////////////////////////////////////

        auto w4 = std::make_shared<julie::nn::var::Tensor<double>> (w4_mat);
        w4->trainable(true);
        auto b4 = std::make_shared<julie::nn::var::Tensor<double>> (b4_mat);
        b4->trainable(true);

        auto conv_4 = std::make_shared<julie::nn::func::Conv2d> (act_3->get_output(), w4, b4, 0, 0, 3, 3);
        auto act_4 = std::make_shared<julie::nn::func::ReLU> (conv_4->get_output());

        ////////////////////////////////////////////////////////////////////////////////////

        auto w5 = std::make_shared<julie::nn::var::Tensor<double>> (w5_mat);
        w5->trainable(true);
        auto b5 = std::make_shared<julie::nn::var::Tensor<double>> (b5_mat);
        b5->trainable(true);

        auto conv_5 = std::make_shared<julie::nn::func::Conv2d> (act_4->get_output(), w5, b5, 0, 0, 1, 1);
        auto act_5 = std::make_shared<julie::nn::func::SoftMax> (conv_5->get_output(), 1);

        ////////////////////////////////////////////////////////////////////////////////////

        auto target = std::make_shared<julie::nn::var::Tensor<double>> ();
        auto loss_func = std::make_shared<julie::nn::func::SoftMax_CrossEntropy> (conv_5->get_output(), target, 1);
        
        auto act_5_output = dynamic_cast<julie::nn::var::Tensor<double>*>(act_5->get_output().get());
        auto loss_output = dynamic_cast<julie::nn::var::Tensor<double>*>(loss_func->get_output().get());
        auto conv_5_output = dynamic_cast<julie::nn::var::Tensor<double>*>(conv_5->get_output().get());

        std::cout << "---------------------- 1 ---------------------" << std::endl;

        julie::op::Graph the_model_graph;
        the_model_graph.new_function(conv_1);
        the_model_graph.new_function(act_1);
        the_model_graph.new_function(conv_2);
        the_model_graph.new_function(act_2);
        the_model_graph.new_function(conv_3);
        the_model_graph.new_function(act_3);
        the_model_graph.new_function(conv_4);
        the_model_graph.new_function(act_4);
        the_model_graph.new_function(conv_5);
        the_model_graph.new_function(act_5);
        the_model_graph.new_function(loss_func);


        the_model_graph.pave_backward_route(w1);
        the_model_graph.pave_backward_route(b1);
        the_model_graph.pave_backward_route(w2);
        the_model_graph.pave_backward_route(b2);
        the_model_graph.pave_backward_route(w3);
        the_model_graph.pave_backward_route(b3);
        the_model_graph.pave_backward_route(w4);
        the_model_graph.pave_backward_route(b4);
        the_model_graph.pave_backward_route(w5);
        the_model_graph.pave_backward_route(b5);


        
        ///////////////////////////////////////////////////////////////////////////////////
        // ----------------------------------------------------------------- //
        // --------------------------- Training ---------------------------- //
        // ----------------------------------------------------------------- //

        std::vector<std::shared_ptr<julie::op::Variable>> params_to_train;
        params_to_train.push_back(w1);
        params_to_train.push_back(b1);

        params_to_train.push_back(w2);
        params_to_train.push_back(b2);

        params_to_train.push_back(w3);
        params_to_train.push_back(b3);

        params_to_train.push_back(w4);
        params_to_train.push_back(b4);

        //std::cout << "---------------------- 2 ---------------------" << std::endl;

        // Define the optimizer
        julie::nn::opt::SGD optimizer {params_to_train, 0.001, 0.5};

        // Define random generators
        std::uniform_int_distribution<lint> train_distribution{ 0, static_cast<lint>(train_set.size() - 1) };
        std::uniform_int_distribution<lint> val_distribution{ 0, static_cast<lint>(val_set.size() - 1) };

        //std::cout << "---------------------- 3 ---------------------" << std::endl;

        lint train_batch_size = 256;
        lint val_batch_size = 1000;

        for (lint itr = 0; itr < 200; ++itr)
        {
            the_model_graph.clear_forwards();
            the_model_graph.clear_backwards();

            std::vector<julie::la::DMatrix<double>> x_batch;
            std::vector<julie::la::DMatrix<double>> t_batch;

            //std::cout << "---------- 4 ---------- Alive SLMatrix items: " << julie::la::SLMatrixTuple<double>::REF_COUNT << std::endl;

            for (lint itr = 0; itr < train_batch_size; ++itr)
            {
                lint index = train_distribution(julie::la::global::global_rand_engine);
                x_batch.push_back(train_set[index]);
                t_batch.push_back(train_label[index]);
            }

            //std::cout << "---------- 5 ---------- Alive SLMatrix items: " << julie::la::SLMatrixTuple<double>::REF_COUNT << std::endl;
            
            julie::la::DMatrix<double> x_mat {x_batch};
            julie::la::DMatrix<double> t_mat {t_batch};

            //std::cout << "---------- 6 ---------- Alive SLMatrix items: " << julie::la::SLMatrixTuple<double>::REF_COUNT << std::endl;

            x_mat.normalize().reshape(julie::la::Shape {train_batch_size, 1, 28, 28});
            t_mat.reshape(julie::la::Shape {train_batch_size, 10, 1, 1});

            //std::cout << "---------- 7 ---------- Alive SLMatrix items: " << julie::la::SLMatrixTuple<double>::REF_COUNT << std::endl;
            // std::cout << x_mat << std::endl;
            
            // std::cout << "The index is: " << index << std::endl; 

            dynamic_cast<julie::nn::var::Tensor<double>*>(x.get())->val( x_mat );
            dynamic_cast<julie::nn::var::Tensor<double>*>(target.get())->val( t_mat );

            //std::cout << "---------- 8 ---------- Alive SLMatrix items: " << julie::la::SLMatrixTuple<double>::REF_COUNT << std::endl;

            the_model_graph.func_forward(loss_func);

            the_model_graph.func_forward(act_5);

            
            //std::cout << "---------- 9 ---------- Alive SLMatrix items: " << julie::la::SLMatrixTuple<double>::REF_COUNT << std::endl;
            
            the_model_graph.func_backward(conv_1);

            //std::cout << "---------- 10 ---------- Alive SLMatrix items: " << julie::la::SLMatrixTuple<double>::REF_COUNT << std::endl;


            auto pred_batch = act_5_output->val()->get_collapsed(0);
            double right = 0;

            for (size_t i = 0; i < pred_batch.size(); ++i)
            {
                if (pred_batch[i].argmax().index() == t_batch[i].argmax().index())
                {
                    right += 1.0;
                }
            }

            //std::cout << "---------- 11 ---------- Alive SLMatrix items: " << julie::la::SLMatrixTuple<double>::REF_COUNT << std::endl;

            std::cout << "Train Accuracy: " << right / train_batch_size << std::endl;

            // Update the weights
            optimizer.step();

            //std::cout << "---------- 12 ---------- Alive SLMatrix items: " << julie::la::SLMatrixTuple<double>::REF_COUNT << std::endl;

            if ((itr + 1) % 100 == 0)
            {
                the_model_graph.clear_forwards();
                the_model_graph.clear_backwards();
                the_model_graph.destroy_backward_route();

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

                x_mat.normalize().reshape(julie::la::Shape {val_batch_size, 1, 28, 28});
                t_mat.reshape(julie::la::Shape {val_batch_size, 10, 1, 1});

                dynamic_cast<julie::nn::var::Tensor<double>*>(x.get())->val( x_mat );
                dynamic_cast<julie::nn::var::Tensor<double>*>(target.get())->val( t_mat );
                
                the_model_graph.func_forward(act_5);

                auto pred_batch = act_5_output->val()->get_collapsed(0);
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

                the_model_graph.pave_backward_route(w1);
                the_model_graph.pave_backward_route(b1);
                the_model_graph.pave_backward_route(w2);
                the_model_graph.pave_backward_route(b2);
                the_model_graph.pave_backward_route(w3);
                the_model_graph.pave_backward_route(b3);
                the_model_graph.pave_backward_route(w4);
                the_model_graph.pave_backward_route(b4);
                the_model_graph.pave_backward_route(w5);
                the_model_graph.pave_backward_route(b5);
                
            }
        }

    }
}  // namespace demo