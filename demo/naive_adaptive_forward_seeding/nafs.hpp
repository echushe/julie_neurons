#pragma once

#include "iMatrix.hpp"
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

#include <random>

namespace demo
{

void fc_train_mnist()
{
    std::cout << "====================== fc_train_mnist =====================" << std::endl;

    // MatrixTypes:
    // julie::CPU    Any calculation of this matrix type will run on CPU only
    // julie::CUDA   Most calculations of this matrix type will run on nvidia GPU
    // julie::la::CL     Most calculations of this matrix type will run via openCL API (still under development)
    julie::MatrixType mat_type;
#ifdef WITH_CUDA
    mat_type = julie::CUDA;
#else
    mat_type = julie::CPU;
#endif

    // ----------------------------------------------------------------- //
    // --------------------- Load the MNIST data set ------------------- //
    // ----------------------------------------------------------------- //

    std::string dataset_dir = "../../../dataset/";

    std::vector<std::shared_ptr<julie::la::iMatrix<float>>> train_set;
    std::vector<std::shared_ptr<julie::la::iMatrix<float>>> train_label;

    std::vector<std::shared_ptr<julie::la::iMatrix<float>>> val_set;
    std::vector<std::shared_ptr<julie::la::iMatrix<float>>> val_label;

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

    julie::la::iMatrix<float> first_w_mat { julie::la::Shape {28 * 28, 10} };
    julie::la::iMatrix<float> first_b_mat { 0, julie::la::Shape {1, 10} };
    first_w_mat.gaussian_random(0, 0.1);

    // Build the network structure

    auto x = std::make_shared<julie::nn::var::Tensor<float>> ();
    auto last_w = std::make_shared<julie::nn::var::Tensor<float>> (first_w_mat);
    last_w->trainable(true);
    auto last_b = std::make_shared<julie::nn::var::Tensor<float>> (first_b_mat);
    last_b->trainable(true);

    auto last_matmul = std::make_shared<julie::nn::func::MatMul> ();
    auto last_add = std::make_shared<julie::nn::func::Add> ();
    auto last_act = std::make_shared<julie::nn::func::Sigmoid> ();

    auto last_target = std::make_shared<julie::nn::var::Tensor<float>> ();
    auto last_loss_func = std::make_shared<julie::nn::func::Sigmoid_CrossEntropy> (1);

    julie::op::Graph the_model_graph;
    the_model_graph.add_node(last_matmul, {x, last_w});
    the_model_graph.add_node(last_add, {last_matmul->get_output(), last_b});
    the_model_graph.add_node(last_act, {last_add->get_output()});
    the_model_graph.add_node(last_loss_func, {last_add->get_output(), last_target});

    the_model_graph.add_input(x);
    the_model_graph.set_device(mat_type);

    auto last_act_output = last_act->get_output();

    std::cout << the_model_graph.to_string() << std::endl;

    auto first_matmul = last_matmul;
    auto first_w = last_w;
    auto first_b = last_b;

    // ----------------------------------------------------------------- //
    // --------------------------- Training ---------------------------- //
    // ----------------------------------------------------------------- //

    std::cout << "---------------------- 5 ---------------------" << std::endl;

    // Define the optimizer
    julie::nn::opt::SGD last_optimizer {the_model_graph, 0.0005, 0.7};

    // Define random generators
    std::uniform_int_distribution<lint> train_distribution{ 0, static_cast<lint>(train_set.size() - 1) };
    std::uniform_int_distribution<lint> val_distribution{ 0, static_cast<lint>(val_set.size() - 1) };

    std::cout << "---------------------- 6 ---------------------" << std::endl;

    lint train_batch_size = 1024;
    lint val_batch_size = 2000;

    std::default_random_engine rand_engine;

    for (lint itr  = 0; itr < 20000; ++itr)
    {
        the_model_graph.clear_forwards();
        the_model_graph.clear_backwards();

        std::vector<std::shared_ptr<julie::la::iMatrix<float>>> x_batch;
        std::vector<std::shared_ptr<julie::la::iMatrix<float>>> t_batch;

        for (lint itr = 0; itr < train_batch_size; ++itr)
        {
            lint index = train_distribution(rand_engine);

            x_batch.push_back(train_set[index]);
            t_batch.push_back(train_label[index]);
        }
        
        julie::la::iMatrix<float> x_mat {x_batch}; x_mat.set_matrix_type(mat_type);
        julie::la::iMatrix<float> t_mat {t_batch}; t_mat.set_matrix_type(mat_type);

        x_mat.normalize().reshape(julie::la::Shape {train_batch_size, 28 * 28});
        t_mat.reshape(julie::la::Shape {train_batch_size, 10});

        // std::cout << "---------------------- 7 ---------------------" << std::endl;
        // std::cout << x_mat << std::endl;
        
        // std::cout << "The index is: " << index << std::endl; 

        x->val( x_mat );
        last_target->val( t_mat );

        // std::cout << "---------------------- 8 ---------------------" << std::endl;

        the_model_graph.forward(last_loss_func->get_output());
        the_model_graph.forward(last_act->get_output());
        
        // std::cout << "---------------------- 9 ---------------------" << std::endl;
        
        the_model_graph.backward(last_w);
        the_model_graph.backward(last_b);

        // std::cout << "---------------------- 11 ---------------------" << std::endl;
        // std::cout << the_model_graph.to_string() << std::endl;

        auto pred_batch = dynamic_cast<julie::nn::var::Tensor<float>*>(last_act_output.get())->val()->argmax(1);
        auto targets    = t_mat.argmax(1);
        float right = 0;

        for (size_t i = 0; i < pred_batch.size(); ++i)
        {
            if (pred_batch[i][1] == targets[i][1])
            {
                right += 1.0;
            }
        }

        std::cout << "Train Accuracy: " << right / train_batch_size << std::endl;

        // Update the weights
        last_optimizer.step();

        if ((itr + 1)% 500 == 0)
        // if (right / train_batch_size > 0.95)
        {
            the_model_graph.clear_forwards();
            the_model_graph.clear_backwards();
            // std::cout << the_model_graph.to_string() << std::endl;

            lint index = val_distribution(rand_engine);

            x_batch.clear();
            t_batch.clear();

            for (lint itr = 0; itr < val_batch_size; ++itr)
            {
                lint index = val_distribution(rand_engine);

                x_batch.push_back(val_set[index]);
                t_batch.push_back(val_label[index]);
            }

            julie::la::iMatrix<float> x_mat {x_batch}; x_mat.set_matrix_type(mat_type);
            julie::la::iMatrix<float> t_mat {t_batch}; t_mat.set_matrix_type(mat_type);

            x_mat.normalize().reshape(julie::la::Shape {val_batch_size, 28 * 28});
            t_mat.reshape(julie::la::Shape {val_batch_size, 10});

            x->val( x_mat );
            last_target->val( t_mat );
            
            the_model_graph.forward(last_act->get_output());

            auto pred_batch = dynamic_cast<julie::nn::var::Tensor<float>*>(last_act_output.get())->val()->argmax(1);
            auto targets    = t_mat.argmax(1);
            float right = 0;

            for (size_t i = 0; i < pred_batch.size(); ++i)
            {
                if (pred_batch[i][1] == targets[i][1])
                {
                    right += 1.0;
                }
            }

            std::cout << "Itr: " << itr << " ============================== " << std::endl;
            // std::cout << "alpha1: " << *(alpha1->val()) << std::endl;
            // std::cout << "alpha2: " << *(alpha2->val()) << std::endl;
            // std::cout << "alpha3: " << *(alpha3->val()) << std::endl;
            std::cout << "Itr: " << itr << " ========== Test Accuracy: " << right / val_batch_size << std::endl;

            // Remove loss function of the previous layer, add a new layer to the network
            {
                the_model_graph.remove_node(last_loss_func);

                julie::la::iMatrix<float> w_mat { julie::la::Shape {10, 10} };
                julie::la::iMatrix<float> b_mat { 0, julie::la::Shape {1, 10} };
                w_mat.gaussian_random(0, 0.1);

                // Build the network structure
                last_w = std::make_shared<julie::nn::var::Tensor<float>> (w_mat);
                last_w->trainable(true);
                last_b = std::make_shared<julie::nn::var::Tensor<float>> (b_mat);
                last_b->trainable(true);

                last_matmul = std::make_shared<julie::nn::func::MatMul> ();   
                last_add = std::make_shared<julie::nn::func::Add> ();
                last_act = std::make_shared<julie::nn::func::Sigmoid> ();

                last_target = std::make_shared<julie::nn::var::Tensor<float>> ();
                last_loss_func = std::make_shared<julie::nn::func::Sigmoid_CrossEntropy> (1);

                the_model_graph.add_node(last_matmul, {last_act_output, last_w});
                the_model_graph.add_node(last_add, {last_matmul->get_output(), last_b});
                the_model_graph.add_node(last_act, {last_add->get_output()});
                the_model_graph.add_node(last_loss_func, {last_add->get_output(), last_target});

                last_act_output = last_act->get_output();

                std::cout << the_model_graph.to_string() << std::endl;

                the_model_graph.set_device(mat_type);
                last_optimizer = julie::nn::opt::SGD{the_model_graph, 0.001, 0.7};
            }
        }
    }

    
}

} // namespace demo