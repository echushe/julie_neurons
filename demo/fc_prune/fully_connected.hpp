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

/******************************************************************************
 * This is a sample program to construct and train a 2-layer fully connected
 * neural network using the MNIST dataset.
 * 
 * To build and run this program, please download the MNIST dataset in advance.
 ******************************************************************************/

#pragma once

#include "iMatrix.hpp"
#include "graph.hpp"
#include "add.hpp"
#include "matmul.hpp"
#include "multiply.hpp"
#include "softmax.hpp"
#include "tanh.hpp"
#include "relu.hpp"
#include "prelu.hpp"
#include "Dataset.hpp"
#include "Mnist.hpp"
#include "softmax_crossentropy.hpp"
#include "sgd.hpp"
#include "iMatrix_func.hpp"

#include <random>

namespace demo
{

void fc_train_mnist(const std::string &dataset_dir)
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
    //                       Load the MNIST data set                     //
    // ----------------------------------------------------------------- //

    // Buffers of training set
    std::vector<std::shared_ptr<julie::la::iMatrix<float>>> train_set;
    std::vector<std::shared_ptr<julie::la::iMatrix<float>>> train_label;

    // Buffers of validation set
    std::vector<std::shared_ptr<julie::la::iMatrix<float>>> val_set;
    std::vector<std::shared_ptr<julie::la::iMatrix<float>>> val_label;

    std::shared_ptr<dataset::Dataset> train_set_getter = std::make_shared<dataset::Mnist>(
        dataset_dir + "/train-images-idx3-ubyte",
        dataset_dir + "/train-labels-idx1-ubyte");
    
    std::shared_ptr<dataset::Dataset> val_set_getter = std::make_shared<dataset::Mnist>(
        dataset_dir + "/t10k-images-idx3-ubyte",
        dataset_dir + "/t10k-labels-idx1-ubyte");

    train_set_getter->get_samples_and_labels(train_set, train_label);
    val_set_getter->get_samples_and_labels(val_set, val_label);
    
    // ----------------------------------------------------------------- //
    //                Initialize tensors and scalars                     //
    // ----------------------------------------------------------------- //

    // Initialize weights of the first layer
    julie::la::iMatrix<float> W_1_mat { julie::la::Shape {28 * 28, 100}};
    julie::la::iMatrix<float> W_1_mask_mat {1, W_1_mat.shape()};
    julie::la::iMatrix<float> B_1_mat { 0, julie::la::Shape {100}};
    W_1_mat.gaussian_random(0, 0.01);
    
    // Initialize weights of the second layer
    julie::la::iMatrix<float> W_2_mat { julie::la::Shape {100, 10}};
    julie::la::iMatrix<float> W_2_mask_mat {1, W_2_mat.shape()};
    julie::la::iMatrix<float> B_2_mat { 0, julie::la::Shape {10}};
    W_2_mat.gaussian_random(0, 0.01);

    // Definition of the input tensor.
    // It will be assigned with batches of training or validation samples later
    auto x = std::make_shared<julie::nn::var::Tensor<float>> ();
    // Definition of the target tensor.
    // It will be assigned with batches of training or validation labels later
    auto target = std::make_shared<julie::nn::var::Tensor<float>> ();

    // Define trainable variables of the first layer and assign them with initial weights
    auto W_1 = std::make_shared<julie::nn::var::Tensor<float>> (W_1_mat);
    W_1->trainable(true);
    auto W_1_mask = std::make_shared<julie::nn::var::Tensor<float>> (W_1_mask_mat);
    auto B_1 = std::make_shared<julie::nn::var::Tensor<float>> (B_1_mat);
    B_1->trainable(true);
    // Alpha is a trainable scalar for the PReLU function
    auto Alpha = std::make_shared<julie::nn::var::Scalar<float>> (0.0);
    Alpha->trainable(true);

    // Define trainable variables of the second layer and assign them with initial weights
    auto W_2 = std::make_shared<julie::nn::var::Tensor<float>> (W_2_mat);
    W_2->trainable(true);
    auto W_2_mask = std::make_shared<julie::nn::var::Tensor<float>> (W_2_mask_mat);
    auto B_2 = std::make_shared<julie::nn::var::Tensor<float>> (B_2_mat);
    B_2->trainable(true);

    // ----------------------------------------------------------------- //
    //              Construct the first layer of network                 //
    // ----------------------------------------------------------------- //

    // The graph is a pool holding all functions and variables
    julie::op::Graph the_model_graph;

    // Define function nodes of the first layer
    auto mul_1 = std::make_shared<julie::nn::func::Multiply> ();
    auto matmul_1 = std::make_shared<julie::nn::func::MatMul> ();
    auto add_1 = std::make_shared<julie::nn::func::Add> ();
    auto act_1 = std::make_shared<julie::nn::func::PReLU> ();

    // Add function nodes with their inputs and outputs to the graph
    the_model_graph.add_node(mul_1, {W_1, W_1_mask});
    the_model_graph.add_node(matmul_1, {x, mul_1->get_output()});
    the_model_graph.add_node(add_1, {matmul_1->get_output(), B_1});
    the_model_graph.add_node(act_1, {add_1->get_output(), Alpha});

    // ----------------------------------------------------------------- //
    //              Construct the second layer of network                //
    // ----------------------------------------------------------------- //

    // Define function nodes of the second layer
    auto mul_2 = std::make_shared<julie::nn::func::Multiply> ();
    auto matmul_2 = std::make_shared<julie::nn::func::MatMul> ();
    auto add_2 = std::make_shared<julie::nn::func::Add> ();
    auto act_2 = std::make_shared<julie::nn::func::SoftMax> (1);

    // Add function nodes with their inputs and outputs to the graph
    the_model_graph.add_node(mul_2, {W_2, W_2_mask});
    the_model_graph.add_node(matmul_2, {act_1->get_output(), mul_2->get_output()});
    the_model_graph.add_node(add_2, {matmul_2->get_output(), B_2});
    the_model_graph.add_node(act_2, {add_2->get_output()});

    // ----------------------------------------------------------------- //
    //               Define loss function of the network                 //
    // ----------------------------------------------------------------- //
    
    // The axis index this loss function will run on
    lint axis_idx = 1;
    // Definition of loss function needs axis index
    // The 1st axis is batch dimension (index = 0),
    // so the loss function should run on the 2nd axis (index = 1)
    auto loss_func = std::make_shared<julie::nn::func::SoftMax_CrossEntropy> (axis_idx);

    // Add the loss function to the graph, using add_2 output and target as inputs
    the_model_graph.add_node(loss_func, {add_2->get_output(), target});

    // Get the handle of softmax output
    auto act_2_output = std::dynamic_pointer_cast<julie::nn::var::Tensor<float>>(act_2->get_output());
    // Get the handle of loss
    auto loss = std::dynamic_pointer_cast<julie::nn::var::Tensor<float>>(loss_func->get_output());

    the_model_graph.add_input(x);
    the_model_graph.set_device(mat_type);

    std::cout << the_model_graph.to_string() << std::endl;

    // ----------------------------------------------------------------- //
    //                             Training                              //
    // ----------------------------------------------------------------- //

    // Define the optimizer with learning rate and momentum
    julie::nn::opt::SGD optimizer {the_model_graph, 0.001, 0.7};

    // Define random generators
    std::uniform_int_distribution<lint> train_distribution{ 0, static_cast<lint>(train_set.size() - 1) };
    std::uniform_int_distribution<lint> val_distribution{ 0, static_cast<lint>(val_set.size() - 1) };
    std::default_random_engine rand_engine;

    lint train_batch_size = 256;
    lint val_batch_size = 10000;

    for (lint itr = 0; itr < 10000; ++itr)
    {
        // Make all nodes forward unvisited
        the_model_graph.clear_forwards();
        // Make all nodes backward unvisited
        the_model_graph.clear_backwards();

        std::vector<std::shared_ptr<julie::la::iMatrix<float>>> x_batch;
        std::vector<std::shared_ptr<julie::la::iMatrix<float>>> t_batch;

        // Use the random selector to create a list of X and a list of targets
        for (lint itr = 0; itr < train_batch_size; ++itr)
        {
            lint index = train_distribution(rand_engine);

            x_batch.push_back(train_set[index]);
            t_batch.push_back(train_label[index]);
        }
        
        // Convert the list of X and the list of targets into atomic batches (matrices)
        julie::la::iMatrix<float> x_mat {x_batch}; x_mat.set_matrix_type(mat_type);
        julie::la::iMatrix<float> t_mat {t_batch}; t_mat.set_matrix_type(mat_type);

        // Reshape X batch and target batch into compatible shapes that the network can accept
        x_mat.normalize().reshape(julie::la::Shape {train_batch_size, 28 * 28});
        t_mat.reshape(julie::la::Shape {train_batch_size, 10});

        // Assign X batch and target batch to x node and target node
        x->val(x_mat);
        target->val(t_mat);

        // Execute forward commands to the graph
        // Each forward command requires a function node as its endpoint
        the_model_graph.forward(loss_func->get_output());
        the_model_graph.forward(act_2->get_output());
        
        // Execute backward command to the gragh
        // Like forward commands, each backward command needs a function node
        // as its endpoint as well
        the_model_graph.backward();

        auto pred_batch = act_2_output->val()->argmax(1);
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
        optimizer.step();

        // Do validation for each 100 training iterations
        if ((itr + 1)% 100 == 0)
        {
            // Make all nodes forward unvisited
            the_model_graph.clear_forwards();
            // Make all nodes backward unvisited
            the_model_graph.clear_backwards();

            x_batch.clear();
            t_batch.clear();

            // Use the random selector to create a list of X and a list of targets
            for (lint i = 0; i < val_batch_size; ++i)
            {
                lint index = val_distribution(rand_engine);

                x_batch.push_back(val_set[index]);
                t_batch.push_back(val_label[index]);
            }

            // Convert the list of X and the list of targets into atomic batches (matrices)
            julie::la::iMatrix<float> x_mat {x_batch}; x_mat.set_matrix_type(mat_type);
            julie::la::iMatrix<float> t_mat {t_batch}; t_mat.set_matrix_type(mat_type);

            // Reshape X batch and target batch into compatible shapes that the network can accept
            x_mat.normalize().reshape(julie::la::Shape {val_batch_size, 28 * 28});
            t_mat.reshape(julie::la::Shape {val_batch_size, 10});

            // Assign X batch and target batch to x node and target node
            x->val( x_mat );
            target->val( t_mat );
            
            // Execute forward command to the graph
            // Each forward command requires a function node as its endpoint
            the_model_graph.forward(act_2->get_output());

            auto pred_batch = act_2_output->val()->argmax(1);
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
            std::cout << "Alpha: " << *(Alpha->val()) << std::endl;
            std::cout << "Itr: " << itr << " ========== Test Accuracy: " << right / val_batch_size << std::endl;
        }

        if ((itr + 1)% 500 == 0)
        {
            julie::la::abs_threshold(*(W_1_mask->val()), *(W_1->val()), 0.02f);
            julie::la::abs_threshold(*(W_2_mask->val()), *(W_2->val()), 0.02f);

            #ifdef WITH_CUDA
            julie::la::cpu::Matrix_CPU<float> w_1_mask_mat {*(W_1_mask->val()->get_cuda_instance())};
            #else
            julie::la::cpu::Matrix_CPU<float> w_1_mask_mat {*(W_1_mask->val()->get_cpu_instance())};
            #endif

            lint zero_count = 0;
            float *data = w_1_mask_mat.m_data;
            for (lint i = 0; i < w_1_mask_mat.shape().size(); ++i)
            {
                if (data[i] == 0)
                {
                    ++zero_count;
                }
            }
            std::cout << "~~~~~~~~~ Layer 1 prunned: " << static_cast<float>(zero_count) / w_1_mask_mat.shape().size() << std::endl;

            #ifdef WITH_CUDA
            julie::la::cpu::Matrix_CPU<float> w_2_mask_mat {*(W_2_mask->val()->get_cuda_instance())};
            #else
            julie::la::cpu::Matrix_CPU<float> w_2_mask_mat {*(W_2_mask->val()->get_cpu_instance())};
            #endif

            zero_count = 0;
            data = w_2_mask_mat.m_data;
            for (lint i = 0; i < w_2_mask_mat.shape().size(); ++i)
            {
                if (data[i] == 0)
                {
                    ++zero_count;
                }
            }
            std::cout << "~~~~~~~~~ Layer 2 prunned: " << static_cast<float>(zero_count) / w_2_mask_mat.shape().size() << std::endl;
        }
    }
}

} // namespace demo