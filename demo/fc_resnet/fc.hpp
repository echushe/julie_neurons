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
 * This is a sample program to construct and train a 4-layer fully connected
 * neural network applying ResNet (Residual neural network) structure.
 * 
 * To build and run this program, please download the MNIST dataset in advance.
 ******************************************************************************/

#pragma once

#include "iMatrix.hpp"
#include "graph.hpp"
#include "add.hpp"
#include "matmul.hpp"
#include "softmax.hpp"
#include "tanh.hpp"
#include "relu.hpp"
#include "prelu.hpp"
#include "Dataset.hpp"
#include "Mnist.hpp"
#include "half_squareerror.hpp"
#include "sum.hpp"
#include "sgd.hpp"

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
    julie::la::iMatrix<float> W_1_mat { julie::la::Shape {28 * 28, 400}};
    W_1_mat.gaussian_random(0, 0.01);
    
    // Initialize weights of the second layer
    julie::la::iMatrix<float> W_2_mat { julie::la::Shape {400, 400}};
    W_2_mat.gaussian_random(0, 0.01);

    // Initialize weights of the third layer
    julie::la::iMatrix<float> W_3_mat { julie::la::Shape {400, 400}};
    W_3_mat.gaussian_random(0, 0.01);

    // Initialize weights of the fourth layer
    julie::la::iMatrix<float> W_4_mat { julie::la::Shape {400, 10}};
    julie::la::iMatrix<float> B_4_mat { 0, julie::la::Shape {10}};
    W_4_mat.gaussian_random(0, 0.01);

    // Definition of the input tensor.
    // It will be assigned with batches of training or validation samples later
    auto x = std::make_shared<julie::nn::var::Tensor<float>> ();
    // Definition of the target tensor.
    // It will be assigned with batches of training or validation labels later
    auto target = std::make_shared<julie::nn::var::Tensor<float>> ();

    // Define trainable variables of the first layer and assign them with initial weights
    auto W_1 = std::make_shared<julie::nn::var::Tensor<float>> (W_1_mat);
    W_1->trainable(true);

    // Define trainable variables of the second layer and assign them with initial weights
    auto W_2 = std::make_shared<julie::nn::var::Tensor<float>> (W_2_mat);
    W_2->trainable(true);

    // Define trainable variables of the third layer and assign them with initial weights
    auto W_3 = std::make_shared<julie::nn::var::Tensor<float>> (W_3_mat);
    W_3->trainable(true);

    // Define trainable variables of the fourth layer and assign them with initial weights
    auto W_4 = std::make_shared<julie::nn::var::Tensor<float>> (W_4_mat);
    W_4->trainable(true);
    auto B_4 = std::make_shared<julie::nn::var::Tensor<float>> (B_4_mat);
    B_4->trainable(true);

    // ----------------------------------------------------------------- //
    //              Construct the first layer of network                 //
    // ----------------------------------------------------------------- //

    // The graph is a pool holding all functions and variables
    julie::op::Graph the_model_graph;

    // Define function nodes of the first layer
    auto matmul_1 = std::make_shared<julie::nn::func::MatMul> ();
    auto act_1 = std::make_shared<julie::nn::func::ReLU> ();

    // Add function nodes with their inputs and outputs to the graph
    the_model_graph.add_node(matmul_1, {x, W_1});
    the_model_graph.add_node(act_1, {matmul_1->get_output()});

    // ----------------------------------------------------------------- //
    //              Construct the second layer of network                //
    // ----------------------------------------------------------------- //

    // Define function nodes of the second layer
    auto matmul_2 = std::make_shared<julie::nn::func::MatMul> ();
    auto act_2 = std::make_shared<julie::nn::func::ReLU> ();

    // Add function nodes with their inputs and outputs to the graph
    the_model_graph.add_node(matmul_2, {act_1->get_output(), W_2});
    the_model_graph.add_node(act_2, {matmul_2->get_output()});

    // ----------------------------------------------------------------- //
    //              Construct the third layer of network                //
    // ----------------------------------------------------------------- //

    // Define function nodes of the second layer
    auto matmul_3 = std::make_shared<julie::nn::func::MatMul> ();

    // Add function nodes with their inputs and outputs to the graph
    the_model_graph.add_node(matmul_3, {act_2->get_output(), W_3});

    // ----------------------------------------------------------------- //
    //              Construct  sum(out_first_layer, out_third_layer)     //
    // ----------------------------------------------------------------- //

    auto add_1_3 = std::make_shared<julie::nn::func::Add> ();
    auto act_1_3 = std::make_shared<julie::nn::func::ReLU> ();

    the_model_graph.add_node(add_1_3, {act_1->get_output(), matmul_3->get_output()});
    the_model_graph.add_node(act_1_3, {add_1_3->get_output()});

    // ----------------------------------------------------------------- //
    //              Construct the fourth layer of network                //
    // ----------------------------------------------------------------- //

    // Define function nodes of the second layer
    auto matmul_4 = std::make_shared<julie::nn::func::MatMul> ();
    auto add_4 = std::make_shared<julie::nn::func::Add> ();

    // Add function nodes with their inputs and outputs to the graph
    the_model_graph.add_node(matmul_4, {act_1_3->get_output(), W_4});
    the_model_graph.add_node(add_4, {matmul_4->get_output(), B_4});

    // ----------------------------------------------------------------- //
    //               Define loss function of the network                 //
    // ----------------------------------------------------------------- //
    
    // The axis index this loss function will run on
    lint axis_idx = 1;
    // Definition of loss function needs axis index
    // The 1st axis is batch dimension (index = 0),
    // so the loss function should run on the 2nd axis (index = 1)
    auto loss_func = std::make_shared<julie::nn::func::HalfSquareError> (axis_idx);
    auto loss_func_sum = std::make_shared<julie::nn::func::Sum> ();
    loss_func_sum->to_loss_function();

    // Add the loss function to the graph, using add_4 output and target as inputs
    the_model_graph.add_node(loss_func, {add_4->get_output(), target});
    the_model_graph.add_node(loss_func_sum, {loss_func->get_output()});

    // Get the handle of network output
    auto add_4_output = std::dynamic_pointer_cast<julie::nn::var::Tensor<float>>(add_4->get_output());

    // Get the handle of loss
    auto loss = std::dynamic_pointer_cast<julie::nn::var::Tensor<float>>(loss_func_sum->get_output());

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
    lint val_batch_size = 1000;

    for (lint itr = 0; itr < 400; ++itr)
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
        the_model_graph.forward(loss_func_sum->get_output());
        
        // Execute backward command to the gragh
        // Like forward commands, each backward command needs a function node
        // as its endpoint as well
        the_model_graph.backward();

        auto pred_batch = add_4_output->val()->argmax(1);
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
            the_model_graph.forward(add_4_output);

            auto pred_batch = add_4_output->val()->argmax(1);
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
            std::cout << "Itr: " << itr << " ========== Test Accuracy: " << right / val_batch_size << std::endl;
        }
    }
}

} // namespace demo