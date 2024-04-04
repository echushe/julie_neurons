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
 * This is a sample program to construct and train a 5-layer convolutional
 * neural network using the MNIST dataset.
 * 
 * To build and run this program, please download the MNIST dataset in advance.
 ******************************************************************************/

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
#include "conv2d_op.hpp"
#include "maxpool.hpp"
#include "avgpool.hpp"

#include "Dataset.hpp"
#include "Mnist.hpp"

#include "softmax_crossentropy.hpp"
#include "sigmoid_crossentropy.hpp"

#include "sgd.hpp"

#include <random>

namespace demo
{

void conv2d_mnist(const std::string &dataset_dir)
{
    std::cout << "====================== train mnist =====================" << std::endl;

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

    std::vector<std::shared_ptr<julie::la::iMatrix<float>>> train_set;
    std::vector<std::shared_ptr<julie::la::iMatrix<float>>> train_label;

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
    // ------------------ Buid the network structure ------------------- //
    // ----------------------------------------------------------------- //

    // Initialize weights
    julie::la::iMatrix<float> W_01_mat {julie::la::Shape{32, 1, 5, 5}};
    W_01_mat.gaussian_random(0, 0.1);
    julie::la::iMatrix<float> B_01_mat {0, julie::la::Shape{32}};

    julie::la::iMatrix<float> W_02_mat {julie::la::Shape{32, 32, 5, 5}};
    W_02_mat.gaussian_random(0, 0.1);
    julie::la::iMatrix<float> B_02_mat {0, julie::la::Shape{32}};

    julie::la::iMatrix<float> W_1_mat {julie::la::Shape{32, 32, 6, 6}};
    W_1_mat.gaussian_random(0, 0.1);
    julie::la::iMatrix<float> B_1_mat {0, julie::la::Shape{32}};

    julie::la::iMatrix<float> W_2_mat {julie::la::Shape{32, 32, 4, 4}};
    W_2_mat.gaussian_random(0, 0.1);
    julie::la::iMatrix<float> B_2_mat {0, julie::la::Shape{32}};

    julie::la::iMatrix<float> W_3_mat {julie::la::Shape{64, 32, 3, 3}};
    W_3_mat.gaussian_random(0, 0.1);
    julie::la::iMatrix<float> B_3_mat {0, julie::la::Shape{64}};

    julie::la::iMatrix<float> W_4_mat {julie::la::Shape{128, 64, 3, 3}};
    W_4_mat.gaussian_random(0, 0.1);
    julie::la::iMatrix<float> B_4_mat {0, julie::la::Shape{128}};

    julie::la::iMatrix<float> W_5_mat {julie::la::Shape{10, 128, 1, 1}};
    W_5_mat.gaussian_random(0, 0.1);
    julie::la::iMatrix<float> B_5_mat {0, julie::la::Shape{10}};

    julie::op::Graph the_model_graph;

    // Construct the first conv layer
    /////////////////////////////////////////////////////////////////////////////////////
    auto x = std::make_shared<julie::nn::var::Tensor<float>> ();

    /////////////////////////////////////////////////////////////////////////////////////

    auto W_01 = std::make_shared<julie::nn::var::Tensor<float>> (W_01_mat);
    W_01->trainable(true);
    auto B_01 = std::make_shared<julie::nn::var::Tensor<float>> (B_01_mat);
    B_01->trainable(true);

    auto conv_01 = std::make_shared<julie::nn::func::Conv2d> (2, 2, 1, 1);
    auto act_01 = std::make_shared<julie::nn::func::ReLU> ();

    the_model_graph.add_node(conv_01, {x, W_01, B_01});
    the_model_graph.add_node(act_01, {conv_01->get_output()});

    /////////////////////////////////////////////////////////////////////////////////////

    auto W_02 = std::make_shared<julie::nn::var::Tensor<float>> (W_02_mat);
    W_02->trainable(true);
    auto B_02 = std::make_shared<julie::nn::var::Tensor<float>> (B_02_mat);
    B_02->trainable(true);

    auto conv_02 = std::make_shared<julie::nn::func::Conv2d> (2, 2, 1, 1);
    auto act_02 = std::make_shared<julie::nn::func::ReLU> ();

    the_model_graph.add_node(conv_02, {act_01->get_output(), W_02, B_02});
    the_model_graph.add_node(act_02, {conv_02->get_output()});

    /////////////////////////////////////////////////////////////////////////////////////

    auto W_1 = std::make_shared<julie::nn::var::Tensor<float>> (W_1_mat);
    W_1->trainable(true);
    auto B_1 = std::make_shared<julie::nn::var::Tensor<float>> (B_1_mat);
    B_1->trainable(true);

    auto conv_1 = std::make_shared<julie::nn::func::Conv2d> (0, 0, 2, 2);
    auto act_1 = std::make_shared<julie::nn::func::ReLU> ();

    the_model_graph.add_node(conv_1, {act_02->get_output(), W_1, B_1});
    the_model_graph.add_node(act_1, {conv_1->get_output()});

    // Construct the second conv layer
    //////////////////////////////////////////////////////////////////////////////////////

    auto W_2 = std::make_shared<julie::nn::var::Tensor<float>> (W_2_mat);
    W_2->trainable(true);
    auto B_2 = std::make_shared<julie::nn::var::Tensor<float>> (B_2_mat);
    B_2->trainable(true);

    auto conv_2 = std::make_shared<julie::nn::func::Conv2d> (0, 0, 2, 2);
    auto act_2 = std::make_shared<julie::nn::func::ReLU> ();

    the_model_graph.add_node(conv_2, {act_1->get_output(), W_2, B_2});
    the_model_graph.add_node(act_2, {conv_2->get_output()});

    // Construct the third conv layer
    /////////////////////////////////////////////////////////////////////////////////////

    auto W_3 = std::make_shared<julie::nn::var::Tensor<float>> (W_3_mat);
    W_3->trainable(true);
    auto B_3 = std::make_shared<julie::nn::var::Tensor<float>> (B_3_mat);
    B_3->trainable(true);

    auto conv_3 = std::make_shared<julie::nn::func::Conv2d> (1, 1, 2, 2);
    auto act_3 = std::make_shared<julie::nn::func::ReLU> ();

    the_model_graph.add_node(conv_3, {act_2->get_output(), W_3, B_3});
    the_model_graph.add_node(act_3, {conv_3->get_output()});

    // Construct the forth conv layer
    ////////////////////////////////////////////////////////////////////////////////////

    auto W_4 = std::make_shared<julie::nn::var::Tensor<float>> (W_4_mat);
    W_4->trainable(true);
    auto B_4 = std::make_shared<julie::nn::var::Tensor<float>> (B_4_mat);
    B_4->trainable(true);

    auto conv_4 = std::make_shared<julie::nn::func::Conv2d> (0, 0, 3, 3);
    auto act_4 = std::make_shared<julie::nn::func::ReLU> ();

    the_model_graph.add_node(conv_4, {act_3->get_output(), W_4, B_4});
    the_model_graph.add_node(act_4, {conv_4->get_output()});

    // Construct the fifth conv layer
    ////////////////////////////////////////////////////////////////////////////////////

    auto W_5 = std::make_shared<julie::nn::var::Tensor<float>> (W_5_mat);
    W_5->trainable(true);
    auto B_5 = std::make_shared<julie::nn::var::Tensor<float>> (B_5_mat);
    B_5->trainable(true);

    auto conv_5 = std::make_shared<julie::nn::func::Conv2d> (0, 0, 1, 1);
    auto act_5 = std::make_shared<julie::nn::func::SoftMax> (1);

    the_model_graph.add_node(conv_5, {act_4->get_output(), W_5, B_5});
    the_model_graph.add_node(act_5, {conv_5->get_output()});

    ////////////////////////////////////////////////////////////////////////////////////

    auto target = std::make_shared<julie::nn::var::Tensor<float>> ();
    auto loss_func = std::make_shared<julie::nn::func::SoftMax_CrossEntropy> (1);

    the_model_graph.add_node(loss_func, {conv_5->get_output(), target});
    
    auto act_1_output = std::dynamic_pointer_cast<julie::nn::var::Tensor<float>> (act_1->get_output());
    auto act_2_output = std::dynamic_pointer_cast<julie::nn::var::Tensor<float>> (act_2->get_output());
    auto act_3_output = std::dynamic_pointer_cast<julie::nn::var::Tensor<float>> (act_3->get_output());
    auto act_4_output = std::dynamic_pointer_cast<julie::nn::var::Tensor<float>> (act_4->get_output());
    auto act_5_output = std::dynamic_pointer_cast<julie::nn::var::Tensor<float>> (act_5->get_output());
    auto loss_output = std::dynamic_pointer_cast<julie::nn::var::Tensor<float>> (loss_func->get_output());

    the_model_graph.add_input(x);
    the_model_graph.set_device(mat_type);

    std::cout << the_model_graph.to_string() << std::endl;
    
    ///////////////////////////////////////////////////////////////////////////////////
    // ----------------------------------------------------------------- //
    //                             Training                              //
    // ----------------------------------------------------------------- //

    // Define the optimizer
    julie::nn::opt::SGD optimizer {the_model_graph, 0.0005, 0.5};

    // Define random generators
    std::uniform_int_distribution<lint> train_distribution{ 0, static_cast<lint>(train_set.size() - 1) };
    std::uniform_int_distribution<lint> val_distribution{ 0, static_cast<lint>(val_set.size() - 1) };

    lint train_batch_size = 100;
    lint val_batch_size = 100;

    std::default_random_engine rand_engine;

    for (lint itr = 0; itr < 600; ++itr)
    {
        the_model_graph.clear_forwards();
        the_model_graph.clear_backwards();

        std::vector<std::shared_ptr<julie::la::iMatrix<float>>> x_batch;
        std::vector<std::shared_ptr<julie::la::iMatrix<float>>> t_batch;

        for (lint i = 0; i < train_batch_size; ++i)
        {
            lint index = train_distribution(rand_engine);

            x_batch.push_back(train_set[index]);
            t_batch.push_back(train_label[index]);
        }
        
        julie::la::iMatrix<float> x_mat {x_batch}; x_mat.set_matrix_type(mat_type);
        julie::la::iMatrix<float> t_mat {t_batch}; t_mat.set_matrix_type(mat_type);

        x_mat.normalize().reshape(julie::la::Shape {train_batch_size, 1, 28, 28});
        t_mat.reshape(julie::la::Shape {train_batch_size, 10, 1, 1});

        x->val( x_mat );
        target->val( t_mat );

        the_model_graph.forward(loss_func->get_output());
        the_model_graph.forward(act_5->get_output());
        
        the_model_graph.backward();

        auto pred_batch = act_5_output->val()->argmax(1);
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
        
        // One validation for each 100 training iterations
        if ((itr + 1) % 100 == 0)
        {
            the_model_graph.clear_forwards();
            the_model_graph.clear_backwards();
            the_model_graph.clear_cache();

            float right = 0;
            for (lint infer_itr = 0; infer_itr < 100; ++infer_itr)
            {
                the_model_graph.clear_forwards();
                lint index = val_distribution(rand_engine);

                x_batch.clear();
                t_batch.clear();

                for (lint i = 0; i < val_batch_size; ++i)
                {
                    lint index = val_distribution(rand_engine);

                    x_batch.push_back(val_set[index]);
                    t_batch.push_back(val_label[index]);
                }

                julie::la::iMatrix<float> x_mat {x_batch}; x_mat.set_matrix_type(mat_type);
                julie::la::iMatrix<float> t_mat {t_batch}; t_mat.set_matrix_type(mat_type);

                x_mat.normalize().reshape(julie::la::Shape {val_batch_size, 1, 28, 28});
                t_mat.reshape(julie::la::Shape {val_batch_size, 10, 1, 1});

                x->val( x_mat );
                target->val( t_mat );
                
                the_model_graph.forward(act_5->get_output());

                auto pred_batch = act_5_output->val()->argmax(1);
                auto targets    = t_mat.argmax(1);

                for (size_t i = 0; i < pred_batch.size(); ++i)
                {
                    if (pred_batch[i][1] == targets[i][1])
                    {
                        right += 1.0;
                    }
                }
            }

            std::cout << "Train Itr: " << itr << " ============================== " << std::endl;
            std::cout << act_1_output->val()->shape() << std::endl;
            std::cout << act_2_output->val()->shape() << std::endl;
            std::cout << act_3_output->val()->shape() << std::endl;
            std::cout << act_4_output->val()->shape() << std::endl;
            std::cout << act_5_output->val()->shape() << std::endl;
            std::cout << "Train Itr: " << itr << " ========== Test Accuracy: " << right / val_batch_size / 100 << std::endl;
        }
    }

}
}  // namespace demo