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
 * This is a sample program to construct and train a convolutional neural network
 * of three GoogleNET inceptions using the MNIST dataset.
 * 
 * To build and run this program, please download the MNIST dataset in advance.
 ******************************************************************************/

#pragma once

#include "iMatrix.hpp"
#include "graph.hpp"
#include "add.hpp"
#include "matmul.hpp"
#include "relu.hpp"
#include "abs.hpp"
#include "scale.hpp"
#include "softmax.hpp"
#include "sigmoid.hpp"
#include "tanh.hpp"
#include "arctan.hpp"
#include "prelu.hpp"
#include "conv2d_op.hpp"
#include "maxpool.hpp"
#include "avgpool.hpp"
#include "concat.hpp"

#include "Dataset.hpp"
#include "Mnist.hpp"

#include "softmax_crossentropy.hpp"
#include "sigmoid_crossentropy.hpp"

#include "sgd.hpp"

#include <random>

namespace demo
{

std::shared_ptr<julie::nn::var::Tensor<float>> build_inception(
    julie::op::Graph &the_model_graph,
    const std::shared_ptr<julie::op::Variable> &input,
    lint input_ch,
    lint ch_1,
    lint ch_21, lint ch_22,
    lint ch_31, lint ch_32,
    lint ch_4 )
{   
    julie::la::iMatrix<float> W_1_mat {julie::la::Shape{ch_1, input_ch, 1, 1}};
    W_1_mat.gaussian_random(0, 0.1);
    julie::la::iMatrix<float> B_1_mat {0, julie::la::Shape{ch_1}};

    julie::la::iMatrix<float> W_21_mat {julie::la::Shape{ch_21, input_ch, 1, 1}};
    W_21_mat.gaussian_random(0, 0.1);
    julie::la::iMatrix<float> B_21_mat {0, julie::la::Shape{ch_21}};

    julie::la::iMatrix<float> W_22_mat {julie::la::Shape{ch_22, ch_21, 3, 3}};
    W_22_mat.gaussian_random(0, 0.1);
    julie::la::iMatrix<float> B_22_mat {0, julie::la::Shape{ch_22}};

    julie::la::iMatrix<float> W_31_mat {julie::la::Shape{ch_31, input_ch, 1, 1}};
    W_31_mat.gaussian_random(0, 0.1);
    julie::la::iMatrix<float> B_31_mat {0, julie::la::Shape{ch_31}};

    julie::la::iMatrix<float> W_32_mat {julie::la::Shape{ch_32, ch_31, 5, 5}};
    W_32_mat.gaussian_random(0, 0.1);
    julie::la::iMatrix<float> B_32_mat {0, julie::la::Shape{ch_32}};

    julie::la::iMatrix<float> W_4_mat {julie::la::Shape{ch_4, input_ch, 1, 1}};
    W_4_mat.gaussian_random(0, 0.1);
    julie::la::iMatrix<float> B_4_mat {0, julie::la::Shape{ch_4}};

    /////////////////////////////////////////////////////////////////////////////
    auto W_1 = std::make_shared<julie::nn::var::Tensor<float>> (W_1_mat);
    W_1->trainable(true);
    auto B_1 = std::make_shared<julie::nn::var::Tensor<float>> (B_1_mat);
    B_1->trainable(true);

    auto conv_1 = std::make_shared<julie::nn::func::Conv2d> (0, 0, 1, 1);
    auto act_1 = std::make_shared<julie::nn::func::ReLU> ();

    the_model_graph.add_node(conv_1, {input, W_1, B_1});
    the_model_graph.add_node(act_1, {conv_1->get_output()});

    /////////////////////////////////////////////////////////////////////////////
    auto W_21 = std::make_shared<julie::nn::var::Tensor<float>> (W_21_mat);
    W_21->trainable(true);
    auto B_21 = std::make_shared<julie::nn::var::Tensor<float>> (B_21_mat);
    B_21->trainable(true);

    auto conv_21 = std::make_shared<julie::nn::func::Conv2d> (0, 0, 1, 1);
    auto act_21 = std::make_shared<julie::nn::func::ReLU> ();

    the_model_graph.add_node(conv_21, {input, W_21, B_21});
    the_model_graph.add_node(act_21, {conv_21->get_output()});

    /////////////////////////////////////////////////////////////////////////////
    auto W_22 = std::make_shared<julie::nn::var::Tensor<float>> (W_22_mat);
    W_22->trainable(true);
    auto B_22 = std::make_shared<julie::nn::var::Tensor<float>> (B_22_mat);
    B_22->trainable(true);

    auto conv_22 = std::make_shared<julie::nn::func::Conv2d> (1, 1, 1, 1);
    auto act_22 = std::make_shared<julie::nn::func::ReLU> ();

    the_model_graph.add_node(conv_22, {act_21->get_output(), W_22, B_22});
    the_model_graph.add_node(act_22, {conv_22->get_output()});

    ////////////////////////////////////////////////////////////////////////////
    auto W_31 = std::make_shared<julie::nn::var::Tensor<float>> (W_31_mat);
    W_31->trainable(true);
    auto B_31 = std::make_shared<julie::nn::var::Tensor<float>> (B_31_mat);
    B_31->trainable(true);

    auto conv_31 = std::make_shared<julie::nn::func::Conv2d> (0, 0, 1, 1);
    auto act_31 = std::make_shared<julie::nn::func::ReLU> ();

    the_model_graph.add_node(conv_31, {input, W_31, B_31});
    the_model_graph.add_node(act_31, {conv_31->get_output()});

    ///////////////////////////////////////////////////////////////////////////
    auto W_32 = std::make_shared<julie::nn::var::Tensor<float>> (W_32_mat);
    W_32->trainable(true);
    auto B_32 = std::make_shared<julie::nn::var::Tensor<float>> (B_32_mat);
    B_32->trainable(true);

    auto conv_32 = std::make_shared<julie::nn::func::Conv2d> (2, 2, 1, 1);
    auto act_32 = std::make_shared<julie::nn::func::ReLU> ();

    the_model_graph.add_node(conv_32, {act_31->get_output(), W_32, B_32});
    the_model_graph.add_node(act_32, {conv_32->get_output()});

    //////////////////////////////////////////////////////////////////////////
    auto W_4 = std::make_shared<julie::nn::var::Tensor<float>> (W_4_mat);
    W_4->trainable(true);
    auto B_4 = std::make_shared<julie::nn::var::Tensor<float>> (B_4_mat);
    B_4->trainable(true);

    auto max_pool = std::make_shared<julie::nn::func::MaxPool> (1, 1, 3, 3, 1, 1);
    auto conv_4 = std::make_shared<julie::nn::func::Conv2d> (0, 0, 1, 1);
    auto act_4 = std::make_shared<julie::nn::func::ReLU> ();
    
    the_model_graph.add_node(max_pool, {input});
    the_model_graph.add_node(conv_4, {max_pool->get_output(), W_4, B_4});
    the_model_graph.add_node(act_4, {conv_4->get_output()});

    ///////////////////////////////////////////////////////////////////////////
    auto concat_1 = std::make_shared<julie::nn::func::Concat> (1);
    the_model_graph.add_node(concat_1, {act_1->get_output(), act_22->get_output()});

    auto concat_2 = std::make_shared<julie::nn::func::Concat> (1);
    the_model_graph.add_node(concat_2, {act_32->get_output(), act_4->get_output()});

    auto concat_3 = std::make_shared<julie::nn::func::Concat> (1);
    the_model_graph.add_node(concat_3, {concat_1->get_output(), concat_2->get_output()});

    return std::dynamic_pointer_cast<julie::nn::var::Tensor<float>> (concat_3->get_output());
}

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


    julie::op::Graph the_model_graph;

    // Construct the first conv layer
    /////////////////////////////////////////////////////////////////////////////////////
    auto x = std::make_shared<julie::nn::var::Tensor<float>> ();
    // Initialize weights
    julie::la::iMatrix<float> W_1_mat {julie::la::Shape{16, 1, 5, 5}};
    W_1_mat.gaussian_random(0, 0.1);
    julie::la::iMatrix<float> B_1_mat {0, julie::la::Shape{16}};

    auto W_1 = std::make_shared<julie::nn::var::Tensor<float>> (W_1_mat);
    W_1->trainable(true);
    auto B_1 = std::make_shared<julie::nn::var::Tensor<float>> (B_1_mat);
    B_1->trainable(true);

    auto conv_1 = std::make_shared<julie::nn::func::Conv2d> (2, 2, 1, 1);
    auto act_1 = std::make_shared<julie::nn::func::ReLU> ();

    the_model_graph.add_node(conv_1, {x, W_1, B_1});
    the_model_graph.add_node(act_1, {conv_1->get_output()});

    auto maxpool_0 = std::make_shared<julie::nn::func::MaxPool> (0, 0, 2, 2, 2, 2);
    the_model_graph.add_node(maxpool_0, {act_1->get_output()});

    auto inception_out = build_inception(the_model_graph, maxpool_0->get_output(), 16, 8, 4, 8, 4, 8, 8);

    inception_out = build_inception(the_model_graph, inception_out, 32, 16, 6, 18, 8, 16, 20);
    
    auto maxpool_1 = std::make_shared<julie::nn::func::MaxPool> (0, 0, 2, 2, 2, 2);
    the_model_graph.add_node(maxpool_1, {inception_out});

    inception_out = build_inception(the_model_graph, maxpool_1->get_output(), 70, 16, 8, 24, 8, 16, 32);

    auto maxpool_2 = std::make_shared<julie::nn::func::MaxPool> (0, 0, 3, 3, 2, 2);
    the_model_graph.add_node(maxpool_2, {inception_out});

    /////////////////////////////////////////////////////////////////////////////////

    julie::la::iMatrix<float> W_2_mat {julie::la::Shape{88, 88, 3, 3}};
    W_2_mat.gaussian_random(0, 0.1);
    julie::la::iMatrix<float> B_2_mat {0, julie::la::Shape{88}};

    auto W_2 = std::make_shared<julie::nn::var::Tensor<float>> (W_2_mat);
    W_2->trainable(true);
    auto B_2 = std::make_shared<julie::nn::var::Tensor<float>> (B_2_mat);
    B_2->trainable(true);

    auto conv_2 = std::make_shared<julie::nn::func::Conv2d> (1, 1, 1, 1);
    auto act_2 = std::make_shared<julie::nn::func::ReLU> ();

    the_model_graph.add_node(conv_2, {maxpool_2->get_output(), W_2, B_2});
    the_model_graph.add_node(act_2, {conv_2->get_output()});

    auto maxpool_3 = std::make_shared<julie::nn::func::MaxPool> (0, 0, 3, 3, 3, 3);
    the_model_graph.add_node(maxpool_3, {act_2->get_output()});

///////////////////////////////////////////////////////////////////////////////////////////

    julie::la::iMatrix<float> W_3_mat {julie::la::Shape{10, 88, 1, 1}};
    W_3_mat.gaussian_random(0, 0.1);
    julie::la::iMatrix<float> B_3_mat {0, julie::la::Shape{10}};

    auto W_3 = std::make_shared<julie::nn::var::Tensor<float>> (W_3_mat);
    W_3->trainable(true);
    auto B_3 = std::make_shared<julie::nn::var::Tensor<float>> (B_3_mat);
    B_3->trainable(true);

    auto conv_3 = std::make_shared<julie::nn::func::Conv2d> (0, 0, 1, 1);
    auto act_3 = std::make_shared<julie::nn::func::SoftMax> (1);

    the_model_graph.add_node(conv_3, {maxpool_3->get_output(), W_3, B_3});
    the_model_graph.add_node(act_3, {conv_3->get_output()});

///////////////////////////////////////////////////////////////////////////////////////////
    
    auto target = std::make_shared<julie::nn::var::Tensor<float>> ();
    auto loss_func = std::make_shared<julie::nn::func::SoftMax_CrossEntropy> (1);

    the_model_graph.add_node(loss_func, {conv_3->get_output(), target});
    
    auto act_1_output = std::dynamic_pointer_cast<julie::nn::var::Tensor<float>> (act_1->get_output());
    auto act_2_output = std::dynamic_pointer_cast<julie::nn::var::Tensor<float>> (act_2->get_output());
    auto act_3_output = std::dynamic_pointer_cast<julie::nn::var::Tensor<float>> (act_3->get_output());
    auto loss_output = std::dynamic_pointer_cast<julie::nn::var::Tensor<float>> (loss_func->get_output());

    the_model_graph.add_input(x);
    the_model_graph.set_device(mat_type);

    std::cout << the_model_graph.to_string() << std::endl;
    
    ///////////////////////////////////////////////////////////////////////////////////
    // ----------------------------------------------------------------- //
    //                             Training                              //
    // ----------------------------------------------------------------- //

    // Define the optimizer
    julie::nn::opt::SGD optimizer {the_model_graph, 0.001, 0.5};

    // Define random generators
    std::uniform_int_distribution<lint> train_distribution{ 0, static_cast<lint>(train_set.size() - 1) };
    std::uniform_int_distribution<lint> val_distribution{ 0, static_cast<lint>(val_set.size() - 1) };

    lint train_batch_size = 100;
    lint val_batch_size = 100;

    std::default_random_engine rand_engine;

    for (lint itr = 0; itr < 10000; ++itr)
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
        the_model_graph.forward(act_3->get_output());
        
        the_model_graph.backward();

        auto pred_batch = act_3_output->val()->argmax(1);
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
                
                the_model_graph.forward(act_3->get_output());

                auto pred_batch = act_3_output->val()->argmax(1);
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
            std::cout << "Train Itr: " << itr << " ========== Test Accuracy: " << right / val_batch_size / 100 << std::endl;
        }
    }

}
}  // namespace demo