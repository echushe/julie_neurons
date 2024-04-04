#pragma once

#include "Conv2dOneDNN.hpp"
#include "relu.hpp"
#include "sigmoid.hpp"
#include "softmax.hpp"

#include "Dataset.hpp"
#include "Mnist.hpp"

namespace test
{
    void onednn_simple_conv()
    {
        std::cout << "====================== simple_conv =====================" << std::endl;
        julie::la::iMatrix<int> input{
            {
                { 1,  1,  1,  1},
                { 1,  1,  1,  1},
                { 1,  1,  1,  1},
                { 1,  1,  1,  1},
                { 1,  1,  1,  1}
            }};

        input.left_extend_shape().left_extend_shape();

        julie::la::iMatrix<int> weights{
            {
                { 1,  1,  1},
                { 1,  1,  1},
                { 1,  1,  1},
            }};

        weights.left_extend_shape().left_extend_shape();

        julie::la::iMatrix<int> bias{0, julie::la::Shape{1}};

        julie::la::Conv2dOneDNN<int> conv2d {3, 2, 1, 1};

        julie::la::iMatrix<int> output;
        conv2d.forward(output, input, weights, bias);

        std::cout << output << std::endl;

        test::ASSERT(output == julie::la::iMatrix<int>{
            {
                {0, 0, 0, 0, 0, 0},
                {1, 2, 3, 3, 2, 1},
                {2, 4, 6, 6, 4, 2},
                {3, 6, 9, 9, 6, 3},
                {3, 6, 9, 9, 6, 3},
                {3, 6, 9, 9, 6, 3},
                {2, 4, 6, 6, 4, 2},
                {1, 2, 3, 3, 2, 1},
                {0, 0, 0, 0, 0, 0}
            }
        }.left_extend_shape().left_extend_shape());

    }

    void onednn_test_stride()
    {
        std::cout << "====================== test_stride =====================" << std::endl;
        julie::la::iMatrix<int> input{
            {
                { 1,  1,  1,  1},
                { 1,  1,  1,  1},
                { 1,  1,  1,  1},
                { 1,  1,  1,  1},
                { 1,  1,  1,  1}
            }};

        input.left_extend_shape().left_extend_shape();

        julie::la::iMatrix<int> weights{
            {
                { 1,  1,  1,  1,  1},
                { 1,  1,  1,  1,  1},
                { 1,  1,  1,  1,  1},
                { 1,  1,  1,  1,  1},
                { 1,  1,  1,  1,  1}
            }};

        weights.left_extend_shape().left_extend_shape();

        julie::la::iMatrix<int> bias{0, julie::la::Shape{1}};

        julie::la::Conv2dOneDNN<int> conv2d {2, 2, 2, 1};

        julie::la::iMatrix<int> output;
        conv2d.forward(output, input, weights, bias);

        std::cout << output << std::endl;

        test::ASSERT(output == julie::la::iMatrix<int>{
            {
                { 9, 12, 12,  9},
                {15, 20, 20, 15},
                { 9, 12, 12,  9}
            }
        }.left_extend_shape().left_extend_shape());
    }

    void onednn_test_input_ch()
    {
        std::cout << "====================== test_input_ch =====================" << std::endl;
        julie::la::iMatrix<int> input_ch1{
            {
                { 1,  1,  1,  1},
                { 1,  1,  1,  1},
                { 1,  1,  1,  1},
                { 1,  1,  1,  1},
                { 1,  1,  1,  1}
            }};

        julie::la::iMatrix<int> input_ch2{
            {
                { 2,  2,  2,  2},
                { 2,  2,  2,  2},
                { 2,  2,  2,  2},
                { 2,  2,  2,  2},
                { 2,  2,  2,  2},
            }};

        julie::la::iMatrix<int> input {{input_ch1, input_ch2}};
        input.left_extend_shape();

        julie::la::iMatrix<int> weights_ch1{
            {
                { 1,  1,  1},
                { 1,  1,  1},
                { 1,  1,  1},
            }};

        julie::la::iMatrix<int> weights_ch2{
            {
                { 2,  2,  2},
                { 2,  2,  2},
                { 2,  2,  2},
            }};

        julie::la::iMatrix<int> weights {{weights_ch1, weights_ch2}};
        weights.left_extend_shape();

        julie::la::iMatrix<int> bias{0, julie::la::Shape{1}};

        julie::la::Conv2dOneDNN<int> conv2d {1, 1, 1, 1};

        julie::la::iMatrix<int> output;
        conv2d.forward(output, input, weights, bias);

        std::cout << output << std::endl;

        test::ASSERT(output == julie::la::iMatrix<int>{
            {
                {20, 30, 30, 20},
                {30, 45, 45, 30},
                {30, 45, 45, 30},
                {30, 45, 45, 30},
                {20, 30, 30, 20}
            }
        }.left_extend_shape().left_extend_shape());

    }


    void onednn_test_output_ch()
    {
        std::cout << "====================== test_output_ch =====================" << std::endl;
        julie::la::iMatrix<int> input_ch1{
            {
                { 1,  1,  1,  1},
                { 1,  1,  1,  1},
                { 1,  1,  1,  1},
                { 1,  1,  1,  1},
                { 1,  1,  1,  1}
            }};

        julie::la::iMatrix<int> input_ch2{
            {
                { 2,  2,  2,  2},
                { 2,  2,  2,  2},
                { 2,  2,  2,  2},
                { 2,  2,  2,  2},
                { 2,  2,  2,  2},
            }};

        julie::la::iMatrix<int> input {{input_ch1, input_ch2}};
        input.left_extend_shape();

        julie::la::iMatrix<int> weights_1_ch1{
            {
                { 1,  1,  1},
                { 1,  1,  1},
                { 1,  1,  1},
            }};

        julie::la::iMatrix<int> weights_1_ch2{
            {
                { 2,  2,  2},
                { 2,  2,  2},
                { 2,  2,  2},
            }};

        julie::la::iMatrix<int> weights_1 {{weights_1_ch1, weights_1_ch2}};

        julie::la::iMatrix<int> weights_2_ch1{
            {
                { 3,  3,  3},
                { 3,  3,  3},
                { 3,  3,  3},
            }};

        julie::la::iMatrix<int> weights_2_ch2{
            {
                std::vector<int>{ 0,  0,  0},
                std::vector<int>{ 0,  0,  0},
                std::vector<int>{ 0,  0,  0},
            }};

        julie::la::iMatrix<int> weights_2 {{weights_2_ch1, weights_2_ch2}};

        julie::la::iMatrix<int> weights_3_ch1{
            {
                { -1,  -1,  -1},
                { -1,  -1,  -1},
                { -1,  -1,  -1}
            }};

        julie::la::iMatrix<int> weights_3_ch2{
            {
                { -2,  -2,  -2},
                { -2,  -2,  -2},
                { -2,  -2,  -2}
            }};

        julie::la::iMatrix<int> weights_3 {{weights_3_ch1, weights_3_ch2}};
        
        julie::la::iMatrix<int> weights {{weights_1, weights_2, weights_3}};

        julie::la::iMatrix<int> bias{0, julie::la::Shape{3}};

        julie::la::Conv2dOneDNN<int> conv2d {1, 1, 1, 1};

        julie::la::iMatrix<int> output;
        conv2d.forward(output, input, weights, bias);

        std::cout << output << std::endl;

        auto to_assert_ch1 = julie::la::iMatrix<int> {
            {
                {20, 30, 30, 20},
                {30, 45, 45, 30},
                {30, 45, 45, 30},
                {30, 45, 45, 30},
                {20, 30, 30, 20}
            }
        };

        auto to_assert_ch2 = julie::la::iMatrix<int> {
            {
                {12, 18, 18, 12},
                {18, 27, 27, 18},
                {18, 27, 27, 18},
                {18, 27, 27, 18},
                {12, 18, 18, 12}
            }
        };

        auto to_assert_ch3 = julie::la::iMatrix<int> {
            {
                {-20, -30, -30, -20},
                {-30, -45, -45, -30},
                {-30, -45, -45, -30},
                {-30, -45, -45, -30},
                {-20, -30, -30, -20}
            }
        };

        auto to_assert = julie::la::iMatrix<int> {{to_assert_ch1, to_assert_ch2, to_assert_ch3}};
        to_assert.left_extend_shape();

        test::ASSERT(output == to_assert);

    }


    void onednn_test_output_batch()
    {
        std::cout << "====================== test_output_batch =====================" << std::endl;
        julie::la::iMatrix<int> input_1_ch1{
            {
                { 1,  1,  1,  1},
                { 1,  1,  1,  1},
                { 1,  1,  1,  1},
                { 1,  1,  1,  1},
                { 1,  1,  1,  1}
            }};

        julie::la::iMatrix<int> input_1_ch2{
            {
                { 2,  2,  2,  2},
                { 2,  2,  2,  2},
                { 2,  2,  2,  2},
                { 2,  2,  2,  2},
                { 2,  2,  2,  2},
            }};

        julie::la::iMatrix<int> input_1 {{input_1_ch1, input_1_ch2}};

        julie::la::iMatrix<int> input_2_ch1{
            {
                { 1,  1,  1,  1},
                { 1,  1,  1,  1},
                { 1,  1,  1,  1},
                { 1,  1,  1,  1},
                { 1,  1,  1,  1}
            }};

        julie::la::iMatrix<int> input_2_ch2{
            {
                { -2, -2, -2, -2},
                { -2, -2, -2, -2},
                { -2, -2, -2, -2},
                { -2, -2, -2, -2},
                { -2, -2, -2, -2}
            }};

        julie::la::iMatrix<int> input_2 {{input_2_ch1, input_2_ch2}};

        auto input = julie::la::iMatrix<int> {{input_1, input_2}};

        julie::la::iMatrix<int> weights_1_ch1{
            {
                { 1,  1,  1},
                { 1,  1,  1},
                { 1,  1,  1},
            }};

        julie::la::iMatrix<int> weights_1_ch2{
            {
                { 2,  2,  2},
                { 2,  2,  2},
                { 2,  2,  2},
            }};

        julie::la::iMatrix<int> weights_1 {{weights_1_ch1, weights_1_ch2}};

        julie::la::iMatrix<int> weights_2_ch1{
            {
                { 3,  3,  3},
                { 3,  3,  3},
                { 3,  3,  3},
            }};

        julie::la::iMatrix<int> weights_2_ch2{
            {
                std::vector<int>{ 0,  0,  0},
                std::vector<int>{ 0,  0,  0},
                std::vector<int>{ 0,  0,  0},
            }};

        julie::la::iMatrix<int> weights_2 {{weights_2_ch1, weights_2_ch2}};

        julie::la::iMatrix<int> weights_3_ch1{
            {
                { -1,  -1,  -1},
                { -1,  -1,  -1},
                { -1,  -1,  -1}
            }};

        julie::la::iMatrix<int> weights_3_ch2{
            {
                { -2,  -2,  -2},
                { -2,  -2,  -2},
                { -2,  -2,  -2}
            }};

        julie::la::iMatrix<int> weights_3 {{weights_3_ch1, weights_3_ch2}};
        
        julie::la::iMatrix<int> weights {{weights_1, weights_2, weights_3}};

        julie::la::iMatrix<int> bias{0, julie::la::Shape{3}};

        julie::la::Conv2dOneDNN<int> conv2d {1, 1, 1, 1};

        julie::la::iMatrix<int> output;
        conv2d.forward(output, input, weights, bias);

        std::cout << output << std::endl;

        auto to_assert_1_ch1 = julie::la::iMatrix<int> {
            {
                {20, 30, 30, 20},
                {30, 45, 45, 30},
                {30, 45, 45, 30},
                {30, 45, 45, 30},
                {20, 30, 30, 20}
            }
        };

        auto to_assert_1_ch2 = julie::la::iMatrix<int> {
            {
                {12, 18, 18, 12},
                {18, 27, 27, 18},
                {18, 27, 27, 18},
                {18, 27, 27, 18},
                {12, 18, 18, 12}
            }
        };

        auto to_assert_1_ch3 = julie::la::iMatrix<int> {
            {
                {-20, -30, -30, -20},
                {-30, -45, -45, -30},
                {-30, -45, -45, -30},
                {-30, -45, -45, -30},
                {-20, -30, -30, -20}
            }
        };

        auto to_assert_1 = julie::la::iMatrix<int> {{to_assert_1_ch1, to_assert_1_ch2, to_assert_1_ch3}};

        auto to_assert_2_ch1 = julie::la::iMatrix<int> {
            {
                {-12, -18, -18, -12},
                {-18, -27, -27, -18},
                {-18, -27, -27, -18},
                {-18, -27, -27, -18},
                {-12, -18, -18, -12}
            }
        };

        auto to_assert_2_ch2 = julie::la::iMatrix<int> {
            {
                {12, 18, 18, 12},
                {18, 27, 27, 18},
                {18, 27, 27, 18},
                {18, 27, 27, 18},
                {12, 18, 18, 12}
            }
        };

        auto to_assert_2_ch3 = julie::la::iMatrix<int> {
            {
                {12, 18, 18, 12},
                {18, 27, 27, 18},
                {18, 27, 27, 18},
                {18, 27, 27, 18},
                {12, 18, 18, 12}
            }
        };

        auto to_assert_2 = julie::la::iMatrix<int> {{to_assert_2_ch1, to_assert_2_ch2, to_assert_2_ch3}};

        auto to_assert = julie::la::iMatrix<int> {{to_assert_1, to_assert_2}};

        test::ASSERT(output == to_assert);
    }


    void onednn_conv_with_bp()
    {
        std::cout << "========================== conv_with_bp =========================" << std::endl;
        julie::la::iMatrix<int> input_ch1{
            {
                { 1,  2,  3,  4},
                { 5,  6,  7,  8},
                { 9, 10, 11, 12},
                {13, 14, 15, 16},
                {17, 18, 19, 20}
            }};

        julie::la::iMatrix<int> input_ch2{
            {
                { -2,  -4,  -6,  -8},
                {-10, -12, -14, -16},
                {-18, -20, -22, -24},
                {-26, -28, -30, -32},
                {-34, -36, -38, -40}
            }};

        auto input = julie::la::iMatrix<int> {{input_ch1, input_ch2}};
        input.left_extend_shape();

        julie::la::iMatrix<int> weight1_ch1{
            {
                {-1, -2, -3},
                {-4, -5, -6},
                {-7, -8, -9}
            }};

        julie::la::iMatrix<int> weight1_ch2{
            {
                { 1,  1,  1},
                { 3,  3,  3},
                { 5,  5,  5}
            }};

        julie::la::iMatrix<int> weight2_ch1{
            {
                {-9,  9, -9},
                {-9,  9, -9},
                {-9,  9, -9}
            }};

        julie::la::iMatrix<int> weight2_ch2{
            {
                {-10, -10, -10},
                { 10,  10,  10},
                {-10, -10, -10}
            }};

        julie::la::iMatrix<int> weight3_ch1{
            {
                {-3,  3, -3},
                {-3,  3, -3},
                {-3,  3, -3}
            }};

        julie::la::iMatrix<int> weight3_ch2{
            {
                {-2, -2, -2},
                { 2,  2,  2},
                {-2, -2, -2}
            }};

        auto weight1 = julie::la::iMatrix<int> {{weight1_ch1, weight1_ch2}};
        auto weight2 = julie::la::iMatrix<int> {{weight2_ch1, weight2_ch2}};
        auto weight3 = julie::la::iMatrix<int> {{weight3_ch1, weight3_ch2}};

        // auto weights = weight1.left_extend_shape();
        auto weights = julie::la::iMatrix<int> {{weight1, weight2, weight3}};

        julie::la::iMatrix<int> bias{0, julie::la::Shape{3}};

        julie::la::Conv2dOneDNN<int> conv2d {1, 1, 1, 1};

        julie::la::iMatrix<int> output;
        conv2d.forward(output, input, weights, bias);

        std::cout << output << std::endl;
    }


    void onednn_conv_with_bp_padding()
    {
        std::cout << "========================== conv_with_bp_padding =========================" << std::endl;
        julie::la::iMatrix<int> input{
            {
                { 1,  2,  3,  4},
                { 5,  6,  7,  8},
                { 9, 10, 11, 12},
                {13, 14, 15, 16},
                {17, 18, 19, 20}
            }};

        input.left_extend_shape().left_extend_shape();

        julie::la::iMatrix<int> weight{
            {
                {-1, -2, -3},
                {-4, -5, -6},
                {-7, -8, -9}
            }};

        weight.left_extend_shape().left_extend_shape();

        julie::la::iMatrix<int> bias{0, julie::la::Shape{1}};

        julie::la::Conv2dOneDNN<int> conv2d {3, 2, 1, 1};


        julie::la::iMatrix<int> output;
        conv2d.forward(output, input, weight, bias);

        std::cout << output << std::endl;
    }



    void onednn_conv_with_bp_stride()
    {
        std::cout << "========================== conv_with_bp_stride =========================" << std::endl;
        julie::la::iMatrix<int> input{
            {
                { 1,  2,  3,  4},
                { 5,  6,  7,  8},
                { 9, 10, 11, 12},
                {13, 14, 15, 16},
                {17, 18, 19, 20}
            }};

        input.left_extend_shape().left_extend_shape();

        julie::la::iMatrix<int> weight{
            {
                {-1, -2, -3},
                {-4, -5, -6},
                {-7, -8, -9}
            }};

        weight.left_extend_shape().left_extend_shape();

        julie::la::iMatrix<int> bias{0, julie::la::Shape{1}};

        julie::la::Conv2dOneDNN<int> conv2d {2, 2, 3, 2};

        julie::la::iMatrix<int> output;
        conv2d.forward(output, input, weight, bias);

        std::cout << output << std::endl;
    }


    void onednn_conv_with_bp_filter_size()
    {
        std::cout << "========================== conv_with_bp_filter_size =========================" << std::endl;
        julie::la::iMatrix<int> input{
            std::vector<int>{
                 1,  2,  3,  4,
                 5,  6,  7,  8,
                 9, 10, 11, 12,
                13, 14, 15, 16,
                17, 18, 19, 20
            }, julie::la::Shape{5, 4}};

        input.left_extend_shape().left_extend_shape();

        julie::la::iMatrix<int> weight{
            std::vector<int>{
                -1, -2, -3,
                -4, -5, -6
            }, julie::la::Shape{2, 3}};

        weight.left_extend_shape().left_extend_shape();

        julie::la::iMatrix<int> bias{0, julie::la::Shape{1}};

        julie::la::Conv2dOneDNN<int> conv2d {1, 1, 1, 1};

        julie::la::iMatrix<int> output;
        conv2d.forward(output, input, weight, bias);

        std::cout << output << std::endl;
    }


    void onednn_conv_with_bp_batch_size()
    {
        std::cout << "========================== conv_with_bp_batch_size =========================" << std::endl;
        julie::la::iMatrix<int> input1{
            {
                { 1,  2,  3,  4},
                { 5,  6,  7,  8},
                { 9, 10, 11, 12},
                {13, 14, 15, 16},
                {17, 18, 19, 20}
            }};

        input1.left_extend_shape();

        julie::la::iMatrix<int> input2; julie::la::multiply(input2, input1, -1);
        julie::la::iMatrix<int> input3; julie::la::multiply(input3, input2, 2);

        auto input = julie::la::iMatrix<int> {{input1, input2, input3}};

        julie::la::iMatrix<int> weight{
            {
                {-1, -2, -3},
                {-4, -5, -6},
                {-7, -8, -9}
            }};

        weight.left_extend_shape().left_extend_shape();

        julie::la::iMatrix<int> bias{0, julie::la::Shape{1}};

        julie::la::Conv2dOneDNN<int> conv2d {1, 1, 1, 1};

        julie::la::iMatrix<int> output;
        conv2d.forward(output, input, weight, bias);

        std::cout << output << std::endl;
    }


    void onednn_conv_with_bp_normal_case()
    {
        std::cout << "========================== conv_with_bp_normal_case =========================" << std::endl;
        auto input = julie::la::iMatrix<int> {2, julie::la::Shape{8, 16, 64, 64}};
        // input.gaussian_random(0, 1.0);

        auto weights = julie::la::iMatrix<int> {3, julie::la::Shape{32, 16, 3, 3}};
        // weights.gaussian_random(0, 0.1);

        julie::la::iMatrix<int> bias{0, julie::la::Shape{32}};

        julie::la::Conv2dOneDNN<int> conv2d {0, 0, 2, 2};

        julie::la::iMatrix<int> output;
        conv2d.forward(output, input, weights, bias);

        // std::cout << output << std::endl;
    }


    void onednn_forward_conv2d_simple()
    {
        std::cout << "====================== forward_conv2d_simple =====================" << std::endl;
        std::cout << "---------------------- 1 ---------------------" << std::endl;
        julie::la::iMatrix<float> w1_mat {
            {
                {0.1, 0.1, 0.1},
                {0.1, 0.1, 0.1},
                {0.1, 0.1, 0.1}
            }
        };
        w1_mat.reshape(julie::la::Shape{1, 1, 3, 3});

        std::cout << "---------------------- 2 ---------------------" << std::endl;
        julie::la::iMatrix<float> b1_mat {
            {0.5},
            true
        };

        auto x = std::make_shared<julie::nn::var::Tensor<float>> ();
        auto w1 = std::make_shared<julie::nn::var::Tensor<float>> (w1_mat);
        auto b1 = std::make_shared<julie::nn::var::Tensor<float>> (b1_mat);

        std::cout << "---------------------- 3 ---------------------" << std::endl;

        // x->needs_grad(false);
        // w->needs_grad(false);

        auto conv_1 = std::make_shared<julie::nn::func::Conv2d> (1, 1, 1, 1);

        std::cout << "---------------------- 4 ---------------------" << std::endl;

        auto act_1 = std::make_shared<julie::nn::func::ReLU> ();

        std::cout << "---------------------- 5 ---------------------" << std::endl;

        julie::la::iMatrix<float> w2_mat {
            {
                {0.2, 0.2, 0.2},
                {0.2, 0.2, 0.2},
                {0.2, 0.2, 0.2},
                {0.2, 0.2, 0.2},
                {0.2, 0.2, 0.2},
                {0.2, 0.2, 0.2}
            }
        };
        w2_mat.reshape(julie::la::Shape{2, 1, 3, 3});

        julie::la::iMatrix<float> b2_mat {
            {-0.5, -0.5},
            true
        };

        std::cout << "---------------------- 6 ---------------------" << std::endl;

        auto w2 = std::make_shared<julie::nn::var::Tensor<float>> (w2_mat);
        auto b2 = std::make_shared<julie::nn::var::Tensor<float>> (b2_mat);

        auto conv_2 = std::make_shared<julie::nn::func::Conv2d> (0, 0, 2, 2);

        auto act_2 = std::make_shared<julie::nn::func::SoftMax> (1);
        
        auto conv_2_output = dynamic_cast<julie::nn::var::Tensor<float>*>(conv_2->get_output().get());
        auto act_2_output = dynamic_cast<julie::nn::var::Tensor<float>*>(act_2->get_output().get());

        std::cout << "---------------------- 6 ---------------------" << std::endl;

        julie::op::Graph the_model_graph;
        the_model_graph.add_node(conv_1, {x, w1, b1});
        the_model_graph.add_node(act_1, {conv_1->get_output()});
        the_model_graph.add_node(conv_2, {act_1->get_output(), w2, b2});
        the_model_graph.add_node(act_2, {conv_2->get_output()});

        std::cout << "---------------------- 7 ---------------------" << std::endl;

        dynamic_cast<julie::nn::var::Tensor<float>*>(x.get())->val(
            julie::la::iMatrix<float> {
            {
                { 1, 1, 1, 1},
                { 1, 1, 1, 1},
                { 1, 1, 1, 1},
                { 1, 1, 1, 1},
                { 1, 1, 1, 1}
            }}.reshape(julie::la::Shape{1, 1, 5, 4})
        );

        std::cout << "---------------------- 8 ---------------------" << std::endl;

        the_model_graph.forward(act_2->get_output());

        std::cout << "---------------------- 9 ---------------------" << std::endl;
        
        if(act_2_output->val())
        {
            std::cout << *(conv_2_output->val()) << std::endl;
            std::cout << *(act_2_output->val()) << std::endl;
        }

        test::ASSERT(
            *(act_2_output->val()) == 
            julie::la::iMatrix<float>{{0.5, 0.5, 0.5, 0.5}, true}.reshape({1, 2, 2, 1})
            );

    }

    void onednn_forward_backward_conv2d_simple()
    {
        std::cout << "====================== forward_backward_conv2d_simple =====================" << std::endl;
        std::cout << "---------------------- 1 ---------------------" << std::endl;
        julie::la::iMatrix<float> w1_mat {
            {
                {0.1, 0.1, 0.1},
                {0.1, 0.1, 0.1},
                {0.1, 0.1, 0.1}
            }
        };
        w1_mat.reshape(julie::la::Shape{1, 1, 3, 3});

        std::cout << "---------------------- 2 ---------------------" << std::endl;
        julie::la::iMatrix<float> b1_mat {
            {0.5},
            true
        };

        auto x = std::make_shared<julie::nn::var::Tensor<float>> ();
        auto w1 = std::make_shared<julie::nn::var::Tensor<float>> (w1_mat);
        w1->trainable(true);
        auto b1 = std::make_shared<julie::nn::var::Tensor<float>> (b1_mat);
        b1->trainable(true);

        std::cout << "---------------------- 3 ---------------------" << std::endl;

        // x->needs_grad(false);
        // w->needs_grad(false);

        auto conv_1 = std::make_shared<julie::nn::func::Conv2d> (1, 1, 1, 1);

        std::cout << "---------------------- 4 ---------------------" << std::endl;

        auto act_1 = std::make_shared<julie::nn::func::ReLU> ();

        std::cout << "---------------------- 5 ---------------------" << std::endl;

        julie::la::iMatrix<float> w2_mat {
            {
                {0.2, 0.2, 0.2},
                {0.2, 0.2, 0.2},
                {0.2, 0.2, 0.2},
                {0.2, 0.2, 0.2},
                {0.2, 0.2, 0.2},
                {0.2, 0.2, 0.2}
            }
        };
        w2_mat.reshape(julie::la::Shape{2, 1, 3, 3});

        julie::la::iMatrix<float> b2_mat {
            {-0.5, -0.5},
            true
        };

        std::cout << "---------------------- 6 ---------------------" << std::endl;

        auto w2 = std::make_shared<julie::nn::var::Tensor<float>> (w2_mat);
        w2->trainable(true);
        auto b2 = std::make_shared<julie::nn::var::Tensor<float>> (b2_mat);
        b2->trainable(true);

        auto conv_2 = std::make_shared<julie::nn::func::Conv2d> (0, 0, 2, 2);

        std::cout << "---------------------- 7 ---------------------" << std::endl;

        auto act_2 = std::make_shared<julie::nn::func::SoftMax> (1);

        auto target = std::make_shared<julie::nn::var::Tensor<float>> ();

        std::cout << "---------------------- 8 ---------------------" << std::endl;

        auto loss_func = std::make_shared<julie::nn::func::SoftMax_CrossEntropy> (1);

        std::cout << "---------------------- 8 ---------------------" << std::endl;
        
        auto act_2_output = dynamic_cast<julie::nn::var::Tensor<float>*>(act_2->get_output().get());
        auto loss_output = dynamic_cast<julie::nn::var::Tensor<float>*>(loss_func->get_output().get());
        auto conv_2_output = dynamic_cast<julie::nn::var::Tensor<float>*>(conv_2->get_output().get());

        std::cout << "---------------------- 9 ---------------------" << std::endl;

        julie::op::Graph the_model_graph;
        the_model_graph.add_node(conv_1, {x, w1, b1, });
        the_model_graph.add_node(act_1, {conv_1->get_output()});
        the_model_graph.add_node(conv_2, {act_1->get_output(), w2, b2});
        the_model_graph.add_node(act_2, {conv_2->get_output()});
        the_model_graph.add_node(loss_func, {conv_2->get_output(), target});

        //the_model_graph.pave_backward_route(w1);
        //the_model_graph.pave_backward_route(b1);
        //the_model_graph.pave_backward_route(w2);
        //the_model_graph.pave_backward_route(b2);

        std::cout << "---------------------- 10 ---------------------" << std::endl;

        dynamic_cast<julie::nn::var::Tensor<float>*>(x.get())->val(
            julie::la::iMatrix<float> {
            {
                { 1, 1, 1, 1},
                { 1, 1, 1, 1},
                { 1, 1, 1, 1},
                { 1, 1, 1, 1},
                { 1, 1, 1, 1}
            }}.reshape(julie::la::Shape{1, 1, 5, 4})
        );

        dynamic_cast<julie::nn::var::Tensor<float>*>(target.get())->val(
            julie::la::iMatrix<float> {{1, 0, 0, 1}, true}.reshape({1, 2, 2, 1})
        );

        std::cout << "---------------------- 11 ---------------------" << std::endl;

        the_model_graph.forward(act_2->get_output());
        the_model_graph.forward(loss_func->get_output());

        std::cout << "---------------------- 11.1 ---------------------" << std::endl;

        the_model_graph.backward(w1);

        std::cout << "---------------------- 12 ---------------------" << std::endl;
        
        std::cout << *(act_2_output->val()) << std::endl;
        test::ASSERT(
            *(act_2_output->val()) == 
            julie::la::iMatrix<float>{{0.5, 0.5, 0.5, 0.5}, true}.reshape({1, 2, 2, 1})
        );

        std::cout << "---------------------- 13 ---------------------" << std::endl;

        std::cout << *(conv_2_output->grad()) << std::endl;
        test::ASSERT(
            *(conv_2_output->grad()) == 
            julie::la::iMatrix<float>{{-0.5, 0.5, 0.5, -0.5}, true}.reshape({1, 2, 2, 1})
        );

        std::cout << *(loss_output->val()) << std::endl;
    }


    void onednn_forward_backward_cross_validate()
    {
        std::cout << "====================== onednn_forward_backward_cross_validate =====================" << std::endl;
        for (lint i = 0; i < 100; ++i)
        {
            lint batch = rand() % 10 + 1;
            lint ch_in = rand() % 20 + 1;
            lint h_in = rand() % 30 + 4;
            lint w_in = rand() % 30 + 4;

            lint pad_h = rand() % 5;
            lint pad_w = rand() % 5;
            lint s_h = rand() % 3 + 1;
            lint s_w = rand() % 3 + 1;

            lint w_h = std::min(lint(rand() % 7 + 1), h_in + pad_h * 2);
            lint w_w = std::min(lint(rand() % 7 + 1), w_in + pad_w * 2);
            lint ch_out = rand() % 40 + 10;

            lint h_out = (h_in + pad_h * 2 - w_h) / s_h + 1;
            lint w_out = (w_in + pad_w * 2 - w_w) / s_w + 1;

            julie::la::iMatrix<float> conv_input {julie::la::Shape{batch, ch_in, h_in, w_in}};
            conv_input.uniform_random(-10, 10);

            julie::la::iMatrix<float> weights {julie::la::Shape{ch_out, ch_in, w_h, w_w}};
            weights.uniform_random(-10, 10);

            julie::la::iMatrix<float> bias {julie::la::Shape{ch_out}};
            bias.uniform_random(-10, 10);

            julie::la::iMatrix<float> gradient_in {julie::la::Shape{batch, ch_out, h_out, w_out}};
            gradient_in.uniform_random(-10, 10);

            julie::la::Conv2d<float> conv1 {pad_h, pad_w, s_h, s_w};
            julie::la::Conv2dOneDNN<float> conv2 {pad_h, pad_w, s_h, s_w};

            julie::la::iMatrix<float> conv_out1;
            julie::la::iMatrix<float> gradient_out1;
            julie::la::iMatrix<float> gradient_w1;
            julie::la::iMatrix<float> gradient_b1;

            julie::la::iMatrix<float> conv_out2;
            julie::la::iMatrix<float> gradient_out2;
            julie::la::iMatrix<float> gradient_w2;
            julie::la::iMatrix<float> gradient_b2;

            std::cout << "run convolution:" << std::endl;
            std::cout << "input: " << conv_input.shape() << std::endl;
            std::cout << "weights: " << weights.shape() << std::endl;
            std::cout << "bias: " << bias.shape() << std::endl;
            std::cout << "pad: " << pad_h << " " << pad_w << std::endl;
            std::cout << "stride: " << s_h << " " << s_w << std::endl;

            conv1.forward(conv_out1, conv_input, weights, bias);
            conv1.backward(gradient_out1, gradient_w1, gradient_b1, gradient_in, conv_input.shape(), weights);

            conv2.forward(conv_out2, conv_input, weights, bias);
            conv2.backward(gradient_out2, gradient_w2, gradient_b2, gradient_in, conv_input.shape(), weights);

            std::cout << "assert conv output ... " << std::endl;

            julie::la::iMatrix<float> sub;
            julie::la::iMatrix<float> distance;
            julie::la::iMatrix<float> base;
            julie::la::subtract(sub, conv_out1, conv_out2);
            //std::cout << sub << std::endl;
            julie::la::abs(distance, sub);
            julie::la::abs(base, conv_out1);
            std::cout << distance.mean() / base.mean() << std::endl;
            test::ASSERT(distance.mean() / base.mean() < 1e-5);

            std::cout << "assert gradient of weights ... " << std::endl;

            julie::la::subtract(sub, gradient_w1, gradient_w2);
            //std::cout << gradient_w1 << std::endl;
            //std::cout << gradient_w2 << std::endl;
            julie::la::abs(distance, sub);
            julie::la::abs(base, gradient_w1);
            std::cout << distance.mean() / base.mean() << std::endl;
            test::ASSERT(distance.mean() / base.mean() < 1e-5);

            std::cout << "assert gradient of bias ... " << std::endl;

            julie::la::subtract(sub, gradient_b1, gradient_b2);
            //std::cout << gradient_b1 << std::endl;
            //std::cout << gradient_b2 << std::endl;
            julie::la::abs(distance, sub);
            julie::la::abs(base, gradient_b1);
            std::cout << distance.mean() / base.mean() << std::endl;
            test::ASSERT(distance.mean() / base.mean() < 1e-5);

            std::cout << "assert gradient of conv input ... " << std::endl;

            julie::la::subtract(sub, gradient_out1, gradient_out2);
            //std::cout << gradient_out1 << std::endl;
            //std::cout << gradient_out2 << std::endl;
            julie::la::abs(distance, sub);
            julie::la::abs(base, gradient_out1);
            std::cout << distance.mean() / base.mean() << std::endl;
            test::ASSERT(distance.mean() / base.mean() < 1e-5);
            std::cout << std::endl;
        }
    }


    void run_conv2d_onednn_cases()
    {
        onednn_simple_conv();
        onednn_test_stride();
        onednn_test_input_ch();
        onednn_test_output_ch();
        onednn_test_output_batch();
        onednn_conv_with_bp();
        onednn_conv_with_bp_padding();
        onednn_conv_with_bp_stride();
        onednn_conv_with_bp_filter_size();
        onednn_conv_with_bp_batch_size();
        onednn_conv_with_bp_normal_case();

        onednn_forward_conv2d_simple();
        onednn_forward_backward_conv2d_simple();

        onednn_forward_backward_cross_validate();
    }

}
