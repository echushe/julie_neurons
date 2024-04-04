#pragma once

#include "Matrix_CPU.hpp"
#include "Matrix_CPU_func.hpp"
#include "Matrix_CPU_func_adv.hpp"

#include "Matrix_CUDA.hpp"

#include "Matrix_CUDA_func.hpp"
#include "Matrix_CUDA_func_adv.hpp"
#include "test_util.hpp"

#include <iostream>
#include <vector>
#include <list>



namespace test
{

void test_cuda_pad_2d()
{
    std::cout << "====================== 2d cuda padding =====================" << std::endl;
    julie::la::cuda::Matrix_CUDA<int> mat_11 {
        {
            { 3,  3,  7,  9},
            { 3,  2,  5,  6},
            {-1, -3, -4, -5},
            { 0,  0, 11, 17},
            {22, 25, -9, 88}
        }};

    julie::la::cuda::Matrix_CUDA<int> mat_12 {
        {
            { 1,  2,  3,  4},
            { 5,  6,  7,  8},
            { 0,  1,  0,  1},
            {-1,  0, -1,  0},
            {11,-11,100,-20}
        }};

    julie::la::cuda::Matrix_CUDA<int> mat_13 {
        {
            { 1,  1,  1,  1},
            {-1, -1, -1, -1},
            { 1,  1,  1,  1},
            {-1, -1, -1, -1},
            { 8,  8,  9, -8}
        }};

    julie::la::cuda::Matrix_CUDA<int> mat_21 {
        {
            {11, 10,  9,  8},
            { 7,  6,  5,  4},
            { 3,  2,  1,  0},
            {-1, -2, -3, -4},
            {-5, -6, -7, -8}
        }};

    julie::la::cuda::Matrix_CUDA<int> mat_22 {
        {
            { 0,  1,  2,  3},
            { 4,  5,  6,  7},
            { 8,  9, 10, 11},
            {12, 13, 14, 15},
            {16, 17, 18, 19}
        }};

    julie::la::cuda::Matrix_CUDA<int> mat_23 {
        {
            { 0,  1,  2,  3},
            { 0,  1,  2,  3},
            { 0,  1,  2,  3},
            { 0,  1,  2,  3},
            { 0,  1,  2,  3}
        }};


    julie::la::cuda::Matrix_CUDA<int> image1 {std::vector<julie::la::cuda::Matrix_CUDA<int>>{mat_11, mat_12, mat_13}};
    julie::la::cuda::Matrix_CUDA<int> image2 {std::vector<julie::la::cuda::Matrix_CUDA<int>>{mat_21, mat_22, mat_23}};
    julie::la::cuda::Matrix_CUDA<int> batch {std::vector<julie::la::cuda::Matrix_CUDA<int>>{image1, image2}};

    julie::la::cuda::Matrix_CUDA<int> pmat_11 {
        {
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  3,  3,  7,  9,  0},
            { 0,  3,  2,  5,  6,  0},
            { 0, -1, -3, -4, -5,  0},
            { 0,  0,  0, 11, 17,  0},
            { 0, 22, 25, -9, 88,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0}
        }};

    julie::la::cuda::Matrix_CUDA<int> pmat_12 {
        {
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  1,  2,  3,  4,  0},
            { 0,  5,  6,  7,  8,  0},
            { 0,  0,  1,  0,  1,  0},
            { 0, -1,  0, -1,  0,  0},
            { 0, 11,-11,100,-20,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0}
        }};

    julie::la::cuda::Matrix_CUDA<int> pmat_13 {
        {
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  1,  1,  1,  1,  0},
            { 0, -1, -1, -1, -1,  0},
            { 0,  1,  1,  1,  1,  0},
            { 0, -1, -1, -1, -1,  0},
            { 0,  8,  8,  9, -8,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0}
        }};

    julie::la::cuda::Matrix_CUDA<int> pmat_21 {
        {
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0, 11, 10,  9,  8,  0},
            { 0,  7,  6,  5,  4,  0},
            { 0,  3,  2,  1,  0,  0},
            { 0, -1, -2, -3, -4,  0},
            { 0, -5, -6, -7, -8,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0}
        }};

    julie::la::cuda::Matrix_CUDA<int> pmat_22 {
        {
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  1,  2,  3,  0},
            { 0,  4,  5,  6,  7,  0},
            { 0,  8,  9, 10, 11,  0},
            { 0, 12, 13, 14, 15,  0},
            { 0, 16, 17, 18, 19,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0}
        }};

    julie::la::cuda::Matrix_CUDA<int> pmat_23 {
        {
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  1,  2,  3,  0},
            { 0,  0,  1,  2,  3,  0},
            { 0,  0,  1,  2,  3,  0},
            { 0,  0,  1,  2,  3,  0},
            { 0,  0,  1,  2,  3,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0}
        }};


    julie::la::cuda::Matrix_CUDA<int> pimage1 {std::vector<julie::la::cuda::Matrix_CUDA<int>>{pmat_11, pmat_12, pmat_13}};
    julie::la::cuda::Matrix_CUDA<int> pimage2 {std::vector<julie::la::cuda::Matrix_CUDA<int>>{pmat_21, pmat_22, pmat_23}};
    julie::la::cuda::Matrix_CUDA<int> pbatch {std::vector<julie::la::cuda::Matrix_CUDA<int>>{pimage1, pimage2}};

    julie::la::cuda::Matrix_CUDA<int> my_pbatch;
    julie::la::cuda::pad_2d(my_pbatch, batch, 2, 1);

    std::cout << batch << std::endl;
    std::cout << my_pbatch << std::endl;
    test::ASSERT(my_pbatch == pbatch);


    julie::la::cpu::Matrix_CPU<int> img_cpu {julie::la::Shape{96, 32, 128, 128}};
    img_cpu.uniform_random(-10, 10);
    julie::la::cuda::Matrix_CUDA<int> img_gpu = img_cpu.get_CUDA();
    test::ASSERT(img_gpu == img_cpu.get_CUDA());

    auto img_pad_cpu = julie::la::cpu::Matrix_CPU<int> 
        {julie::la::Shape{img_cpu.shape()[0], img_cpu.shape()[1], img_cpu.shape()[2] + 8, img_cpu.shape()[3] + 14}};

    auto img_pad_gpu = julie::la::cuda::Matrix_CUDA<int> 
        {julie::la::Shape{img_cpu.shape()[0], img_cpu.shape()[1], img_cpu.shape()[2] + 8, img_cpu.shape()[3] + 14}};

    julie::la::cuda::pad_2d(img_pad_gpu, img_gpu, 4, 7);
    julie::la::cpu::pad_2d(img_pad_cpu, img_cpu, 4, 7);

    julie::la::cpu::Matrix_CPU<int> tmp1, fused_cpu; img_pad_cpu.get_reduce_sum(tmp1, 0); tmp1.get_reduce_sum(fused_cpu, 0);
    julie::la::cuda::Matrix_CUDA<int> tmp2, fused_gpu; img_pad_gpu.get_reduce_sum(tmp2, 0); tmp2.get_reduce_sum(fused_gpu, 0);

    std::cout << fused_cpu << std::endl;
    std::cout << fused_gpu << std::endl;

    test::ASSERT(img_pad_gpu == img_pad_cpu.get_CUDA());

    lint time1 = get_time_in_milliseconds();
    for (int i = 0; i < 10; ++i)
    {
        std::cout << "gpu pad_2d: " << i << std::endl;
        julie::la::cuda::pad_2d(img_pad_gpu, img_gpu, 4, 7);
    }
    lint time2 = get_time_in_milliseconds();
    for (int i = 0; i < 10; ++i)
    {
        std::cout << "cpu pad_2d: " << i << std::endl;
        julie::la::cpu::pad_2d(img_pad_cpu, img_cpu, 4, 7);
    }
    lint time3 = get_time_in_milliseconds();

    std::cout << "GPU compute time: " << time2 - time1 << std::endl;
    std::cout << "CPU compute time: " << time3 - time2 << std::endl;


    for (int i = 0; i < 100; ++i)
    {
        int b = rand() % 50 + 1;
        int c = rand() % 50 + 1;
        int h = rand() % 50 + 1;
        int w = rand() % 50 + 1;
        int p_h = rand() % 50;
        int p_w = rand() % 50;

        julie::la::cpu::Matrix_CPU<int> img_cpu2 {julie::la::Shape{b, c, h, w}};
        img_cpu2.uniform_random(-100, 100);
        auto img_gpu2 = img_cpu2.get_CUDA();

        std::cout << "random mat test " << img_gpu2.shape() << " " << p_h << " " << p_w << std::endl;

        julie::la::cpu::Matrix_CPU<int> cpu_pad_2d_out;
        julie::la::cpu::pad_2d(cpu_pad_2d_out, img_cpu2, p_h, p_w);

        julie::la::cuda::Matrix_CUDA<int> cuda_pad_2d_out;
        julie::la::cuda::pad_2d(cuda_pad_2d_out, img_gpu2, p_h, p_w);
        test::ASSERT(cpu_pad_2d_out.get_CUDA() == cuda_pad_2d_out);
    }
}


void test_cuda_pad_2d_backward()
{
    std::cout << "====================== 2d cuda padding backward =====================" << std::endl;
    julie::la::cuda::Matrix_CUDA<int> mat_11 {
        {
            { 3,  3,  7,  9},
            { 3,  2,  5,  6},
            {-1, -3, -4, -5},
            { 0,  0, 11, 17},
            {22, 25, -9, 88}
        }};

    julie::la::cuda::Matrix_CUDA<int> mat_12 {
        {
            { 1,  2,  3,  4},
            { 5,  6,  7,  8},
            { 0,  1,  0,  1},
            {-1,  0, -1,  0},
            {11,-11,100,-20}
        }};

    julie::la::cuda::Matrix_CUDA<int> mat_13 {
        {
            { 1,  1,  1,  1},
            {-1, -1, -1, -1},
            { 1,  1,  1,  1},
            {-1, -1, -1, -1},
            { 8,  8,  9, -8}
        }};

    julie::la::cuda::Matrix_CUDA<int> mat_21 {
        {
            {11, 10,  9,  8},
            { 7,  6,  5,  4},
            { 3,  2,  1,  0},
            {-1, -2, -3, -4},
            {-5, -6, -7, -8}
        }};

    julie::la::cuda::Matrix_CUDA<int> mat_22 {
        {
            { 0,  1,  2,  3},
            { 4,  5,  6,  7},
            { 8,  9, 10, 11},
            {12, 13, 14, 15},
            {16, 17, 18, 19}
        }};

    julie::la::cuda::Matrix_CUDA<int> mat_23 {
        {
            { 0,  1,  2,  3},
            { 0,  1,  2,  3},
            { 0,  1,  2,  3},
            { 0,  1,  2,  3},
            { 0,  1,  2,  3}
        }};


    julie::la::cuda::Matrix_CUDA<int> image1 {std::vector<julie::la::cuda::Matrix_CUDA<int>>{mat_11, mat_12, mat_13}};
    julie::la::cuda::Matrix_CUDA<int> image2 {std::vector<julie::la::cuda::Matrix_CUDA<int>>{mat_21, mat_22, mat_23}};
    julie::la::cuda::Matrix_CUDA<int> batch {std::vector<julie::la::cuda::Matrix_CUDA<int>>{image1, image2}};

    julie::la::cuda::Matrix_CUDA<int> pmat_11 {
        {
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  3,  3,  7,  9,  0},
            { 0,  3,  2,  5,  6,  0},
            { 0, -1, -3, -4, -5,  0},
            { 0,  0,  0, 11, 17,  0},
            { 0, 22, 25, -9, 88,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0}
        }};

    julie::la::cuda::Matrix_CUDA<int> pmat_12 {
        {
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  1,  2,  3,  4,  0},
            { 0,  5,  6,  7,  8,  0},
            { 0,  0,  1,  0,  1,  0},
            { 0, -1,  0, -1,  0,  0},
            { 0, 11,-11,100,-20,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0}
        }};

    julie::la::cuda::Matrix_CUDA<int> pmat_13 {
        {
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  1,  1,  1,  1,  0},
            { 0, -1, -1, -1, -1,  0},
            { 0,  1,  1,  1,  1,  0},
            { 0, -1, -1, -1, -1,  0},
            { 0,  8,  8,  9, -8,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0}
        }};

    julie::la::cuda::Matrix_CUDA<int> pmat_21 {
        {
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0, 11, 10,  9,  8,  0},
            { 0,  7,  6,  5,  4,  0},
            { 0,  3,  2,  1,  0,  0},
            { 0, -1, -2, -3, -4,  0},
            { 0, -5, -6, -7, -8,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0}
        }};

    julie::la::cuda::Matrix_CUDA<int> pmat_22 {
        {
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  1,  2,  3,  0},
            { 0,  4,  5,  6,  7,  0},
            { 0,  8,  9, 10, 11,  0},
            { 0, 12, 13, 14, 15,  0},
            { 0, 16, 17, 18, 19,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0}
        }};

    julie::la::cuda::Matrix_CUDA<int> pmat_23 {
        {
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  1,  2,  3,  0},
            { 0,  0,  1,  2,  3,  0},
            { 0,  0,  1,  2,  3,  0},
            { 0,  0,  1,  2,  3,  0},
            { 0,  0,  1,  2,  3,  0},
            { 0,  0,  0,  0,  0,  0},
            { 0,  0,  0,  0,  0,  0}
        }};


    julie::la::cuda::Matrix_CUDA<int> pimage1 {std::vector<julie::la::cuda::Matrix_CUDA<int>>{pmat_11, pmat_12, pmat_13}};
    julie::la::cuda::Matrix_CUDA<int> pimage2 {std::vector<julie::la::cuda::Matrix_CUDA<int>>{pmat_21, pmat_22, pmat_23}};
    julie::la::cuda::Matrix_CUDA<int> pbatch {std::vector<julie::la::cuda::Matrix_CUDA<int>>{pimage1, pimage2}};

    julie::la::cuda::Matrix_CUDA<int> my_batch;
    julie::la::cuda::pad_2d_backward(my_batch, pbatch, 2, 1);

    std::cout << batch << std::endl;
    std::cout << my_batch << std::endl;
    test::ASSERT(my_batch == batch);

    for (int i = 0; i < 100; ++i)
    {
        int b = rand() % 50 + 1;
        int c = rand() % 50 + 1;
        int h = rand() % 50 + 1;
        int w = rand() % 50 + 1;
        int p_h = std::min(rand() % 50, (h - 1) / 2);
        int p_w = std::min(rand() % 50, (w - 1) / 2);

        julie::la::cpu::Matrix_CPU<int> img_cpu2 {julie::la::Shape{b, c, h, w}};
        img_cpu2.uniform_random(-100, 100);
        auto img_gpu2 = img_cpu2.get_CUDA();

        std::cout << "random mat test " << img_gpu2.shape() << " " << p_h << " " << p_w << std::endl;

        julie::la::cpu::Matrix_CPU<int> out_cpu {julie::la::Shape{b, c, h - p_h * 2, w - p_w * 2}};
        julie::la::cuda::Matrix_CUDA<int> out_gpu {julie::la::Shape{b, c, h - p_h * 2, w - p_w * 2}};

        julie::la::cpu::pad_2d_backward(out_cpu, img_cpu2, p_h, p_w);
        julie::la::cuda::pad_2d_backward(out_gpu, img_gpu2, p_h, p_w);
        test::ASSERT(out_cpu.get_CUDA() == out_gpu);
    }

}

void img2row_test_filter_size_gpu()
{
    std::cout << "====================== img2row_test_filter_size_gpu =====================" << std::endl;

    julie::la::cuda::Matrix_CUDA<int> input_ch1 {
        {
            {0,  1,  2,  3},
            {4,  5,  6,  7},
            {8,  9,  10, 11}
        }
    };

    julie::la::cuda::Matrix_CUDA<int> input_ch2 {
        {
            { 0,  2,  4,  6},
            { 8, 10, 12, 14},
            {16, 18, 24, 28}
        }
    };

    julie::la::cuda::Matrix_CUDA<int> input_ch3 {
        {
            { 0, -1, -2, -3},
            {-4, -5, -6, -7},
            {-8, -9,-10,-11}
        }
    };

    julie::la::cuda::Matrix_CUDA<int> input {{input_ch1, input_ch2, input_ch3}};
    input.left_extend_shape();

    julie::la::cpu::Matrix_CPU<int> input_cpu {input};

    julie::la::cuda::Matrix_CUDA<int> img2row_output;
    julie::la::cuda::img2row_2d(img2row_output, input, 1, 1, 2, 3);
    julie::la::cpu::Matrix_CPU<int> img2row_output_cpu;
    julie::la::cpu::img2row_2d(img2row_output_cpu, input_cpu, 1, 1, 2, 3);
    std::cout << img2row_output << std::endl;

    test::ASSERT(img2row_output == img2row_output_cpu.get_CUDA());


    julie::la::cuda::Matrix_CUDA<int> img2col_output;
    julie::la::cuda::img2col_2d(img2col_output, input, 1, 1, 2, 3);
    julie::la::cpu::Matrix_CPU<int> img2col_output_cpu;
    julie::la::cpu::img2col_2d(img2col_output_cpu, input_cpu, 1, 1, 2, 3);
    std::cout << img2col_output << std::endl;

    test::ASSERT(img2col_output == img2col_output_cpu.get_CUDA());


    julie::la::cuda::Matrix_CUDA<int> img2row_backward;
    julie::la::cuda::img2row_2d_backward(img2row_backward, input.shape(), img2row_output, 1, 1, 2, 3);
    julie::la::cpu::Matrix_CPU<int> img2row_backward_cpu;
    julie::la::cpu::img2row_2d_backward(img2row_backward_cpu, input_cpu.shape(), img2row_output_cpu, 1, 1, 2, 3);
    std::cout << img2row_backward << std::endl;

    test::ASSERT(img2row_backward == img2row_backward_cpu.get_CUDA());
    
    

    julie::la::cuda::Matrix_CUDA<int> img2col_backward;
    julie::la::cuda::img2col_2d_backward(img2col_backward, input.shape(), img2col_output, 1, 1, 2, 3);
    julie::la::cpu::Matrix_CPU<int> img2col_backward_cpu;
    julie::la::cpu::img2col_2d_backward(img2col_backward_cpu, input_cpu.shape(), img2col_output_cpu, 1, 1, 2, 3);
    std::cout << img2col_backward << std::endl;

    test::ASSERT(img2col_backward == img2col_backward_cpu.get_CUDA());

    test::ASSERT(img2row_backward == img2col_backward);

}

void img2row_test_filter_stride_gpu()
{
    std::cout << "====================== img2row_test_filter_stride_gpu =====================" << std::endl;

    julie::la::cuda::Matrix_CUDA<int> input_ch1 {
        {
            {0,  1,  2,  3},
            {4,  5,  6,  7},
            {8,  9,  10, 11}
        }
    };

    julie::la::cuda::Matrix_CUDA<int> input_ch2 {
        {
            { 0,  2,  4,  6},
            { 8, 10, 12, 14},
            {16, 18, 24, 28}
        }
    };

    julie::la::cuda::Matrix_CUDA<int> input_ch3 {
        {
            { 0, -1, -2, -3},
            {-4, -5, -6, -7},
            {-8, -9,-10,-11}
        }
    };

    julie::la::cuda::Matrix_CUDA<int> input {{input_ch1, input_ch2, input_ch3}};
    input.left_extend_shape();
    julie::la::cpu::Matrix_CPU<int> input_cpu {input};

    julie::la::cuda::Matrix_CUDA<int> img2row_output;
    julie::la::cuda::img2row_2d(img2row_output, input, 1, 2, 2, 2);
    julie::la::cpu::Matrix_CPU<int> img2row_output_cpu;
    julie::la::cpu::img2row_2d(img2row_output_cpu, input_cpu, 1, 2, 2, 2);
    std::cout << img2row_output << std::endl;
    std::cout << img2row_output_cpu << std::endl;

    test::ASSERT(img2row_output == img2row_output_cpu.get_CUDA());


    julie::la::cuda::Matrix_CUDA<int> img2col_output;
    julie::la::cuda::img2col_2d(img2col_output, input, 1, 2, 2, 2);
    julie::la::cpu::Matrix_CPU<int> img2col_output_cpu;
    julie::la::cpu::img2col_2d(img2col_output_cpu, input_cpu, 1, 2, 2, 2);
    std::cout << img2col_output << std::endl;

    test::ASSERT(img2col_output == img2col_output_cpu.get_CUDA());


    julie::la::cuda::Matrix_CUDA<int> img2row_backward;
    julie::la::cuda::img2row_2d_backward(img2row_backward, input.shape(), img2row_output, 1, 2, 2, 2);
    julie::la::cpu::Matrix_CPU<int> img2row_backward_cpu;
    julie::la::cpu::img2row_2d_backward(img2row_backward_cpu, input_cpu.shape(), img2row_output_cpu, 1, 2, 2, 2);
    std::cout << img2row_backward << std::endl;

    test::ASSERT(img2row_backward == img2row_backward_cpu.get_CUDA());


    julie::la::cuda::Matrix_CUDA<int> img2col_backward;
    julie::la::cuda::img2col_2d_backward(img2col_backward, input.shape(), img2col_output, 1, 2, 2, 2);
    julie::la::cpu::Matrix_CPU<int> img2col_backward_cpu;
    julie::la::cpu::img2col_2d_backward(img2col_backward_cpu, input_cpu.shape(), img2col_output_cpu, 1, 2, 2, 2);
    std::cout << img2col_backward << std::endl;

    test::ASSERT(img2col_backward == img2col_backward_cpu.get_CUDA());

    test::ASSERT(img2row_backward == img2col_backward);
}

void img2row_test_batch_gpu()
{
    std::cout << "====================== img2row_test_batch_gpu =====================" << std::endl;

    julie::la::cuda::Matrix_CUDA<int> input_1_ch1 {
        {
            {0,  1,  2,  3},
            {4,  5,  6,  7},
            {8,  9,  10, 11}
        }
    };

    julie::la::cuda::Matrix_CUDA<int> input_1_ch2 {
        {
            { 0,  2,  4,  6},
            { 8, 10, 12, 14},
            {16, 18, 24, 28}
        }
    };

    julie::la::cuda::Matrix_CUDA<int> input_1_ch3 {
        {
            { 0, -1, -2, -3},
            {-4, -5, -6, -7},
            {-8, -9,-10,-11}
        }
    };

    julie::la::cuda::Matrix_CUDA<int> input_2_ch1 {
        {
            {0,  0,  0,  0},
            {1,  1,  1,  1},
            {2,  2,  2,  2}
        }
    };

    julie::la::cuda::Matrix_CUDA<int> input_2_ch2 {
        {
            { 0,  2,  4,  6},
            { 0,  2,  4,  6},
            { 0,  2,  4,  6}
        }
    };

    julie::la::cuda::Matrix_CUDA<int> input_2_ch3 {
        {
            { 0,  0,  0,  0},
            {-1, -1, -1, -1},
            {-2, -2, -2, -2}
        }
    };

    julie::la::cuda::Matrix_CUDA<int> input_1 {{input_1_ch1, input_1_ch2, input_1_ch3}};
    julie::la::cuda::Matrix_CUDA<int> input_2 {{input_2_ch1, input_2_ch2, input_2_ch3}};

    auto input = julie::la::cuda::Matrix_CUDA<int> {{input_1, input_2}};
    auto input_cpu = julie::la::cpu::Matrix_CPU<int> {input};

    julie::la::cuda::Matrix_CUDA<int> img2row_output;
    julie::la::cuda::img2row_2d(img2row_output, input, 1, 1, 3, 3);
    julie::la::cpu::Matrix_CPU<int> img2row_output_cpu;
    julie::la::cpu::img2row_2d(img2row_output_cpu, input_cpu, 1, 1, 3, 3);
    std::cout << img2row_output << std::endl;
    test::ASSERT(img2row_output == img2row_output_cpu.get_CUDA());

    julie::la::cuda::Matrix_CUDA<int> img2col_output;
    julie::la::cuda::img2col_2d(img2col_output, input, 1, 1, 3, 3);
    julie::la::cpu::Matrix_CPU<int> img2col_output_cpu;
    julie::la::cpu::img2col_2d(img2col_output_cpu, input_cpu, 1, 1, 3, 3);
    std::cout << img2col_output << std::endl;
    test::ASSERT(img2col_output == img2col_output_cpu.get_CUDA());

    julie::la::cuda::Matrix_CUDA<int> img2row_backward;
    julie::la::cuda::img2row_2d_backward(img2row_backward, input.shape(), img2row_output, 1, 1, 3, 3);
    julie::la::cpu::Matrix_CPU<int> img2row_backward_cpu;
    julie::la::cpu::img2row_2d_backward(img2row_backward_cpu, input_cpu.shape(), img2row_output_cpu, 1, 1, 3, 3);
    std::cout << img2row_backward << std::endl;
    test::ASSERT(img2row_backward_cpu.get_CUDA() == img2row_backward);

    julie::la::cuda::Matrix_CUDA<int> img2col_backward;
    julie::la::cuda::img2col_2d_backward(img2col_backward, input.shape(), img2col_output, 1, 1, 3, 3);
    julie::la::cpu::Matrix_CPU<int> img2col_backward_cpu;
    julie::la::cpu::img2col_2d_backward(img2col_backward_cpu, input_cpu.shape(), img2col_output_cpu, 1, 1, 3, 3);
    std::cout << img2col_backward << std::endl;
    test::ASSERT(img2col_backward_cpu.get_CUDA() == img2col_backward);

    test::ASSERT(img2row_backward == img2col_backward);
}

void img2row_test_random_gpu()
{
    std::cout << "====================== img2row_test_random_gpu =====================" << std::endl;

    for (int i = 0; i < 100; ++i)
    {
        int b = rand() % 20 + 1;
        int c = rand() % 20 + 1;
        int h = rand() % 50 + 1;
        int w = rand() % 50 + 1;
        int w_h = std::min(rand() % 10 + 1, h);
        int w_w = std::min(rand() % 10 + 1, w);
        int s_h = std::min(rand() % 10 + 1, h);
        int s_w = std::min(rand() % 10 + 1, w);

        std::cout << "----------------- random case " << i << "-------------\n";
        std::cout << "batch: " << b << " ch: " << c << " h: " << h << " w: " << w << "\n";
        std::cout << "Conv filter h: " << w_h << " w: " << w_w << " Stride h: " << s_h << " w: " << s_w << std::endl;

        julie::la::cuda::Matrix_CUDA<int> input {julie::la::Shape{b, c, h, w}};
        input.uniform_random(-30, 30);
        auto input_cpu = julie::la::cpu::Matrix_CPU<int> {input};

        //std::cout << input << std::endl;

        julie::la::cuda::Matrix_CUDA<int> img2row_output;
        julie::la::cuda::img2row_2d(img2row_output, input, s_h, s_w, w_h, w_w);
        julie::la::cpu::Matrix_CPU<int> img2row_output_cpu;
        julie::la::cpu::img2row_2d(img2row_output_cpu, input_cpu, s_h, s_w, w_h, w_w);
        //std::cout << img2row_output << std::endl;
        test::ASSERT(img2row_output == img2row_output_cpu.get_CUDA());

        julie::la::cuda::Matrix_CUDA<int> img2col_output;
        julie::la::cuda::img2col_2d(img2col_output, input, s_h, s_w, w_h, w_w);
        julie::la::cpu::Matrix_CPU<int> img2col_output_cpu;
        julie::la::cpu::img2col_2d(img2col_output_cpu, input_cpu, s_h, s_w, w_h, w_w);
        //std::cout << img2col_output << std::endl;
        test::ASSERT(img2col_output == img2col_output_cpu.get_CUDA());

        julie::la::cuda::Matrix_CUDA<int> img2row_backward;
        julie::la::cuda::img2row_2d_backward(img2row_backward, input.shape(), img2row_output, s_h, s_w, w_h, w_w);
        julie::la::cpu::Matrix_CPU<int> img2row_backward_cpu;
        julie::la::cpu::img2row_2d_backward(img2row_backward_cpu, input_cpu.shape(), img2row_output_cpu, s_h, s_w, w_h, w_w);
        //std::cout << img2row_backward << std::endl;
        //std::cout << img2row_backward_cpu << std::endl;
        test::ASSERT(img2row_backward_cpu.get_CUDA() == img2row_backward);

        julie::la::cuda::Matrix_CUDA<int> img2col_backward;
        julie::la::cuda::img2col_2d_backward(img2col_backward, input.shape(), img2col_output, s_h, s_w, w_h, w_w);
        julie::la::cpu::Matrix_CPU<int> img2col_backward_cpu;
        julie::la::cpu::img2col_2d_backward(img2col_backward_cpu, input_cpu.shape(), img2col_output_cpu, s_h, s_w, w_h, w_w);
        //std::cout << img2col_backward << std::endl;
        test::ASSERT(img2col_backward_cpu.get_CUDA() == img2col_backward);
        test::ASSERT(img2row_backward_cpu == img2col_backward_cpu);
    }
}

void img2row_test_speed_gpu()
{
    std::cout << "====================== img2row_test_speed_gpu =====================" << std::endl;

    int b = 32;
    int c = 3;
    int h = 256;
    int w = 256;
    int w_h = 5;
    int w_w = 5;
    int s_h = 2;
    int s_w = 2;

    julie::la::cuda::Matrix_CUDA<int> img2row_output;
    julie::la::cuda::Matrix_CUDA<int> img2row_backward;

    julie::la::cpu::Matrix_CPU<int> img2row_output_cpu;
    julie::la::cpu::Matrix_CPU<int> img2row_backward_cpu;

    for (int i = 0; i< 5; ++i)
    {
        julie::la::cuda::Matrix_CUDA<int> input {julie::la::Shape{b, c, h, w}};
        input.uniform_random(-30, 30);
        auto input_cpu = julie::la::cpu::Matrix_CPU<int> {input};

        lint time1 = get_time_in_milliseconds();
        julie::la::cuda::img2row_2d(img2row_output, input, s_h, s_w, w_h, w_w);
        lint time2 = get_time_in_milliseconds();
        julie::la::cuda::img2row_2d_backward(img2row_backward, input.shape(), img2row_output, s_h, s_w, w_h, w_w);
        lint time3 = get_time_in_milliseconds();

        std::cout << "GPU time: forward: " << time2 - time1 << " backward: " << time3 - time2 << std::endl;

        lint time4 = get_time_in_milliseconds();
        julie::la::cpu::img2row_2d(img2row_output_cpu, input_cpu, s_h, s_w, w_h, w_w);
        lint time5 = get_time_in_milliseconds();
        julie::la::cpu::img2row_2d_backward(img2row_backward_cpu, input_cpu.shape(), img2row_output_cpu, s_h, s_w, w_h, w_w);
        lint time6 = get_time_in_milliseconds();

        std::cout << "CPU time: forward: " << time5 - time4 << " backward: " << time6 - time5 << std::endl;

        //std::cout << img2row_backward << std::endl;
        //std::cout << img2row_backward_cpu << std::endl;
        //std::cout << julie::la::cpu::Matrix_CPU<int> {img2row_backward} << std::endl;

        test::ASSERT(img2row_output == img2row_output_cpu.get_CUDA());
        test::ASSERT(img2row_backward == img2row_backward_cpu.get_CUDA());
        test::ASSERT(img2row_backward_cpu == julie::la::cpu::Matrix_CPU<int> {img2row_backward});
    }
    
}

void maxpool_2d_test_gpu()
{
    std::cout << "====================== maxpool_2d_test_gpu =====================" << std::endl;
    julie::la::cuda::Matrix_CUDA<float> input_1_ch1 {
        {
            { 0,  1,   2,  3},
            { 4,  5,   6,  7},
            { 8,  9,  10, 11},
            {12, 13,  14, 15},
            {16, 17,  18, 19},
            {20, 21,  22, 23}
        }
    };

    julie::la::cuda::Matrix_CUDA<float> input_1_ch2 {
        {
            { 0,  2,  4,  6},
            { 8, 10, 12, 14},
            {16, 18, 20, 22},
            {24, 26, 28, 30},
            {32, 34, 36, 38},
            {40, 42, 44, 46}
        }
    };

    julie::la::cuda::Matrix_CUDA<float> input_1_ch3 {
        {
            {  0,  -1,   -2,  -3},
            { -4,  -5,   -6,  -7},
            { -8,  -9,  -10, -11},
            {-12, -13,  -14, -15},
            {-16, -17,  -18, -19},
            {-20, -21,  -22, -23}
        }
    };

    ///////////////////////////////////////////////////////

    julie::la::cuda::Matrix_CUDA<float> output_1_ch1 {
        {
            { 14, 15},
            { 22, 23}
        }
    };

    julie::la::cuda::Matrix_CUDA<float> output_1_ch2 {
        {
            { 28, 30},
            { 44, 46},
        }
    };

    julie::la::cuda::Matrix_CUDA<float> output_1_ch3 {
        {
            { 0, -1},
            {-8, -9}
        }
    };

    ///////////////////////////////////////////////////////

    julie::la::cuda::Matrix_CUDA<float> diff_1_ch1 {
        {
            { 0, 0, 0, 0, 0, 0},
            { 0, 0, 0, 0, 0, 0},
            { 0, 0, 0, 0, 0, 0},
            { 0, 0, 1, 0, 0, 1},
            { 0, 0, 0, 0, 0, 0},
            { 0, 0, 0, 0, 0, 0},
            { 0, 0, 0, 0, 0, 0},
            { 0, 0, 1, 0, 0, 1}
        }
    };

    julie::la::cuda::Matrix_CUDA<float> diff_1_ch2 {
        {
            { 0, 0, 0, 0, 0, 0},
            { 0, 0, 0, 0, 0, 0},
            { 0, 0, 0, 0, 0, 0},
            { 0, 0, 1, 0, 0, 1},
            { 0, 0, 0, 0, 0, 0},
            { 0, 0, 0, 0, 0, 0},
            { 0, 0, 0, 0, 0, 0},
            { 0, 0, 1, 0, 0, 1}
        }
    };

    julie::la::cuda::Matrix_CUDA<float> diff_1_ch3 {
        {
            { 1, 0, 0, 1, 0, 0},
            { 0, 0, 0, 0, 0, 0},
            { 0, 0, 0, 0, 0, 0},
            { 0, 0, 0, 0, 0, 0},
            { 1, 0, 0, 1, 0, 0},
            { 0, 0, 0, 0, 0, 0},
            { 0, 0, 0, 0, 0, 0},
            { 0, 0, 0, 0, 0, 0}
        }
    };

    julie::la::cuda::Matrix_CUDA<float> input {{input_1_ch1, input_1_ch2, input_1_ch3}};
    julie::la::cuda::Matrix_CUDA<float> output_assert {{output_1_ch1, output_1_ch2, output_1_ch3}};
    julie::la::cuda::Matrix_CUDA<float> diff_assert {{diff_1_ch1, diff_1_ch2, diff_1_ch3}};

    input.left_extend_shape(); // 3 to 4 dimensions
    output_assert.left_extend_shape(); // 3 to 4 dimensions
    diff_assert.left_extend_shape(); // 3 to 4 dimensions

    julie::la::cuda::Matrix_CUDA<float> maxpool_out, maxpool_diff;
    julie::la::cuda::maxpool_2d(maxpool_out, maxpool_diff, input, 2, 1, 4, 3);

    std::cout << maxpool_out << std::endl;
    std::cout << maxpool_diff << std::endl;

    julie::la::cuda::Matrix_CUDA<float> distance;
    julie::la::cuda::subtract(distance, output_assert, maxpool_out);
    test::ASSERT(distance.euclidean_norm() < 0.0001);

    julie::la::cuda::subtract(distance, diff_assert, maxpool_diff);
    test::ASSERT(distance.euclidean_norm() < 0.0001);

    //////////////////////////////////////////////////////////////////

    julie::la::cuda::Matrix_CUDA<float> gradient_1_ch1 {
        {
            { 1, 2},
            { 3, 4}
        }
    };

    julie::la::cuda::Matrix_CUDA<float> gradient_1_ch2 {
        {
            { 2, 4},
            { 6, 8},
        }
    };

    julie::la::cuda::Matrix_CUDA<float> gradient_1_ch3 {
        {
            { -1, -2},
            { -3, -4}
        }
    };

    julie::la::cuda::Matrix_CUDA<float> in_g_1_ch1 {
        {
            { 0, 0, 0, 0},
            { 0, 0, 0, 0},
            { 0, 0, 0, 0},
            { 0, 0, 1, 2},
            { 0, 0, 0, 0},
            { 0, 0, 3, 4}
        }
    };

    julie::la::cuda::Matrix_CUDA<float> in_g_1_ch2 {
        {
            { 0, 0, 0, 0},
            { 0, 0, 0, 0},
            { 0, 0, 0, 0},
            { 0, 0, 2, 4},
            { 0, 0, 0, 0},
            { 0, 0, 6, 8}
        }
    };

    julie::la::cuda::Matrix_CUDA<float> in_g_1_ch3 {
        {
            {-1,-2, 0, 0},
            { 0, 0, 0, 0},
            {-3,-4, 0, 0},
            { 0, 0, 0, 0},
            { 0, 0, 0, 0},
            { 0, 0, 0, 0}
        }
    };

    julie::la::cuda::Matrix_CUDA<float> gradient {{gradient_1_ch1, gradient_1_ch2, gradient_1_ch3}};
    julie::la::cuda::Matrix_CUDA<float> in_gradient_assert {{in_g_1_ch1, in_g_1_ch2, in_g_1_ch3}};

    gradient.left_extend_shape();
    in_gradient_assert.left_extend_shape();

    julie::la::cuda::Matrix_CUDA<float> in_gradient, gradient_cache;
    julie::la::cuda::pool_2d_backward(in_gradient, gradient_cache, input.shape(), maxpool_diff, gradient, 2, 1, 4, 3);

    std::cout << in_gradient << std::endl;
    std::cout << gradient_cache << std::endl;
    
    julie::la::cuda::subtract(distance, in_gradient, in_gradient_assert);
    test::ASSERT(distance.euclidean_norm() < 0.0001);
}

void maxpool_2d_cross_test_gpu()
{
    std::cout << "====================== maxpool_2d_cross_test_gpu =====================" << std::endl;

    for (int i = 0; i < 100; ++i)
    {
        int b = rand() % 50 + 1;
        int c = rand() % 50 + 1;
        int h = rand() % 50 + 1;
        int w = rand() % 50 + 1;
        int k_h = std::min(rand() % 10 + 1, h);
        int k_w = std::min(rand() % 10 + 1, w);
        int s_h = std::min(rand() % 10 + 1, k_h);
        int s_w = std::min(rand() % 10 + 1, k_w);

        julie::la::cpu::Matrix_CPU<int> cpu_in {{b, c, h, w}};
        cpu_in.uniform_random(-20, 20);
        julie::la::cuda::Matrix_CUDA<int> cuda_in = cpu_in.get_CUDA();

        julie::la::cpu::Matrix_CPU<int> cpu_diff;
        julie::la::cuda::Matrix_CUDA<int> cuda_diff;

        julie::la::cpu::Matrix_CPU<int> cpu_cache;
        julie::la::cuda::Matrix_CUDA<int> cuda_cache;

        julie::la::cpu::Matrix_CPU<int> cpu_out;
        julie::la::cuda::Matrix_CUDA<int> cuda_out;

        julie::la::cpu::maxpool_2d(cpu_out, cpu_diff, cpu_in, s_h, s_w, k_h, k_w);
        julie::la::cuda::maxpool_2d(cuda_out, cuda_diff, cuda_in, s_h, s_w, k_h, k_w);
        
        julie::la::cpu::Matrix_CPU<int> cpu_in_grad;
        julie::la::cuda::Matrix_CUDA<int> cuda_in_grad;

        julie::la::cpu::Matrix_CPU<int> cpu_grad {{ b, c, (h - k_h) / s_h + 1, (w - k_w) / s_w + 1 }};
        cpu_grad.uniform_random(-20, 20);
        julie::la::cuda::Matrix_CUDA<int> cuda_grad = cpu_grad.get_CUDA();

        julie::la::cpu::pool_2d_backward(cpu_in_grad, cpu_cache, {b, c, h, w}, cpu_diff, cpu_grad, s_h, s_w, k_h, k_w);
        julie::la::cuda::pool_2d_backward(cuda_in_grad, cuda_cache, {b, c, h, w}, cuda_diff, cuda_grad, s_h, s_w, k_h, k_w);

        std::cout << "itr " << i << std::endl;
        std::cout << "input shape: " << cpu_in.shape() << std::endl;
        std::cout << "kernel: " << k_h << " " << k_w << std::endl;
        std::cout << "stride: " << s_h << " " << s_w << std::endl;

        //std::cout << cpu_in_grad << std::endl;
        //std::cout << cuda_in_grad << std::endl;

        test::ASSERT(cpu_out.get_CUDA() == cuda_out);
        //std::cout << "-------------A--------------" << std::endl;
        test::ASSERT(cpu_diff.get_CUDA() == cuda_diff);
        //std::cout << "-------------B--------------" << std::endl;
        test::ASSERT(cpu_cache.get_CUDA() == cuda_cache);
        //std::cout << "-------------C--------------" << std::endl;
        test::ASSERT(cpu_in_grad.get_CUDA() == cuda_in_grad);
        //std::cout << "-------------D--------------" << std::endl;
    }
}

void avgpool_2d_test_gpu()
{
    std::cout << "====================== avgpool_2d_test_gpu =====================" << std::endl;
    julie::la::cuda::Matrix_CUDA<float> input_1_ch1 {
        {
            { 0,  1,   2,  3},
            { 4,  5,   6,  7},
            { 8,  9,  10, 11},
            {12, 13,  14, 15},
            {16, 17,  18, 19},
            {20, 21,  22, 23}
        }
    };

    julie::la::cuda::Matrix_CUDA<float> input_1_ch2 {
        {
            { 0,  2,  4,  6},
            { 8, 10, 12, 14},
            {16, 18, 20, 22},
            {24, 26, 28, 30},
            {32, 34, 36, 38},
            {40, 42, 44, 46}
        }
    };

    julie::la::cuda::Matrix_CUDA<float> input_1_ch3 {
        {
            {  0,  -1,   -2,  -3},
            { -4,  -5,   -6,  -7},
            { -8,  -9,  -10, -11},
            {-12, -13,  -14, -15},
            {-16, -17,  -18, -19},
            {-20, -21,  -22, -23}
        }
    };

    ///////////////////////////////////////////////////////

    julie::la::cuda::Matrix_CUDA<float> output_1_ch1 {
        {
            { 7,  8},
            {15, 16}
        }
    };

    julie::la::cuda::Matrix_CUDA<float> output_1_ch2 {
        {
            { 14, 16},
            { 30, 32},
        }
    };

    julie::la::cuda::Matrix_CUDA<float> output_1_ch3 {
        {
            { -7,  -8},
            {-15, -16}
        }
    };

    ///////////////////////////////////////////////////////

    julie::la::cuda::Matrix_CUDA<float> diff_1_ch1 {1.0 / 12, julie::la::Shape{8, 6}};

    julie::la::cuda::Matrix_CUDA<float> diff_1_ch2 {1.0 / 12, julie::la::Shape{8, 6}};
 
    julie::la::cuda::Matrix_CUDA<float> diff_1_ch3 {1.0 / 12, julie::la::Shape{8, 6}};

    julie::la::cuda::Matrix_CUDA<float> input {{input_1_ch1, input_1_ch2, input_1_ch3}};
    julie::la::cuda::Matrix_CUDA<float> output_assert {{output_1_ch1, output_1_ch2, output_1_ch3}};
    julie::la::cuda::Matrix_CUDA<float> diff_assert {{diff_1_ch1, diff_1_ch2, diff_1_ch3}};

    input.left_extend_shape(); // 3 to 4 dimensions
    output_assert.left_extend_shape(); // 3 to 4 dimensions
    diff_assert.left_extend_shape(); // 3 to 4 dimensions

    julie::la::cuda::Matrix_CUDA<float> avgpool_out, avgpool_diff;
    julie::la::cuda::avgpool_2d(avgpool_out, avgpool_diff, input, 2, 1, 4, 3);

    std::cout << avgpool_out << std::endl;
    std::cout << avgpool_diff << std::endl;

    julie::la::cuda::Matrix_CUDA<float> distance;
    julie::la::cuda::subtract(distance, output_assert, avgpool_out);
    test::ASSERT(distance.euclidean_norm() < 0.0001);

    julie::la::cuda::subtract(distance, diff_assert, avgpool_diff);
    test::ASSERT(distance.euclidean_norm() < 0.0001);

//////////////////////////////////////////////////////////////////

    julie::la::cuda::Matrix_CUDA<float> gradient_1_ch1 {
        {
            { 1, 2},
            { 3, 4}
        }
    };

    julie::la::cuda::Matrix_CUDA<float> gradient_1_ch2 {
        {
            { 2, 4},
            { 6, 8},
        }
    };

    julie::la::cuda::Matrix_CUDA<float> gradient_1_ch3 {
        {
            { -1, -2},
            { -3, -4}
        }
    };

    julie::la::cuda::Matrix_CUDA<float> in_g_1_ch1 {
        {
            {      1,         1 + 2,         1 + 2,      2 },
            {      1,         1 + 2,         1 + 2,      2 },
            {  1 + 3, 1 + 2 + 3 + 4, 1 + 2 + 3 + 4,  2 + 4 },
            {  1 + 3, 1 + 2 + 3 + 4, 1 + 2 + 3 + 4,  2 + 4 },
            {      3,         3 + 4,         3 + 4,      4 },
            {      3,         3 + 4,         3 + 4,      4 },
        }
    };

    julie::la::cuda::Matrix_CUDA<float> in_g_1_ch2 {
        {
            {      2,         2 + 4,         2 + 4,      4 },
            {      2,         2 + 4,         2 + 4,      4 },
            {  2 + 6, 2 + 4 + 6 + 8, 2 + 4 + 6 + 8,  4 + 8 },
            {  2 + 6, 2 + 4 + 6 + 8, 2 + 4 + 6 + 8,  4 + 8 },
            {      6,         6 + 8,         6 + 8,      8 },
            {      6,         6 + 8,         6 + 8,      8 },
        }
    };

    julie::la::cuda::Matrix_CUDA<float> in_g_1_ch3 {
        {
            {      1,         1 + 2,         1 + 2,      2 },
            {      1,         1 + 2,         1 + 2,      2 },
            {  1 + 3, 1 + 2 + 3 + 4, 1 + 2 + 3 + 4,  2 + 4 },
            {  1 + 3, 1 + 2 + 3 + 4, 1 + 2 + 3 + 4,  2 + 4 },
            {      3,         3 + 4,         3 + 4,      4 },
            {      3,         3 + 4,         3 + 4,      4 },
        }
    };
    in_g_1_ch3 *= -1;

    julie::la::cuda::Matrix_CUDA<float> gradient {{gradient_1_ch1, gradient_1_ch2, gradient_1_ch3}};
    julie::la::cuda::Matrix_CUDA<float> in_gradient_assert {{in_g_1_ch1, in_g_1_ch2, in_g_1_ch3}};

    gradient.left_extend_shape();
    in_gradient_assert.left_extend_shape(); in_gradient_assert /= 12;

    julie::la::cuda::Matrix_CUDA<float> in_gradient, gradient_cache;
    julie::la::cuda::pool_2d_backward(in_gradient, gradient_cache, input.shape(), avgpool_diff, gradient, 2, 1, 4, 3);

    std::cout << in_gradient << std::endl;
    std::cout << gradient_cache << std::endl;
    
    julie::la::cuda::subtract(distance, in_gradient, in_gradient_assert);
    test::ASSERT(distance.euclidean_norm() < 0.0001);
}

void avgpool_2d_cross_test_gpu()
{
    std::cout << "====================== avgpool_2d_cross_test_gpu =====================" << std::endl;

    for (int i = 0; i < 100; ++i)
    {
        int b = rand() % 50 + 1;
        int c = rand() % 50 + 1;
        int h = rand() % 50 + 1;
        int w = rand() % 50 + 1;
        int k_h = std::min(rand() % 10 + 1, h);
        int k_w = std::min(rand() % 10 + 1, w);
        int s_h = std::min(rand() % 10 + 1, k_h);
        int s_w = std::min(rand() % 10 + 1, k_w);

        julie::la::cpu::Matrix_CPU<int> cpu_in {{b, c, h, w}};
        cpu_in.uniform_random(-20, 20);
        julie::la::cuda::Matrix_CUDA<int> cuda_in = cpu_in.get_CUDA();

        julie::la::cpu::Matrix_CPU<int> cpu_diff;
        julie::la::cuda::Matrix_CUDA<int> cuda_diff;

        julie::la::cpu::Matrix_CPU<int> cpu_cache;
        julie::la::cuda::Matrix_CUDA<int> cuda_cache;

        julie::la::cpu::Matrix_CPU<int> cpu_out;
        julie::la::cuda::Matrix_CUDA<int> cuda_out;

        julie::la::cpu::avgpool_2d(cpu_out, cpu_diff, cpu_in, s_h, s_w, k_h, k_w);
        julie::la::cuda::avgpool_2d(cuda_out, cuda_diff, cuda_in, s_h, s_w, k_h, k_w);
        
        julie::la::cpu::Matrix_CPU<int> cpu_in_grad;
        julie::la::cuda::Matrix_CUDA<int> cuda_in_grad;

        julie::la::cpu::Matrix_CPU<int> cpu_grad {{ b, c, (h - k_h) / s_h + 1, (w - k_w) / s_w + 1 }};
        cpu_grad.uniform_random(-20, 20);
        julie::la::cuda::Matrix_CUDA<int> cuda_grad = cpu_grad.get_CUDA();

        julie::la::cpu::pool_2d_backward(cpu_in_grad, cpu_cache, {b, c, h, w}, cpu_diff, cpu_grad, s_h, s_w, k_h, k_w);
        julie::la::cuda::pool_2d_backward(cuda_in_grad, cuda_cache, {b, c, h, w}, cuda_diff, cuda_grad, s_h, s_w, k_h, k_w);

        std::cout << "itr " << i << std::endl;
        std::cout << "input shape: " << cpu_in.shape() << std::endl;
        std::cout << "kernel: " << k_h << " " << k_w << std::endl;
        std::cout << "stride: " << s_h << " " << s_w << std::endl;

        //std::cout << cpu_in_grad << std::endl;
        //std::cout << cuda_in_grad << std::endl;

        test::ASSERT(cpu_out.get_CUDA() == cuda_out);
        //std::cout << "-------------A--------------" << std::endl;
        test::ASSERT(cpu_diff.get_CUDA() == cuda_diff);
        //std::cout << "-------------B--------------" << std::endl;
        test::ASSERT(cpu_cache.get_CUDA() == cuda_cache);
        //std::cout << "-------------C--------------" << std::endl;
        test::ASSERT(cpu_in_grad.get_CUDA() == cuda_in_grad);
        //std::cout << "-------------D--------------" << std::endl;
    }
}


void test_of_Matrix_CUDA_adv()
{
    test_cuda_pad_2d();
    test_cuda_pad_2d_backward();
    img2row_test_filter_size_gpu();
    img2row_test_filter_stride_gpu();
    img2row_test_batch_gpu();
    img2row_test_random_gpu();
    img2row_test_speed_gpu();

    maxpool_2d_test_gpu();
    maxpool_2d_cross_test_gpu();
    avgpool_2d_test_gpu();
    avgpool_2d_cross_test_gpu();
}

} // namespace test
