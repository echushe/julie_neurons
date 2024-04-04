#pragma once

#include "Matrix_CPU.hpp"

#include "Matrix_CPU_func.hpp"
#include "Matrix_CPU_func_adv.hpp"
#include "test_util.hpp"

#include <iostream>
#include <vector>
#include <list>



namespace test
{

void test_pad_2d()
{
    std::cout << "====================== 2d padding =====================" << std::endl;
    julie::la::cpu::Matrix_CPU<int> mat_11 {
        {
            { 3,  3,  7,  9},
            { 3,  2,  5,  6},
            {-1, -3, -4, -5},
            { 0,  0, 11, 17},
            {22, 25, -9, 88}
        }};

    julie::la::cpu::Matrix_CPU<int> mat_12 {
        {
            { 1,  2,  3,  4},
            { 5,  6,  7,  8},
            { 0,  1,  0,  1},
            {-1,  0, -1,  0},
            {11,-11,100,-20}
        }};

    julie::la::cpu::Matrix_CPU<int> mat_13 {
        {
            { 1,  1,  1,  1},
            {-1, -1, -1, -1},
            { 1,  1,  1,  1},
            {-1, -1, -1, -1},
            { 8,  8,  9, -8}
        }};

    julie::la::cpu::Matrix_CPU<int> mat_21 {
        {
            {11, 10,  9,  8},
            { 7,  6,  5,  4},
            { 3,  2,  1,  0},
            {-1, -2, -3, -4},
            {-5, -6, -7, -8}
        }};

    julie::la::cpu::Matrix_CPU<int> mat_22 {
        {
            { 0,  1,  2,  3},
            { 4,  5,  6,  7},
            { 8,  9, 10, 11},
            {12, 13, 14, 15},
            {16, 17, 18, 19}
        }};

    julie::la::cpu::Matrix_CPU<int> mat_23 {
        {
            { 0,  1,  2,  3},
            { 0,  1,  2,  3},
            { 0,  1,  2,  3},
            { 0,  1,  2,  3},
            { 0,  1,  2,  3}
        }};


    julie::la::cpu::Matrix_CPU<int> image1 {std::vector<julie::la::cpu::Matrix_CPU<int>>{mat_11, mat_12, mat_13}};
    julie::la::cpu::Matrix_CPU<int> image2 {std::vector<julie::la::cpu::Matrix_CPU<int>>{mat_21, mat_22, mat_23}};
    julie::la::cpu::Matrix_CPU<int> batch {std::vector<julie::la::cpu::Matrix_CPU<int>>{image1, image2}};

    julie::la::cpu::Matrix_CPU<int> pmat_11 {
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

    julie::la::cpu::Matrix_CPU<int> pmat_12 {
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

    julie::la::cpu::Matrix_CPU<int> pmat_13 {
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

    julie::la::cpu::Matrix_CPU<int> pmat_21 {
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

    julie::la::cpu::Matrix_CPU<int> pmat_22 {
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

    julie::la::cpu::Matrix_CPU<int> pmat_23 {
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


    julie::la::cpu::Matrix_CPU<int> pimage1 {std::vector<julie::la::cpu::Matrix_CPU<int>>{pmat_11, pmat_12, pmat_13}};
    julie::la::cpu::Matrix_CPU<int> pimage2 {std::vector<julie::la::cpu::Matrix_CPU<int>>{pmat_21, pmat_22, pmat_23}};
    julie::la::cpu::Matrix_CPU<int> pbatch {std::vector<julie::la::cpu::Matrix_CPU<int>>{pimage1, pimage2}};

    julie::la::cpu::Matrix_CPU<int> my_pbatch;
    julie::la::cpu::pad_2d(my_pbatch, batch, 2, 1);

    std::cout << batch << std::endl;
    std::cout << my_pbatch << std::endl;
    test::ASSERT(my_pbatch == pbatch);

}

void test_pad_2d_backward()
{
    std::cout << "====================== 2d padding backward =====================" << std::endl;
    julie::la::cpu::Matrix_CPU<int> mat_11 {
        {
            { 3,  3,  7,  9},
            { 3,  2,  5,  6},
            {-1, -3, -4, -5},
            { 0,  0, 11, 17},
            {22, 25, -9, 88}
        }};

    julie::la::cpu::Matrix_CPU<int> mat_12 {
        {
            { 1,  2,  3,  4},
            { 5,  6,  7,  8},
            { 0,  1,  0,  1},
            {-1,  0, -1,  0},
            {11,-11,100,-20}
        }};

    julie::la::cpu::Matrix_CPU<int> mat_13 {
        {
            { 1,  1,  1,  1},
            {-1, -1, -1, -1},
            { 1,  1,  1,  1},
            {-1, -1, -1, -1},
            { 8,  8,  9, -8}
        }};

    julie::la::cpu::Matrix_CPU<int> mat_21 {
        {
            {11, 10,  9,  8},
            { 7,  6,  5,  4},
            { 3,  2,  1,  0},
            {-1, -2, -3, -4},
            {-5, -6, -7, -8}
        }};

    julie::la::cpu::Matrix_CPU<int> mat_22 {
        {
            { 0,  1,  2,  3},
            { 4,  5,  6,  7},
            { 8,  9, 10, 11},
            {12, 13, 14, 15},
            {16, 17, 18, 19}
        }};

    julie::la::cpu::Matrix_CPU<int> mat_23 {
        {
            { 0,  1,  2,  3},
            { 0,  1,  2,  3},
            { 0,  1,  2,  3},
            { 0,  1,  2,  3},
            { 0,  1,  2,  3}
        }};


    julie::la::cpu::Matrix_CPU<int> image1 {std::vector<julie::la::cpu::Matrix_CPU<int>>{mat_11, mat_12, mat_13}};
    julie::la::cpu::Matrix_CPU<int> image2 {std::vector<julie::la::cpu::Matrix_CPU<int>>{mat_21, mat_22, mat_23}};
    julie::la::cpu::Matrix_CPU<int> batch {std::vector<julie::la::cpu::Matrix_CPU<int>>{image1, image2}};

    julie::la::cpu::Matrix_CPU<int> pmat_11 {
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

    julie::la::cpu::Matrix_CPU<int> pmat_12 {
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

    julie::la::cpu::Matrix_CPU<int> pmat_13 {
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

    julie::la::cpu::Matrix_CPU<int> pmat_21 {
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

    julie::la::cpu::Matrix_CPU<int> pmat_22 {
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

    julie::la::cpu::Matrix_CPU<int> pmat_23 {
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


    julie::la::cpu::Matrix_CPU<int> pimage1 {std::vector<julie::la::cpu::Matrix_CPU<int>>{pmat_11, pmat_12, pmat_13}};
    julie::la::cpu::Matrix_CPU<int> pimage2 {std::vector<julie::la::cpu::Matrix_CPU<int>>{pmat_21, pmat_22, pmat_23}};
    julie::la::cpu::Matrix_CPU<int> pbatch {std::vector<julie::la::cpu::Matrix_CPU<int>>{pimage1, pimage2}};

    julie::la::cpu::Matrix_CPU<int> my_batch;
    julie::la::cpu::pad_2d_backward(my_batch, pbatch, 2, 1);

    std::cout << batch << std::endl;
    std::cout << my_batch << std::endl;
    test::ASSERT(my_batch == batch);

}


void img2row_test_filter_size()
{
    std::cout << "====================== img2row_test_filter_size =====================" << std::endl;
    julie::la::cpu::Matrix_CPU<int> input_ch1 {
        {
            {0,  1,  2,  3},
            {4,  5,  6,  7},
            {8,  9,  10, 11}
        }
    };

    julie::la::cpu::Matrix_CPU<int> input_ch2 {
        {
            { 0,  2,  4,  6},
            { 8, 10, 12, 14},
            {16, 18, 24, 28}
        }
    };

    julie::la::cpu::Matrix_CPU<int> input_ch3 {
        {
            { 0, -1, -2, -3},
            {-4, -5, -6, -7},
            {-8, -9,-10,-11}
        }
    };

    julie::la::cpu::Matrix_CPU<int> input {{input_ch1, input_ch2, input_ch3}};
    input.left_extend_shape();

    julie::la::cpu::Matrix_CPU<int> img2row_output;
    julie::la::cpu::img2row_2d(img2row_output, input, 1, 1, 2, 3);
    std::cout << img2row_output << std::endl;

    julie::la::cpu::Matrix_CPU<int> img2col_output;
    julie::la::cpu::img2col_2d(img2col_output, input, 1, 1, 2, 3);
    std::cout << img2col_output << std::endl;

    julie::la::cpu::Matrix_CPU<int> img2row_backward;
    julie::la::cpu::img2row_2d_backward(img2row_backward, input.shape(), img2row_output, 1, 1, 2, 3);
    std::cout << img2row_backward << std::endl;

    julie::la::cpu::Matrix_CPU<int> img2col_backward;
    julie::la::cpu::img2col_2d_backward(img2col_backward, input.shape(), img2col_output, 1, 1, 2, 3);
    std::cout << img2col_backward << std::endl;

    test::ASSERT(img2row_backward == img2col_backward);
}

void img2row_test_filter_stride()
{
    std::cout << "====================== img2row_test_filter_stride =====================" << std::endl;
    julie::la::cpu::Matrix_CPU<int> input_ch1 {
        {
            {0,  1,  2,  3},
            {4,  5,  6,  7},
            {8,  9,  10, 11}
        }
    };

    julie::la::cpu::Matrix_CPU<int> input_ch2 {
        {
            { 0,  2,  4,  6},
            { 8, 10, 12, 14},
            {16, 18, 24, 28}
        }
    };

    julie::la::cpu::Matrix_CPU<int> input_ch3 {
        {
            { 0, -1, -2, -3},
            {-4, -5, -6, -7},
            {-8, -9,-10,-11}
        }
    };

    julie::la::cpu::Matrix_CPU<int> input {{input_ch1, input_ch2, input_ch3}};
    input.left_extend_shape();

    julie::la::cpu::Matrix_CPU<int> img2row_output;
    julie::la::cpu::img2row_2d(img2row_output, input, 1, 2, 2, 2);
    std::cout << img2row_output << std::endl;

    julie::la::cpu::Matrix_CPU<int> img2col_output;
    julie::la::cpu::img2col_2d(img2col_output, input, 1, 2, 2, 2);
    std::cout << img2col_output << std::endl;

    julie::la::cpu::Matrix_CPU<int> img2row_backward;
    julie::la::cpu::img2row_2d_backward(img2row_backward, input.shape(), img2row_output, 1, 2, 2, 2);
    std::cout << img2row_backward << std::endl;

    julie::la::cpu::Matrix_CPU<int> img2col_backward;
    julie::la::cpu::img2col_2d_backward(img2col_backward, input.shape(), img2col_output, 1, 2, 2, 2);
    std::cout << img2col_backward << std::endl;

    test::ASSERT(img2row_backward == img2col_backward);
}

void img2row_test_batch()
{
    std::cout << "====================== img2row_test_batch =====================" << std::endl;
    julie::la::cpu::Matrix_CPU<int> input_1_ch1 {
        {
            {0,  1,  2,  3},
            {4,  5,  6,  7},
            {8,  9,  10, 11}
        }
    };

    julie::la::cpu::Matrix_CPU<int> input_1_ch2 {
        {
            { 0,  2,  4,  6},
            { 8, 10, 12, 14},
            {16, 18, 24, 28}
        }
    };

    julie::la::cpu::Matrix_CPU<int> input_1_ch3 {
        {
            { 0, -1, -2, -3},
            {-4, -5, -6, -7},
            {-8, -9,-10,-11}
        }
    };

    julie::la::cpu::Matrix_CPU<int> input_2_ch1 {
        {
            {0,  0,  0,  0},
            {1,  1,  1,  1},
            {2,  2,  2,  2}
        }
    };

    julie::la::cpu::Matrix_CPU<int> input_2_ch2 {
        {
            { 0,  2,  4,  6},
            { 0,  2,  4,  6},
            { 0,  2,  4,  6}
        }
    };

    julie::la::cpu::Matrix_CPU<int> input_2_ch3 {
        {
            { 0,  0,  0,  0},
            {-1, -1, -1, -1},
            {-2, -2, -2, -2}
        }
    };

    julie::la::cpu::Matrix_CPU<int> input_1 {{input_1_ch1, input_1_ch2, input_1_ch3}};
    julie::la::cpu::Matrix_CPU<int> input_2 {{input_2_ch1, input_2_ch2, input_2_ch3}};

    auto input = julie::la::cpu::Matrix_CPU<int> {{input_1, input_2}};

    julie::la::cpu::Matrix_CPU<int> img2row_output;
    julie::la::cpu::img2row_2d(img2row_output, input, 1, 1, 3, 3);
    std::cout << img2row_output << std::endl;

    julie::la::cpu::Matrix_CPU<int> img2col_output;
    julie::la::cpu::img2col_2d(img2col_output, input, 1, 1, 3, 3);
    std::cout << img2col_output << std::endl;

    julie::la::cpu::Matrix_CPU<int> img2row_backward;
    julie::la::cpu::img2row_2d_backward(img2row_backward, input.shape(), img2row_output, 1, 1, 3, 3);
    std::cout << img2row_backward << std::endl;

    julie::la::cpu::Matrix_CPU<int> img2col_backward;
    julie::la::cpu::img2col_2d_backward(img2col_backward, input.shape(), img2col_output, 1, 1, 3, 3);
    std::cout << img2col_backward << std::endl;

    test::ASSERT(img2row_backward == img2col_backward);
}

void maxpool_2d_test()
{
    std::cout << "====================== maxpool_2d_test =====================" << std::endl;
    julie::la::cpu::Matrix_CPU<float> input_1_ch1 {
        {
            { 0,  1,   2,  3},
            { 4,  5,   6,  7},
            { 8,  9,  10, 11},
            {12, 13,  14, 15},
            {16, 17,  18, 19},
            {20, 21,  22, 23}
        }
    };

    julie::la::cpu::Matrix_CPU<float> input_1_ch2 {
        {
            { 0,  2,  4,  6},
            { 8, 10, 12, 14},
            {16, 18, 20, 22},
            {24, 26, 28, 30},
            {32, 34, 36, 38},
            {40, 42, 44, 46}
        }
    };

    julie::la::cpu::Matrix_CPU<float> input_1_ch3 {
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

    julie::la::cpu::Matrix_CPU<float> output_1_ch1 {
        std::vector<float>{
             14, 15,
             22, 23
        },
        julie::la::Shape{2, 2}
    };

    julie::la::cpu::Matrix_CPU<float> output_1_ch2 {
        std::vector<float>{
             28, 30,
             44, 46,
        },
        julie::la::Shape{2, 2}
    };

    julie::la::cpu::Matrix_CPU<float> output_1_ch3 {
        std::vector<float>{
             0, -1,
            -8, -9
        },
        julie::la::Shape{2, 2}
    };

    ///////////////////////////////////////////////////////

    julie::la::cpu::Matrix_CPU<float> diff_1_ch1 {
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

    julie::la::cpu::Matrix_CPU<float> diff_1_ch2 {
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

    julie::la::cpu::Matrix_CPU<float> diff_1_ch3 {
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

    julie::la::cpu::Matrix_CPU<float> input {{input_1_ch1, input_1_ch2, input_1_ch3}};
    julie::la::cpu::Matrix_CPU<float> output_assert {{output_1_ch1, output_1_ch2, output_1_ch3}};
    julie::la::cpu::Matrix_CPU<float> diff_assert {{diff_1_ch1, diff_1_ch2, diff_1_ch3}};

    input.left_extend_shape(); // 3 to 4 dimensions
    output_assert.left_extend_shape(); // 3 to 4 dimensions
    diff_assert.left_extend_shape(); // 3 to 4 dimensions

    julie::la::cpu::Matrix_CPU<float> maxpool_out, maxpool_diff;
    julie::la::cpu::maxpool_2d(maxpool_out, maxpool_diff, input, 2, 1, 4, 3);

    std::cout << maxpool_out << std::endl;
    std::cout << maxpool_diff << std::endl;

    julie::la::cpu::Matrix_CPU<float> distance;
    julie::la::cpu::subtract(distance, output_assert, maxpool_out);
    test::ASSERT(distance.euclidean_norm() < 0.0001);

    julie::la::cpu::subtract(distance, diff_assert, maxpool_diff);
    test::ASSERT(distance.euclidean_norm() < 0.0001);
//////////////////////////////////////////////////////////////////

    julie::la::cpu::Matrix_CPU<float> gradient_1_ch1 {
        std::vector<float>{
             1, 2,
             3, 4
        },
        julie::la::Shape{2, 2}
    };

    julie::la::cpu::Matrix_CPU<float> gradient_1_ch2 {
        std::vector<float>{
             2, 4,
             6, 8
        },
        julie::la::Shape{2, 2}
    };

    julie::la::cpu::Matrix_CPU<float> gradient_1_ch3 {
        std::vector<float>{
             -1, -2,
             -3, -4
        },
        julie::la::Shape{2, 2}
    };

    julie::la::cpu::Matrix_CPU<float> in_g_1_ch1 {
        {
            { 0, 0, 0, 0},
            { 0, 0, 0, 0},
            { 0, 0, 0, 0},
            { 0, 0, 1, 2},
            { 0, 0, 0, 0},
            { 0, 0, 3, 4}
        }
    };

    julie::la::cpu::Matrix_CPU<float> in_g_1_ch2 {
        {
            { 0, 0, 0, 0},
            { 0, 0, 0, 0},
            { 0, 0, 0, 0},
            { 0, 0, 2, 4},
            { 0, 0, 0, 0},
            { 0, 0, 6, 8}
        }
    };

    julie::la::cpu::Matrix_CPU<float> in_g_1_ch3 {
        {
            {-1,-2, 0, 0},
            { 0, 0, 0, 0},
            {-3,-4, 0, 0},
            { 0, 0, 0, 0},
            { 0, 0, 0, 0},
            { 0, 0, 0, 0}
        }
    };

    julie::la::cpu::Matrix_CPU<float> gradient {{gradient_1_ch1, gradient_1_ch2, gradient_1_ch3}};
    julie::la::cpu::Matrix_CPU<float> in_gradient_assert {{in_g_1_ch1, in_g_1_ch2, in_g_1_ch3}};

    gradient.left_extend_shape();
    in_gradient_assert.left_extend_shape();

    julie::la::cpu::Matrix_CPU<float> in_gradient, gradient_cache;
    julie::la::cpu::pool_2d_backward(in_gradient, gradient_cache, input.shape(), maxpool_diff, gradient, 2, 1, 4, 3);

    std::cout << in_gradient << std::endl;
    std::cout << gradient_cache << std::endl;
    
    julie::la::cpu::subtract(distance, in_gradient, in_gradient_assert);
    test::ASSERT(distance.euclidean_norm() < 0.0001);
}

void avgpool_2d_test()
{
    std::cout << "====================== avgpool_2d_test =====================" << std::endl;
    julie::la::cpu::Matrix_CPU<float> input_1_ch1 {
        {
            { 0,  1,   2,  3},
            { 4,  5,   6,  7},
            { 8,  9,  10, 11},
            {12, 13,  14, 15},
            {16, 17,  18, 19},
            {20, 21,  22, 23}
        }
    };

    julie::la::cpu::Matrix_CPU<float> input_1_ch2 {
        {
            { 0,  2,  4,  6},
            { 8, 10, 12, 14},
            {16, 18, 20, 22},
            {24, 26, 28, 30},
            {32, 34, 36, 38},
            {40, 42, 44, 46}
        }
    };

    julie::la::cpu::Matrix_CPU<float> input_1_ch3 {
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

    julie::la::cpu::Matrix_CPU<float> output_1_ch1 {
        std::vector<float>{
             7,  8,
            15, 16
        },
        julie::la::Shape{2, 2}
    };

    julie::la::cpu::Matrix_CPU<float> output_1_ch2 {
        std::vector<float>{
             14, 16,
             30, 32,
        },
        julie::la::Shape{2, 2}
    };

    julie::la::cpu::Matrix_CPU<float> output_1_ch3 {
        std::vector<float>{
             -7,  -8,
            -15, -16
        },
        julie::la::Shape{2, 2}
    };

    ///////////////////////////////////////////////////////

    julie::la::cpu::Matrix_CPU<float> diff_1_ch1 {1.0 / 12, julie::la::Shape{8, 6}};

    julie::la::cpu::Matrix_CPU<float> diff_1_ch2 {1.0 / 12, julie::la::Shape{8, 6}};
 
    julie::la::cpu::Matrix_CPU<float> diff_1_ch3 {1.0 / 12, julie::la::Shape{8, 6}};

    julie::la::cpu::Matrix_CPU<float> input {{input_1_ch1, input_1_ch2, input_1_ch3}};
    julie::la::cpu::Matrix_CPU<float> output_assert {{output_1_ch1, output_1_ch2, output_1_ch3}};
    julie::la::cpu::Matrix_CPU<float> diff_assert {{diff_1_ch1, diff_1_ch2, diff_1_ch3}};

    input.left_extend_shape(); // 3 to 4 dimensions
    output_assert.left_extend_shape(); // 3 to 4 dimensions
    diff_assert.left_extend_shape(); // 3 to 4 dimensions

    julie::la::cpu::Matrix_CPU<float> avgpool_out, avgpool_diff;
    julie::la::cpu::avgpool_2d(avgpool_out, avgpool_diff, input, 2, 1, 4, 3);

    std::cout << avgpool_out << std::endl;
    std::cout << avgpool_diff << std::endl;

    julie::la::cpu::Matrix_CPU<float> distance;
    julie::la::cpu::subtract(distance, output_assert, avgpool_out);
    test::ASSERT(distance.euclidean_norm() < 0.0001);

    julie::la::cpu::subtract(distance, diff_assert, avgpool_diff);
    test::ASSERT(distance.euclidean_norm() < 0.0001);

//////////////////////////////////////////////////////////////////

    julie::la::cpu::Matrix_CPU<float> gradient_1_ch1 {
        std::vector<float>{
             1, 2,
             3, 4
        },
        julie::la::Shape{2, 2}
    };

    julie::la::cpu::Matrix_CPU<float> gradient_1_ch2 {
        std::vector<float>{
             2, 4,
             6, 8,
        },
        julie::la::Shape{2, 2}
    };

    julie::la::cpu::Matrix_CPU<float> gradient_1_ch3 {
        std::vector<float>{
             -1, -2,
             -3, -4
        },
        julie::la::Shape{2, 2}
    };

    julie::la::cpu::Matrix_CPU<float> in_g_1_ch1 {
        {
            {      1,         1 + 2,         1 + 2,      2 },
            {      1,         1 + 2,         1 + 2,      2 },
            {  1 + 3, 1 + 2 + 3 + 4, 1 + 2 + 3 + 4,  2 + 4 },
            {  1 + 3, 1 + 2 + 3 + 4, 1 + 2 + 3 + 4,  2 + 4 },
            {      3,         3 + 4,         3 + 4,      4 },
            {      3,         3 + 4,         3 + 4,      4 },
        }
    };

    julie::la::cpu::Matrix_CPU<float> in_g_1_ch2 {
        {
            {      2,         2 + 4,         2 + 4,      4 },
            {      2,         2 + 4,         2 + 4,      4 },
            {  2 + 6, 2 + 4 + 6 + 8, 2 + 4 + 6 + 8,  4 + 8 },
            {  2 + 6, 2 + 4 + 6 + 8, 2 + 4 + 6 + 8,  4 + 8 },
            {      6,         6 + 8,         6 + 8,      8 },
            {      6,         6 + 8,         6 + 8,      8 },
        }
    };

    julie::la::cpu::Matrix_CPU<float> in_g_1_ch3 {
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

    julie::la::cpu::Matrix_CPU<float> gradient {{gradient_1_ch1, gradient_1_ch2, gradient_1_ch3}};
    julie::la::cpu::Matrix_CPU<float> in_gradient_assert {{in_g_1_ch1, in_g_1_ch2, in_g_1_ch3}};

    gradient.left_extend_shape();
    in_gradient_assert.left_extend_shape(); in_gradient_assert /= 12;

    julie::la::cpu::Matrix_CPU<float> in_gradient, gradient_cache;
    julie::la::cpu::pool_2d_backward(in_gradient, gradient_cache, input.shape(), avgpool_diff, gradient, 2, 1, 4, 3);

    std::cout << in_gradient << std::endl;
    std::cout << gradient_cache << std::endl;
    
    julie::la::cpu::subtract(distance, in_gradient, in_gradient_assert);
    test::ASSERT(distance.euclidean_norm() < 0.0001);
}



void test_of_Matrix_CPU_adv()
{
    test_pad_2d();
    test_pad_2d_backward();
    img2row_test_filter_size();
    img2row_test_filter_stride();
    img2row_test_batch();

    maxpool_2d_test();
    avgpool_2d_test();
}

} // namespace test
