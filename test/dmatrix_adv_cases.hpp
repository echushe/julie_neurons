#pragma once

#include "DMatrix.hpp"


#include "DMatrix_adv.hpp"
#include "test_util.hpp"

#include <iostream>
#include <vector>
#include <list>



namespace test
{

void test_pad_2d()
{
    std::cout << "====================== 2d padding =====================" << std::endl;
    julie::la::DMatrix<int> mat_11 {
        {
            { 3,  3,  7,  9},
            { 3,  2,  5,  6},
            {-1, -3, -4, -5},
            { 0,  0, 11, 17},
            {22, 25, -9, 88}
        }};

    julie::la::DMatrix<int> mat_12 {
        {
            { 1,  2,  3,  4},
            { 5,  6,  7,  8},
            { 0,  1,  0,  1},
            {-1,  0, -1,  0},
            {11,-11,100,-20}
        }};

    julie::la::DMatrix<int> mat_13 {
        {
            { 1,  1,  1,  1},
            {-1, -1, -1, -1},
            { 1,  1,  1,  1},
            {-1, -1, -1, -1},
            { 8,  8,  9, -8}
        }};

    julie::la::DMatrix<int> mat_21 {
        {
            {11, 10,  9,  8},
            { 7,  6,  5,  4},
            { 3,  2,  1,  0},
            {-1, -2, -3, -4},
            {-5, -6, -7, -8}
        }};

    julie::la::DMatrix<int> mat_22 {
        {
            { 0,  1,  2,  3},
            { 4,  5,  6,  7},
            { 8,  9, 10, 11},
            {12, 13, 14, 15},
            {16, 17, 18, 19}
        }};

    julie::la::DMatrix<int> mat_23 {
        {
            { 0,  1,  2,  3},
            { 0,  1,  2,  3},
            { 0,  1,  2,  3},
            { 0,  1,  2,  3},
            { 0,  1,  2,  3}
        }};


    julie::la::DMatrix<int> image1 {std::vector<julie::la::DMatrix<int>>{mat_11, mat_12, mat_13}};
    julie::la::DMatrix<int> image2 {std::vector<julie::la::DMatrix<int>>{mat_21, mat_22, mat_23}};
    julie::la::DMatrix<int> batch {std::vector<julie::la::DMatrix<int>>{image1, image2}};

    julie::la::DMatrix<int> pmat_11 {
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

    julie::la::DMatrix<int> pmat_12 {
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

    julie::la::DMatrix<int> pmat_13 {
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

    julie::la::DMatrix<int> pmat_21 {
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

    julie::la::DMatrix<int> pmat_22 {
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

    julie::la::DMatrix<int> pmat_23 {
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


    julie::la::DMatrix<int> pimage1 {std::vector<julie::la::DMatrix<int>>{pmat_11, pmat_12, pmat_13}};
    julie::la::DMatrix<int> pimage2 {std::vector<julie::la::DMatrix<int>>{pmat_21, pmat_22, pmat_23}};
    julie::la::DMatrix<int> pbatch {std::vector<julie::la::DMatrix<int>>{pimage1, pimage2}};

    auto my_pbatch = julie::la::pad_2d(batch, 2, 1);

    std::cout << batch << std::endl;
    std::cout << my_pbatch << std::endl;
    test::ASSERT(my_pbatch == pbatch);

}

void test_pad_2d_backward()
{
    std::cout << "====================== 2d padding =====================" << std::endl;
    julie::la::DMatrix<int> mat_11 {
        {
            { 3,  3,  7,  9},
            { 3,  2,  5,  6},
            {-1, -3, -4, -5},
            { 0,  0, 11, 17},
            {22, 25, -9, 88}
        }};

    julie::la::DMatrix<int> mat_12 {
        {
            { 1,  2,  3,  4},
            { 5,  6,  7,  8},
            { 0,  1,  0,  1},
            {-1,  0, -1,  0},
            {11,-11,100,-20}
        }};

    julie::la::DMatrix<int> mat_13 {
        {
            { 1,  1,  1,  1},
            {-1, -1, -1, -1},
            { 1,  1,  1,  1},
            {-1, -1, -1, -1},
            { 8,  8,  9, -8}
        }};

    julie::la::DMatrix<int> mat_21 {
        {
            {11, 10,  9,  8},
            { 7,  6,  5,  4},
            { 3,  2,  1,  0},
            {-1, -2, -3, -4},
            {-5, -6, -7, -8}
        }};

    julie::la::DMatrix<int> mat_22 {
        {
            { 0,  1,  2,  3},
            { 4,  5,  6,  7},
            { 8,  9, 10, 11},
            {12, 13, 14, 15},
            {16, 17, 18, 19}
        }};

    julie::la::DMatrix<int> mat_23 {
        {
            { 0,  1,  2,  3},
            { 0,  1,  2,  3},
            { 0,  1,  2,  3},
            { 0,  1,  2,  3},
            { 0,  1,  2,  3}
        }};


    julie::la::DMatrix<int> image1 {std::vector<julie::la::DMatrix<int>>{mat_11, mat_12, mat_13}};
    julie::la::DMatrix<int> image2 {std::vector<julie::la::DMatrix<int>>{mat_21, mat_22, mat_23}};
    julie::la::DMatrix<int> batch {std::vector<julie::la::DMatrix<int>>{image1, image2}};

    julie::la::DMatrix<int> pmat_11 {
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

    julie::la::DMatrix<int> pmat_12 {
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

    julie::la::DMatrix<int> pmat_13 {
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

    julie::la::DMatrix<int> pmat_21 {
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

    julie::la::DMatrix<int> pmat_22 {
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

    julie::la::DMatrix<int> pmat_23 {
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


    julie::la::DMatrix<int> pimage1 {std::vector<julie::la::DMatrix<int>>{pmat_11, pmat_12, pmat_13}};
    julie::la::DMatrix<int> pimage2 {std::vector<julie::la::DMatrix<int>>{pmat_21, pmat_22, pmat_23}};
    julie::la::DMatrix<int> pbatch {std::vector<julie::la::DMatrix<int>>{pimage1, pimage2}};

    auto my_batch = julie::la::pad_2d_backward(pbatch, 2, 1);

    std::cout << batch << std::endl;
    std::cout << my_batch << std::endl;
    test::ASSERT(my_batch == batch);

}


void img2row_test_filter_size()
{
    julie::la::DMatrix<int> input_ch1 {
        {
            {0,  1,  2,  3},
            {4,  5,  6,  7},
            {8,  9,  10, 11}
        }
    };

    julie::la::DMatrix<int> input_ch2 {
        {
            { 0,  2,  4,  6},
            { 8, 10, 12, 14},
            {16, 18, 24, 28}
        }
    };

    julie::la::DMatrix<int> input_ch3 {
        {
            { 0, -1, -2, -3},
            {-4, -5, -6, -7},
            {-8, -9,-10,-11}
        }
    };

    julie::la::DMatrix<int> input {{input_ch1, input_ch2, input_ch3}};
    input.left_extend_shape();

    auto img2row_output = julie::la::img2row_2d(input, 1, 1, 2, 3);
    std::cout << img2row_output << std::endl;

    auto img2col_output = julie::la::img2col_2d(input, 1, 1, 2, 3);
    std::cout << img2col_output << std::endl;

    auto img2row_backward = julie::la::img2row_2d_backward(input.shape(), img2row_output, 1, 1, 2, 3);
    std::cout << img2row_backward << std::endl;

    auto img2col_backward = julie::la::img2col_2d_backward(input.shape(), img2col_output, 1, 1, 2, 3);
    std::cout << img2col_backward << std::endl;

    test::ASSERT(img2row_backward == img2col_backward);
}

void img2row_test_filter_stride()
{
    julie::la::DMatrix<int> input_ch1 {
        {
            {0,  1,  2,  3},
            {4,  5,  6,  7},
            {8,  9,  10, 11}
        }
    };

    julie::la::DMatrix<int> input_ch2 {
        {
            { 0,  2,  4,  6},
            { 8, 10, 12, 14},
            {16, 18, 24, 28}
        }
    };

    julie::la::DMatrix<int> input_ch3 {
        {
            { 0, -1, -2, -3},
            {-4, -5, -6, -7},
            {-8, -9,-10,-11}
        }
    };

    julie::la::DMatrix<int> input {{input_ch1, input_ch2, input_ch3}};
    input.left_extend_shape();

    auto img2row_output = julie::la::img2row_2d(input, 1, 2, 2, 2);
    std::cout << img2row_output << std::endl;

    auto img2col_output = julie::la::img2col_2d(input, 1, 2, 2, 2);
    std::cout << img2col_output << std::endl;

    auto img2row_backward = julie::la::img2row_2d_backward(input.shape(), img2row_output, 1, 2, 2, 2);
    std::cout << img2row_backward << std::endl;

    auto img2col_backward = julie::la::img2col_2d_backward(input.shape(), img2col_output, 1, 2, 2, 2);
    std::cout << img2col_backward << std::endl;

    test::ASSERT(img2row_backward == img2col_backward);
}

void img2row_test_batch()
{
    julie::la::DMatrix<int> input_1_ch1 {
        {
            {0,  1,  2,  3},
            {4,  5,  6,  7},
            {8,  9,  10, 11}
        }
    };

    julie::la::DMatrix<int> input_1_ch2 {
        {
            { 0,  2,  4,  6},
            { 8, 10, 12, 14},
            {16, 18, 24, 28}
        }
    };

    julie::la::DMatrix<int> input_1_ch3 {
        {
            { 0, -1, -2, -3},
            {-4, -5, -6, -7},
            {-8, -9,-10,-11}
        }
    };

    julie::la::DMatrix<int> input_2_ch1 {
        {
            {0,  0,  0,  0},
            {1,  1,  1,  1},
            {2,  2,  2,  2}
        }
    };

    julie::la::DMatrix<int> input_2_ch2 {
        {
            { 0,  2,  4,  6},
            { 0,  2,  4,  6},
            { 0,  2,  4,  6}
        }
    };

    julie::la::DMatrix<int> input_2_ch3 {
        {
            { 0,  0,  0,  0},
            {-1, -1, -1, -1},
            {-2, -2, -2, -2}
        }
    };

    julie::la::DMatrix<int> input_1 {{input_1_ch1, input_1_ch2, input_1_ch3}};
    julie::la::DMatrix<int> input_2 {{input_2_ch1, input_2_ch2, input_2_ch3}};

    auto input = julie::la::DMatrix<int> {{input_1, input_2}};

    auto img2row_output = julie::la::img2row_2d(input, 1, 1, 3, 3);
    std::cout << img2row_output << std::endl;

    auto img2col_output = julie::la::img2col_2d(input, 1, 1, 3, 3);
    std::cout << img2col_output << std::endl;

    auto img2row_backward = julie::la::img2row_2d_backward(input.shape(), img2row_output, 1, 1, 3, 3);
    std::cout << img2row_backward << std::endl;

    auto img2col_backward = julie::la::img2col_2d_backward(input.shape(), img2col_output, 1, 1, 3, 3);
    std::cout << img2col_backward << std::endl;

    test::ASSERT(img2row_backward == img2col_backward);
}






void test_of_dmatrix_adv()
{
    test_pad_2d();
    img2row_test_filter_size();
    img2row_test_filter_stride();
    img2row_test_batch();
}

} // namespace test
