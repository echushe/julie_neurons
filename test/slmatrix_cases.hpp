#pragma once
#include "Matrix_CPU.hpp"
#include "SLMatrix.hpp"
#include "test_util.hpp"
#include "random_smatrix.hpp"

#include <iostream>

namespace test
{
    void SLMatrixTuple_Cases()
    {
        std::cout << "=================== Matrix Tuple Cases ==================\n";

        auto sh = julie::la::Shape{12, 18};

        julie::la::cpu::SLMatrixTuple<float> tuple_float {2, 9, 0.1};

        julie::la::cpu::SLMatrixTuple<int> tuple_int {2, 9, 1};

        tuple_float += 0.1;
        
        test::ASSERT(tuple_float.m_val == 0.1 + 0.1);

        tuple_float -= 0.2;

        test::ASSERT(tuple_float.m_val == 0.1 + 0.1 - 0.2);

        tuple_int *= 8;

        test::ASSERT(tuple_int.m_val == 8);

        tuple_int /= 2;

        std::cout << "tuple_int: " << tuple_int << std::endl;

        test::ASSERT(tuple_int.m_val == 4);

        julie::la::cpu::SLMatrixTuple<int> tuple_int1 = std::move(tuple_int);
        
        test::ASSERT(tuple_int.m_col == 0);

        test::ASSERT(tuple_int1.m_val == 4);

        test::ASSERT(tuple_int1.m_col == 9);

        auto tuple_int2 = tuple_int1 * 3;
        
        test::ASSERT(tuple_int2.m_val == 12);

        tuple_int2 = tuple_int2 / 4;

        test::ASSERT(tuple_int2.m_val == 3);

        auto tuple_int3 = 6 / tuple_int2;

        test::ASSERT(tuple_int3.m_val == 2);

        std::cout << "tuple_int3: " << tuple_int3 << std::endl;

        tuple_int3 = 8 - tuple_int2;

        test::ASSERT(tuple_int3.m_val == 5);
        
    }

    void SLMatrix_and_Matrix_CPU()
    {
        std::cout << "==================== SLMatrix_and_Matrix_CPU ====================\n";

        julie::la::cpu::Matrix_CPU<int> int_dmat{
            {{0, 0, 1},
             {0, 2, 0},
             {3, 0, 0}}};

        std::cout << "aaaa\n";

        julie::la::cpu::SLMatrix<int> int_smat{int_dmat};

        std::cout << int_smat << std::endl;

        std::cout << "bbbb\n";

        julie::la::cpu::Matrix_CPU<int> int_dmat_cp = int_smat.to_Matrix_CPU();

        std::cout << "cccc\n";

        test::ASSERT(int_dmat_cp == julie::la::cpu::Matrix_CPU<int> {{{0, 0, 1},
                                                          {0, 2, 0},
                                                          {3, 0, 0}}});
        
        std::cout << int_dmat << std::endl;

        std::cout << "dddd\n";

        std::cout << int_smat << std::endl;

        std::cout << "eeee\n";

        std::cout << int_dmat_cp << std::endl;

        std::cout << "ffff\n";
    }

    void SLMatrix_add()
    {
        std::cout << "==================== SLMatrix add ====================\n";

        julie::la::cpu::Matrix_CPU<int> dmat_a{
            {{0, 0, 1},
             {0, 2, 0},
             {3, 0, 0}}};

        julie::la::cpu::Matrix_CPU<int> dmat_b{
            {{1,  0,  0},
             {0, -2,  0},
             {0,  0, -4}}};

        julie::la::cpu::Matrix_CPU<int> tmp_mat_sum;
        julie::la::cpu::add(tmp_mat_sum, dmat_a, dmat_b);
        julie::la::cpu::SLMatrix<int> smat_sum {tmp_mat_sum};

        std::cout << smat_sum << std::endl;

        julie::la::cpu::Matrix_CPU<int> dmat_sum = smat_sum.to_Matrix_CPU();;
        test::ASSERT(dmat_sum == julie::la::cpu::Matrix_CPU<int>{  {{1,  0,  1},
                                                        {0,  0,  0},
                                                        {3,  0, -4}}});

        julie::la::cpu::SLMatrix<int> smat_a{dmat_a};
        julie::la::cpu::SLMatrix<int> smat_b{dmat_b};

        smat_sum = smat_a + smat_b;

        std::cout << smat_sum << std::endl;

        dmat_sum = smat_sum.to_Matrix_CPU();;
        test::ASSERT(dmat_sum == julie::la::cpu::Matrix_CPU<int>{  {{1,  0,  1},
                                                        {0,  0,  0},
                                                        {3,  0, -4}}});
    }


    void SLMatrix_sub()
    {
        std::cout << "==================== SLMatrix sub ====================\n";

        julie::la::cpu::Matrix_CPU<int> dmat_a{
            {{0, 0, 1},
             {0, 2, 0},
             {3, 0, 0}}};

        julie::la::cpu::Matrix_CPU<int> dmat_b{
            {{1,  0,  0},
             {0,  2,  0},
             {0,  0,  4}}};

        julie::la::cpu::Matrix_CPU<int> tmp_mat_sub;
        julie::la::cpu::subtract(tmp_mat_sub, dmat_a, dmat_b);
        julie::la::cpu::SLMatrix<int> smat_sub {tmp_mat_sub};

        std::cout << smat_sub << std::endl;

        julie::la::cpu::Matrix_CPU<int> dmat_sub = smat_sub.to_Matrix_CPU();;
        test::ASSERT(dmat_sub == julie::la::cpu::Matrix_CPU<int>{  {{-1,  0,  1},
                                                        { 0,  0,  0},
                                                        { 3,  0, -4}}});

        julie::la::cpu::SLMatrix<int> smat_a{dmat_a};
        julie::la::cpu::SLMatrix<int> smat_b{dmat_b};

        smat_sub = smat_a - smat_b;
        
        std::cout << smat_sub << std::endl;

        dmat_sub = smat_sub.to_Matrix_CPU();;
        test::ASSERT(dmat_sub == julie::la::cpu::Matrix_CPU<int>{  {{-1,  0,  1},
                                                        { 0,  0,  0},
                                                        { 3,  0, -4}}});
    }


    void SLMatrix_multiply()
    {
        std::cout << "==================== SLMatrix Multiply ====================\n";

        julie::la::cpu::Matrix_CPU<int> dmat_a{
            {{0, 0, 1},
             {0, 2, 0},
             {3, 0, 0}}};

        julie::la::cpu::Matrix_CPU<int> dmat_b{
            {{1,  0,  0},
             {0,  2,  0},
             {0,  0,  4}}};

        julie::la::cpu::Matrix_CPU<int> dmat_tmp;
        julie::la::cpu::multiply(dmat_tmp, dmat_a, dmat_b);
        julie::la::cpu::SLMatrix<int> smat_mul {dmat_tmp};

        std::cout << smat_mul << std::endl;

        julie::la::cpu::Matrix_CPU<int> dmat_mul = smat_mul.to_Matrix_CPU();
        test::ASSERT(dmat_mul == julie::la::cpu::Matrix_CPU<int>{
            {
                std::vector<int>{ 0,  0,  0},
                std::vector<int>{ 0,  4,  0},
                std::vector<int>{ 0,  0,  0}
            }});

        julie::la::cpu::SLMatrix<int> smat_a{dmat_a};
        julie::la::cpu::SLMatrix<int> smat_b{dmat_b};

        smat_mul = smat_a * smat_b;
        
        std::cout << smat_mul << std::endl;

        dmat_mul = smat_mul.to_Matrix_CPU();
        test::ASSERT(dmat_mul == julie::la::cpu::Matrix_CPU<int>{
            {
                std::vector<int>{ 0,  0,  0},
                std::vector<int>{ 0,  4,  0},
                std::vector<int>{ 0,  0,  0}
            }});
    }


    void SLMatrix_transpose()
    {
        std::cout << "==================== SLMatrix get_transpose ====================\n";

        julie::la::cpu::Matrix_CPU<int> dmat_a{
            {{0,  0,  1},
             {0,  2,  0},
             {3,  0,  0},
             {0,  0,  4},
             {0, -1, -2}}};

        julie::la::cpu::Matrix_CPU<int> dmat_b{
            std::vector<int>{
                1,  7,  0,  6,
                0,  2,  0,  0},
            julie::la::Shape{2, 4}};

        julie::la::cpu::Matrix_CPU<int> dmat_a_t; dmat_a.get_transpose(dmat_a_t, 1);
        julie::la::cpu::Matrix_CPU<int> dmat_b_t; dmat_b.get_transpose(dmat_b_t, 1);

        test::ASSERT(dmat_a_t == julie::la::cpu::Matrix_CPU<int>{  {{0,  0,  3,  0,  0},
                                                        {0,  2,  0,  0, -1},
                                                        {1,  0,  0,  4, -2}}});

        test::ASSERT(dmat_b_t == julie::la::cpu::Matrix_CPU<int>{  {{1,  0},
                                                        {7,  2},
                                                        {0,  0},
                                                        {6,  0}}});

        julie::la::cpu::SLMatrix<int> smat_a = dmat_a;
        julie::la::cpu::SLMatrix<int> smat_b = dmat_b;

        auto smat_a_t = smat_a.get_transpose();

        std::cout << smat_a_t << std::endl;

        auto smat_b_t = smat_b.get_transpose();

        std::cout << smat_b_t << std::endl;

        dmat_a_t = smat_a_t.to_Matrix_CPU();

        std::cout << dmat_a_t << std::endl;

        dmat_b_t = smat_b_t.to_Matrix_CPU();

        std::cout << dmat_b_t << std::endl;

        test::ASSERT(dmat_a_t == julie::la::cpu::Matrix_CPU<int>{  {{0,  0,  3,  0,  0},
                                                        {0,  2,  0,  0, -1},
                                                        {1,  0,  0,  4, -2}}});

        test::ASSERT(dmat_b_t == julie::la::cpu::Matrix_CPU<int>{  {{1,  0},
                                                        {7,  2},
                                                        {0,  0},
                                                        {6,  0}}});

    }

    void SLMatrix_MatMul()
    {
        std::cout << "==================== SLMatrix MatMul ====================\n";

        julie::la::cpu::Matrix_CPU<int> dmat_a{
            {
                std::vector<int>{ 0,  0,  1},
                std::vector<int>{ 0,  2,  0},
                std::vector<int>{ 0,  0,  0},
                std::vector<int>{ 0,  0,  4},
                std::vector<int>{ 0, -1, -2},
                std::vector<int>{ 0,  0,  0},
                std::vector<int>{ 0,  0,  0},
                std::vector<int>{ 0,  0,  0},
                std::vector<int>{ 0,  7,  0},
                std::vector<int>{ 0,  0, -3},
                std::vector<int>{ 0,  0,  0}
            }
        };

        julie::la::cpu::Matrix_CPU<int> dmat_b{
            {
                { 0,  0,  6,  0,  1,  0},
                { 0, -2,  2,  0,  0,  0},
                { 0,  3,  0,  0,  0,  8},
            }
        };

        julie::la::cpu::Matrix_CPU<int> dmat_c;
        julie::la::cpu::matmul(dmat_c, dmat_a, dmat_b);

        std::cout << dmat_c << std::endl;

        julie::la::cpu::SLMatrix<int> smat_a {dmat_a};
        julie::la::cpu::SLMatrix<int> smat_b {dmat_b};

        std::cout << "smat_a" << std::endl;
        std::cout << smat_a << std::endl;
        std::cout << "smat_b" << std::endl;
        std::cout << smat_b << std::endl;

        julie::la::cpu::SLMatrix<int> smat_c = julie::la::cpu::matmul(smat_a, smat_b);

        std::cout << smat_c << std::endl;

        std::cout << smat_c.to_Matrix_CPU() << std::endl;

        test::ASSERT(dmat_c == smat_c.to_Matrix_CPU());
    }

    void SLMatrix_random_mat_add()
    {
        std::cout << "==================== SLMatrix random mat add ====================\n";

        for (lint i = 1; i < 50; ++i)
        {
            for (lint j = 1; j < 50; ++j)
            {
                auto smat1 = test::RandSMatrix<int>::generate_random_SLMatrix(0.2, julie::la::Shape{i, j});
                auto dmat1 = smat1.to_Matrix_CPU();

                auto smat2 = test::RandSMatrix<int>::generate_random_SLMatrix(0.2, julie::la::Shape{i, j});
                auto dmat2 = smat2.to_Matrix_CPU();

                julie::la::cpu::SLMatrix<int> smat_sum = smat1 + smat2;
                julie::la::cpu::Matrix_CPU<int> dmat_sum;
                julie::la::cpu::add(dmat_sum, dmat1, dmat2);

                //std::cout << smat_sum << std::endl;
                //std::cout << smat_sum.to_Matrix_CPU() << std::endl;
                //std::cout << dmat_sum << std::endl;

                test::ASSERT(smat_sum.to_Matrix_CPU() == dmat_sum);
            }
        }
    }

    void SLMatrix_random_mat_sub()
    {
        std::cout << "==================== SLMatrix random mat sub ====================\n";

        for (lint i = 1; i < 50; ++i)
        {
            for (lint j = 1; j < 50; ++j)
            {
                auto smat1 = test::RandSMatrix<int>::generate_random_SLMatrix(0.2, julie::la::Shape{i, j});
                auto dmat1 = smat1.to_Matrix_CPU();

                auto smat2 = test::RandSMatrix<int>::generate_random_SLMatrix(0.2, julie::la::Shape{i, j});
                auto dmat2 = smat2.to_Matrix_CPU();

                julie::la::cpu::SLMatrix<int> smat_sub = smat1 - smat2;
                julie::la::cpu::Matrix_CPU<int> dmat_sub;
                julie::la::cpu::subtract(dmat_sub, dmat1, dmat2);

                test::ASSERT(smat_sub.to_Matrix_CPU() == dmat_sub);
            }
        }
    }

    void SLMatrix_random_mat_multiply()
    {
        std::cout << "==================== SLMatrix random mat multiply ====================\n";

        for (lint i = 1; i < 50; ++i)
        {
            for (lint j = 1; j < 50; ++j)
            {
                auto smat1 = test::RandSMatrix<int>::generate_random_SLMatrix(0.2, julie::la::Shape{i, j});
                auto dmat1 = smat1.to_Matrix_CPU();

                auto smat2 = test::RandSMatrix<int>::generate_random_SLMatrix(0.2, julie::la::Shape{i, j});
                auto dmat2 = smat2.to_Matrix_CPU();

                julie::la::cpu::SLMatrix<int> smat_mul = smat1 * smat2;
                julie::la::cpu::Matrix_CPU<int> dmat_mul;
                julie::la::cpu::multiply(dmat_mul, dmat1, dmat2);

                test::ASSERT(smat_mul.to_Matrix_CPU() == dmat_mul);
            }
        }
    }

    void SLMatrix_random_mat_MatMul()
    {
        std::cout << "==================== SLMatrix random mat MatMul ====================\n";

        for (lint i = 1; i < 20; ++i)
        {
            for (lint j = 1; j < 20; ++j)
            {
                auto smat1 = test::RandSMatrix<int>::generate_random_SLMatrix(0.2, julie::la::Shape{i, j});
                auto dmat1 = smat1.to_Matrix_CPU();

                for (lint k = 1; k < 20; ++k)
                {
                    // std::cout << "--------------------------- MATMUL ----------------------------\n";
                    auto smat2 = test::RandSMatrix<int>::generate_random_SLMatrix(0.2, julie::la::Shape{j, k});
                    auto dmat2 = smat2.to_Matrix_CPU();

                    julie::la::cpu::SLMatrix<int> smat_mul = julie::la::cpu::matmul(smat1, smat2);
                    julie::la::cpu::Matrix_CPU<int> dmat_mul;
                    julie::la::cpu::matmul(dmat_mul, dmat1, dmat2);

                    test::ASSERT(smat_mul.to_Matrix_CPU() == dmat_mul);

                    // std::cout << dmat1 << std::endl;
                    // std::cout << dmat2 << std::endl;

                    // std::cout << dmat_mul << std::endl;
                }
            }
        }
    }

    void SLMatrix_random_mat()
    {
        std::cout << "==================== SLMatrix random mat ====================\n";

        for (lint i = 1; i < 5; ++i)
        {
            auto smat1 = test::RandSMatrix<int>::generate_random_SLMatrix(0.2, julie::la::Shape{30, 20}, -1000, 1000);
            std::cout << smat1.to_Matrix_CPU() << std::endl;
        }

        for (lint i = 1; i < 5; ++i)
        {
            auto smat1 = test::RandSMatrix<int>::generate_random_SLMatrix(0.3, julie::la::Shape{30, 20}, 0, 1000);
            std::cout << smat1.to_Matrix_CPU() << std::endl;
        }

        for (lint i = 1; i < 5; ++i)
        {
            auto smat1 = test::RandSMatrix<float>::generate_random_SLMatrix(0.3, julie::la::Shape{15, 10}, -1000, 1000);
            std::cout << smat1.to_Matrix_CPU() << std::endl;
        }

        for (lint i = 1; i < 5; ++i)
        {
            auto smat1 = test::RandSMatrix<float>::generate_random_SLMatrix(0.5, julie::la::Shape{15, 6}, -1e10, 1e10);
            std::cout << smat1.to_Matrix_CPU() << std::endl;
        }
    }

    void SLMatrix_transpose_speed()
    {
        std::cout << "==================== SLMatrix transpose Speed ====================\n";

         
        for (int i = 0; i <= 999; i += 100)
        {
            auto smat1 = test::RandSMatrix<int>::generate_random_SLMatrix(0.001 * i, julie::la::Shape{1024, 1024}, -100, 100);
            
            lint time1 = test::get_time_in_milliseconds();
            auto smat1_t = smat1.get_transpose();
            lint time2 = test::get_time_in_milliseconds();

            std::cout << i << " Time spent in transpose: " << time2 - time1 << std::endl;
        }
    }


    void SLMatrix_MatMul_speed(float rate)
    {
        std::cout << "==================== SLMatrix MatMul Speed Rate: ";
        std::cout << rate << " ====================\n";


        std::vector<julie::la::cpu::SLMatrix<int>> smat1_list;
        std::vector<julie::la::cpu::SLMatrix<int>> smat2_list;
        std::vector<julie::la::cpu::SLMatrix<int>> smat_mul_list;

        std::vector<julie::la::cpu::Matrix_CPU<int>> dmat1_list;
        std::vector<julie::la::cpu::Matrix_CPU<int>> dmat2_list;
        std::vector<julie::la::cpu::Matrix_CPU<int>> dmat_mul_list;

        for (int i = 0; i < 1; ++i)
        {
            auto smat1 = test::RandSMatrix<int>::generate_random_SLMatrix(rate, julie::la::Shape{2048, 1024}, -100, 100);
            auto smat2 = test::RandSMatrix<int>::generate_random_SLMatrix(rate, julie::la::Shape{1024, 512}, -100, 100);
            
            smat1_list.push_back(smat1);
            smat2_list.push_back(smat2);

            auto dmat1 = smat1.to_Matrix_CPU();
            auto dmat2 = smat2.to_Matrix_CPU();

            dmat1_list.push_back(dmat1);
            dmat2_list.push_back(dmat2);
        }

        lint time1 = test::get_time_in_milliseconds();

        for (int i = 0; i < smat1_list.size(); ++i)
        {
            smat_mul_list.push_back(julie::la::cpu::matmul(smat1_list[i], smat2_list[i]));
        }

        lint time2 = test::get_time_in_milliseconds();

        for (int i = 0; i < dmat1_list.size(); ++i)
        {
            julie::la::cpu::Matrix_CPU<int> matmul_out;
            julie::la::cpu::matmul(matmul_out, dmat1_list[i], dmat2_list[i]);
            dmat_mul_list.push_back(matmul_out);
        }

        lint time3 = test::get_time_in_milliseconds();

        std::cout << "SMatrices used " << time2 - time1 << " milliseconds\n";
        std::cout << "DMatrices used " << time3 - time2 << " milliseconds\n";

        for (int i = 0; i < smat1_list.size(); ++i)
        {
            test::ASSERT(smat_mul_list[i].to_Matrix_CPU() == dmat_mul_list[i]);
        }
    }

    void SLMatrix_huge_MatMul_speed(float rate, lint size)
    {
        std::cout << "==================== Huge SMatrix2D MatMul Speed Rate: ";
        std::cout << rate << " ====================\n";

        auto smat1 = test::RandSMatrix<int>::generate_random_SLMatrix(rate, julie::la::Shape{size * size, size * size}, -100, 100);
        auto smat2 = test::RandSMatrix<int>::generate_random_SLMatrix(rate, julie::la::Shape{size * size, 1}, -100, 100);

        lint time1 = test::get_time_in_milliseconds();

        auto smat_mul = julie::la::cpu::matmul(smat1, smat2);

        lint time2 = test::get_time_in_milliseconds();

        std::cout << "SMatrices used " << time2 - time1 << " milliseconds\n";
    }


    void test_of_SLMatrix_operations()
    {
        SLMatrixTuple_Cases();
        SLMatrix_and_Matrix_CPU();
        SLMatrix_add();
        SLMatrix_sub();
        SLMatrix_multiply();
        SLMatrix_transpose();

        SLMatrix_MatMul();
        SLMatrix_random_mat_add();
        SLMatrix_random_mat_sub();
        SLMatrix_random_mat_multiply();
        SLMatrix_random_mat_MatMul();
        SLMatrix_random_mat();

        SLMatrix_transpose_speed();

        SLMatrix_MatMul_speed(0.0001);
        SLMatrix_MatMul_speed(0.001);
        SLMatrix_MatMul_speed(0.01);
        SLMatrix_MatMul_speed(0.1);
        //SLMatrix_MatMul_speed(0.9);

        SLMatrix_huge_MatMul_speed(0.01, 64);
        SLMatrix_huge_MatMul_speed(0.01, 128);
        SLMatrix_huge_MatMul_speed(0.001, 256);
        //SLMatrix_huge_MatMul_speed(0.0001, 512);
    }
    
} // namespace test