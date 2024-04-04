#pragma once
#include "Matrix_CPU.hpp"
#include "Matrix_CPU_func.hpp"
#include "test_util.hpp"

#include <iostream>
#include <vector>
#include <list>



namespace test
{
void test_matrix_constructor()
{
    std::cout << "=================== test_matrix_constructor ==================" << "\n";

    julie::la::cpu::Matrix_CPU<> mat1(3, julie::la::Shape{ 6 });
    std::cout << "mat1:\n" << mat1 << '\n';

    julie::la::cpu::Matrix_CPU<> mat2{3, julie::la::Shape{ 1, 6 }};
    std::cout << "mat2:\n" << mat2 << '\n';

    julie::la::cpu::Matrix_CPU<> mat3{4.4, julie::la::Shape{ 6, 1 }};
    std::cout << "mat3:\n" << mat3 << '\n';

    julie::la::cpu::Matrix_CPU<> mat4{-3, julie::la::Shape{ 6, 5, 4 }};
    std::cout << "mat4:\n" << mat4 << '\n';

    try
    {
        julie::la::cpu::Matrix_CPU<> mat6{-9, julie::la::Shape{ 3, 0, 9 }};
    }
    catch (std::exception & e)
    {
        std::cout << e.what() << '\n';
    }

    julie::la::cpu::Matrix_CPU<> mat7(std::vector<float>{ 1, 3, 5, 7, 9, 11, 13 }, false);
    std::cout << "mat7:\n" << mat7 << '\n';

    julie::la::cpu::Matrix_CPU<> mat8(std::vector<float>{ 1, 3, 5, 7, 9, 11, 13 }, true);
    std::cout << "mat8:\n" << mat8 << '\n';

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                mat4[{i, j, k}] = i * j * k;
            }
        }
    }

    julie::la::cpu::Matrix_CPU<> mat9{ mat4 };
    std::cout << "mat9:\n" << mat9 << '\n';

    julie::la::cpu::Matrix_CPU<> mat10{ std::move(mat4) };
    std::cout << "mat10:\n" << mat10 << '\n';
    std::cout << "mat4:\n" << mat4 << '\n';

    julie::la::cpu::Matrix_CPU<> mat11{-1, julie::la::Shape{ 9, 5 }};
    mat11 = mat10;
    std::cout << "mat11:\n" << mat11 << '\n';

    julie::la::cpu::Matrix_CPU<> mat12{-1, julie::la::Shape{ 2, 5 }};
    mat12 = std::move(mat10);
    std::cout << "mat12:\n" << mat12 << '\n';
    std::cout << "mat10:\n" << mat10 << '\n';

    julie::la::cpu::Matrix_CPU<> mat13{-1, julie::la::Shape{ 7, 8 }};
    julie::la::cpu::Matrix_CPU<> mat14{ 3, julie::la::Shape{ 7, 8 }};
    julie::la::cpu::Matrix_CPU<> mat15{ 9, { 7, 8 }};
    std::vector<julie::la::cpu::Matrix_CPU<>> vec;
    vec.push_back(mat13);
    vec.push_back(mat14);
    vec.push_back(mat15);

    julie::la::cpu::Matrix_CPU<> mat16{ vec };
    std::cout << "mat16:\n" << mat16 << '\n';

    julie::la::cpu::Matrix_CPU<> mat17{{{9, 8, 7}, 
                            {6, 5, 4},
                            {3, 2, 1}}};

    std::cout << "mat17:\n" << mat17 << '\n';

    julie::la::cpu::Matrix_CPU<> mat18{std::vector<float>{9, 8, 7, 3.3, 2.2, 1.1}, julie::la::Shape{2, 3}};
    std::cout << "mat18:\n" << mat18 << '\n';

    auto vect = std::vector<std::vector<float>>{{9, 8, 7}};
    julie::la::cpu::Matrix_CPU<> mat19{vect};
    std::cout << "mat19:\n" << mat19 << '\n';

    vect = std::vector<std::vector<float>>{{9}, {8}, {7}};
    julie::la::cpu::Matrix_CPU<> mat20{vect};
    std::cout << "mat20:\n" << mat20 << '\n';

    julie::la::cpu::Matrix_CPU<> mat21{{{9}, {8}, {7}}, true};
    std::cout << "mat21:\n" << mat21 << '\n';

    julie::la::cpu::Matrix_CPU<> mat22{{0, 1, 2, 3, 4}, false};
    std::cout << "mat22:\n" << mat22 << '\n';

    try
    {
        julie::la::cpu::Matrix_CPU<> mat23{{{9, 8, 7}, {3.3, 2.2, 1.1}, {99, 88}}};
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    try
    {
        vect = std::vector<std::vector<float>>{{}, {}, {}};
        julie::la::cpu::Matrix_CPU<> mat24{vect};
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    try
    {
        julie::la::cpu::Matrix_CPU<> mat25{{}, false};
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    try
    {
        vect = std::vector<std::vector<float>>{};
        julie::la::cpu::Matrix_CPU<> mat25{vect};
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
}


void test_matrix_indexing()
{
    std::cout << "=================== test_matrix_indexing ==================" << "\n";
    julie::la::cpu::Matrix_CPU<> mat1{8.88, julie::la::Shape{ 6, 7 }};
    std::cout << mat1 << '\n';

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 7; ++j)
        {
            mat1[{i, j}] = i * j;
        }
    }

    std::cout << mat1 << '\n';

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 7; ++j)
        {
            mat1[julie::la::Coordinate{ {i, j}, mat1.shape() }] = i - j;
        }
    }

    std::cout << mat1 << '\n';
}

void test_matrix_iterator()
{
    std::cout << "=================== test_matrix_iterator ==================" << "\n";
    julie::la::cpu::Matrix_CPU<> mat1{999.9, julie::la::Shape{ 3, 2, 4 } };
    
    // std::cout << mat1 << '\n';

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                mat1[{i, j, k}] = i + j * k;
            }
        }
    }

    std::cout << mat1 << '\n';

    for (auto itr = mat1.begin(); itr != mat1.end(); ++itr)
    {
        std::cout << *itr << " ";
    }

    std::cout << std::endl;
    auto itr = mat1.end();

    do
    {
        --itr;
        std::cout << *itr << " ";
    } while (itr != mat1.begin());

    std::cout << std::endl;

    for (auto & item : mat1)
    {
        std::cout << item << " ";
    }

    std::cout << std::endl;

    for (auto item : mat1)
    {
        std::cout << item << " ";
    }

    std::cout << std::endl;
}

void test_matrix_self_cal()
{
    std::cout << "=================== test_matrix_self_cal ==================" << "\n";
    julie::la::cpu::Matrix_CPU<> mat1{8, julie::la::Shape{ 6, 5, 5 }};
    julie::la::cpu::Matrix_CPU<> mat2{2, julie::la::Shape{ 6, 5, 5 }};
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            for (int k = 0; k < 5; ++k)
            {
                mat1[julie::la::Coordinate{ {i, j, k}, mat1.shape() }] = j - k;
            }
        }
    }

    mat1 += mat2;
    std::cout << "mat1:\n" << mat1 << '\n';

    mat1 -= mat2;
    std::cout << "mat1:\n" << mat1 << '\n';

    mat1 *= 2;
    std::cout << "mat1:\n" << mat1 << '\n';

    mat1 /= 2;
    std::cout << "mat1:\n" << mat1 << '\n';
}

void test_matrix_int_self_cal()
{
    std::cout << "=================== test_matrix_int_self_cal ==================" << "\n";
    julie::la::cpu::Matrix_CPU<int> mat1{8, julie::la::Shape{ 6, 5, 5 }};
    julie::la::cpu::Matrix_CPU<int> mat2{2, julie::la::Shape{ 6, 5, 5 }};
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            for (int k = 0; k < 5; ++k)
            {
                mat1[julie::la::Coordinate{ {i, j, k}, mat1.shape() }] = j - k;
            }
        }
    }

    mat1 += mat2;
    std::cout << "mat1:\n" << mat1 << '\n';

    mat1 -= mat2;
    std::cout << "mat1:\n" << mat1 << '\n';

    mat1 *= 2;
    std::cout << "mat1:\n" << mat1 << '\n';

    mat1 /= 2;
    std::cout << "mat1:\n" << mat1 << '\n';
}

void test_matrix_mul()
{
    std::cout << "=================== test_matmul ==================" << "\n";

    julie::la::cpu::Matrix_CPU<> mat1{ 33.333, julie::la::Shape{ 6, 5 } };
    julie::la::cpu::Matrix_CPU<> mat2{ 66.666, julie::la::Shape{ 5, 3 } };

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            mat1[{i, j}] = i * 5 + j;
        }
    }

    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            mat2[{i, j}] = i * 3 + j;
        }
    }

    julie::la::cpu::Matrix_CPU<> mat3;
    julie::la::cpu::matmul(mat3, mat1, mat2);

    std::cout << "mat3:\n" << mat3 << '\n';

    try
    {
        julie::la::cpu::Matrix_CPU<> mat4{ 7, julie::la::Shape{ 5 }};
        julie::la::cpu::Matrix_CPU<> mat5{ 4, julie::la::Shape{ 3 }};
        julie::la::cpu::matmul(mat3, mat4, mat5);
        std::cout << "mat3:\n" << mat3 << '\n';
    }
    catch (std::invalid_argument & ex)
    {
        std::cout << ex.what() << '\n';
    }

    julie::la::cpu::Matrix_CPU<> mat6{ 7, julie::la::Shape{ 5, 1 }};
    julie::la::cpu::Matrix_CPU<> mat7{ 4, julie::la::Shape{ 1, 3 }};
    julie::la::cpu::matmul(mat3, mat6, mat7);
    std::cout << "mat3:\n" << mat3 << '\n';

    julie::la::cpu::Matrix_CPU<> mat8{ 7, julie::la::Shape{ 1, 5 }};
    julie::la::cpu::Matrix_CPU<> mat9{ 4, julie::la::Shape{ 5, 1 }};
    julie::la::cpu::matmul(mat3, mat8, mat9);
    std::cout << "mat3:\n" << mat3 << '\n';

    julie::la::cpu::Matrix_CPU<> mat10{ -888, julie::la::Shape{ 7, 2, 6 }};
    julie::la::cpu::Matrix_CPU<> mat11{ -999, julie::la::Shape{ 3, 4, 5 }};

    for (int i = 0; i < 7; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < 6; ++k)
            {
                mat10[{i, j, k}] = i * j * k;
            }
        }
    }

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            for (int k = 0; k < 5; ++k)
            {
                mat11[{i, j, k}] = i * j * k;
            }
        }
    }

    julie::la::cpu::matmul(mat3, mat10, mat11);
    std::cout << "mat10:\n" << mat10 << '\n';
    std::cout << "mat11:\n" << mat11 << '\n';
    std::cout << "mat3:\n" << mat3 << '\n';

    julie::la::cpu::Matrix_CPU<> mat12{ 1, julie::la::Shape{ 5, 4, 6, 3 }};
    julie::la::cpu::Matrix_CPU<> mat13{ 1, julie::la::Shape{ 2, 9, 2, 2, 7 }};
    julie::la::cpu::matmul(mat3, mat12, mat13);
    std::cout << "mat12:\n" << mat12 << '\n';
    std::cout << "mat13:\n" << mat13 << '\n';
    std::cout << "mat3:\n" << mat3 << '\n';
    julie::la::cpu::matmul(mat3, mat12, mat13, 3, 4);
    std::cout << "mat3:\n" << mat3 << '\n';
}



void test_multiply_and_dot_product()
{
    std::cout << "=================== test_multiply_and_dot_product ==================" << "\n";

    julie::la::cpu::Matrix_CPU<> mat10{ 3, julie::la::Shape{ 4, 3 }};
    julie::la::cpu::Matrix_CPU<> mat11{ 5, julie::la::Shape{ 4, 3 }};

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            mat10[{i, j}] = i * 3 + j;
            mat11[{i, j}] = i * 3 + j;
        }
    }

    julie::la::cpu::Matrix_CPU<> mat3;
    julie::la::cpu::multiply(mat3, mat10, mat11);
    
    std::cout << "mat3:\n" << mat3 << '\n';

    julie::la::cpu::Matrix_CPU<> mat12{ 3, julie::la::Shape{ 4, 3 }};
    julie::la::cpu::Matrix_CPU<> mat13{ 2, julie::la::Shape{ 2, 3, 2, 1 }};

    julie::la::cpu::Matrix_CPU<> mul_cache;
    float dot = julie::la::cpu::dot_product(mul_cache, mat12, mat13);
    std::cout << "results of dot product:\n" << dot << '\n';

}

void test_matrix_dim_scale_up()
{
    std::cout << "=================== test_matrix_dim_scale_up ==================" << "\n";

    julie::la::cpu::Matrix_CPU<> mat1{ 1, julie::la::Shape{ 5, 6 }};
    std::vector<float> vec1{ 1, 2, 3, 4, 5, 6 };
    mat1.scale_one_dimension(1, vec1);

    std::cout << "mat1:\n" << mat1 << '\n';

    julie::la::cpu::Matrix_CPU<> mat2{ 1, julie::la::Shape{ 5, 6 }};
    std::vector<float> vec2{ 1, 2, 3, 4, 5 };
    mat2.scale_one_dimension(0, vec2);

    std::cout << "mat2:\n" << mat2 << '\n';

    julie::la::cpu::Matrix_CPU<> mat3{ 1, julie::la::Shape{ 4, 5, 6 }};
    std::vector<float> vec3{ 5, 4, 3, 2, 1 };
    mat3.scale_one_dimension(1, vec3);

    std::cout << "mat3:\n" << mat3 << '\n';

    julie::la::cpu::Matrix_CPU<> mat4{ 1, julie::la::Shape{ 4, 5, 6 }};
    std::vector<float> vec4{ 7, 6, 5, 4, 3, 2 };
    mat4.scale_one_dimension(2, vec4);

    std::cout << "mat3:\n" << mat4 << '\n';
}

void test_matrix_other_cal()
{
    std::cout << "=================== test_matrix_other_cal ==================" << "\n";

    julie::la::cpu::Matrix_CPU<> mat1{ -1e10, julie::la::Shape{ 6, 5 } };
    julie::la::cpu::Matrix_CPU<> mat2{ -1e11, julie::la::Shape{ 6, 5 } };

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            mat1[{i, j}] = i * 5 + j;
        }
    }

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            mat2[{i, j}] = (i * 5 + j) * (-1);
        }
    }

    julie::la::cpu::Matrix_CPU<> mat3;
    julie::la::cpu::add(mat3, mat1, mat2);
    std::cout << "mat1:\n" << mat1 << '\n';
    std::cout << "mat2:\n" << mat2 << '\n';
    std::cout << "mat3:\n" << mat3 << '\n';

    julie::la::cpu::subtract(mat3, mat1, mat2);
    std::cout << "mat3:\n" << mat3 << '\n';

    julie::la::cpu::Matrix_CPU<> mat_out;
    julie::la::cpu::multiply(mat_out, mat3, 3.0f);
    std::cout << "mat_out:\n" << mat_out << '\n';

    julie::la::cpu::divide(mat_out, mat3, 2.0f);
    std::cout << "mat_out:\n" << mat_out << '\n';
}

void test_matrix_random_normal()
{
    std::cout << "=================== test_matrix_random_normal ==================" << "\n";
    julie::la::cpu::Matrix_CPU<> mat1{ 1e10, julie::la::Shape{ 5, 4, 3 } };
    mat1.uniform_random(0, 1);
    std::cout << "mat1:\n" << mat1 << '\n';
    mat1.gaussian_random(0, 1);
    std::cout << "mat1:\n" << mat1 << '\n';
    mat1.normalize(0, 100);
    std::cout << "mat1:\n" << mat1 << '\n';
}


void test_matrix_transpose()
{
    std::cout << "=================== test_matrix_transpose ==================" << "\n";
    julie::la::cpu::Matrix_CPU<> mat1{ 1e11, julie::la::Shape{ 4, 3 } };
    mat1.uniform_random(0, 1);
    std::cout << "mat1:\n" << mat1 << '\n';
    julie::la::cpu::Matrix_CPU<> mat2;
    julie::la::cpu::transpose(mat2, mat1, 1);
    std::cout << "mat2 is transpose of mat1:\n" << mat2 << '\n';
}


void test_matrix_collapse()
{
    std::cout << "=================== test_matrix_collapse ==================" << "\n";
    julie::la::cpu::Matrix_CPU<> mat1{ 1e12, julie::la::Shape{ 5, 4, 3, 6 } };
    lint index = 0;
    for (lint i = 0; i < 5; ++i)
    {
        for (lint j = 0; j < 4; ++j)
        {
            for (lint k = 0; k < 3; ++k)
            {
                for (lint l = 0; l < 6; ++l)
                {
                    mat1[{i, j, k, l}] = static_cast<float>(index++);
                }
            }
        }
    }

    std::vector<julie::la::cpu::Matrix_CPU<>> vec_1 = mat1.get_collapsed(0);
    std::vector<julie::la::cpu::Matrix_CPU<>> vec_2 = mat1.get_collapsed(1);
    std::vector<julie::la::cpu::Matrix_CPU<>> vec_3 = mat1.get_collapsed(2);
    std::vector<julie::la::cpu::Matrix_CPU<>> vec_4 = mat1.get_collapsed(3);

    for (size_t i = 0; i < vec_1.size(); ++i)
    {
        std::cout << vec_1[i] << '\n';
    }

    test::ASSERT(vec_1[0].shape() == julie::la::Shape{4, 3, 6});

    std::cout << "-------------------------------------" << '\n';

    for (size_t i = 0; i < vec_2.size(); ++i)
    {
        std::cout << vec_2[i] << '\n';
    }

    test::ASSERT(vec_2[0].shape() == julie::la::Shape{5, 3, 6});

    std::cout << "-------------------------------------" << '\n';

    for (size_t i = 0; i < vec_3.size(); ++i)
    {
        std::cout << vec_3[i] << '\n';
    }

    test::ASSERT(vec_3[0].shape() == julie::la::Shape{5, 4, 6});
    std::cout << "-------------------------------------" << '\n';

    for (size_t i = 0; i < vec_4.size(); ++i)
    {
        std::cout << vec_4[i] << '\n';
    }

    test::ASSERT(vec_4[0].shape() == julie::la::Shape{5, 4, 3});
}


void test_matrix_fuse()
{
    std::cout << "=================== test_matrix_fuse ==================" << "\n";
    julie::la::cpu::Matrix_CPU<> mat1{ -1e9, julie::la::Shape{ 5, 4, 3, 6 } };
    lint index = 0;
    for (lint i = 0; i < 5; ++i)
    {
        for (lint j = 0; j < 4; ++j)
        {
            for (lint k = 0; k < 3; ++k)
            {
                for (lint l = 0; l < 6; ++l)
                {
                    mat1[{i, j, k, l}] = static_cast<float>(index++);
                }
            }
        }
    }

    julie::la::cpu::Matrix_CPU<> f_1; mat1.get_reduce_sum(f_1, 0);
    julie::la::cpu::Matrix_CPU<> f_2; mat1.get_reduce_sum(f_2, 1);
    julie::la::cpu::Matrix_CPU<> f_3; mat1.get_reduce_sum(f_3, 2);
    julie::la::cpu::Matrix_CPU<> f_4; mat1.get_reduce_sum(f_4, 3);

    std::cout << "mat1:\n" << mat1 << '\n';
    std::cout << "fuse 1\n" << f_1 << '\n';
    std::cout << "fuse 2\n" << f_2 << '\n';
    std::cout << "fuse 3\n" << f_3 << '\n';
    std::cout << "fuse 4\n" << f_4 << '\n';
}

void test_adv_coordinate_cases()
{
    std::cout << "=================== Advanced coordinate cases ==================" << "\n";

    julie::la::Coordinate co_1{};
    test::ASSERT(0 == co_1.dim());

    julie::la::Coordinate co_2{{0, 0, 0, 1}, julie::la::Shape{4, 3, 2, 5}};
    std::cout << co_2 << std::endl;

    --co_2;
    test::ASSERT(co_2 == julie::la::Coordinate{{0, 0, 0, 0}, julie::la::Shape{4, 3, 2, 5}});
    std::cout << co_2 << std::endl;
    test::ASSERT(co_2 != julie::la::Coordinate{{0, 0, 0, 0}, julie::la::Shape{3, 4, 2, 5}});
    std::cout << co_2 << std::endl;

    --co_2;
    test::ASSERT(co_2 == julie::la::Coordinate{{3, 2, 1, 4}, julie::la::Shape{4, 3, 2, 5}});
    std::cout << co_2 << std::endl;

    for (int i = 0; i < 4 * 3 * 2 * 5 - 1; ++i)
    {
        --co_2;
        std::cout << co_2 << std::endl;
    }

    test::ASSERT(co_2 == julie::la::Coordinate{{0, 0, 0, 0}, julie::la::Shape{4, 3, 2, 5}});

    for (int i = 0; i < 4 * 3 * 2 * 5; ++i)
    {
        ++co_2;
        std::cout << co_2 << std::endl;
    }

    test::ASSERT(co_2 == julie::la::Coordinate{{0, 0, 0, 0}, julie::la::Shape{4, 3, 2, 5}});

    julie::la::Coordinate bound = julie::la::Coordinate{{7, 1, 0}, julie::la::Shape{8, 13, 6}};
    julie::la::Coordinate co_3 {{0, 0, 0}, julie::la::Shape{8, 13, 6}};
    julie::la::Coordinate co_4 {{7, 12, 5}, julie::la::Shape{8, 13, 6}};

    while (co_3 < bound)
    {
        co_3++;
        std::cout << co_3 << std::endl;
    }

    test::ASSERT(co_3 == bound);

    while (co_4 > bound)
    {
        co_4--;
        std::cout << co_4 << std::endl;
    }

    test::ASSERT(co_4 == bound);


    julie::la::Coordinate co_5 {julie::la::Shape{18, 3, 9}};
    co_5 = 110;

    test::ASSERT(co_5 == julie::la::Coordinate{{4, 0, 2}, {18, 3, 9}});
    std::cout << co_5 << std::endl;

    test::ASSERT(110 == co_5.index());

    auto co_6 = julie::la::Coordinate{120, {4, 3, 10}};
    test::ASSERT(co_6 == julie::la::Coordinate{{0, 0, 0}, {4, 3, 10}});

    co_6 = julie::la::Coordinate{251, {4, 3, 10}};
    test::ASSERT(co_6 == julie::la::Coordinate{{0, 1, 1}, {4, 3, 10}});

    co_6 = julie::la::Coordinate{-251, {4, 3, 10}};
    test::ASSERT(co_6 == julie::la::Coordinate{{3, 1, 9}, {4, 3, 10}});

    co_6 = 360;
    test::ASSERT(co_6 == julie::la::Coordinate{{0, 0, 0}, {4, 3, 10}});

    co_6 = -361;
    test::ASSERT(co_6 == julie::la::Coordinate{{3, 2, 9}, {4, 3, 10}});

    co_6 = 128;
    test::ASSERT(co_6 == julie::la::Coordinate{{0, 0, 8}, {4, 3, 10}});
}

void test_transpose_neighboring_dim_pair()
{
    std::cout << "=================== test_transpose_neighboring_dim_pair ==================" << "\n";

    julie::la::cpu::Matrix_CPU<int> mat{
        {
            { 1,  2,  3,  4},
            { 5,  6,  7,  8},
            { 9, 10, 11, 12},
            {13, 14, 15, 16},
            {17, 18, 19, 20}
        }
        };

    julie::la::cpu::Matrix_CPU<int> mat_trans1;
    julie::la::cpu::Matrix_CPU<int> mat_trans2;
    julie::la::cpu::Matrix_CPU<int> mat_trans3;

    julie::la::cpu::transpose_neighboring_dims(mat_trans1, mat, 0, 0, 1, 1);
    julie::la::cpu::transpose(mat_trans2, mat, 1);

    std::cout << mat_trans1 << std::endl;

    test::ASSERT(mat_trans1 == mat_trans2);

    julie::la::cpu::Matrix_CPU<int> mat_1{
        {
            { -1,  -2,  -3,  -4},
            { -5,  -6,  -7,  -8},
            { -9, -10, -11, -12},
            {-13, -14, -15, -16},
            {-17, -18, -19, -20}
        }
        };

    julie::la::cpu::Matrix_CPU<int> ch2_mat {{mat, mat_1}};

    julie::la::cpu::transpose_neighboring_dims(mat_trans1, ch2_mat, 0, 0, 1, 1);
    julie::la::cpu::transpose_neighboring_dims(mat_trans2, mat_trans1, 1, 1, 2, 2);
    julie::la::cpu::transpose(mat_trans3, ch2_mat, 1);

    std::cout << mat_trans2 << std::endl;

    test::ASSERT(mat_trans2 == mat_trans3);

}

void test_argmax_of_one_dimension()
{
    std::cout << "=================== test_argmax_of_one_dimension ==================" << "\n";

    julie::la::cpu::Matrix_CPU<int> mat_ch1{
        {
            {3, 0, 7,  9,  8},
            {9, 7, 0,  4,  1},
            {3, 3, 4,  8, 11},
            {0, 6, 5, 12,  2}
        }};

    julie::la::cpu::Matrix_CPU<int> mat_ch2{
        {
            {17, 8,  8, 22,  6},
            { 9, 7,  0,  4,  4},
            {33, 1, 12, 18,  2},
            {15, 0, 36, 17,  1}
        }};

    julie::la::cpu::Matrix_CPU<int> mat {{mat_ch1, mat_ch2}};
    std::cout << mat << std::endl;
    std::cout << mat.shape() << std::endl;

    auto coordinates = mat.argmax(1);

    std::cout << "argmax of dimension 1: " << std::endl;
    std::cout << coordinates[0] << std::endl;
    test::ASSERT(coordinates[0] == julie::la::Coordinate {{0, 1, 0}, {2, 4, 5}});
    std::cout << coordinates[1] << std::endl;
    test::ASSERT(coordinates[1] == julie::la::Coordinate {{0, 1, 1}, {2, 4, 5}});
    std::cout << coordinates[2] << std::endl;
    test::ASSERT(coordinates[2] == julie::la::Coordinate {{0, 0, 2}, {2, 4, 5}});
    std::cout << coordinates[3] << std::endl;
    test::ASSERT(coordinates[3] == julie::la::Coordinate {{0, 3, 3}, {2, 4, 5}});
    std::cout << coordinates[4] << std::endl;
    test::ASSERT(coordinates[4] == julie::la::Coordinate {{0, 2, 4}, {2, 4, 5}});

    std::cout << coordinates[5] << std::endl;
    test::ASSERT(coordinates[5] == julie::la::Coordinate {{1, 2, 0}, {2, 4, 5}});
    std::cout << coordinates[6] << std::endl;
    test::ASSERT(coordinates[6] == julie::la::Coordinate {{1, 0, 1}, {2, 4, 5}});
    std::cout << coordinates[7] << std::endl;
    test::ASSERT(coordinates[7] == julie::la::Coordinate {{1, 3, 2}, {2, 4, 5}});
    std::cout << coordinates[8] << std::endl;
    test::ASSERT(coordinates[8] == julie::la::Coordinate {{1, 0, 3}, {2, 4, 5}});
    std::cout << coordinates[9] << std::endl;
    test::ASSERT(coordinates[9] == julie::la::Coordinate {{1, 0, 4}, {2, 4, 5}});

    coordinates = mat.argmax(0);
    std::cout << "argmax of dimension 0: " << std::endl;
    std::vector<lint> idx_list { 
        1, 1, 1, 1, 0,
        0, 0, 0, 0, 1,
        1, 0, 1, 1, 0,
        1, 0, 1, 1, 0 };

    lint idx = 0;
    for (lint i = 0; i < 4; ++i)
    {
        for (lint j = 0; j < 5; ++j)
        {
            std::cout << coordinates[idx] << std::endl;
            test::ASSERT(coordinates[idx] == julie::la::Coordinate {{idx_list[idx], i, j}, {2, 4, 5}});
            ++idx;
        }
    }

    coordinates = mat.argmax(2);
    std::cout << "argmax of dimension 2: " << std::endl;
    idx_list = std::vector<lint> {3, 0, 4, 3, 3, 0, 0, 2};
    idx = 0;
    for (lint i = 0; i < 2; ++i)
    {
        for (lint j = 0; j < 4; ++j)
        {
            std::cout << coordinates[idx] << std::endl;
            test::ASSERT(coordinates[idx] == julie::la::Coordinate {{i, j, idx_list[idx]}, {2, 4, 5}});
            ++idx;
        }
    }
}


void test_argmin_of_one_dimension()
{
    std::cout << "=================== test_argmin_of_one_dimension ==================" << "\n";

    julie::la::cpu::Matrix_CPU<int> mat_ch1{
        {
            {3, 0, 7,  9,  8},
            {9, 7, 0,  4,  1},
            {3, 3, 4,  8, 11},
            {0, 6, 5, 12,  2}
        }};

    julie::la::cpu::Matrix_CPU<int> mat_ch2{
        {
            {17, 8,  8, 22,  6},
            { 9, 7,  0,  4,  4},
            {33, 1, 12, 18,  2},
            {15, 0, 36, 17,  1}
        }};

    julie::la::cpu::Matrix_CPU<int> mat {{mat_ch1, mat_ch2}};
    mat *= -1;
    std::cout << mat << std::endl;
    std::cout << mat.shape() << std::endl;

    auto coordinates = mat.argmin(1);

    std::cout << "argmin of dimension 1: " << std::endl;
    std::cout << coordinates[0] << std::endl;
    test::ASSERT(coordinates[0] == julie::la::Coordinate {{0, 1, 0}, {2, 4, 5}});
    std::cout << coordinates[1] << std::endl;
    test::ASSERT(coordinates[1] == julie::la::Coordinate {{0, 1, 1}, {2, 4, 5}});
    std::cout << coordinates[2] << std::endl;
    test::ASSERT(coordinates[2] == julie::la::Coordinate {{0, 0, 2}, {2, 4, 5}});
    std::cout << coordinates[3] << std::endl;
    test::ASSERT(coordinates[3] == julie::la::Coordinate {{0, 3, 3}, {2, 4, 5}});
    std::cout << coordinates[4] << std::endl;
    test::ASSERT(coordinates[4] == julie::la::Coordinate {{0, 2, 4}, {2, 4, 5}});

    std::cout << coordinates[5] << std::endl;
    test::ASSERT(coordinates[5] == julie::la::Coordinate {{1, 2, 0}, {2, 4, 5}});
    std::cout << coordinates[6] << std::endl;
    test::ASSERT(coordinates[6] == julie::la::Coordinate {{1, 0, 1}, {2, 4, 5}});
    std::cout << coordinates[7] << std::endl;
    test::ASSERT(coordinates[7] == julie::la::Coordinate {{1, 3, 2}, {2, 4, 5}});
    std::cout << coordinates[8] << std::endl;
    test::ASSERT(coordinates[8] == julie::la::Coordinate {{1, 0, 3}, {2, 4, 5}});
    std::cout << coordinates[9] << std::endl;
    test::ASSERT(coordinates[9] == julie::la::Coordinate {{1, 0, 4}, {2, 4, 5}});

    coordinates = mat.argmin(0);
    std::cout << "argmin of dimension 0: " << std::endl;
    std::vector<lint> idx_list { 
        1, 1, 1, 1, 0,
        0, 0, 0, 0, 1,
        1, 0, 1, 1, 0,
        1, 0, 1, 1, 0 };

    lint idx = 0;
    for (lint i = 0; i < 4; ++i)
    {
        for (lint j = 0; j < 5; ++j)
        {
            std::cout << coordinates[idx] << std::endl;
            test::ASSERT(coordinates[idx] == julie::la::Coordinate {{idx_list[idx], i, j}, {2, 4, 5}});
            ++idx;
        }
    }

    coordinates = mat.argmin(2);
    std::cout << "argmin of dimension 2: " << std::endl;
    idx_list = std::vector<lint> {3, 0, 4, 3, 3, 0, 0, 2};
    idx = 0;
    for (lint i = 0; i < 2; ++i)
    {
        for (lint j = 0; j < 4; ++j)
        {
            std::cout << coordinates[idx] << std::endl;
            test::ASSERT(coordinates[idx] == julie::la::Coordinate {{i, j, idx_list[idx]}, {2, 4, 5}});
            ++idx;
        }
    }
}

void test_concat()
{
    std::cout << "=================== test_concat ==================" << "\n";

    julie::la::cpu::Matrix_CPU<int> mat1{
        {
            {3, 0, 7,  9,  8},
            {9, 7, 0,  4,  1},
            {3, 3, 4,  8, 11},
            {0, 6, 5, 12,  2}
        }};

    julie::la::cpu::Matrix_CPU<int> mat2{
        {
            {17, 8,  8, 22,  6},
            { 9, 7,  0,  4,  4},
            {33, 1, 12, 18,  2}
        }};

    julie::la::cpu::Matrix_CPU<int> cat;

    std::cout << "concat of " << mat1.shape() << " and " << mat2.shape() << std::endl;

    julie::la::cpu::concatenate(cat, mat1, mat2, 0);
    test::ASSERT(cat == julie::la::cpu::Matrix_CPU<int> {
        {
            { 3, 0,  7,  9,  8},
            { 9, 7,  0,  4,  1},
            { 3, 3,  4,  8, 11},
            { 0, 6,  5, 12,  2},
            {17, 8,  8, 22,  6},
            { 9, 7,  0,  4,  4},
            {33, 1, 12, 18,  2}
        }
    });

    julie::la::cpu::Matrix_CPU<int> mat3{
        {
            {3, 0, 7},
            {9, 7, 0},
            {3, 3, 4},
            {0, 6, 5}
        }};

    julie::la::cpu::Matrix_CPU<int> mat4{
        {
            {17, 8,  8, 22,  6},
            { 9, 7,  0,  4,  4},
            {33, 1, 12, 18,  2},
            {15, 0, 36, 17,  1}
        }};

    std::cout << "concat of " << mat3.shape() << " and " << mat4.shape() << std::endl;

    julie::la::cpu::concatenate(cat, mat3, mat4, 1);
    test::ASSERT(cat == julie::la::cpu::Matrix_CPU<int> {
        {
            {3, 0, 7, 17, 8,  8, 22,  6},
            {9, 7, 0,  9, 7,  0,  4,  4},
            {3, 3, 4, 33, 1, 12, 18,  2},
            {0, 6, 5, 15, 0, 36, 17,  1}
        }
    });

    julie::la::cpu::Matrix_CPU<int> mat5{std::vector<int>{1, 2, 3, 4, 5}, true};
    mat5.reshape(julie::la::Shape{5});
    julie::la::cpu::Matrix_CPU<int> mat6{std::vector<int>{10, 11, 12, 13, 14, 15, 16, 17}, true};
    mat6.reshape(julie::la::Shape{8});

    std::cout << "concat of " << mat5.shape() << " and " << mat6.shape() << std::endl;
    julie::la::cpu::concatenate(cat, mat5, mat6, 0);

    test::ASSERT(cat == julie::la::cpu::Matrix_CPU<int> {std::vector<int>{1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17}, true}.reshape(julie::la::Shape{13}));

    julie::la::cpu::Matrix_CPU<int> mat7{std::vector<int>{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}, true};
    mat7.reshape(julie::la::Shape{3, 3, 3});
    julie::la::cpu::Matrix_CPU<int> mat8{std::vector<int>{
        -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18}, true};
    mat8.reshape(julie::la::Shape{3, 2, 3});

    std::cout << "concat of " << mat7.shape() << " and " << mat8.shape() << std::endl;
    julie::la::cpu::concatenate(cat, mat7, mat8, 1);

    test::ASSERT(cat == julie::la::cpu::Matrix_CPU<int> {std::vector<int>{
        1,  2,  3,  4,  5,  6,  7,  8,  9,  -1,  -2,  -3,  -4,  -5,  -6,
        10, 11, 12, 13, 14, 15, 16, 17, 18, -7,  -8,  -9,  -10, -11, -12,
        19, 20, 21, 22, 23, 24, 25, 26, 27, -13, -14, -15, -16, -17, -18}, true}.reshape(julie::la::Shape{3, 5, 3}));

    mat8.reshape(julie::la::Shape{3, 3, 2});
    std::cout << "concat of " << mat7.shape() << " and " << mat8.shape() << std::endl;
    julie::la::cpu::concatenate(cat, mat7, mat8, 2);

    test::ASSERT(cat == julie::la::cpu::Matrix_CPU<int> {std::vector<int>{
        1,  2,  3,  -1,  -2,
        4,  5,  6,  -3,  -4,
        7,  8,  9,  -5,  -6,
        10, 11, 12, -7,  -8,
        13, 14, 15, -9,  -10,
        16, 17, 18, -11, -12,
        19, 20, 21, -13, -14,
        22, 23, 24, -15, -16,
        25, 26, 27, -17, -18
        }, true}.reshape(julie::la::Shape{3, 3, 5}));

    mat8.reshape(julie::la::Shape{2, 3, 3});
    std::cout << "concat of " << mat7.shape() << " and " << mat8.shape() << std::endl;
    julie::la::cpu::concatenate(cat, mat7, mat8, 0);

    test::ASSERT(cat == julie::la::cpu::Matrix_CPU<int> {std::vector<int>{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18
        }, julie::la::Shape{5, 3, 3}});
}


void test_slice()
{
    std::cout << "=================== test_slice ==================" << "\n";
    julie::la::cpu::Matrix_CPU<int> input{
        std::vector<int>{
            1,  2,  3,  4,  5,
            6,  7,  8,  9,  10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20
        },
        julie::la::Shape{4, 5}};

    julie::la::cpu::Matrix_CPU<int> slice;
    julie::la::cpu::slice(slice, input, 0, 0, 3);
    std::cout << slice << std::endl;
    test::ASSERT(slice == julie::la::cpu::Matrix_CPU<int> {
        std::vector<int>{    
            1,  2,  3,  4,  5,
            6,  7,  8,  9,  10,
            11, 12, 13, 14, 15
        },
        julie::la::Shape{3, 5}});

    julie::la::cpu::slice(slice, input, 1, 2, 2);
    std::cout << slice << std::endl;
    test::ASSERT(slice == julie::la::cpu::Matrix_CPU<int> {
        std::vector<int>{    
            3,  4,
            8,  9,
            13, 14,
            18, 19
        },
        julie::la::Shape{4, 2}});

    input.reshape(julie::la::Shape{20});

    julie::la::cpu::slice(slice, input, 0, 13, 7);
    std::cout << slice << std::endl;
    test::ASSERT(slice == julie::la::cpu::Matrix_CPU<int> {
        std::vector<int>{    
            14, 15, 16, 17, 18, 19, 20
        },
        julie::la::Shape{7}});
}


void test_repeat()
{
    std::cout << "=================== test_repeat ==================" << "\n";
    julie::la::cpu::Matrix_CPU<int> input{
        std::vector<int>{
            1,  2,  3,  4,  5,
            6,  7,  8,  9,  10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20
        },
        julie::la::Shape{4, 5}};

    julie::la::cpu::Matrix_CPU<int> repeat;
    julie::la::cpu::repeat(repeat, input, 0, 3);
    std::cout << repeat << std::endl;
    test::ASSERT(repeat == julie::la::cpu::Matrix_CPU<int> {
        std::vector<int>{    
            1,  2,  3,  4,  5,
            6,  7,  8,  9,  10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20,
            1,  2,  3,  4,  5,
            6,  7,  8,  9,  10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20,
            1,  2,  3,  4,  5,
            6,  7,  8,  9,  10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20
        },
        julie::la::Shape{12, 5}});

    julie::la::cpu::repeat(repeat, input, 1, 2);
    std::cout << repeat << std::endl;
    test::ASSERT(repeat == julie::la::cpu::Matrix_CPU<int> {
        std::vector<int>{    
            1,  2,  3,  4,  5,  1,  2,  3,  4,  5,
            6,  7,  8,  9,  10, 6,  7,  8,  9,  10,
            11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 16, 17, 18, 19, 20
        },
        julie::la::Shape{4, 10}});

    input.reshape(julie::la::Shape{20});

    julie::la::cpu::repeat(repeat, input, 0, 4);
    std::cout << repeat << std::endl;
    test::ASSERT(repeat == julie::la::cpu::Matrix_CPU<int> {
        std::vector<int>{    
            1,  2,  3,  4,  5,
            6,  7,  8,  9,  10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20,
            1,  2,  3,  4,  5,
            6,  7,  8,  9,  10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20,
            1,  2,  3,  4,  5,
            6,  7,  8,  9,  10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20,
            1,  2,  3,  4,  5,
            6,  7,  8,  9,  10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20
        },
        julie::la::Shape{80}});
}


void test_of_Matrix_CPU_operations()
{
    test_matrix_constructor();

    test_matrix_indexing();

    test_matrix_iterator();

    test_matrix_self_cal();
    test_matrix_int_self_cal();

    test_matrix_mul();
    test_multiply_and_dot_product();

    test_matrix_dim_scale_up();
    test_matrix_other_cal();

    test_matrix_random_normal();

    test_matrix_transpose();

    test_matrix_collapse();
    test_matrix_fuse();

    test_adv_coordinate_cases();
    
    test_transpose_neighboring_dim_pair();

    test_argmax_of_one_dimension();
    test_argmin_of_one_dimension();

    test_concat();
    test_slice();
    test_repeat();
}

} // namespace julie
