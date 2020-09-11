#pragma once
#include "Vector.hpp"
#include "DMatrix.hpp"
#include "test_util.hpp"

#include <iostream>
#include <vector>
#include <list>



namespace test
{

void vector_cases()
{
    julie::la::Vector a(2);

    test::ASSERT(a == julie::la::Vector{0, 0});

    std::list<double> l{ 1,2,3 };
    julie::la::Vector b(l.begin(), l.end());
    
    test::ASSERT(b == julie::la::Vector{1, 2, 3});

    std::vector<double> v2{ 4,5,6,7 };
    julie::la::Vector c{ v2.begin(),v2.end() };

    test::ASSERT(c == julie::la::Vector{4, 5, 6, 7});

    std::vector<double> a1{ 5,4,3,2,1 };
    julie::la::Vector d{ a1.begin(),a1.end() };

    test::ASSERT(d == julie::la::Vector{5, 4, 3, 2, 1});

    std::list<double> a2{ 9,0,8,6,7 };
    julie::la::Vector e{ a2.begin(),a2.end() };

    test::ASSERT(e == julie::la::Vector{9, 0, 8, 6, 7});

    // use the copy constructor
    julie::la::Vector f{ e };

    test::ASSERT(f == julie::la::Vector{9, 0, 8, 6, 7});

    test::ASSERT(2 == a.getNumDimensions());

    std::cout << c << " Euclidean Norm = " << c.getEuclideanNorm() << "\n";
    std::cout << d << " Unit Vector: " << d.createUnitVector() << " L = " << d.createUnitVector().getEuclideanNorm() << "\n";
    std::cout << e << "\n";
    std::cout << f << "\n";

    // test the move constructor
    julie::la::Vector g = std::move(f);
    std::cout << g << "\n";
    std::cout << f << "\n";

    test::ASSERT(0 == f.dim());
    test::ASSERT(g == julie::la::Vector{9, 0, 8, 6, 7});

    // try operator overloading
    e += d;
    std::cout << e << "\n";

    julie::la::Vector h = e - g;
    std::cout << h << "\n";

    // test scalar multiplication
    h *= 2;
    std::cout << h << "\n";

    julie::la::Vector j = b / 2;
    std::cout << j << "\n";

    std::cout << "dot product = " << j * b << "\n";

    if (g == (e - d)) std::cout << "true" << "\n";
    if (j != b) std::cout << "false" << "\n";

    j[0] = 1;
    std::cout << j << "\n";

    // type cast from Vector to a std::vector
    std::vector<double> vj = j;

    // type cast from Vector to a std::vector
    std::list<double> lj = j;

    for (auto d : lj)
    {
        std::cout << d << "\n";
    }

    // list initialisation
    julie::la::Vector k{ 1, 2, 3 };
    std::cout << k << "\n";

    std::cout << "========================= my own test cases =========================" << "\n";

    julie::la::Vector scn(2, 7);
    julie::la::Vector scn0{ 3, 3 };
    julie::la::Vector scn1{ 9, 8, 7, 6, 5 };
    julie::la::Vector scn2{ 1, 2, 3, 4, 5, 6 };
    julie::la::Vector scn3(scn1);
    julie::la::Vector scn4;

    std::cout << scn << "\n";
    scn0 = std::move(scn1);
    std::cout << scn0 << "\n";
    std::cout << scn1 << "\n";

    scn0 = -1 * scn0;
    std::cout << scn0 << "\n";
    scn0 = scn2 * (scn3 * scn0);
    std::cout << scn0 << "\n";
    std::cout << scn3 << "\n";

    scn4 = scn2;

    std::cout << scn4 << "\n";

    scn4 = (scn2 + (scn0 - scn2) * 3.33) / 12.33;

    std::cout << scn4 << "\n";

    scn4 += scn2;

    std::cout << scn4 << "\n";

    scn4 *= 2;

    std::cout << scn4 << "\n";

    scn4 /= 100;

    std::cout << scn4 << "\n";

    scn4 -= scn0;

    std::cout << scn4 << "\n";

    scn4 = scn4.createUnitVector();

    std::cout << scn4 << "\n";

    try
    {
        julie::la::Vector scn5 = scn + scn2;
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        scn += scn2;
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        julie::la::Vector scn5 = scn - scn2;
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        scn -= scn2;
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        double dot = scn * scn2;
        std::cout << dot << "\n";
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        scn1 = scn1;
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        scn1 = std::move(scn1);
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        std::cout << scn0[100] << "\n";
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        std::cout << scn0.get(-2) << "\n";
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        std::cout << (scn0 /= 0) << "\n";
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        std::cout << (scn0 / 0) << "\n";
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }

    try
    {
        julie::la::Vector zero_v{ 0, 0, 0, 0, 0, 0 };
        std::cout << (zero_v.createUnitVector()) << "\n";
    }
    catch (std::exception &ex)
    {
        std::cout << ex.what() << "\n";
    }
}


void test_matrix_constructor()
{
    std::cout << "=================== test_matrix_constructor ==================" << "\n";

    julie::la::DMatrix<> mat1(3, julie::la::Shape{ 6 });
    std::cout << "mat1:\n" << mat1 << '\n';

    julie::la::DMatrix<> mat2{3, julie::la::Shape{ 1, 6 }};
    std::cout << "mat2:\n" << mat2 << '\n';

    julie::la::DMatrix<> mat3{4.4, julie::la::Shape{ 6, 1 }};
    std::cout << "mat3:\n" << mat3 << '\n';

    julie::la::DMatrix<> mat4{-3, julie::la::Shape{ 6, 5, 4 }};
    std::cout << "mat4:\n" << mat4 << '\n';

    try
    {
        julie::la::DMatrix<> mat6{-9, julie::la::Shape{ 3, 0, 9 }};
    }
    catch (std::exception & e)
    {
        std::cout << e.what() << '\n';
    }

    julie::la::DMatrix<> mat7(julie::la::Vector{ 1, 3, 5, 7, 9, 11, 13 }, false);
    std::cout << "mat7:\n" << mat7 << '\n';

    julie::la::DMatrix<> mat8(julie::la::Vector{ 1, 3, 5, 7, 9, 11, 13 }, true);
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

    julie::la::DMatrix<> mat9{ mat4 };
    std::cout << "mat9:\n" << mat9 << '\n';

    julie::la::DMatrix<> mat10{ std::move(mat4) };
    std::cout << "mat10:\n" << mat10 << '\n';
    std::cout << "mat4:\n" << mat4 << '\n';

    julie::la::DMatrix<> mat11{-1, julie::la::Shape{ 9, 5 }};
    mat11 = mat10;
    std::cout << "mat11:\n" << mat11 << '\n';

    julie::la::DMatrix<> mat12{-1, julie::la::Shape{ 2, 5 }};
    mat12 = std::move(mat10);
    std::cout << "mat12:\n" << mat12 << '\n';
    std::cout << "mat10:\n" << mat10 << '\n';

    julie::la::DMatrix<> mat13{-1, julie::la::Shape{ 7, 8 }};
    julie::la::DMatrix<> mat14{ 3, julie::la::Shape{ 7, 8 }};
    julie::la::DMatrix<> mat15{ 9, { 7, 8 }};
    std::vector<julie::la::DMatrix<>> vec;
    vec.push_back(mat13);
    vec.push_back(mat14);
    vec.push_back(mat15);

    julie::la::DMatrix<> mat16{ vec };
    std::cout << "mat16:\n" << mat16 << '\n';

    julie::la::DMatrix<> mat17{{{9, 8, 7}, 
                            {6, 5, 4},
                            {3, 2, 1}}};

    std::cout << "mat17:\n" << mat17 << '\n';

    julie::la::DMatrix<> mat18{{{9, 8, 7}, {3.3, 2.2, 1.1}}};
    std::cout << "mat18:\n" << mat18 << '\n';

    auto vect = std::vector<std::vector<double>>{{9, 8, 7}};
    julie::la::DMatrix<> mat19{vect};
    std::cout << "mat19:\n" << mat19 << '\n';

    vect = std::vector<std::vector<double>>{{9}, {8}, {7}};
    julie::la::DMatrix<> mat20{vect};
    std::cout << "mat20:\n" << mat20 << '\n';

    julie::la::DMatrix<> mat21{{{9}, {8}, {7}}, true};
    std::cout << "mat21:\n" << mat21 << '\n';

    julie::la::DMatrix<> mat22{{0, 1, 2, 3, 4}, false};
    std::cout << "mat22:\n" << mat22 << '\n';

    try
    {
        julie::la::DMatrix<> mat23{{{9, 8, 7}, {3.3, 2.2, 1.1}, {99, 88}}};
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    try
    {
        vect = std::vector<std::vector<double>>{{}, {}, {}};
        julie::la::DMatrix<> mat24{vect};
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    try
    {
        julie::la::DMatrix<> mat25{{}, false};
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    try
    {
        vect = std::vector<std::vector<double>>{};
        julie::la::DMatrix<> mat25{vect};
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
}


void test_matrix_indexing()
{
    std::cout << "=================== test_matrix_indexing ==================" << "\n";
    julie::la::DMatrix<> mat1{8.88, julie::la::Shape{ 6, 7 }};
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
    julie::la::DMatrix<> mat1{999.9, julie::la::Shape{ 3, 2, 4 } };
    
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
    julie::la::DMatrix<> mat1{8, julie::la::Shape{ 6, 5, 5 }};
    julie::la::DMatrix<> mat2{2, julie::la::Shape{ 6, 5, 5 }};
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

    julie::la::DMatrix<> mat1{ 33.333, julie::la::Shape{ 6, 5 } };
    julie::la::DMatrix<> mat2{ 66.666, julie::la::Shape{ 5, 3 } };

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

    julie::la::DMatrix<> mat3 = julie::la::matmul(mat1, mat2);

    std::cout << "mat3:\n" << mat3 << '\n';

    try
    {
        julie::la::DMatrix<> mat4{ 7, julie::la::Shape{ 5 }};
        julie::la::DMatrix<> mat5{ 4, julie::la::Shape{ 3 }};
        mat3 = julie::la::matmul(mat4, mat5);
        std::cout << "mat3:\n" << mat3 << '\n';
    }
    catch (std::invalid_argument & ex)
    {
        std::cout << ex.what() << '\n';
    }

    julie::la::DMatrix<> mat6{ 7, julie::la::Shape{ 5, 1 }};
    julie::la::DMatrix<> mat7{ 4, julie::la::Shape{ 1, 3 }};
    mat3 = julie::la::matmul(mat6, mat7);
    std::cout << "mat3:\n" << mat3 << '\n';

    julie::la::DMatrix<> mat8{ 7, julie::la::Shape{ 1, 5 }};
    julie::la::DMatrix<> mat9{ 4, julie::la::Shape{ 5, 1 }};
    mat3 = julie::la::matmul(mat8, mat9);
    std::cout << "mat3:\n" << mat3 << '\n';

    julie::la::DMatrix<> mat10{ -888, julie::la::Shape{ 7, 2, 6 }};
    julie::la::DMatrix<> mat11{ -999, julie::la::Shape{ 3, 4, 5 }};

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

    mat3 = julie::la::matmul(mat10, mat11);
    std::cout << "mat10:\n" << mat10 << '\n';
    std::cout << "mat11:\n" << mat11 << '\n';
    std::cout << "mat3:\n" << mat3 << '\n';

    julie::la::DMatrix<> mat12{ 1, julie::la::Shape{ 5, 4, 6, 3 }};
    julie::la::DMatrix<> mat13{ 1, julie::la::Shape{ 2, 9, 2, 2, 7 }};
    mat3 = julie::la::matmul(mat12, mat13);
    std::cout << "mat12:\n" << mat12 << '\n';
    std::cout << "mat13:\n" << mat13 << '\n';
    std::cout << "mat3:\n" << mat3 << '\n';
    mat3 = julie::la::matmul(mat12, mat13, 3, 4);
    std::cout << "mat3:\n" << mat3 << '\n';
}



void test_multiply_and_dot_product()
{
    std::cout << "=================== test_multiply_and_dot_product ==================" << "\n";

    julie::la::DMatrix<> mat10{ 3, julie::la::Shape{ 4, 3 }};
    julie::la::DMatrix<> mat11{ 5, julie::la::Shape{ 4, 3 }};

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            mat10[{i, j}] = i * 3 + j;
            mat11[{i, j}] = i * 3 + j;
        }
    }

    julie::la::DMatrix<> mat3 = julie::la::multiply(mat10, mat11);
    std::cout << "mat3:\n" << mat3 << '\n';

    julie::la::DMatrix<> mat12{ 3, julie::la::Shape{ 4, 3 }};
    julie::la::DMatrix<> mat13{ 2, julie::la::Shape{ 2, 3, 2, 1 }};
    double dot = julie::la::dot_product(mat12, mat13);
    std::cout << "results of dot product:\n" << dot << '\n';

}

void test_matrix_dim_scale_up()
{
    std::cout << "=================== test_matrix_dim_scale_up ==================" << "\n";

    julie::la::DMatrix<> mat1{ 1, julie::la::Shape{ 5, 6 }};
    julie::la::Vector vec1{ 1, 2, 3, 4, 5, 6 };
    mat1.scale_one_dimension(1, vec1);

    std::cout << "mat1:\n" << mat1 << '\n';

    julie::la::DMatrix<> mat2{ 1, julie::la::Shape{ 5, 6 }};
    julie::la::Vector vec2{ 1, 2, 3, 4, 5 };
    mat2.scale_one_dimension(0, vec2);

    std::cout << "mat2:\n" << mat2 << '\n';

    julie::la::DMatrix<> mat3{ 1, julie::la::Shape{ 4, 5, 6 }};
    julie::la::Vector vec3{ 5, 4, 3, 2, 1 };
    mat3.scale_one_dimension(1, vec3);

    std::cout << "mat3:\n" << mat3 << '\n';

    julie::la::DMatrix<> mat4{ 1, julie::la::Shape{ 4, 5, 6 }};
    julie::la::Vector vec4{ 7, 6, 5, 4, 3, 2 };
    mat4.scale_one_dimension(2, vec4);

    std::cout << "mat3:\n" << mat4 << '\n';
}

void test_matrix_other_cal()
{
    std::cout << "=================== test_matrix_other_cal ==================" << "\n";

    julie::la::DMatrix<> mat1{ -1e10, julie::la::Shape{ 6, 5 } };
    julie::la::DMatrix<> mat2{ -1e11, julie::la::Shape{ 6, 5 } };

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

    julie::la::DMatrix<> mat3 = mat1 + mat2;
    std::cout << "mat1:\n" << mat1 << '\n';
    std::cout << "mat2:\n" << mat2 << '\n';
    std::cout << "mat3:\n" << mat3 << '\n';

    mat3 = mat1 - mat2;
    std::cout << "mat3:\n" << mat3 << '\n';

    mat3 = mat3 * 3.0;
    std::cout << "mat3:\n" << mat3 << '\n';

    mat3 = 2.0 * mat3;
    std::cout << "mat3:\n" << mat3 << '\n';

    mat3 = mat3 / 2.0;
    std::cout << "mat3:\n" << mat3 << '\n';
}

void test_matrix_random_normal()
{
    std::cout << "=================== test_matrix_random_normal ==================" << "\n";
    julie::la::DMatrix<> mat1{ 1e10, julie::la::Shape{ 5, 4, 3 } };
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
    julie::la::DMatrix<> mat1{ 1e11, julie::la::Shape{ 4, 3 } };
    mat1.uniform_random(0, 1);
    std::cout << "mat1:\n" << mat1 << '\n';
    julie::la::DMatrix<> mat2 = julie::la::transpose(mat1, 1);
    std::cout << "mat2 is transpose of mat1:\n" << mat2 << '\n';
}


void test_matrix_collapse()
{
    std::cout << "=================== test_matrix_collapse ==================" << "\n";
    julie::la::DMatrix<> mat1{ 1e12, julie::la::Shape{ 5, 4, 3, 6 } };
    lint index = 0;
    for (lint i = 0; i < 5; ++i)
    {
        for (lint j = 0; j < 4; ++j)
        {
            for (lint k = 0; k < 3; ++k)
            {
                for (lint l = 0; l < 6; ++l)
                {
                    mat1[{i, j, k, l}] = static_cast<double>(index++);
                }
            }
        }
    }

    std::vector<julie::la::DMatrix<>> vec_1 = mat1.get_collapsed(0);
    std::vector<julie::la::DMatrix<>> vec_2 = mat1.get_collapsed(1);
    std::vector<julie::la::DMatrix<>> vec_3 = mat1.get_collapsed(2);
    std::vector<julie::la::DMatrix<>> vec_4 = mat1.get_collapsed(3);

    for (size_t i = 0; i < vec_1.size(); ++i)
    {
        std::cout << vec_1[i] << '\n';
    }
    std::cout << "-------------------------------------" << '\n';

    for (size_t i = 0; i < vec_2.size(); ++i)
    {
        std::cout << vec_2[i] << '\n';
    }
    std::cout << "-------------------------------------" << '\n';

    for (size_t i = 0; i < vec_3.size(); ++i)
    {
        std::cout << vec_3[i] << '\n';
    }
    std::cout << "-------------------------------------" << '\n';

    for (size_t i = 0; i < vec_4.size(); ++i)
    {
        std::cout << vec_4[i] << '\n';
    }
}


void test_matrix_fuse()
{
    std::cout << "=================== test_matrix_fuse ==================" << "\n";
    julie::la::DMatrix<> mat1{ -1e9, julie::la::Shape{ 5, 4, 3, 6 } };
    lint index = 0;
    for (lint i = 0; i < 5; ++i)
    {
        for (lint j = 0; j < 4; ++j)
        {
            for (lint k = 0; k < 3; ++k)
            {
                for (lint l = 0; l < 6; ++l)
                {
                    mat1[{i, j, k, l}] = static_cast<double>(index++);
                }
            }
        }
    }

    julie::la::DMatrix<> f_1 = mat1.get_fused(0);
    julie::la::DMatrix<> f_2 = mat1.get_fused(1);
    julie::la::DMatrix<> f_3 = mat1.get_fused(2);
    julie::la::DMatrix<> f_4 = mat1.get_fused(3);

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
}

void test_of_dmatrix_operations()
{
    vector_cases();

    test_matrix_constructor();

    test_matrix_indexing();

    test_matrix_iterator();

    test_matrix_self_cal();
    test_matrix_mul();
    test_multiply_and_dot_product();

    test_matrix_dim_scale_up();
    test_matrix_other_cal();

    test_matrix_random_normal();

    /*
    test_matrix_transpose();

    test_matrix_collapse();
    test_matrix_fuse();

    test_adv_coordinate_cases();
    */
}

} // namespace julie
