#pragma once
#include "Matrix_CUDA.hpp"
#include "Matrix_CUDA_func.hpp"
#include "Matrix_CPU.hpp"
#include "Matrix_CPU_func.hpp"
#include "test_util.hpp"

#include <math.h>
#include <iostream>
#include <vector>
#include <list>



namespace test
{

void test_matrix_gpu_constructor()
{
    std::cout << "=================== test_matrix_gpu_constructor ==================" << "\n";

    julie::la::cuda::Matrix_CUDA<> mat1(3, julie::la::Shape{ 6 });
    std::cout << "mat1:\n" << mat1 << '\n';

    julie::la::cuda::Matrix_CUDA<> mat2{3, julie::la::Shape{ 1, 6 }};
    std::cout << "mat2:\n" << mat2 << '\n';

    julie::la::cuda::Matrix_CUDA<> mat3{4.4, julie::la::Shape{ 6, 1 }};
    std::cout << "mat3:\n" << mat3 << '\n';

    julie::la::cuda::Matrix_CUDA<> mat4{-3, julie::la::Shape{ 6, 5, 4 }};
    std::cout << "mat4:\n" << mat4 << '\n';

    try
    {
        julie::la::cuda::Matrix_CUDA<> mat6{-9, julie::la::Shape{ 3, 0, 9 }};
    }
    catch (std::exception & e)
    {
        std::cout << e.what() << '\n';
    }

    julie::la::cuda::Matrix_CUDA<> mat7(std::vector<float>{ 1, 3, 5, 7, 9, 11, 13 }, false);
    std::cout << "mat7:\n" << mat7 << '\n';

    julie::la::cuda::Matrix_CUDA<> mat8(std::vector<float>{ 1, 3, 5, 7, 9, 11, 13 }, true);
    std::cout << "mat8:\n" << mat8 << '\n';

    julie::la::cpu::Matrix_CPU<> mat4_cpu {mat4};
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                mat4_cpu[{i, j, k}] = i * j * k;
            }
        }
    }
    mat4 = mat4_cpu.get_CUDA();

    julie::la::cuda::Matrix_CUDA<> mat9{ mat4 };
    std::cout << "mat9:\n" << mat9 << '\n';

    julie::la::cuda::Matrix_CUDA<> mat10{ std::move(mat4) };
    std::cout << "mat10:\n" << mat10 << '\n';
    std::cout << "mat4:\n" << mat4 << '\n';

    julie::la::cuda::Matrix_CUDA<> mat11{-1, julie::la::Shape{ 9, 5 }};
    mat11 = mat10;
    std::cout << "mat11:\n" << mat11 << '\n';

    julie::la::cuda::Matrix_CUDA<> mat12{-1, julie::la::Shape{ 2, 5 }};
    mat12 = std::move(mat10);
    std::cout << "mat12:\n" << mat12 << '\n';
    std::cout << "mat10:\n" << mat10 << '\n';

    julie::la::cuda::Matrix_CUDA<> mat13{-1, julie::la::Shape{ 7, 8 }};
    julie::la::cuda::Matrix_CUDA<> mat14{ 3, julie::la::Shape{ 7, 8 }};
    julie::la::cuda::Matrix_CUDA<> mat15{ 9, { 7, 8 }};
    std::vector<julie::la::cuda::Matrix_CUDA<>> vec;
    vec.push_back(mat13);
    vec.push_back(mat14);
    vec.push_back(mat15);

    julie::la::cuda::Matrix_CUDA<> mat16{ vec };
    std::cout << "mat16:\n" << mat16 << '\n';

    julie::la::cuda::Matrix_CUDA<> mat17{{{9, 8, 7}, 
                            {6, 5, 4},
                            {3, 2, 1}}};

    std::cout << "mat17:\n" << mat17 << '\n';

    julie::la::cuda::Matrix_CUDA<> mat18{{{9, 8, 7}, {3.3, 2.2, 1.1}}};
    std::cout << "mat18:\n" << mat18 << '\n';

    auto vect = std::vector<std::vector<float>>{{9, 8, 7}};
    julie::la::cuda::Matrix_CUDA<> mat19{vect};
    std::cout << "mat19:\n" << mat19 << '\n';

    vect = std::vector<std::vector<float>>{{9}, {8}, {7}};
    julie::la::cuda::Matrix_CUDA<> mat20{vect};
    std::cout << "mat20:\n" << mat20 << '\n';

    julie::la::cuda::Matrix_CUDA<> mat21{{{9}, {8}, {7}}, true};
    std::cout << "mat21:\n" << mat21 << '\n';

    julie::la::cuda::Matrix_CUDA<> mat22{{0, 1, 2, 3, 4}, false};
    std::cout << "mat22:\n" << mat22 << '\n';

    try
    {
        julie::la::cuda::Matrix_CUDA<> mat23{{{9, 8, 7}, {3.3, 2.2, 1.1}, {99, 88}}};
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    try
    {
        vect = std::vector<std::vector<float>>{{}, {}, {}};
        julie::la::cuda::Matrix_CUDA<> mat24{vect};
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    try
    {
        julie::la::cuda::Matrix_CUDA<> mat25{{}, false};
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    try
    {
        vect = std::vector<std::vector<float>>{};
        julie::la::cuda::Matrix_CUDA<> mat25{vect};
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
}

void test_matrix_gpu_self_cal()
{
    std::cout << "=================== test_matrix_gpu_self_cal ==================" << "\n";
    julie::la::cuda::Matrix_CUDA<int> gpu_mat1{8, julie::la::Shape{ 6, 5, 5 }};
    julie::la::cuda::Matrix_CUDA<int> gpu_mat2{2, julie::la::Shape{ 6, 5, 5 }};

    julie::la::cpu::Matrix_CPU<int> cpu_mat1 {gpu_mat1};
    julie::la::cpu::Matrix_CPU<int> cpu_mat2 {gpu_mat2};

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            for (int k = 0; k < 5; ++k)
            {
                cpu_mat1[julie::la::Coordinate{ {i, j, k}, cpu_mat1.shape() }] = j - k;
            }
        }
    }
    
    gpu_mat1 = cpu_mat1.get_CUDA();

    gpu_mat1 += gpu_mat2;
    cpu_mat1 += cpu_mat2;
    std::cout << "mat1:\n" << gpu_mat1 << '\n';
    test::ASSERT(gpu_mat1 == cpu_mat1.get_CUDA());

    gpu_mat1 -= gpu_mat2;
    cpu_mat1 -= cpu_mat2;
    std::cout << "mat1:\n" << gpu_mat1 << '\n';
    test::ASSERT(gpu_mat1 == cpu_mat1.get_CUDA());

    gpu_mat1 *= 2;
    cpu_mat1 *= 2;
    std::cout << "mat1:\n" << gpu_mat1 << '\n';
    test::ASSERT(gpu_mat1 == cpu_mat1.get_CUDA());

    gpu_mat1 /= 2;
    cpu_mat1 /= 2;
    std::cout << "mat1:\n" << gpu_mat1 << '\n';
    test::ASSERT(gpu_mat1 == cpu_mat1.get_CUDA());
}


void test_cuda_min_max()
{
    std::cout << "=================== test_cuda_min_max ================" << std::endl;


    julie::la::cuda::Matrix_CUDA<int> gpu_mat1{ {
        {5, -1, -8, -100, -1000},
        {7, -2, -5,   -3,    11},
        {9,  7, 99,   33,   105}
    } };

    std::cout << gpu_mat1 << std::endl;

    std::cout << "Calculate min ..." << std::endl;

    int min = gpu_mat1.min();
    std::cout << "min: " << min << std::endl;

    std::cout << gpu_mat1 << std::endl;

    std::cout << "Calculate max ..." << std::endl;

    int max = gpu_mat1.max();
    std::cout << "max: " << max << std::endl;

    std::cout << gpu_mat1 << std::endl;

    std::cout << "ASSERT 1" << std::endl;

    test::ASSERT(-1000 == min);
    
    std::cout << "ASSERT 2" << std::endl;

    test::ASSERT(105 == max);

    julie::la::cuda::Matrix_CUDA<float> gpu_mat2 {julie::la::Shape{16, 128, 128, 128}};
    gpu_mat2.uniform_random(0, 1);
    julie::la::cpu::Matrix_CPU<float> cpu_mat2 {gpu_mat2};

    std::cout << "Test Random Matrix of " << gpu_mat2.shape().size() << " elements" << std::endl;

    std::cout << "ASSERT max" << std::endl;
    test::ASSERT(gpu_mat2.max() == cpu_mat2.max());
    std::cout << "ASSERT min" << std::endl;
    test::ASSERT(gpu_mat2.min() == cpu_mat2.min());

    lint time1 = get_time_in_milliseconds();
    for (int i = 0; i < 10; ++i)
    {
        std::cout << "gpu max " << i << std::endl;
        gpu_mat2.max();
    }
    lint time2 = get_time_in_milliseconds();

    for (int i = 0; i < 10; ++i)
    {
        std::cout << "cpu max " << i << std::endl;
        cpu_mat2.max();
    }
    lint time3 = get_time_in_milliseconds();

    std::cout << "GPU compute time: " << time2 - time1 << std::endl;
    std::cout << "CPU compute time: " << time3 - time2 << std::endl;

    julie::la::cuda::Matrix_CUDA<int> gpu_mat3;
    julie::la::cpu::Matrix_CPU<int> cpu_mat3;

    for (int i = 0; i < 100; ++i)
    {
        int size = rand() % 10000;
        if (size == 0)
        {
            continue;
        }
        gpu_mat3 = julie::la::cuda::Matrix_CUDA<int> {julie::la::Shape {size}};
        gpu_mat3.uniform_random((1 << 20) * (-1), 1 << 20);
        cpu_mat3 = gpu_mat3;

        std::cout << "Random Matrix of size " << size << std::endl;

        int gpu_min = gpu_mat3.min();
        int cpu_min = cpu_mat3.min();

        std::cout << "gpu min: " << gpu_min << std::endl;
        std::cout << "cpu min: " << cpu_min << std::endl;
        test::ASSERT(gpu_min == cpu_min);
    }
}


void test_cuda_argmin_argmax()
{
    std::cout << "=================== test_cuda_argmin_argmax ================" << std::endl;

    julie::la::cuda::Matrix_CUDA<int> gpu_mat1{ {
        {5, -1, -8, -100, -1000},
        {7, -2, -5,   -3,    11},
        {9,  7, 99,   33,   105}
    } };

    std::cout << gpu_mat1 << std::endl;

    std::cout << "Calculate argmin ..." << std::endl;

    auto argmin = gpu_mat1.argmin();
    std::cout << "argmin: " << argmin << std::endl;

    std::cout << gpu_mat1 << std::endl;

    std::cout << "Calculate argmax ..." << std::endl;

    auto argmax = gpu_mat1.argmax();
    std::cout << "argmax: " << argmax << std::endl;

    std::cout << gpu_mat1 << std::endl;

    std::cout << "ASSERT 1" << std::endl;

    test::ASSERT(julie::la::Coordinate{{0, 4}, julie::la::Shape{3, 5}} == argmin);
    
    std::cout << "ASSERT 2" << std::endl;

    test::ASSERT(julie::la::Coordinate{{2, 4}, julie::la::Shape{3, 5}} == argmax);


    julie::la::cuda::Matrix_CUDA<float> gpu_mat2 {julie::la::Shape{16, 128, 128, 128}};
    gpu_mat2.uniform_random(0, 1);
    julie::la::cpu::Matrix_CPU<float> cpu_mat2 {gpu_mat2};

    std::cout << "Test Random Matrix of " << gpu_mat2.shape().size() << " elements" << std::endl;

    std::cout << "ASSERT argmax" << std::endl;
    test::ASSERT(gpu_mat2.argmax() == cpu_mat2.argmax());
    std::cout << "ASSERT argmin" << std::endl;
    test::ASSERT(gpu_mat2.argmin() == cpu_mat2.argmin());

    lint time1 = get_time_in_milliseconds();
    for (int i = 0; i < 10; ++i)
    {
        std::cout << "gpu argmax " << i << std::endl;
        gpu_mat2.argmax();
    }
    lint time2 = get_time_in_milliseconds();

    for (int i = 0; i < 10; ++i)
    {
        std::cout << "cpu argmax " << i << std::endl;
        cpu_mat2.argmax();
    }
    lint time3 = get_time_in_milliseconds();

    std::cout << "GPU compute time: " << time2 - time1 << std::endl;
    std::cout << "CPU compute time: " << time3 - time2 << std::endl;

    julie::la::cuda::Matrix_CUDA<int> gpu_mat3;
    julie::la::cpu::Matrix_CPU<int> cpu_mat3;

    for (int i = 0; i < 100; ++i)
    {
        int size = rand() % 10000;
        if (size == 0)
        {
            continue;
        }
        gpu_mat3 = julie::la::cuda::Matrix_CUDA<int> {julie::la::Shape {size}};
        gpu_mat3.uniform_random((1 << 20) * (-1), 1 << 20);
        cpu_mat3 = gpu_mat3;

        std::cout << "Random Matrix of size " << size << std::endl;

        auto gpu_argmin = gpu_mat3.argmin();
        auto cpu_argmin = cpu_mat3.argmin();

        std::cout << "gpu argmin: " << gpu_argmin << std::endl;
        std::cout << "cpu argmin: " << cpu_argmin << std::endl;
        test::ASSERT(gpu_argmin == cpu_argmin);
    } 
}


void test_cuda_sum()
{
    std::cout << "=================== test_cuda_sum ================" << std::endl;

    julie::la::cuda::Matrix_CUDA<int> gpu_mat1{ 1,  julie::la::Shape{64, 3, 32, 32} };
    test::ASSERT(64 * 3 * 32 * 32 == gpu_mat1.sum());

    julie::la::cuda::Matrix_CUDA<int> gpu_mat3;
    julie::la::cpu::Matrix_CPU<int> cpu_mat3;

    for (int i = 0; i < 100; ++i)
    {
        int size = rand() % 10000;
        if (size == 0)
        {
            continue;
        }
        gpu_mat3 = julie::la::cuda::Matrix_CUDA<int> {julie::la::Shape {size}};
        gpu_mat3.uniform_random(-128, 128);
        cpu_mat3 = gpu_mat3;

        std::cout << "Random Matrix of size " << size << std::endl;

        auto gpu_sum = gpu_mat3.sum();
        auto cpu_sum = cpu_mat3.sum();

        std::cout << "gpu sum: " << gpu_sum << std::endl;
        std::cout << "cpu sum: " << cpu_sum << std::endl;
        test::ASSERT(gpu_sum == cpu_sum);
    }

    julie::la::cuda::Matrix_CUDA<int> gpu_mat2 {1, julie::la::Shape{16, 128, 128, 128}};
    // gpu_mat2.uniform_random(0, 1);
    julie::la::cpu::Matrix_CPU<int> cpu_mat2 {gpu_mat2};

    std::cout << "Test Random Matrix of " << gpu_mat2.shape().size() << " elements" << std::endl;

    auto cpu_sum = cpu_mat2.sum();
    auto gpu_sum = gpu_mat2.sum();
    std::cout << "ASSERT sum: " << cpu_sum << " " << gpu_sum << std::endl;

    test::ASSERT(gpu_sum == cpu_sum);

    lint time1 = get_time_in_milliseconds();
    for (int i = 0; i < 10; ++i)
    {
        std::cout << "gpu sum " << i << std::endl;
        gpu_mat2.sum();
    }
    lint time2 = get_time_in_milliseconds();

    for (int i = 0; i < 10; ++i)
    {
        std::cout << "cpu sum " << i << std::endl;
        cpu_mat2.sum();
    }
    lint time3 = get_time_in_milliseconds();

    std::cout << "GPU compute time: " << time2 - time1 << std::endl;
    std::cout << "CPU compute time: " << time3 - time2 << std::endl;

}


void test_cuda_stat()
{
    std::cout << "=================== test_cuda_stat ================" << std::endl;

    julie::la::cuda::Matrix_CUDA<float> mat_gpu {julie::la::Shape{778, 321, 42}};

    mat_gpu.gaussian_random(15, 25);
    auto mat_cpu = julie::la::cpu::Matrix_CPU<float> {mat_gpu};

    std::cout << "gpu Mean: " << mat_gpu.mean() << std::endl;
    std::cout << "gpu Variance: " << mat_gpu.variance() << std::endl;

    std::cout << "cpu Mean: " << mat_cpu.mean() << std::endl;
    std::cout << "cpu Variance: " << mat_cpu.variance() << std::endl;

    test::ASSERT(fabs(mat_gpu.mean() - 15) < 0.1);
    test::ASSERT(fabs(mat_gpu.variance() - 25 * 25) < 1);

    mat_gpu.normalize();
    mat_cpu.normalize();

    std::cout << "gpu Mean: " << mat_gpu.mean() << std::endl;
    std::cout << "gpu Variance: " << mat_gpu.variance() << std::endl;

    std::cout << "cpu Mean: " << mat_cpu.mean() << std::endl;
    std::cout << "cpu Variance: " << mat_cpu.variance() << std::endl;

    test::ASSERT(fabs(mat_gpu.mean()) < 0.00001);
    test::ASSERT(fabs(mat_gpu.variance() - 1) < 0.00001);
}

void test_cuda_collapse()
{
    std::cout << "=================== test_cuda_collapse ================" << std::endl;

    julie::la::cuda::Matrix_CUDA<float> mat_gpu{julie::la::Shape{13, 15, 11, 9, 7, 88}};
    mat_gpu.gaussian_random(1, 10);

    julie::la::cpu::Matrix_CPU<float> mat_cpu {mat_gpu};

    auto mat_gpu_list = mat_gpu.get_collapsed(2);
    auto mat_cpu_list = mat_cpu.get_collapsed(2);

    for (size_t i = 0; i < mat_gpu_list.size(); ++i)
    {
        test::ASSERT(mat_gpu_list[i] == mat_cpu_list[i].get_CUDA());
        // std::cout << mat_gpu_list[i] << std::endl;
    }

    julie::la::cuda::Matrix_CUDA<float> mat_fused_gpu; mat_gpu.get_reduce_sum(mat_fused_gpu, 3);
    julie::la::cpu::Matrix_CPU<float> mat_fused_cpu; mat_cpu.get_reduce_sum(mat_fused_cpu, 3);

    // std::cout << "1" << std::endl;
    test::ASSERT(mat_fused_cpu.get_CUDA() == mat_fused_gpu);

    julie::la::cuda::Matrix_CUDA<float> mat_reducemean_gpu; mat_gpu.get_reduce_mean(mat_reducemean_gpu, 0);
    julie::la::cpu::Matrix_CPU<float> mat_reducemean_cpu; mat_cpu.get_reduce_mean(mat_reducemean_cpu, 0);
    // std::cout << "2" << std::endl;
    test::ASSERT(mat_reducemean_cpu.get_CUDA() == mat_reducemean_gpu);

    julie::la::cuda::Matrix_CUDA<float> mat1_gpu {
        {
            {9, 8, 7},
            {6, 5, 4},
            {3, 2, 1}
        }
    };

    julie::la::cuda::Matrix_CUDA<float> tmp;
    mat1_gpu.get_reduce_sum(tmp, 0);
    std::cout << tmp << std::endl;
    mat1_gpu.get_reduce_sum(tmp, 1);
    std::cout << tmp << std::endl;

    mat1_gpu.get_reduce_mean(mat_reducemean_gpu, 0);
    std::cout << mat_reducemean_gpu << std::endl;

    mat1_gpu.get_reduce_mean(mat_reducemean_gpu, 1);
    std::cout << mat_reducemean_gpu << std::endl;


    julie::la::cuda::Matrix_CUDA<int> mat2_gpu{julie::la::Shape{128, 16, 128, 128}};
    mat2_gpu.uniform_random(0, 100);
    julie::la::cpu::Matrix_CPU<int> mat2_cpu{mat2_gpu};

    julie::la::cuda::Matrix_CUDA<int> mat_reducemean_gpu_i;
    julie::la::cpu::Matrix_CPU<int> mat_reducemean_cpu_i;

    std::cout << "3" << std::endl;
    mat2_gpu.get_reduce_mean(mat_reducemean_gpu_i, 1);
    mat2_cpu.get_reduce_mean(mat_reducemean_cpu_i, 1);
    test::ASSERT(mat_reducemean_gpu_i == mat_reducemean_cpu_i.get_CUDA());

    lint time1 = get_time_in_milliseconds();
    for (int i = 0; i < 10; ++i)
    {
        std::cout << "gpu reduce mean: " << i << std::endl;
        mat2_gpu.get_reduce_mean(mat_reducemean_gpu_i, 1);
    }

    lint time2 = get_time_in_milliseconds();
    for (int i = 0; i < 10; ++i)
    {
        std::cout << "cpu reduce mean: " << i << std::endl;
        mat2_cpu.get_reduce_mean(mat_reducemean_cpu_i, 1);
    }

    lint time3 = get_time_in_milliseconds();

    std::cout << "test_cuda_collapse finish" << std::endl;

    std::cout << "GPU compute time: " << time2 - time1 << std::endl;
    std::cout << "CPU compute time: " << time3 - time2 << std::endl;
}


void test_cuda_euclidean_norm()
{
    std::cout << "=================== test_cuda_euclidean_norm ================" << std::endl;

    julie::la::cuda::Matrix_CUDA<float> mat1_gpu {std::vector<float> {2, 2, 2, 2}, true};

    std::cout << "Euclidien norm: " << mat1_gpu.euclidean_norm() << std::endl;

    test::ASSERT(fabs(mat1_gpu.euclidean_norm() - 4.0) < 0.00001);

    julie::la::cuda::Matrix_CUDA<float> mat2_gpu;
    julie::la::cpu::Matrix_CPU<float> mat2_cpu;

    for (int i = 0; i < 100; ++i)
    {
        int size = rand() % 10000;
        if (size == 0)
        {
            continue;
        }
        mat2_gpu = julie::la::cuda::Matrix_CUDA<float> {julie::la::Shape {size}};
        mat2_gpu.uniform_random(-2, 2);
        mat2_cpu = mat2_gpu;

        std::cout << "Random Matrix of size " << size << std::endl;

        auto gpu_norm = mat2_gpu.euclidean_norm();
        auto cpu_norm = mat2_cpu.euclidean_norm();

        std::cout << "gpu norm: " << gpu_norm << std::endl;
        std::cout << "cpu norm: " << cpu_norm << std::endl;
        test::ASSERT(fabs(gpu_norm - cpu_norm) < 0.00001);
    }

    julie::la::cuda::Matrix_CUDA<float> mat3_gpu{julie::la::Shape{128, 16, 128, 128}};
    mat3_gpu.gaussian_random(0, 1);
    julie::la::cpu::Matrix_CPU<float> mat3_cpu{mat3_gpu};

    lint time1 = get_time_in_milliseconds();
    for (int i = 0; i < 10; ++i)
    {
        std::cout << "gpu euclidean_norm " << i << std::endl;
        mat3_gpu.euclidean_norm();
    }
    lint time2 = get_time_in_milliseconds();

    for (int i = 0; i < 10; ++i)
    {
        std::cout << "cpu euclidean_norm " << i << std::endl;
        mat3_cpu.euclidean_norm();
    }
    lint time3 = get_time_in_milliseconds();

    std::cout << "GPU compute time: " << time2 - time1 << std::endl;
    std::cout << "CPU compute time: " << time3 - time2 << std::endl;
}


void test_cuda_transpose()
{
    std::cout << "=================== test_cuda_transpose ================" << std::endl;

    julie::la::cuda::Matrix_CUDA<float> mat1_gpu{julie::la::Shape{34, 17, 9, 13}};
    mat1_gpu.gaussian_random(-1, 10);
    julie::la::cpu::Matrix_CPU<float> mat1_cpu{mat1_gpu};

    julie::la::cuda::Matrix_CUDA<float> mat1_gpu_trans; julie::la::cuda::transpose(mat1_gpu_trans, mat1_gpu, 2);
    julie::la::cpu::Matrix_CPU<float> mat1_cpu_trans; julie::la::cpu::transpose(mat1_cpu_trans, mat1_cpu, 2);

    test::ASSERT(mat1_gpu_trans == mat1_cpu_trans.get_CUDA());

    julie::la::cuda::Matrix_CUDA<float> mat2_gpu{julie::la::Shape{128, 16, 128, 128}};
    mat2_gpu.gaussian_random(-1, 100);
    julie::la::cpu::Matrix_CPU<float> mat2_cpu{mat2_gpu};

    julie::la::cuda::Matrix_CUDA<float> gpu_trans;
    julie::la::cpu::Matrix_CPU<float> cpu_trans;
    mat2_gpu.get_transpose(gpu_trans, 2);
    mat2_cpu.get_transpose(cpu_trans, 2);
    test::ASSERT( gpu_trans == cpu_trans.get_CUDA());

    lint time1 = get_time_in_milliseconds();
    for (int i = 0; i < 10; ++i)
    {
        std::cout << "gpu transpose " << i << std::endl;
        mat2_gpu.get_transpose(gpu_trans, 3);
    }
    lint time2 = get_time_in_milliseconds();

    for (int i = 0; i < 10; ++i)
    {
        std::cout << "cpu transpose " << i << std::endl;
        mat2_cpu.get_transpose(cpu_trans, 3);
    }
    lint time3 = get_time_in_milliseconds();

    std::cout << "GPU compute time: " << time2 - time1 << std::endl;
    std::cout << "CPU compute time: " << time3 - time2 << std::endl;
}


void test_cuda_matmul()
{
    std::cout << "=================== test_cuda_matmul ================" << std::endl;

    julie::la::cuda::Matrix_CUDA<int> mat1_gpu {
        {
            {1, 1, 1, 1},
            {1, 1, 1, 1},
            {1, 1, 1, 1}
        }
    };

    julie::la::cuda::Matrix_CUDA<int> mat2_gpu {
        {
            {1, 1},
            {1, 1},
            {1, 1},
            {1, 1}
        }
    };

    julie::la::cuda::Matrix_CUDA<int> mat_out_gpu;
    julie::la::cuda::matmul(mat_out_gpu, mat1_gpu, mat2_gpu);

    test::ASSERT(mat_out_gpu == julie::la::cuda::Matrix_CUDA<int> {
        {
            {4, 4},
            {4, 4},
            {4, 4}
        }
    });

    julie::la::cuda::Matrix_CUDA<int> mat3_gpu;
    julie::la::cuda::Matrix_CUDA<int> mat4_gpu;
    julie::la::cpu::Matrix_CPU<int> mat3_cpu;
    julie::la::cpu::Matrix_CPU<int> mat4_cpu;

    julie::la::cpu::Matrix_CPU<int> mat_out_cpu;

    for(int i = 0; i < 100; i++)
    {
        int a = rand() % 200;
        int b = rand() % 200;
        int c = rand() % 200;

        if(a * b * c == 0)
        {
            continue;
        }

        mat3_gpu = julie::la::cuda::Matrix_CUDA<int> {julie::la::Shape{a, b}};
        mat4_gpu = julie::la::cuda::Matrix_CUDA<int> {julie::la::Shape{b, c}};

        mat3_gpu.uniform_random(-100, 100);
        mat4_gpu.uniform_random(-100, 100);

        mat3_cpu = julie::la::cpu::Matrix_CPU<int> {mat3_gpu};
        mat4_cpu = julie::la::cpu::Matrix_CPU<int> {mat4_gpu};

        std::cout << "[ " << a << ", " << b << " ] and [ " << b << ", " << c << " ]" << std::endl;

        julie::la::cuda::matmul(mat_out_gpu, mat3_gpu, mat4_gpu);
        julie::la::cpu::matmul(mat_out_cpu, mat3_cpu, mat4_cpu);

        //std::cout << mat_out_gpu;
        //std::cout << mat_out_cpu;

        test::ASSERT(mat_out_gpu == mat_out_cpu.get_CUDA());
    }

    mat3_gpu = julie::la::cuda::Matrix_CUDA<int> {julie::la::Shape{16, 64, 128}};
    mat4_gpu = julie::la::cuda::Matrix_CUDA<int> {julie::la::Shape{64, 2, 96, 22}};

    mat3_gpu.uniform_random(-100, 100);
    mat4_gpu.uniform_random(-100, 100);

    mat3_cpu = julie::la::cpu::Matrix_CPU<int> {mat3_gpu};
    mat4_cpu = julie::la::cpu::Matrix_CPU<int> {mat4_gpu};

    julie::la::cuda::matmul(mat_out_gpu, mat3_gpu, mat4_gpu, 1, 2);
    julie::la::cpu::matmul(mat_out_cpu, mat3_cpu, mat4_cpu, 1, 2);

    test::ASSERT(mat_out_gpu == mat_out_cpu.get_CUDA());

    lint time1 = get_time_in_milliseconds();
    for(int i = 0; i < 10; ++i)
    {
        std::cout << "gpu matmul " << i << std::endl;
        julie::la::cuda::matmul(mat_out_gpu, mat3_gpu, mat4_gpu);
    }

    lint time2 = get_time_in_milliseconds();
    for(int i = 0; i < 10; ++i)
    {
        std::cout << "cpu matmul " << i << std::endl;
        julie::la::cpu::matmul(mat_out_cpu, mat3_cpu, mat4_cpu);
    }

    lint time3 = get_time_in_milliseconds();

    std::cout << "GPU compute time: " << time2 - time1 << std::endl;
    std::cout << "CPU compute time: " << time3 - time2 << std::endl;

////////////////////////////////////////////////////////////////////////////////////////
    mat3_gpu = julie::la::cuda::Matrix_CUDA<int> {julie::la::Shape{4, 2000, 355}};
    mat4_gpu = julie::la::cuda::Matrix_CUDA<int> {julie::la::Shape{2000, 355, 8}};

    mat3_gpu.uniform_random(-100, 100);
    mat4_gpu.uniform_random(-100, 100);

    mat3_cpu = julie::la::cpu::Matrix_CPU<int> {mat3_gpu};
    mat4_cpu = julie::la::cpu::Matrix_CPU<int> {mat4_gpu};

    julie::la::cuda::matmul(mat_out_gpu, mat3_gpu, mat4_gpu, 2, 2);
    julie::la::cpu::matmul(mat_out_cpu, mat3_cpu, mat4_cpu, 2, 2);

    test::ASSERT(mat_out_gpu == mat_out_cpu.get_CUDA());

    lint time4 = get_time_in_milliseconds();
    for(int i = 0; i < 10; ++i)
    {
        std::cout << "gpu matmul " << i << std::endl;
        julie::la::cuda::matmul(mat_out_gpu, mat3_gpu, mat4_gpu);
    }

    lint time5 = get_time_in_milliseconds();
    for(int i = 0; i < 10; ++i)
    {
        std::cout << "cpu matmul " << i << std::endl;
        julie::la::cpu::matmul(mat_out_cpu, mat3_cpu, mat4_cpu);
    }

    lint time6 = get_time_in_milliseconds();

    std::cout << "GPU compute time: " << time5 - time4 << std::endl;
    std::cout << "CPU compute time: " << time6 - time5 << std::endl;
}

void test_cuda_transpose_neighboring_dim_pair()
{
    std::cout << "=================== test_cuda_transpose_neighboring_dim_pair ==================" << "\n";

    julie::la::cuda::Matrix_CUDA<int> mat{
        {
            { 1,  2,  3,  4},
            { 5,  6,  7,  8},
            { 9, 10, 11, 12},
            {13, 14, 15, 16},
            {17, 18, 19, 20}
        }
        };

    julie::la::cuda::Matrix_CUDA<int> mat_trans1;
    julie::la::cuda::Matrix_CUDA<int> mat_trans2;
    julie::la::cuda::Matrix_CUDA<int> mat_trans3;

    julie::la::cuda::transpose_neighboring_dims(mat_trans1, mat, 0, 0, 1, 1);
    julie::la::cuda::transpose(mat_trans2, mat, 1);

    std::cout << mat_trans1 << std::endl;

    test::ASSERT(mat_trans1 == mat_trans2);

    julie::la::cuda::Matrix_CUDA<int> mat_1{
        {
            { -1,  -2,  -3,  -4},
            { -5,  -6,  -7,  -8},
            { -9, -10, -11, -12},
            {-13, -14, -15, -16},
            {-17, -18, -19, -20}
        }
        };

    julie::la::cuda::Matrix_CUDA<int> ch2_mat {{mat, mat_1}};

    julie::la::cuda::transpose_neighboring_dims(mat_trans1, ch2_mat, 0, 0, 1, 1);
    julie::la::cuda::transpose_neighboring_dims(mat_trans2, mat_trans1, 1, 1, 2, 2);
    julie::la::cuda::transpose(mat_trans3, ch2_mat, 1);

    std::cout << mat_trans2 << std::endl;

    test::ASSERT(mat_trans2 == mat_trans3);

    // Continue to do correctness test
    for (int i = 0; i < 100; ++i)
    {
        lint dim1 = rand() % 20 + 1;
        lint dim2 = rand() % 20 + 1;
        lint dim3 = rand() % 20 + 1;
        lint dim4 = rand() % 20 + 1;
        lint dim5 = rand() % 20 + 1;
        lint dim6 = rand() % 20 + 1;

        lint dim_idx = rand() % 5;

        julie::la::cpu::Matrix_CPU<int> cpu_mat {julie::la::Shape{dim1, dim2, dim3, dim4, dim5, dim6}};
        cpu_mat.uniform_random(-100, 100);

        std::cout << "correctness: shape: " << cpu_mat.shape() << " dims to transpose: " << dim_idx << " and " << dim_idx + 1 << std::endl; 

        julie::la::cuda::Matrix_CUDA<int> cuda_mat = cpu_mat.get_CUDA();

        julie::la::cpu::Matrix_CPU<int> cpu_mat_trans;
        julie::la::cpu::transpose_neighboring_dims(cpu_mat_trans, cpu_mat, dim_idx, dim_idx, dim_idx + 1, dim_idx + 1);

        julie::la::cuda::Matrix_CUDA<int> cuda_mat_trans;
        julie::la::cuda::transpose_neighboring_dims(cuda_mat_trans, cuda_mat, dim_idx, dim_idx, dim_idx + 1, dim_idx + 1);

        test::ASSERT(cpu_mat_trans.get_CUDA() == cuda_mat_trans);
    }

}

void test_cuda_argmax_of_one_dimension()
{
    std::cout << "=================== test_cuda_argmax_of_one_dimension ==================" << "\n";

    julie::la::cuda::Matrix_CUDA<int> mat_ch1{
        {
            {3, 0, 7,  9,  8},
            {9, 7, 0,  4,  1},
            {3, 3, 4,  8, 11},
            {0, 6, 5, 12,  2}
        }};

    julie::la::cuda::Matrix_CUDA<int> mat_ch2{
        {
            {17, 8,  8, 22,  6},
            { 9, 7,  0,  4,  4},
            {33, 1, 12, 18,  2},
            {15, 0, 36, 17,  1}
        }};

    julie::la::cuda::Matrix_CUDA<int> mat {{mat_ch1, mat_ch2}};
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


void test_cuda_argmax_of_one_dimension_cross_validate()
{
    std::cout << "=================== test_cuda_argmax_of_one_dimension_cross_validate ==================" << "\n";

    for (int i = 0; i < 100; ++i)
    {
        int a = rand() % 50 + 1;
        int b = rand() % 50 + 1;
        int c = rand() % 50 + 1;
        int d = rand() % 50 + 1;

        julie::la::cpu::Matrix_CPU<float> cpu_mat {julie::la::Shape{a, b, c, d}};
        cpu_mat.gaussian_random(5, 10);
        julie::la::cuda::Matrix_CUDA<float> gpu_mat = cpu_mat.get_CUDA();

        std::cout << "argmax of each dimension for shape: " << cpu_mat.shape() << std::endl; 

        for (int dim = 0; dim < 4; ++dim)
        {
            auto co_list_cpu = cpu_mat.argmax(dim);
            auto co_list_gpu = gpu_mat.argmax(dim);

            for (size_t s = 0; s < co_list_cpu.size(); ++s)
            {
                test::ASSERT(co_list_cpu[s] == co_list_gpu[s]);
            }
        }
    }
}

void test_cuda_argmin_of_one_dimension()
{
    std::cout << "=================== test_cuda_argmin_of_one_dimension ==================" << "\n";

    julie::la::cuda::Matrix_CUDA<int> mat_ch1{
        {
            {3, 0, 7,  9,  8},
            {9, 7, 0,  4,  1},
            {3, 3, 4,  8, 11},
            {0, 6, 5, 12,  2}
        }};

    julie::la::cuda::Matrix_CUDA<int> mat_ch2{
        {
            {17, 8,  8, 22,  6},
            { 9, 7,  0,  4,  4},
            {33, 1, 12, 18,  2},
            {15, 0, 36, 17,  1}
        }};

    julie::la::cuda::Matrix_CUDA<int> mat {{mat_ch1, mat_ch2}};
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


void test_cuda_argmin_of_one_dimension_cross_validate()
{
    std::cout << "=================== test_cuda_argmin_of_one_dimension_cross_validate ==================" << "\n";

    for (int i = 0; i < 100; ++i)
    {
        int a = rand() % 50 + 1;
        int b = rand() % 50 + 1;
        int c = rand() % 50 + 1;
        int d = rand() % 50 + 1;

        julie::la::cpu::Matrix_CPU<float> cpu_mat {julie::la::Shape{a, b, c, d}};
        cpu_mat.gaussian_random(5, 10);
        julie::la::cuda::Matrix_CUDA<float> gpu_mat = cpu_mat.get_CUDA();

        std::cout << "argmin of each dimension for shape: " << cpu_mat.shape() << std::endl; 

        for (int dim = 0; dim < 4; ++dim)
        {
            auto co_list_cpu = cpu_mat.argmin(dim);
            auto co_list_gpu = gpu_mat.argmin(dim);

            for (size_t s = 0; s < co_list_cpu.size(); ++s)
            {
                test::ASSERT(co_list_cpu[s] == co_list_gpu[s]);
            }
        }
    }
}


void test_cuda_concat()
{
    std::cout << "=================== test_cuda_concat ==================" << "\n";

    julie::la::cuda::Matrix_CUDA<int> mat1{
        {
            {3, 0, 7,  9,  8},
            {9, 7, 0,  4,  1},
            {3, 3, 4,  8, 11},
            {0, 6, 5, 12,  2}
        }};

    julie::la::cuda::Matrix_CUDA<int> mat2{
        {
            {17, 8,  8, 22,  6},
            { 9, 7,  0,  4,  4},
            {33, 1, 12, 18,  2}
        }};

    julie::la::cuda::Matrix_CUDA<int> cat;

    std::cout << "concat of " << mat1.shape() << " and " << mat2.shape() << std::endl;

    julie::la::cuda::concatenate(cat, mat1, mat2, 0);
    test::ASSERT(cat == julie::la::cuda::Matrix_CUDA<int> {
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

    julie::la::cuda::Matrix_CUDA<int> mat3{
        {
            {3, 0, 7},
            {9, 7, 0},
            {3, 3, 4},
            {0, 6, 5}
        }};

    julie::la::cuda::Matrix_CUDA<int> mat4{
        {
            {17, 8,  8, 22,  6},
            { 9, 7,  0,  4,  4},
            {33, 1, 12, 18,  2},
            {15, 0, 36, 17,  1}
        }};

    std::cout << "concat of " << mat3.shape() << " and " << mat4.shape() << std::endl;

    julie::la::cuda::concatenate(cat, mat3, mat4, 1);
    test::ASSERT(cat == julie::la::cuda::Matrix_CUDA<int> {
        {
            {3, 0, 7, 17, 8,  8, 22,  6},
            {9, 7, 0,  9, 7,  0,  4,  4},
            {3, 3, 4, 33, 1, 12, 18,  2},
            {0, 6, 5, 15, 0, 36, 17,  1}
        }
    });

    julie::la::cuda::Matrix_CUDA<int> mat5{std::vector<int>{1, 2, 3, 4, 5}, true};
    mat5.reshape(julie::la::Shape{5});
    julie::la::cuda::Matrix_CUDA<int> mat6{std::vector<int>{10, 11, 12, 13, 14, 15, 16, 17}, true};
    mat6.reshape(julie::la::Shape{8});

    std::cout << "concat of " << mat5.shape() << " and " << mat6.shape() << std::endl;
    julie::la::cuda::concatenate(cat, mat5, mat6, 0);

    test::ASSERT(cat == julie::la::cuda::Matrix_CUDA<int> {std::vector<int>{1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17}, true}.reshape(julie::la::Shape{13}));

    julie::la::cuda::Matrix_CUDA<int> mat7{std::vector<int>{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}, true};
    mat7.reshape(julie::la::Shape{3, 3, 3});
    julie::la::cuda::Matrix_CUDA<int> mat8{std::vector<int>{
        -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18}, true};
    mat8.reshape(julie::la::Shape{3, 2, 3});

    std::cout << "concat of " << mat7.shape() << " and " << mat8.shape() << std::endl;
    julie::la::cuda::concatenate(cat, mat7, mat8, 1);

    test::ASSERT(cat == julie::la::cuda::Matrix_CUDA<int> {std::vector<int>{
        1,  2,  3,  4,  5,  6,  7,  8,  9,  -1,  -2,  -3,  -4,  -5,  -6,
        10, 11, 12, 13, 14, 15, 16, 17, 18, -7,  -8,  -9,  -10, -11, -12,
        19, 20, 21, 22, 23, 24, 25, 26, 27, -13, -14, -15, -16, -17, -18}, true}.reshape(julie::la::Shape{3, 5, 3}));

    mat8.reshape(julie::la::Shape{3, 3, 2});
    std::cout << "concat of " << mat7.shape() << " and " << mat8.shape() << std::endl;
    julie::la::cuda::concatenate(cat, mat7, mat8, 2);

    test::ASSERT(cat == julie::la::cuda::Matrix_CUDA<int> {std::vector<int>{
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
    julie::la::cuda::concatenate(cat, mat7, mat8, 0);

    test::ASSERT(cat == julie::la::cuda::Matrix_CUDA<int> {std::vector<int>{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18
        }, julie::la::Shape{5, 3, 3}});
}


void test_cuda_concat_cross_validate()
{
    std::cout << "=================== test_cuda_concat_cross_validate ==================" << "\n";

    for (int i = 0; i < 100; ++i)
    {
        int a = rand() % 50 + 1;
        int b = rand() % 50 + 1;
        int c = rand() % 50 + 1;
        int d = rand() % 50 + 1;
        int e = rand() % 50 + 1;
        int f = rand() % 50 + 1;
        int g = rand() % 50 + 1;
        int h = rand() % 50 + 1;

        int dim_list[8] = {a, b, c, d, e, f, g, h};

        julie::la::cpu::Matrix_CPU<float> cpu_mat_cat;
        julie::la::cuda::Matrix_CUDA<float> gpu_mat_cat;

        julie::la::cpu::Matrix_CPU<float> cpu_mat {julie::la::Shape{dim_list[0], dim_list[1], dim_list[2], dim_list[3]}};
        cpu_mat.gaussian_random(0, 10);
        julie::la::cuda::Matrix_CUDA<float> gpu_mat = cpu_mat.get_CUDA();

        auto cpu_mat_b = julie::la::cpu::Matrix_CPU<float>{julie::la::Shape{dim_list[4], dim_list[1], dim_list[2], dim_list[3]}};
        auto gpu_mat_b = cpu_mat_b.get_CUDA();

        std::cout << "concat " << cpu_mat.shape() << " and " << cpu_mat_b.shape() << std::endl;

        julie::la::cpu::concatenate(cpu_mat_cat, cpu_mat, cpu_mat_b, 0);
        julie::la::cuda::concatenate(gpu_mat_cat, gpu_mat, gpu_mat_b, 0);
    
        test::ASSERT(cpu_mat_cat == julie::la::cpu::Matrix_CPU<float>{gpu_mat_cat});

        cpu_mat_b = julie::la::cpu::Matrix_CPU<float>{julie::la::Shape{dim_list[0], dim_list[5], dim_list[2], dim_list[3]}};
        gpu_mat_b = cpu_mat_b.get_CUDA();

        std::cout << "concat " << cpu_mat.shape() << " and " << cpu_mat_b.shape() << std::endl;

        julie::la::cpu::concatenate(cpu_mat_cat, cpu_mat, cpu_mat_b, 1);
        julie::la::cuda::concatenate(gpu_mat_cat, gpu_mat, gpu_mat_b, 1);
    
        test::ASSERT(cpu_mat_cat == julie::la::cpu::Matrix_CPU<float>{gpu_mat_cat});

        cpu_mat_b = julie::la::cpu::Matrix_CPU<float>{julie::la::Shape{dim_list[0], dim_list[1], dim_list[6], dim_list[3]}};
        gpu_mat_b = cpu_mat_b.get_CUDA();

        std::cout << "concat " << cpu_mat.shape() << " and " << cpu_mat_b.shape() << std::endl;

        julie::la::cpu::concatenate(cpu_mat_cat, cpu_mat, cpu_mat_b, 2);
        julie::la::cuda::concatenate(gpu_mat_cat, gpu_mat, gpu_mat_b, 2);
    
        test::ASSERT(cpu_mat_cat == julie::la::cpu::Matrix_CPU<float>{gpu_mat_cat});

        cpu_mat_b = julie::la::cpu::Matrix_CPU<float>{julie::la::Shape{dim_list[0], dim_list[1], dim_list[2], dim_list[7]}};
        gpu_mat_b = cpu_mat_b.get_CUDA();

        std::cout << "concat " << cpu_mat.shape() << " and " << cpu_mat_b.shape() << std::endl;

        julie::la::cpu::concatenate(cpu_mat_cat, cpu_mat, cpu_mat_b, 3);
        julie::la::cuda::concatenate(gpu_mat_cat, gpu_mat, gpu_mat_b, 3);
    
        test::ASSERT(cpu_mat_cat == julie::la::cpu::Matrix_CPU<float>{gpu_mat_cat});
    }
}


void test_cuda_slice()
{
    std::cout << "=================== test_cuda_slice ==================" << "\n";
    julie::la::cuda::Matrix_CUDA<int> input{
        std::vector<int>{
            1,  2,  3,  4,  5,
            6,  7,  8,  9,  10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20
        },
        julie::la::Shape{4, 5}};

    julie::la::cuda::Matrix_CUDA<int> slice;
    julie::la::cuda::slice(slice, input, 0, 0, 3);
    std::cout << slice << std::endl;
    test::ASSERT(slice == julie::la::cuda::Matrix_CUDA<int> {
        std::vector<int>{    
            1,  2,  3,  4,  5,
            6,  7,  8,  9,  10,
            11, 12, 13, 14, 15
        },
        julie::la::Shape{3, 5}});

    julie::la::cuda::slice(slice, input, 1, 2, 2);
    std::cout << slice << std::endl;
    test::ASSERT(slice == julie::la::cuda::Matrix_CUDA<int> {
        std::vector<int>{    
            3,  4,
            8,  9,
            13, 14,
            18, 19
        },
        julie::la::Shape{4, 2}});

    input.reshape(julie::la::Shape{20});

    julie::la::cuda::slice(slice, input, 0, 13, 7);
    std::cout << slice << std::endl;
    test::ASSERT(slice == julie::la::cuda::Matrix_CUDA<int> {
        std::vector<int>{    
            14, 15, 16, 17, 18, 19, 20
        },
        julie::la::Shape{7}});
}


void test_cuda_slice_cross_validate()
{
    std::cout << "=================== test_cuda_slice_cross_validate ==================" << "\n";

    for (int i = 0; i < 100; ++i)
    {
        int a = rand() % 50 + 1;
        int b = rand() % 50 + 1;
        int c = rand() % 50 + 1;
        int d = rand() % 50 + 1;

        julie::la::cpu::Matrix_CPU<float> cpu_mat_slice;
        julie::la::cuda::Matrix_CUDA<float> gpu_mat_slice;

        julie::la::cpu::Matrix_CPU<float> cpu_mat {julie::la::Shape{a, b, c, d}};
        cpu_mat.gaussian_random(0, 10);
        julie::la::cuda::Matrix_CUDA<float> gpu_mat = cpu_mat.get_CUDA();

        int idx = std::min(a - 1, rand() % 50);
        int slice_size = std::min(a - idx, rand() % 50 + 1);
        julie::la::cpu::slice(cpu_mat_slice, cpu_mat, 0, idx, slice_size);
        julie::la::cuda::slice(gpu_mat_slice, gpu_mat, 0, idx, slice_size);
        std::cout << "Slice " << cpu_mat_slice.shape() << " from " << cpu_mat.shape() << std::endl;
        test::ASSERT(cpu_mat_slice == julie::la::cpu::Matrix_CPU<float>{gpu_mat_slice});

        idx = std::min(b - 1, rand() % 50);
        slice_size = std::min(b - idx, rand() % 50 + 1);
        julie::la::cpu::slice(cpu_mat_slice, cpu_mat, 1, idx, slice_size);
        julie::la::cuda::slice(gpu_mat_slice, gpu_mat, 1, idx, slice_size);
        std::cout << "Slice " << cpu_mat_slice.shape() << " from " << cpu_mat.shape() << std::endl;
        test::ASSERT(cpu_mat_slice == julie::la::cpu::Matrix_CPU<float>{gpu_mat_slice});

        idx = std::min(c - 1, rand() % 50);
        slice_size = std::min(c - idx, rand() % 50 + 1);
        julie::la::cpu::slice(cpu_mat_slice, cpu_mat, 2, idx, slice_size);
        julie::la::cuda::slice(gpu_mat_slice, gpu_mat, 2, idx, slice_size);
        std::cout << "Slice " << cpu_mat_slice.shape() << " from " << cpu_mat.shape() << std::endl;
        test::ASSERT(cpu_mat_slice == julie::la::cpu::Matrix_CPU<float>{gpu_mat_slice});

        idx = std::min(d - 1, rand() % 50);
        slice_size = std::min(d - idx, rand() % 50 + 1);
        julie::la::cpu::slice(cpu_mat_slice, cpu_mat, 3, idx, slice_size);
        julie::la::cuda::slice(gpu_mat_slice, gpu_mat, 3, idx, slice_size);
        std::cout << "Slice " << cpu_mat_slice.shape() << " from " << cpu_mat.shape() << std::endl;
        test::ASSERT(cpu_mat_slice == julie::la::cpu::Matrix_CPU<float>{gpu_mat_slice});
    }
}


void test_cuda_repeat()
{
    std::cout << "=================== test_cuda_repeat ==================" << "\n";
    julie::la::cuda::Matrix_CUDA<int> input{
        std::vector<int>{
            1,  2,  3,  4,  5,
            6,  7,  8,  9,  10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20
        },
        julie::la::Shape{4, 5}};

    julie::la::cuda::Matrix_CUDA<int> repeat;
    julie::la::cuda::repeat(repeat, input, 0, 3);
    std::cout << repeat << std::endl;
    test::ASSERT(repeat == julie::la::cuda::Matrix_CUDA<int> {
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

    julie::la::cuda::repeat(repeat, input, 1, 2);
    std::cout << repeat << std::endl;
    test::ASSERT(repeat == julie::la::cuda::Matrix_CUDA<int> {
        std::vector<int>{    
            1,  2,  3,  4,  5,  1,  2,  3,  4,  5,
            6,  7,  8,  9,  10, 6,  7,  8,  9,  10,
            11, 12, 13, 14, 15, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 16, 17, 18, 19, 20
        },
        julie::la::Shape{4, 10}});

    input.reshape(julie::la::Shape{20});

    julie::la::cuda::repeat(repeat, input, 0, 4);
    std::cout << repeat << std::endl;
    test::ASSERT(repeat == julie::la::cuda::Matrix_CUDA<int> {
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


void test_cuda_repeat_cross_validate()
{
    std::cout << "=================== test_cuda_repeat_cross_validate ==================" << "\n";

    for (int i = 0; i < 100; ++i)
    {
        int a = rand() % 20 + 1;
        int b = rand() % 20 + 1;
        int c = rand() % 20 + 1;
        int d = rand() % 20 + 1;

        julie::la::cpu::Matrix_CPU<float> cpu_mat_repeat;
        julie::la::cuda::Matrix_CUDA<float> gpu_mat_repeat;

        julie::la::cpu::Matrix_CPU<float> cpu_mat {julie::la::Shape{a, b, c, d}};
        cpu_mat.gaussian_random(0, 10);
        julie::la::cuda::Matrix_CUDA<float> gpu_mat = cpu_mat.get_CUDA();

        int repeat_size = rand() % 20 + 1;
        julie::la::cpu::repeat(cpu_mat_repeat, cpu_mat, 0, repeat_size);
        julie::la::cuda::repeat(gpu_mat_repeat, gpu_mat, 0, repeat_size);
        std::cout << "Repeat " << cpu_mat_repeat.shape() << " from " << cpu_mat.shape() << std::endl;
        test::ASSERT(cpu_mat_repeat == julie::la::cpu::Matrix_CPU<float>{gpu_mat_repeat});

        repeat_size = rand() % 20 + 1;
        julie::la::cpu::repeat(cpu_mat_repeat, cpu_mat, 1, repeat_size);
        julie::la::cuda::repeat(gpu_mat_repeat, gpu_mat, 1, repeat_size);
        std::cout << "Repeat " << cpu_mat_repeat.shape() << " from " << cpu_mat.shape() << std::endl;
        test::ASSERT(cpu_mat_repeat == julie::la::cpu::Matrix_CPU<float>{gpu_mat_repeat});

        repeat_size = rand() % 20 + 1;
        julie::la::cpu::repeat(cpu_mat_repeat, cpu_mat, 2, repeat_size);
        julie::la::cuda::repeat(gpu_mat_repeat, gpu_mat, 2, repeat_size);
        std::cout << "Repeat " << cpu_mat_repeat.shape() << " from " << cpu_mat.shape() << std::endl;
        test::ASSERT(cpu_mat_repeat == julie::la::cpu::Matrix_CPU<float>{gpu_mat_repeat});

        repeat_size = rand() % 20 + 1;
        julie::la::cpu::repeat(cpu_mat_repeat, cpu_mat, 3, repeat_size);
        julie::la::cuda::repeat(gpu_mat_repeat, gpu_mat, 3, repeat_size);
        std::cout << "Repeat " << cpu_mat_repeat.shape() << " from " << cpu_mat.shape() << std::endl;
        test::ASSERT(cpu_mat_repeat == julie::la::cpu::Matrix_CPU<float>{gpu_mat_repeat});
    }
}


void test_of_Matrix_CUDA_operations()
{
    test_matrix_gpu_constructor();
    test_matrix_gpu_self_cal();
    test_cuda_min_max();
    test_cuda_argmin_argmax();
    test_cuda_sum();
    test_cuda_stat();
    test_cuda_collapse();
    test_cuda_euclidean_norm();
    test_cuda_transpose();
    test_cuda_matmul();

    test_cuda_transpose_neighboring_dim_pair();
    test_cuda_argmax_of_one_dimension();
    test_cuda_argmax_of_one_dimension_cross_validate();
    test_cuda_argmin_of_one_dimension();
    test_cuda_argmin_of_one_dimension_cross_validate();

    test_cuda_concat();
    test_cuda_concat_cross_validate();
    test_cuda_slice();
    test_cuda_slice_cross_validate();
    test_cuda_repeat();
    test_cuda_repeat_cross_validate();
}

} // namespace julie
