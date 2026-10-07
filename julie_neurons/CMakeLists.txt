cmake_minimum_required(VERSION 3.9)

set(CUDA_COMPILER "")
if (WITH_CUDA)
    set(CUDA_COMPILER CUDA)

    # Select nvcc: -DCMAKE_CUDA_COMPILER, -DCUDA_TOOLKIT_ROOT_DIR (e.g. /usr/local/cuda-12.4),
    # the CUDACXX environment variable, or nvcc in PATH / /usr/local/cuda/bin
    if (NOT CMAKE_CUDA_COMPILER)
        if (CUDA_TOOLKIT_ROOT_DIR)
            set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
        elseif (DEFINED ENV{CUDACXX})
            set(CMAKE_CUDA_COMPILER $ENV{CUDACXX})
        else()
            find_program(CMAKE_CUDA_COMPILER nvcc PATHS /usr/local/cuda/bin)
        endif()
    endif()

    # Default GPU architectures depend on what this nvcc release supports
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_CUDA_COMPILER)
        execute_process(COMMAND ${CMAKE_CUDA_COMPILER} --version OUTPUT_VARIABLE NVCC_VERSION_OUTPUT)
        string(REGEX MATCH "release ([0-9]+\\.[0-9]+)" NVCC_VERSION_MATCH "${NVCC_VERSION_OUTPUT}")
        set(NVCC_VERSION ${CMAKE_MATCH_1})
        if (NVCC_VERSION VERSION_GREATER_EQUAL 13.0)
            # CUDA 13 dropped sm_50 - sm_72 (Maxwell, Pascal, Volta)
            set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
        elseif (NVCC_VERSION VERSION_GREATER_EQUAL 11.1)
            set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86)
        elseif (NVCC_VERSION VERSION_GREATER_EQUAL 11.0)
            set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 61 70 75)
        endif()
        message(STATUS "nvcc ${NVCC_VERSION}: default CUDA architectures ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()

project(julie_neurons LANGUAGES CXX ${CUDA_COMPILER} VERSION 1.0.1 DESCRIPTION "a simple nn framework")

set(CMAKE_SYSTEM_NAME Linux)

# In case this program should run on aarch64 (ARM) platform
if (ARCH_ARM)
    MESSAGE(STATUS "ARCH_ARM")
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    add_definitions(-DARCH_ARM=1)
    #add_compile_options(-march=armv8-a)
else ()
    add_compile_options(-m64)
endif()

set(CMAKE_C_COMPILER g++)
set(CMAKE_CXX_COMPILER g++)

add_compile_options(-std=c++14)

set(CUDA_INCLUDE_DIRS "")

if (WITH_CUDA)
    # Toolkit that owns the selected nvcc
    get_filename_component(CUDA_BIN_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
    get_filename_component(CUDA_TOOLKIT_ROOT_DIR ${CUDA_BIN_DIR} DIRECTORY)
    message(STATUS "CUDA toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
    set(
        CUDA_INCLUDE_DIRS
        ${CUDA_TOOLKIT_ROOT_DIR}/include
        julie/la/cuda
    )
    add_definitions(-DWITH_CUDA=1)
endif()

set (OPENBLAS_INCLUDE_DIRS, "")

if (WITH_OPENBLAS)  
    set(OPENBLAS_INCLUDE_DIRS ../OpenBLAS/include)
    add_definitions(-DWITH_OPENBLAS=1)
endif()

set (ONEDNN_INCLUDE_DIRS, "")

if (WITH_ONEDNN)
    set(ONEDNN_INCLUDE_DIRS ../oneDNN/include)
    add_definitions(-DWITH_ONEDNN=1)
endif()

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OPENBLAS_INCLUDE_DIRS}
    ${ONEDNN_INCLUDE_DIRS}
    julie
    julie/la
    julie/la/cpu
    julie/op
    julie/nn
    julie/nn/var
    julie/nn/func
    julie/nn/opt
    )

set(CUBLAS_LIB "")

if (WITH_CUDA)
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    set(
        CUDA_BUILD_FILES
        julie/la/cuda/nsqrt.cpp
        julie/la/cuda/Matrix_CUDA.cu
        julie/la/cuda/Matrix_CUDA_func.cu
        julie/la/cuda/Matrix_CUDA_func_adv.cu
        julie/la/cuda/Activations_CUDA.cu
        julie/la/cuda/Activations_CUDA_adv.cu
        julie/la/cuda/Losses_CUDA.cu
        julie/la/cpu/Matrix_CPU.cu
    )
    set(CUBLAS_LIB -lcublas)
endif()

set(CUDNN_LIB "")

if (WITH_CUDNN)
    add_definitions(-DWITH_CUDNN=1)
    set(
        CUDNN_BUILD_FILES
        julie/la/Conv2dCuDNN.cu
    )
    set(CUDNN_LIB -lcudnn)
endif()

if (WITH_OPENBLAS)
    find_library(OPENBLAS_LIB openblas HINTS ../OpenBLAS/lib)
endif()

if (WITH_ONEDNN)
    find_library(ONEDNN_LIB dnnl HINTS ../oneDNN/lib)
    set(
        ONEDNN_BUILD_FILES
        julie/la/OneDNNHelper.cpp
        julie/la/Conv2dOneDNN.cpp
        julie/la/Pool2dOneDNN.cpp
    )
endif()

add_library(julie_neurons SHARED

    julie/la/Shape.cpp
    julie/la/Coordinate.cpp
    julie/la/iMatrix.cpp
    julie/la/iMatrix_func.cpp
    julie/la/iMatrix_func_adv.cpp
    julie/la/Activations.cpp
    julie/la/Activations_adv.cpp
    julie/la/Losses.cpp
    julie/la/Conv2d.cpp

    ${ONEDNN_BUILD_FILES}

    julie/la/cpu/Matrix_CPU.cpp
    julie/la/cpu/Matrix_CPU_func.cpp
    julie/la/cpu/Matrix_CPU_func_adv.cpp
    julie/la/cpu/Activations_CPU.cpp
    julie/la/cpu/Activations_CPU_adv.cpp
    julie/la/cpu/Losses_CPU.cpp
    julie/la/cpu/SLMatrix.cpp
    
    ${CUDA_BUILD_FILES}

    ${CUDNN_BUILD_FILES}

    julie/op/variable.cpp
    julie/op/function.cpp
    julie/op/graph.cpp

    julie/nn/var/tensor.cpp
    julie/nn/var/scalar.cpp

    julie/nn/func/matmul.cpp
    julie/nn/func/add.cpp
    julie/nn/func/scalar_add.cpp
    julie/nn/func/multiply.cpp
    julie/nn/func/scale.cpp
    julie/nn/func/tanh.cpp
    julie/nn/func/arctan.cpp
    julie/nn/func/relu.cpp
    julie/nn/func/prelu.cpp
    julie/nn/func/softmax.cpp
    julie/nn/func/sigmoid.cpp
    julie/nn/func/conv2d_op.cpp
    julie/nn/func/maxpool.cpp
    julie/nn/func/avgpool.cpp
    julie/nn/func/concat.cpp
    julie/nn/func/sum.cpp
    julie/nn/func/abs.cpp

    julie/nn/func/softmax_crossentropy.cpp
    julie/nn/func/sigmoid_crossentropy.cpp
    julie/nn/func/half_squareerror.cpp

    julie/nn/opt/optimizer.cpp
    julie/nn/opt/sgd.cpp
)

target_link_libraries(
    julie_neurons
    ${OPENBLAS_LIB}
    ${ONEDNN_LIB}
    ${CUBLAS_LIB}
    ${CUDNN_LIB}
#     -lcudadevrt
    )

if (WITH_CUDA)
    # target_compile_options(julie_neurons PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true -gencode arch=compute_61,code=sm_61>)
    if (ARCH_ARM)
        target_compile_options(julie_neurons PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_53,code=sm_53>)
    endif()

    #install(TARGETS cpp_warmup RUNTIME DESTINATION bin)
endif()
