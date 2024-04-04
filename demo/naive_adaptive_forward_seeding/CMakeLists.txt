cmake_minimum_required(VERSION 3.9)
project(fully_connected VERSION 1.0.1 DESCRIPTION "fc network via a simple nn framework")

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
    set(
        CUDA_INCLUDE_DIRS
        /usr/local/cuda/include
        ../../julie_neurons/julie/la/cuda
    )
    add_definitions(-DWITH_CUDA=1)
endif()

if (WITH_OPENBLAS)  
    add_definitions(-DWITH_OPENBLAS=1)
endif()

if (WITH_ONEDNN)
    add_definitions(-DWITH_ONEDNN=1)
endif()

include_directories( 
    ${CUDA_INCLUDE_DIRS}
    ../../julie_neurons/julie
    ../../julie_neurons/julie/la
    ../../julie_neurons/julie/la/cpu
    ../../julie_neurons/julie/op
    ../../julie_neurons/julie/nn
    ../../julie_neurons/julie/nn/var
    ../../julie_neurons/julie/nn/func
    ../../julie_neurons/julie/nn/opt
    ../../dataset
    )

if (WITH_CUDA)
    link_directories(/usr/local/cuda/lib64)
endif()

if (WITH_CUDNN)
    add_definitions(-DWITH_CUDNN=1)
endif()

find_library(JULIE_NEURONS_LIB julie_neurons HINTS ../../julie_neurons/build)

add_executable(nafs_demo
    main.cpp
    ../../dataset/Mnist.cpp
)

target_link_libraries(
    nafs_demo
    ${JULIE_NEURONS_LIB}
)
