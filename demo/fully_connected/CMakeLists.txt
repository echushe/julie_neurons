cmake_minimum_required(VERSION 3.9)
project(julie_neurons VERSION 1.0.1 DESCRIPTION "a simple nn framework")

SET(CMAKE_C_COMPILER g++)
SET(CMAK_CXX_COMPILER g++)
add_compile_options(-std=c++14)
add_compile_options(-m64)

include_directories( 
    /usr/local/cuda/include
    ../../julie_neurons/julie/la
    ../../julie_neurons/julie/op
    ../../julie_neurons/julie/nn
    ../../julie_neurons/julie/nn/var
    ../../julie_neurons/julie/nn/func
    ../../julie_neurons/julie/nn/opt
    ../../dataset
    )

link_directories(
    /usr/local/cuda/lib64)

find_library(JULIE_NEURONS_LIB julie_neurons HINTS ../../julie_neurons/build)

add_executable(fc_demo
    main.cpp
    ../../dataset/Mnist.cpp
)

target_link_libraries(
    fc_demo
    ${JULIE_NEURONS_LIB}
)
