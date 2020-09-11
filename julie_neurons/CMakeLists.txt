cmake_minimum_required(VERSION 3.9)
project(julie_neurons VERSION 1.0.1 DESCRIPTION "a simple nn framework")

SET(CMAKE_C_COMPILER g++)
SET(CMAK_CXX_COMPILER g++)
add_compile_options(-std=c++14)
add_compile_options(-m64)

include_directories( 
    /usr/local/cuda/include
    ../julie_neurons/julie/la
    ../julie_neurons/julie/op
    ../julie_neurons/julie/nn
    ../julie_neurons/julie/nn/var
    ../julie_neurons/julie/nn/func
    ../julie_neurons/julie/nn/opt
    )

link_directories(
    /usr/local/cuda/lib64)


add_library(julie_neurons SHARED

    julie/la/Shape.cpp
    julie/la/Coordinate.cpp
    julie/la/Vector.cpp
    julie/la/DMatrix.cpp

    julie/op/variable.cpp
    julie/op/function.cpp
    julie/op/graph.cpp

    julie/nn/var/tensor.cpp
    julie/nn/var/scalar.cpp

    julie/nn/func/matmul.cpp
    julie/nn/func/add.cpp
    julie/nn/func/scale.cpp
    julie/nn/func/tanh.cpp
    julie/nn/func/arctan.cpp
    julie/nn/func/relu.cpp
    julie/nn/func/prelu.cpp
    julie/nn/func/softmax.cpp
    julie/nn/func/sigmoid.cpp
    julie/nn/func/conv2d_op.cpp

    julie/nn/func/softmax_crossentropy.cpp
    julie/nn/func/sigmoid_crossentropy.cpp

    julie/nn/opt/optimizer.cpp
    julie/nn/opt/sgd.cpp
)

target_link_libraries(
    julie_neurons
    )

#install(TARGETS cpp_warmup RUNTIME DESTINATION bin)
