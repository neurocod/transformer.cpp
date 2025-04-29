cmake_minimum_required(VERSION 3.10)

project(TransformerCPP)

include_directories(include)

add_executable(neural_network
    src/main.cpp
    src/utils/Tensor.cpp
    src/layers/Linear.cpp
    src/layers/Activations.cpp
    src/utils/Optimizer.cpp
    src/utils/LossFunction.cpp
    src/layers/LayerNorm.cpp
    src/layers/PositionwiseFeedForward.cpp
    src/layers/MultiHeadAttention.cpp
    src/layers/Dropout.cpp
)

add_executable(test_tensor
    tests/test_tensor.cpp
    src/utils/Tensor.cpp
    src/layers/Linear.cpp
)
