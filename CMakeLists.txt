cmake_minimum_required(VERSION 3.10)

project(TransformerCPP)

include_directories(include)

add_executable(test_tensor
    tests/test_tensor.cpp
    src/utils/Tensor.cpp
)
