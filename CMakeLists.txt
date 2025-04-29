cmake_minimum_required(VERSION 3.10)

project(TransformerCPP)

# Include directories where header files are located
include_directories(include)

# Define the source files for the main application executable
add_executable(neural_network
    src/main.cpp
    src/utils/Tensor.cpp
    src/layers/Linear.cpp
    src/layers/Activations.cpp
    src/utils/Optimizer.cpp
)

# Define the source files for the test executable
add_executable(test_tensor
    tests/test_tensor.cpp
    src/utils/Tensor.cpp
    src/layers/Linear.cpp
)

# If you later add a test framework like GoogleTest, you can link it like this:
# target_link_libraries(test_tensor gtest pthread)
