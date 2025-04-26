// tests/test_tensor.cpp
#include "utils/Tensor.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <algorithm> // For std::equal

// Helper function to print tensor details
void print_tensor(const Tensor& t, const std::string& name) {
    std::cout << "--- Tensor: " << name << " ---" << std::endl;
    std::cout << "Shape: [";
    for (size_t i = 0; i < t.get_shape().size(); ++i) {
        std::cout << t.get_shape()[i] << (i == t.get_shape().size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;

    std::cout << "Data: [";
    const auto& data = t.get_data();
    size_t print_limit = std::min((size_t)20, data.size()); // Print up to 20 elements
    for (size_t i = 0; i < print_limit; ++i) {
        std::cout << data[i] << (i == print_limit - 1 ? "" : ", ");
    }
    if (data.size() > print_limit) {
        std::cout << ", ...";
    }
    std::cout << "]" << std::endl;

    std::cout << "Grad: [";
    const auto& grad = t.get_grad();
    print_limit = std::min((size_t)20, grad.size()); // Print up to 20 elements
    for (size_t i = 0; i < print_limit; ++i) {
        std::cout << grad[i] << (i == print_limit - 1 ? "" : ", ");
    }
    if (grad.size() > print_limit) {
        std::cout << ", ...";
    }
    std::cout << "]" << std::endl;
    std::cout << "--------------------------" << std::endl;
}

// Helper function to check if two tensors are approximately equal
bool are_tensors_equal(const Tensor& t1, const Tensor& t2, float tolerance = 1e-9) {
    if (t1.get_shape() != t2.get_shape()) {
        std::cerr << "Shape mismatch!" << std::endl;
        return false;
    }
    const auto& data1 = t1.get_data();
    const auto& data2 = t2.get_data();
    if (data1.size() != data2.size()) {
         std::cerr << "Data size mismatch!" << std::endl;
         return false;
    }

    for (size_t i = 0; i < data1.size(); ++i) {
        if (std::abs(data1[i] - data2[i]) > tolerance) {
             std::cerr << "Data mismatch at index " << i << ": " << data1[i] << " vs " << data2[i] << std::endl;
            return false;
        }
    }
    return true;
}


int main() {
    std::cout << "Starting Tensor class tests..." << std::endl;

    // Test Case: Addition with Broadcasting
    std::cout << "\nTest Case: Addition" << std::endl;
    // Broadcasting scalar-like tensor (1x1) with 2D tensor
    Tensor scalar_like({1, 1}, {2.0});
    Tensor matrix_2d({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor add_result1 = scalar_like + matrix_2d;
    assert(add_result1.get_shape() == std::vector<int>({2, 3}));
    assert(are_tensors_equal(add_result1, Tensor({2, 3}, {3.0, 4.0, 5.0, 6.0, 7.0, 8.0})));
    std::cout << "Addition broadcasting 1x1 to 2x3 test passed." << std::endl;

    // Broadcasting 1D tensor with 2D tensor
    Tensor vector_1d({1, 3}, {1.0, 2.0, 3.0}); // 1x3 vector
    Tensor add_result2 = vector_1d + matrix_2d;
    assert(add_result2.get_shape() == std::vector<int>({2, 3}));
    assert(are_tensors_equal(add_result2, Tensor({2, 3}, {2.0, 4.0, 6.0, 5.0, 7.0, 9.0})));
    std::cout << "Addition broadcasting 1x3 to 2x3 test passed." << std::endl;

    // Broadcasting with 3D tensors
    Tensor tensor_3d_a({2, 1, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor tensor_3d_b({2, 2, 1}, {1.0, 2.0, 3.0, 4.0});
    Tensor add_result3 = tensor_3d_a + tensor_3d_b;
    assert(add_result3.get_shape() == std::vector<int>({2, 2, 3}));
    assert(are_tensors_equal(add_result3, Tensor({2, 2, 3}, {
        2.0, 3.0, 4.0,  // batch 0, row 0
        3.0, 4.0, 5.0,  // batch 0, row 1
        7.0, 8.0, 9.0,  // batch 1, row 0
        8.0, 9.0, 10.0  // batch 1, row 1
    })));
    std::cout << "Addition broadcasting 3D tensors test passed." << std::endl;

    // Subtraction with Broadcasting
    std::cout << "\nTest Case: Subtraction" << std::endl;
    // Broadcasting scalar-like tensor with 2D tensor
    Tensor sub_result1 = matrix_2d - scalar_like;
    assert(sub_result1.get_shape() == std::vector<int>({2, 3}));
    assert(are_tensors_equal(sub_result1, Tensor({2, 3}, {-1.0, 0.0, 1.0, 2.0, 3.0, 4.0})));
    std::cout << "Subtraction broadcasting 1x1 from 2x3 test passed." << std::endl;

    // Broadcasting 1D tensor from 2D tensor
    Tensor sub_result2 = matrix_2d - vector_1d;
    assert(sub_result2.get_shape() == std::vector<int>({2, 3}));
    assert(are_tensors_equal(sub_result2, Tensor({2, 3}, {0.0, 0.0, 0.0, 3.0, 3.0, 3.0})));
    std::cout << "Subtraction broadcasting 1x3 from 2x3 test passed." << std::endl;

    // Test 3: Broadcasting with 3D tensors
    Tensor sub_result3 = tensor_3d_a - tensor_3d_b;
    assert(sub_result3.get_shape() == std::vector<int>({2, 2, 3}));
    assert(are_tensors_equal(sub_result3, Tensor({2, 2, 3}, {
        0.0, 1.0, 2.0,   // batch 0, row 0
        -1.0, 0.0, 1.0,  // batch 0, row 1
        1.0, 2.0, 3.0,   // batch 1, row 0
        0.0, 1.0, 2.0    // batch 1, row 1
    })));
    std::cout << "Subtraction broadcasting 3D tensors test passed." << std::endl;

    // Test Case: Dot Product (Matrix Multiplication)
    std::cout << "\nTest Case: Dot Product (Matrix Multiplication)" << std::endl;
    // 2D Matrix Multiplication
    Tensor mat_a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}); // 2x3 matrix
    Tensor mat_b({3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}); // 3x2 matrix

    Tensor dot_result_2d = mat_a.dot(mat_b); // Result should be 2x2
    assert(dot_result_2d.get_shape() == std::vector<int>({2, 2}));
    // Expected result:
    // (1*7 + 2*9 + 3*11) = 7 + 18 + 33 = 58
    // (1*8 + 2*10 + 3*12) = 8 + 20 + 36 = 64
    // (4*7 + 5*9 + 6*11) = 28 + 45 + 66 = 139
    // (4*8 + 5*10 + 6*12) = 32 + 50 + 72 = 154
    assert(are_tensors_equal(dot_result_2d, Tensor({2, 2}, {58.0f, 64.0f, 139.0f, 154.0f})));
    std::cout << "2D Matrix multiplication test passed." << std::endl;

    // 3D x 3D Batched Matrix Multiplication
    // Batch size 2, (2x3) * (3x2)
    Tensor batch_mat_a({2, 2, 3}, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, // First batch (2x3)
        -1.0f, -2.0f, -3.0f,
        -4.0f, -5.0f, -6.0f  // Second batch (2x3)
    });
    Tensor batch_mat_b({2, 3, 2}, {
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f, // First batch (3x2)
        -7.0f, -8.0f,
        -9.0f, -10.0f,
        -11.0f, -12.0f // Second batch (3x2)
    });

    Tensor dot_result_3d = batch_mat_a.dot(batch_mat_b); // Result should be 2x2x2
    assert(dot_result_3d.get_shape() == std::vector<int>({2, 2, 2}));
    // Expected result for first batch (same as 2D case):
    // {58.0f, 64.0f, 139.0f, 154.0f}
    // Expected result for second batch:
    // (-1*-7 + -2*-9 + -3*-11) = 7 + 18 + 33 = 58
    // (-1*-8 + -2*-10 + -3*-12) = 8 + 20 + 36 = 64
    // (-4*-7 + -5*-9 + -6*-11) = 28 + 45 + 66 = 139
    // (-4*-8 + -5*-10 + -6*-12) = 32 + 50 + 72 = 154
    assert(are_tensors_equal(dot_result_3d, Tensor({2, 2, 2}, {
        58.0f, 64.0f,
        139.0f, 154.0f,
        58.0f, 64.0f,
        139.0f, 154.0f
    })));
    std::cout << "3D x 3D Batched matrix multiplication test passed." << std::endl;

    // 2D x 3D Batched Matrix Multiplication (Broadcasting)
    // (2x3) * (2, 3x2) -> Result should be (2, 2x2)
    Tensor mat_a_broadcast({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}); // 2x3 matrix
    Tensor batch_mat_b_broadcast({2, 3, 2}, {
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f, // First batch (3x2)
        -7.0f, -8.0f,
        -9.0f, -10.0f,
        -11.0f, -12.0f // Second batch (3x2)
    });

    Tensor dot_result_broadcast = mat_a_broadcast.dot(batch_mat_b_broadcast); // Result should be 2x2x2
    assert(dot_result_broadcast.get_shape() == std::vector<int>({2, 2, 2}));
     // Expected result: The 2x3 matrix should be broadcast across the batch dimension of the 3x2x2 tensor.
     // This means the same 2x3 matrix is multiplied with each 3x2 matrix in the batch.
     // The results for each batch should be the same as the 2D case.
     assert(are_tensors_equal(dot_result_broadcast, Tensor({2, 2, 2}, {
        58.0f, 64.0f,
        139.0f, 154.0f,
        -58.0f, -64.0f,
        -139.0f, -154.0f
    })));
    std::cout << "2D x 3D Batched matrix multiplication (broadcasting) test passed." << std::endl;


    // Test Case: Transpose
    std::cout << "\nTest Case: Transpose" << std::endl;
    Tensor trans_t1({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}); // 2x3 tensor
    Tensor trans_result1 = trans_t1.transpose({1, 0}); // Transpose to 3x2
    assert(trans_result1.get_shape() == std::vector<int>({3, 2}));
    assert(are_tensors_equal(trans_result1, Tensor({3, 2}, {1.0, 4.0, 2.0, 5.0, 3.0, 6.0})));
    std::cout << "2D Transpose test passed." << std::endl;

    Tensor trans_t2({2, 3, 4}); // 2x3x4 tensor
    // Permute dimensions 0 and 2: result shape 4x3x2
    Tensor trans_result2 = trans_t2.transpose({2, 1, 0});
     assert(trans_result2.get_shape() == std::vector<int>({4, 3, 2}));
     // (Manual check of a few elements could be done here if needed)
     std::cout << "3D Transpose test passed." << std::endl;


    // Test Case: Reshape
    std::cout << "\nTest Case: Reshape" << std::endl;
    Tensor reshape_t1({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}); // 2x3 tensor
    Tensor reshape_result1 = reshape_t1.reshape({6, 1}); // Reshape to 6x1
    assert(reshape_result1.get_shape() == std::vector<int>({6, 1}));
    assert(are_tensors_equal(reshape_result1, Tensor({6, 1}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})));
    std::cout << "Reshape 2x3 to 6x1 test passed." << std::endl;

    Tensor reshape_result2 = reshape_t1.reshape({3, 2}); // Reshape to 3x2
    assert(reshape_result2.get_shape() == std::vector<int>({3, 2}));
    assert(are_tensors_equal(reshape_result2, Tensor({3, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})));
    std::cout << "Reshape 2x3 to 3x2 test passed." << std::endl;

    Tensor reshape_t2({2, 2, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}); // 2x2x2 tensor
    Tensor reshape_result3 = reshape_t2.reshape({4, 2}); // Reshape to 4x2
    assert(reshape_result3.get_shape() == std::vector<int>({4, 2}));
     assert(are_tensors_equal(reshape_result3, Tensor({4, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0})));
    std::cout << "Reshape 2x2x2 to 4x2 test passed." << std::endl;

    std::cout << "\nAll Tensor class tests completed." << std::endl;

    return 0;
}
