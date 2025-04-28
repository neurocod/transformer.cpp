// tests/test_tensor.cpp
#include "utils/Tensor.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <algorithm> // For std::equal

// Helper function to print tensor details
void print_tensor(const Tensor &t, const std::string &name)
{
    std::cout << "--- Tensor: " << name << " ---" << std::endl;
    std::cout << "Shape: [";
    for (size_t i = 0; i < t.get_shape().size(); ++i)
    {
        std::cout << t.get_shape()[i] << (i == t.get_shape().size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;

    std::cout << "Data: [";
    const auto &data = t.get_data();
    size_t print_limit = std::min((size_t)20, data.size()); // Print up to 20 elements
    for (size_t i = 0; i < print_limit; ++i)
    {
        std::cout << data[i] << (i == print_limit - 1 ? "" : ", ");
    }
    if (data.size() > print_limit)
    {
        std::cout << ", ...";
    }
    std::cout << "]" << std::endl;

    std::cout << "Grad: [";
    const auto &grad = t.get_grad();
    print_limit = std::min((size_t)20, grad.size()); // Print up to 20 elements
    for (size_t i = 0; i < print_limit; ++i)
    {
        std::cout << grad[i] << (i == print_limit - 1 ? "" : ", ");
    }
    if (grad.size() > print_limit)
    {
        std::cout << ", ...";
    }
    std::cout << "]" << std::endl;
    std::cout << "--------------------------" << std::endl;
}

// Helper function to check if two tensors are approximately equal
bool are_tensors_equal(const Tensor &t1, const Tensor &t2, float tolerance = 1e-9)
{
    if (t1.get_shape() != t2.get_shape())
    {
        std::cerr << "Shape mismatch!" << std::endl;
        return false;
    }
    const auto &data1 = t1.get_data();
    const auto &data2 = t2.get_data();
    if (data1.size() != data2.size())
    {
        std::cerr << "Data size mismatch!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < data1.size(); ++i)
    {
        if (std::abs(data1[i] - data2[i]) > tolerance)
        {
            std::cerr << "Data mismatch at index " << i << ": " << data1[i] << " vs " << data2[i] << std::endl;
            return false;
        }
    }
    return true;
}

void test_addition()
{
    std::cout << "\nTest Case: Addition" << std::endl;
    Tensor scalar_like({1, 1}, {{2.0}});
    Tensor matrix_2d({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor add_result1 = scalar_like + matrix_2d;
    assert(add_result1.get_shape() == std::vector<int>({2, 3}));
    assert(are_tensors_equal(add_result1, Tensor({2, 3}, {3.0, 4.0, 5.0, 6.0, 7.0, 8.0})));
    std::cout << "Addition broadcasting 1x1 to 2x3 test passed." << std::endl;

    Tensor vector_1d({1, 3}, {1.0, 2.0, 3.0});
    Tensor add_result2 = vector_1d + matrix_2d;
    assert(add_result2.get_shape() == std::vector<int>({2, 3}));
    assert(are_tensors_equal(add_result2, Tensor({2, 3}, {2.0, 4.0, 6.0, 5.0, 7.0, 9.0})));
    std::cout << "Addition broadcasting 1x3 to 2x3 test passed." << std::endl;

    Tensor tensor_3d_a({2, 1, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor tensor_3d_b({2, 2, 1}, {1.0, 2.0, 3.0, 4.0});
    Tensor add_result3 = tensor_3d_a + tensor_3d_b;
    assert(add_result3.get_shape() == std::vector<int>({2, 2, 3}));
    assert(are_tensors_equal(add_result3, Tensor({2, 2, 3}, {2.0, 3.0, 4.0,
                                                             3.0, 4.0, 5.0,
                                                             7.0, 8.0, 9.0,
                                                             8.0, 9.0, 10.0})));
    std::cout << "Addition broadcasting 3D tensors test passed." << std::endl;
}

void test_subtraction()
{
    std::cout << "\nTest Case: Subtraction" << std::endl;
    Tensor scalar_like({1, 1}, {{2.0}});
    Tensor matrix_2d({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor sub_result1 = matrix_2d - scalar_like;
    assert(sub_result1.get_shape() == std::vector<int>({2, 3}));
    assert(are_tensors_equal(sub_result1, Tensor({2, 3}, {-1.0, 0.0, 1.0, 2.0, 3.0, 4.0})));
    std::cout << "Subtraction broadcasting 1x1 from 2x3 test passed." << std::endl;

    Tensor vector_1d({1, 3}, {1.0, 2.0, 3.0});
    Tensor sub_result2 = matrix_2d - vector_1d;
    assert(sub_result2.get_shape() == std::vector<int>({2, 3}));
    assert(are_tensors_equal(sub_result2, Tensor({2, 3}, {0.0, 0.0, 0.0, 3.0, 3.0, 3.0})));
    std::cout << "Subtraction broadcasting 1x3 from 2x3 test passed." << std::endl;

    Tensor tensor_3d_a({2, 1, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor tensor_3d_b({2, 2, 1}, {1.0, 2.0, 3.0, 4.0});
    Tensor sub_result3 = tensor_3d_a - tensor_3d_b;
    assert(sub_result3.get_shape() == std::vector<int>({2, 2, 3}));
    assert(are_tensors_equal(sub_result3, Tensor({2, 2, 3}, {0.0, 1.0, 2.0,
                                                             -1.0, 0.0, 1.0,
                                                             1.0, 2.0, 3.0,
                                                             0.0, 1.0, 2.0})));
    std::cout << "Subtraction broadcasting 3D tensors test passed." << std::endl;
}

void test_dot_product()
{
    std::cout << "\nTest Case: Dot Product (Matrix Multiplication)" << std::endl;
    Tensor mat_a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor mat_b({3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    Tensor dot_result_2d = mat_a.dot(mat_b);
    assert(dot_result_2d.get_shape() == std::vector<int>({2, 2}));
    assert(are_tensors_equal(dot_result_2d, Tensor({2, 2}, {58.0f, 64.0f, 139.0f, 154.0f})));
    std::cout << "2D Matrix multiplication test passed." << std::endl;

    Tensor batch_mat_a({2, 2, 3}, {1.0f, 2.0f, 3.0f,
                                   4.0f, 5.0f, 6.0f,
                                   -1.0f, -2.0f, -3.0f,
                                   -4.0f, -5.0f, -6.0f});
    Tensor batch_mat_b({2, 3, 2}, {7.0f, 8.0f,
                                   9.0f, 10.0f,
                                   11.0f, 12.0f,
                                   -7.0f, -8.0f,
                                   -9.0f, -10.0f,
                                   -11.0f, -12.0f});

    Tensor dot_result_3d = batch_mat_a.dot(batch_mat_b);
    assert(dot_result_3d.get_shape() == std::vector<int>({2, 2, 2}));
    assert(are_tensors_equal(dot_result_3d, Tensor({2, 2, 2}, {58.0f, 64.0f,
                                                               139.0f, 154.0f,
                                                               58.0f, 64.0f,
                                                               139.0f, 154.0f})));
    std::cout << "3D x 3D Batched matrix multiplication test passed." << std::endl;

    Tensor mat_a_broadcast({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor batch_mat_b_broadcast({2, 3, 2}, {7.0f, 8.0f,
                                             9.0f, 10.0f,
                                             11.0f, 12.0f,
                                             -7.0f, -8.0f,
                                             -9.0f, -10.0f,
                                             -11.0f, -12.0f});

    Tensor dot_result_broadcast = mat_a_broadcast.dot(batch_mat_b_broadcast);
    assert(dot_result_broadcast.get_shape() == std::vector<int>({2, 2, 2}));
    assert(are_tensors_equal(dot_result_broadcast, Tensor({2, 2, 2}, {58.0f, 64.0f,
                                                                      139.0f, 154.0f,
                                                                      -58.0f, -64.0f,
                                                                      -139.0f, -154.0f})));
    std::cout << "2D x 3D Batched matrix multiplication (broadcasting) test passed." << std::endl;

    Tensor A({2, 2, 3}, {1, 2, 3,
                         4, 5, 6,
                         -1, -2, -3,
                         -4, -5, -6});

    Tensor B({3, 2}, {
                         7,
                         8,
                         9,
                         10,
                         11,
                         12,
                     });
    Tensor C = A.dot(B);

    assert(C.get_shape() == std::vector<int>({2, 2, 2}));
    assert(are_tensors_equal(C, Tensor({2, 2, 2}, {58, 64,
                                                   139, 154,
                                                   -58, -64,
                                                   -139, -154})));
    std::cout << "3D x 2D Batched matrix multiplication (broadcasting) test passed." << std::endl;
}

void test_transpose()
{
    std::cout << "\nTest Case: Transpose" << std::endl;
    Tensor trans_t1({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor trans_result1 = trans_t1.transpose({1, 0});
    assert(trans_result1.get_shape() == std::vector<int>({3, 2}));
    assert(are_tensors_equal(trans_result1, Tensor({3, 2}, {1.0, 4.0, 2.0, 5.0, 3.0, 6.0})));
    std::cout << "2D Transpose test passed." << std::endl;

    Tensor trans_t2({2, 3, 4});
    Tensor trans_result2 = trans_t2.transpose({2, 1, 0});
    assert(trans_result2.get_shape() == std::vector<int>({4, 3, 2}));
    std::cout << "3D Transpose test passed." << std::endl;
}

void test_reshape()
{
    std::cout << "\nTest Case: Reshape" << std::endl;
    Tensor reshape_t1({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor reshape_result1 = reshape_t1.reshape({6, 1});
    assert(reshape_result1.get_shape() == std::vector<int>({6, 1}));
    assert(are_tensors_equal(reshape_result1, Tensor({6, 1}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})));
    std::cout << "Reshape 2x3 to 6x1 test passed." << std::endl;

    Tensor reshape_result2 = reshape_t1.reshape({3, 2});
    assert(reshape_result2.get_shape() == std::vector<int>({3, 2}));
    assert(are_tensors_equal(reshape_result2, Tensor({3, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0})));
    std::cout << "Reshape 2x3 to 3x2 test passed." << std::endl;

    Tensor reshape_t2({2, 2, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    Tensor reshape_result3 = reshape_t2.reshape({4, 2});
    assert(reshape_result3.get_shape() == std::vector<int>({4, 2}));
    assert(are_tensors_equal(reshape_result3, Tensor({4, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0})));
    std::cout << "Reshape 2x2x2 to 4x2 test passed." << std::endl;
}

void test_backward_pass()
{
    std::cout << "\nTest Case: Backward Pass through a Chain of Operations" << std::endl;

    Tensor A({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor B({2, 2}, {0.5f, 1.0f, 1.5f, 2.0f});
    Tensor C({2, 2}, {2.0f, 1.0f, 1.0f, 2.0f});
    Tensor D({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});

    Tensor E = A + B;
    Tensor F = E * C;
    Tensor G = F.dot(D);
    Tensor H = G.transpose({1, 0});
    Tensor I = H.reshape({4, 1});

    Tensor grad_I({4, 1}, {1.0f, 1.0f, 1.0f, 1.0f});
    I.backward(grad_I);

    Tensor expected_grad_A({2, 2}, {2.0f, 1.0f, 1.0f, 2.0f});
    Tensor expected_grad_B({2, 2}, {2.0f, 1.0f, 1.0f, 2.0f});
    Tensor expected_grad_C({2, 2}, {1.5f, 3.0f, 4.5f, 6.0f});
    Tensor expected_grad_D({2, 2}, {7.5f, 7.5f, 15.0f, 15.0f});

    assert(are_tensors_equal(Tensor(A.get_shape(), A.get_grad()), expected_grad_A));
    std::cout << "  Gradient of A verified." << std::endl;
    assert(are_tensors_equal(Tensor(B.get_shape(), B.get_grad()), expected_grad_B));
    std::cout << "  Gradient of B verified." << std::endl;
    assert(are_tensors_equal(Tensor(C.get_shape(), C.get_grad()), expected_grad_C));
    std::cout << "  Gradient of C verified." << std::endl;
    assert(are_tensors_equal(Tensor(D.get_shape(), D.get_grad()), expected_grad_D));
    std::cout << "  Gradient of D verified." << std::endl;

    std::cout << "Backward Pass through a Chain of Operations test passed." << std::endl;
}

void test_backward_pass_with_broadcasting_operations()
{
    std::cout << "\nTest Case: Backward Pass with Broadcasting" << std::endl;

    Tensor A({2, 3}, {1.0f, 2.0f, 3.0f,
                      4.0f, 5.0f, 6.0f});
    Tensor B({3}, {0.5f, 1.0f, 1.5f});
    Tensor C({2, 1}, {2.0f, 3.0f});
    Tensor D({1, 3}, {1.0f, 2.0f, 3.0f});

    Tensor E = A + B;
    Tensor F = E - D;
    Tensor G = F * C;
    Tensor H = G.reshape({6, 1});

    Tensor grad_H({6, 1}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    H.backward(grad_H);

    Tensor expected_grad_A({2, 3}, {2.0f, 2.0f, 2.0, 3.0f, 3.0f, 3.0f});
    Tensor expected_grad_B({3}, {5.0f, 5.0f, 5.0f});
    Tensor expected_grad_C({2, 1}, {3.0f, 12.0f});
    Tensor expected_grad_D({1, 3}, {-5.0f, -5.0f, -5.0f});

    assert(are_tensors_equal(Tensor(A.get_shape(), A.get_grad()), expected_grad_A));
    std::cout << "  Gradient of A verified." << std::endl;
    assert(are_tensors_equal(Tensor(B.get_shape(), B.get_grad()), expected_grad_B));
    std::cout << "  Gradient of B verified." << std::endl;
    assert(are_tensors_equal(Tensor(C.get_shape(), C.get_grad()), expected_grad_C));
    std::cout << "  Gradient of C verified." << std::endl;
    assert(are_tensors_equal(Tensor(D.get_shape(), D.get_grad()), expected_grad_D));
    std::cout << "  Gradient of D verified." << std::endl;

    std::cout << "Backward Pass with Broadcasting over test passed." << std::endl;
}

int main()
{
    std::cout << "Starting Tensor class tests..." << std::endl;

    test_addition();
    test_subtraction();
    test_dot_product();
    test_transpose();
    test_reshape();
    test_backward_pass();
    test_backward_pass_with_broadcasting_operations();

    std::cout << "\nAll Tensor class tests completed." << std::endl;
    return 0;
}
