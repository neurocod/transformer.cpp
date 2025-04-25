#include "utils/Tensor.h"
#include <stdexcept>
#include <cassert>
#include <iostream>
#include <numeric>
#include <algorithm>

// Default constructor
Tensor::Tensor() : shape_{{}}, data_{{}}, grad_{{}} {}

// Constructor with shape
Tensor::Tensor(const std::vector<int>& shape) : shape_{shape} {
    size_t total_elements = num_elements();
    data_.resize(total_elements, 0.0);
    grad_.resize(total_elements, 0.0);
}

// Constructor with shape and data
Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data) : shape_{shape}, data_{data} {
    size_t total_elements = num_elements();
    if (data_.size() != total_elements) {
         throw std::runtime_error("Data size does not match the specified shape in constructor.");
    }
    grad_.resize(total_elements, 0.0);
}

// Destructor
Tensor::~Tensor() {}

// Getters
const std::vector<int>& Tensor::get_shape() const {
    return shape_;
}

const std::vector<float>& Tensor::get_data() const {
    return data_;
}

const std::vector<float>& Tensor::get_grad() const {
    return grad_;
}

// Setter for data
void Tensor::set_data(const std::vector<float>& data) {
    if (data.size() != num_elements()) {
        throw std::runtime_error("Data size mismatch in set_data.");
    }
    data_ = data;
    grad_.assign(num_elements(), 0.0); // Reset gradient on setting new data
}

// Helper to calculate the linear index from multi-dimensional indices
size_t Tensor::get_linear_index(const std::vector<int>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::runtime_error("Number of indices must match tensor dimensions.");
    }
    size_t linear_index = 0;
    size_t stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            throw std::runtime_error("Index out of bounds.");
        }
        linear_index += indices[i] * stride;
        stride *= shape_[i];
    }
    return linear_index;
}

// Get element by multi-dimensional index
float Tensor::get(const std::vector<int>& indices) const {
    return data_[get_linear_index(indices)];
}

// Set element by multi-dimensional index
void Tensor::set(const std::vector<int>& indices, float value) {
    data_[get_linear_index(indices)] = value;
}


// Basic tensor operations

Tensor Tensor::operator+(const Tensor& other) const {
    if (!are_shapes_compatible(other)) {
        throw std::runtime_error("Tensor shapes do not match for addition.");
    }

    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (!are_shapes_compatible(other)) {
        throw std::runtime_error("Tensor shapes do not match for subtraction.");
    }

    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const { // Element-wise multiplication
     if (!are_shapes_compatible(other)) {
        throw std::runtime_error("Tensor shapes do not match for element-wise multiplication.");
    }
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

// Matrix multiplication (simplified for 2D for now, needs expansion for general tensor contraction)
Tensor Tensor::dot(const Tensor& other) const {
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::runtime_error("Dot product currently only supported for 2D tensors (matrices).");
    }
    if (shape_[1] != other.shape_[0]) {
        throw std::runtime_error("Tensor shapes are not compatible for matrix multiplication.");
    }

    int rows_a = shape_[0];
    int cols_a = shape_[1];
    int cols_b = other.shape_[1];

    Tensor result({rows_a, cols_b}); // Result shape (rows_a, cols_b)

    // Manual matrix multiplication
    for (int i = 0; i < rows_a; ++i) {
        for (int j = 0; j < cols_b; ++j) {
            float sum = 0.0;
            for (int k = 0; k < cols_a; ++k) {
                // Get elements from original tensors using 2D indices
                sum += this->get({i, k}) * other.get({k, j});
            }
            // Set element in the result tensor using 2D indices
            result.set({i, j}, sum);
        }
    }

    return result;
}

Tensor Tensor::transpose(const std::vector<int>& permutation) const {
    if (permutation.size() != shape_.size()) {
        throw std::runtime_error("Permutation size must match tensor dimension.");
    }
    // Check if permutation is valid (contains all dimensions exactly once)
    std::vector<int> sorted_permutation = permutation;
    std::sort(sorted_permutation.begin(), sorted_permutation.end());
    for(size_t i = 0; i < sorted_permutation.size(); ++i) {
        if (sorted_permutation[i] != i) {
            throw std::runtime_error("Invalid permutation for transpose.");
        }
    }

    // Calculate the new shape based on the permutation
    std::vector<int> new_shape(shape_.size());
    for(size_t i = 0; i < shape_.size(); ++i) {
        new_shape[i] = shape_[permutation[i]];
    }

    Tensor result(new_shape);

    // Manually copy elements based on the permutation
    std::vector<int> original_indices(shape_.size());
    std::vector<int> transposed_indices(shape_.size());

    // Iterate through all elements in the original tensor
    size_t total_elements = num_elements();
    for (size_t i = 0; i < total_elements; ++i) {
        // Convert linear index i back to original multi-dimensional indices
        size_t temp_index = i;
        size_t stride = 1;
        for (int d = shape_.size() - 1; d >= 0; --d) {
            original_indices[d] = (temp_index / stride) % shape_[d];
            stride *= shape_[d];
        }

        // Calculate the corresponding indices in the transposed tensor based on permutation
        for(size_t d = 0; d < shape_.size(); ++d) {
            transposed_indices[d] = original_indices[permutation[d]];
        }

        // Get the value from the original tensor and set it in the result tensor
        result.set(transposed_indices, data_[i]);
    }

    return result;
}

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    size_t new_num_elements = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
    if (num_elements() != new_num_elements) {
        throw std::runtime_error("Total number of elements must remain the same during reshape.");
    }

    Tensor result(new_shape, data_);

    return result;
}

// Gradient handling
void Tensor::zero_grad() {
    grad_.assign(num_elements(), 0.0);
}

void Tensor::backward(const Tensor& grad_output) {
    // This is a placeholder
    if (!are_shapes_compatible(grad_output)) {
         throw std::runtime_error("Gradient shape mismatch in backward.");
    }
    // This simplified version just adds the incoming gradient
    for(size_t i = 0; i < grad_.size(); ++i) {
        grad_[i] += grad_output.data_[i];
    }
}
