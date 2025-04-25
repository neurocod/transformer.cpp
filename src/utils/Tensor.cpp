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

Tensor Tensor::dot(const Tensor& other) const {
    if (shape_.size() < 2 || other.shape_.size() < 2) {
        throw std::runtime_error("Tensors must be at least 2D for matrix multiplication.");
    }

    // Extract batch dimensions (if any) and matrix dimensions
    std::vector<int> batch_dims_a(shape_.begin(), shape_.end() - 2);
    std::vector<int> batch_dims_b(other.shape_.begin(), other.shape_.end() - 2);
    
    int rows_a = shape_[shape_.size() - 2];
    int cols_a = shape_[shape_.size() - 1];
    int cols_b = other.shape_[other.shape_.size() - 1];
    
    if (cols_a != other.shape_[other.shape_.size() - 2]) {
        throw std::runtime_error("Inner matrix dimensions must match.");
    }

    // Determine output batch dimensions
    std::vector<int> result_shape;
    if (batch_dims_a.empty() && !batch_dims_b.empty()) {
        result_shape = batch_dims_b;
    } else if (!batch_dims_a.empty() && batch_dims_b.empty()) {
        result_shape = batch_dims_a;
    } else if (batch_dims_a == batch_dims_b) {
        result_shape = batch_dims_a;
    } else {
        throw std::runtime_error("Incompatible batch dimensions.");
    }
    
    // Add matrix dimensions to result shape
    result_shape.push_back(rows_a);
    result_shape.push_back(cols_b);
    
    Tensor result(result_shape);
    
    // Calculate total number of batches
    size_t total_batches = result_shape.size() <= 2 ? 1 :
        std::accumulate(result_shape.begin(), result_shape.end() - 2, 1, std::multiplies<int>());
    
    // For each batch
    for (size_t batch = 0; batch < total_batches; ++batch) {
        // Calculate batch indices
        std::vector<int> batch_idx;
        size_t temp_batch = batch;
        for (size_t i = 0; i < result_shape.size() - 2; ++i) {
            batch_idx.push_back(temp_batch % result_shape[i]);
            temp_batch /= result_shape[i];
        }
        
        // Perform matrix multiplication for this batch
        for (int i = 0; i < rows_a; ++i) {
            for (int j = 0; j < cols_b; ++j) {
                float sum = 0.0;
                for (int k = 0; k < cols_a; ++k) {
                    // Create full indices for both tensors
                    std::vector<int> idx_a = batch_dims_a.empty() ? 
                        std::vector<int>{i, k} : 
                        std::vector<int>(batch_idx.begin(), batch_idx.end());
                    std::vector<int> idx_b = batch_dims_b.empty() ? 
                        std::vector<int>{k, j} : 
                        std::vector<int>(batch_idx.begin(), batch_idx.end());
                    
                    if (!batch_dims_a.empty()) {
                        idx_a.push_back(i);
                        idx_a.push_back(k);
                    }
                    if (!batch_dims_b.empty()) {
                        idx_b.push_back(k);
                        idx_b.push_back(j);
                    }
                    
                    sum += get(idx_a) * other.get(idx_b);
                }
                
                // Create full index for result
                std::vector<int> result_idx(batch_idx);
                result_idx.push_back(i);
                result_idx.push_back(j);
                result.set(result_idx, sum);
            }
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
