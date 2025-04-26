#include "utils/Tensor.h"
#include <stdexcept>
#include <cassert>
#include <iostream>
#include <numeric>
#include <algorithm>

// Default constructor
Tensor::Tensor() : shape_{{}}, data_{{}}, grad_{{}}, creator_op_(OperationType::None), parents_{{}} {}

// Constructor with shape
Tensor::Tensor(const std::vector<int> &shape) : shape_{shape}, creator_op_(OperationType::None), parents_{{}}
{
    size_t total_elements = num_elements();
    data_.resize(total_elements, 0.0f);
    grad_.resize(total_elements, 0.0f);
}

// Constructor with shape and data
Tensor::Tensor(const std::vector<int> &shape, const std::vector<float> &data) : shape_{shape}, data_{data}, creator_op_(OperationType::None), parents_{{}}
{
    size_t total_elements = num_elements();
    if (data_.size() != total_elements)
    {
        throw std::runtime_error("Data size does not match the specified shape in constructor.");
    }
    grad_.resize(total_elements, 0.0f);
}

// Destructor
Tensor::~Tensor() {}

// Getters
const std::vector<int> &Tensor::get_shape() const
{
    return shape_;
}

const std::vector<float> &Tensor::get_data() const
{
    return data_;
}

const std::vector<float> &Tensor::get_grad() const
{
    return grad_;
}

// Setter for data
void Tensor::set_data(const std::vector<float> &data)
{
    if (data.size() != num_elements())
    {
        throw std::runtime_error("Data size mismatch in set_data.");
    }
    data_ = data;
    grad_.assign(num_elements(), 0.0f);
    // When data is explicitly set, this tensor is a leaf node, not derived from an operation.
    creator_op_ = OperationType::None;
    parents_.clear();
}

// Helper to calculate the linear index from multi-dimensional indices
size_t Tensor::get_linear_index(const std::vector<int> &indices) const
{
    if (indices.size() != shape_.size())
    {
        throw std::runtime_error("Number of indices must match tensor dimensions.");
    }
    size_t linear_index = 0;
    size_t stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i)
    {
        if (indices[i] < 0 || indices[i] >= shape_[i])
        {
            throw std::runtime_error("Index out of bounds.");
        }
        linear_index += indices[i] * stride;
        stride *= shape_[i];
    }
    return linear_index;
}

// Get element by multi-dimensional index
float Tensor::get(const std::vector<int> &indices) const
{
    return data_[get_linear_index(indices)];
}

// Set element by multi-dimensional index
void Tensor::set(const std::vector<int> &indices, float value)
{
    data_[get_linear_index(indices)] = value;
}

// Basic tensor operations

bool Tensor::is_broadcastable(const std::vector<int> &shape1, const std::vector<int> &shape2)
{
    int max_dims = std::max(shape1.size(), shape2.size());
    std::vector<int> padded1(max_dims, 1);
    std::vector<int> padded2(max_dims, 1);

    // Pad shapes with 1s from the left
    std::copy(shape1.rbegin(), shape1.rend(), padded1.rbegin());
    std::copy(shape2.rbegin(), shape2.rend(), padded2.rbegin());

    for (int i = 0; i < max_dims; ++i)
    {
        if (padded1[i] != padded2[i] && padded1[i] != 1 && padded2[i] != 1)
        {
            return false;
        }
    }
    return true;
}

std::vector<int> Tensor::calculate_broadcast_shape(const std::vector<int> &shape1, const std::vector<int> &shape2)
{
    int max_dims = std::max(shape1.size(), shape2.size());
    std::vector<int> padded1(max_dims, 1);
    std::vector<int> padded2(max_dims, 1);
    std::vector<int> result_shape(max_dims);

    std::copy(shape1.rbegin(), shape1.rend(), padded1.rbegin());
    std::copy(shape2.rbegin(), shape2.rend(), padded2.rbegin());

    for (int i = 0; i < max_dims; ++i)
    {
        if (padded1[i] != padded2[i] && padded1[i] != 1 && padded2[i] != 1)
        {
            throw std::runtime_error("Shapes are not broadcastable");
        }
        result_shape[i] = std::max(padded1[i], padded2[i]);
    }

    return result_shape;
}

Tensor Tensor::broadcast_to(const std::vector<int> &new_shape) const
{
    if (!is_broadcastable(shape_, new_shape))
    {
        throw std::runtime_error("Cannot broadcast to target shape");
    }

    Tensor result(new_shape);
    std::vector<int> input_idx(shape_.size());
    std::vector<int> output_idx(new_shape.size());

    // Iterate through all elements in the result tensor
    size_t total_elements = result.num_elements();
    for (size_t i = 0; i < total_elements; ++i)
    {
        // Calculate output indices
        size_t temp = i;
        for (int d = new_shape.size() - 1; d >= 0; --d)
        {
            output_idx[d] = temp % new_shape[d];
            temp /= new_shape[d];
        }

        // Calculate corresponding input indices
        for (int d = 0; d < shape_.size(); ++d)
        {
            int offset = new_shape.size() - shape_.size();
            input_idx[d] = shape_[d] == 1 ? 0 : output_idx[d + offset];
        }

        result.set(output_idx, get(input_idx));
    }

    return result;
}

Tensor Tensor::operator+(const Tensor &other) const
{
    // Broadcasting logic for forward pass
    std::vector<int> result_shape = calculate_broadcast_shape(shape_, other.shape_);
    Tensor broadcasted_a = broadcast_to(result_shape);
    Tensor broadcasted_b = other.broadcast_to(result_shape);

    Tensor result(result_shape);
    // Record operation and parents for backward pass
    result.creator_op_ = OperationType::Add;
    result.parents_.push_back(this);
    result.parents_.push_back(&other);

    for (size_t i = 0; i < result.data_.size(); ++i)
    {
        result.data_[i] = broadcasted_a.data_[i] + broadcasted_b.data_[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor &other) const
{
    std::vector<int> result_shape = calculate_broadcast_shape(shape_, other.shape_);
    Tensor broadcasted_a = broadcast_to(result_shape);
    Tensor broadcasted_b = other.broadcast_to(result_shape);

    Tensor result(result_shape);
    result.creator_op_ = OperationType::Sub;
    result.parents_.push_back(this);
    result.parents_.push_back(&other);

    for (size_t i = 0; i < result.data_.size(); ++i)
    {
        result.data_[i] = broadcasted_a.data_[i] - broadcasted_b.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor &other) const
{
    std::vector<int> result_shape = calculate_broadcast_shape(shape_, other.shape_);
    Tensor broadcasted_a = broadcast_to(result_shape);
    Tensor broadcasted_b = other.broadcast_to(result_shape);

    Tensor result(result_shape);
    result.creator_op_ = OperationType::Mul;
    result.parents_.push_back(this);
    result.parents_.push_back(&other);

    for (size_t i = 0; i < result.data_.size(); ++i)
    {
        result.data_[i] = broadcasted_a.data_[i] * broadcasted_b.data_[i];
    }
    return result;
}

Tensor Tensor::dot(const Tensor &other) const
{
    if (shape_.size() < 2 || other.shape_.size() < 2)
    {
        throw std::runtime_error("Tensors must be at least 2D for matrix multiplication.");
    }

    std::vector<int> batch_dims_a(shape_.begin(), shape_.end() - 2);
    std::vector<int> batch_dims_b(other.shape_.begin(), other.shape_.end() - 2);

    int rows_a = shape_[shape_.size() - 2];
    int cols_a = shape_[shape_.size() - 1];
    int rows_b = other.shape_[other.shape_.size() - 2];
    int cols_b = other.shape_[other.shape_.size() - 1];

    if (cols_a != rows_b)
    {
        throw std::runtime_error("Inner matrix dimensions must match.");
    }

    // Determine output batch dimensions using broadcasting rules
    std::vector<int> batch_shape = calculate_broadcast_shape(batch_dims_a, batch_dims_b);

    std::vector<int> result_shape = batch_shape;
    result_shape.push_back(rows_a);
    result_shape.push_back(cols_b);

    Tensor result(result_shape);
    result.creator_op_ = OperationType::Dot;
    result.parents_.push_back(this);
    result.parents_.push_back(&other);

    // Calculate total number of batches
    size_t total_batches = batch_shape.empty() ? 1 : std::accumulate(batch_shape.begin(), batch_shape.end(), 1, std::multiplies<int>());

    // Calculate strides for efficient indexing
    std::vector<size_t> stride_a(shape_.size());
    std::vector<size_t> stride_b(other.shape_.size());
    std::vector<size_t> stride_result(result_shape.size());

    // Calculate strides for the original tensors
    stride_a.back() = 1;
    for (int i = shape_.size() - 2; i >= 0; --i)
        stride_a[i] = stride_a[i + 1] * shape_[i + 1];
    stride_b.back() = 1;
    for (int i = other.shape_.size() - 2; i >= 0; --i)
        stride_b[i] = stride_b[i + 1] * other.shape_[i + 1];
    // Calculate strides for the result tensor
    stride_result.back() = 1;
    for (int i = result_shape.size() - 2; i >= 0; --i)
        stride_result[i] = stride_result[i + 1] * result_shape[i + 1];

    // For each batch
    for (size_t batch_linear_idx = 0; batch_linear_idx < total_batches; ++batch_linear_idx)
    {
        // Calculate batch indices for result
        std::vector<int> batch_idx(batch_shape.size());
        size_t temp_batch = batch_linear_idx;
        for (int i = batch_shape.size() - 1; i >= 0; --i)
        {
            batch_idx[i] = temp_batch % batch_shape[i];
            temp_batch /= batch_shape[i];
        }

        // Calculate linear offset for the current batch in result tensor
        size_t result_batch_offset = 0;
        for (size_t i = 0; i < batch_idx.size(); ++i)
        {
            result_batch_offset += batch_idx[i] * stride_result[i];
        }

        for (int i = 0; i < rows_a; ++i)
        {
            for (int j = 0; j < cols_b; ++j)
            {
                float sum = 0.0f;
                for (int k = 0; k < cols_a; ++k)
                {
                    // Calculate linear indices for elements in the current batch
                    size_t linear_idx_a = 0;
                    size_t linear_idx_b = 0;

                    // Handle broadcasting for batch dimensions
                    for (size_t d = 0; d < batch_dims_a.size(); ++d)
                    {
                        // If the batch dimension of tensor A is 1, always use index 0 for that dimension
                        linear_idx_a += (batch_dims_a[d] == 1 ? 0 : batch_idx[d]) * stride_a[d];
                    }
                    for (size_t d = 0; d < batch_dims_b.size(); ++d)
                    {
                        // If the batch dimension of tensor B is 1, always use index 0 for that dimension
                        linear_idx_b += (batch_dims_b[d] == 1 ? 0 : batch_idx[d]) * stride_b[d];
                    }

                    // Add matrix indices
                    linear_idx_a += i * stride_a[shape_.size() - 2] + k * stride_a[shape_.size() - 1];
                    linear_idx_b += k * stride_b[other.shape_.size() - 2] + j * stride_b[other.shape_.size() - 1];

                    sum += data_[linear_idx_a] * other.data_[linear_idx_b];
                }

                // Calculate linear index for result element
                size_t result_linear_idx = result_batch_offset + i * stride_result[result_shape.size() - 2] + j * stride_result[result_shape.size() - 1];
                result.data_[result_linear_idx] = sum;
            }
        }
    }

    return result;
}

Tensor Tensor::transpose(const std::vector<int> &permutation) const
{
    if (permutation.size() != shape_.size())
    {
        throw std::runtime_error("Permutation size must match tensor dimension.");
    }
    // Check if permutation is valid (contains all dimensions exactly once)
    std::vector<int> sorted_permutation = permutation;
    std::sort(sorted_permutation.begin(), sorted_permutation.end());
    for (size_t i = 0; i < sorted_permutation.size(); ++i)
    {
        if (sorted_permutation[i] != i)
        {
            throw std::runtime_error("Invalid permutation for transpose.");
        }
    }

    // Calculate the new shape based on the permutation
    std::vector<int> new_shape(shape_.size());
    for (size_t i = 0; i < shape_.size(); ++i)
    {
        new_shape[i] = shape_[permutation[i]];
    }

    Tensor result(new_shape);
    result.creator_op_ = OperationType::Transpose;
    result.parents_.push_back(this);

    // Manually copy elements based on the permutation
    std::vector<int> original_indices(shape_.size());
    std::vector<int> transposed_indices(shape_.size());

    // Iterate through all elements in the original tensor
    size_t total_elements = num_elements();
    for (size_t i = 0; i < total_elements; ++i)
    {
        // Convert linear index i back to original multi-dimensional indices
        size_t temp_index = i;
        size_t stride = 1;
        for (int d = shape_.size() - 1; d >= 0; --d)
        {
            original_indices[d] = (temp_index / stride) % shape_[d];
            stride *= shape_[d];
        }

        // Calculate the corresponding indices in the transposed tensor based on permutation
        for (size_t d = 0; d < shape_.size(); ++d)
        {
            transposed_indices[d] = original_indices[permutation[d]];
        }

        // Get the value from the original tensor and set it in the result tensor
        result.set(transposed_indices, data_[i]);
    }

    return result;
}

Tensor Tensor::reshape(const std::vector<int> &new_shape) const
{
    size_t new_num_elements = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
    if (num_elements() != new_num_elements)
    {
        throw std::runtime_error("Total number of elements must remain the same during reshape.");
    }

    Tensor result(new_shape, data_);
    result.creator_op_ = OperationType::Reshape;
    result.parents_.push_back(this);

    return result;
}

// Gradient handling
void Tensor::zero_grad()
{
    grad_.assign(num_elements(), 0.0f);
}

void Tensor::backward(const Tensor &grad_output)
{
    if (shape_ != grad_output.get_shape())
    {
        throw std::runtime_error("Gradient shape mismatch in backward.");
    }

    // Accumulate the incoming gradient
    for (size_t i = 0; i < grad_.size(); ++i)
    {
        grad_[i] += grad_output.data_[i];
    }

    // If this tensor was the result of an operation, propagate the gradient to its parents
    if (creator_op_ != OperationType::None && !parents_.empty())
    {
        switch (creator_op_)
        {
        case OperationType::Add:
            backward_add(grad_output);
            break;
        case OperationType::Sub:
            backward_sub(grad_output);
            break;
        case OperationType::Mul:
            backward_mul(grad_output);
            break;
        case OperationType::Dot:
            backward_dot(grad_output);
            break;
        case OperationType::Transpose:
            backward_transpose(grad_output);
            break;
        case OperationType::Reshape:
            backward_reshape(grad_output);
            break;
        }
    }
}

// Helper to reduce gradient for broadcasting
void Tensor::reduce_gradient(const Tensor &grad_output, Tensor &parent_grad, const std::vector<int> &parent_shape)
{
    // This function sums the gradient of grad_output along dimensions that were broadcast to match the parent's original shape.

    std::vector<int> grad_shape = grad_output.get_shape();
    int max_dims = std::max(grad_shape.size(), parent_shape.size());

    std::vector<int> padded_grad_shape(max_dims, 1);
    std::vector<int> padded_parent_shape(max_dims, 1);

    std::copy(grad_shape.rbegin(), grad_shape.rend(), padded_grad_shape.rbegin());
    std::copy(parent_shape.rbegin(), parent_shape.rend(), padded_parent_shape.rbegin());

    // Initialize parent_grad with zeros and the correct shape
    parent_grad = Tensor(parent_shape);
    parent_grad.zero_grad();

    // Calculate strides for gradient and parent shapes
    std::vector<size_t> grad_stride(max_dims);
    std::vector<size_t> parent_stride(max_dims);

    if (max_dims > 0)
    {
        grad_stride.back() = 1;
        for (int i = max_dims - 2; i >= 0; --i)
            grad_stride[i] = grad_stride[i + 1] * padded_grad_shape[i + 1];
        parent_stride.back() = 1;
        for (int i = max_dims - 2; i >= 0; --i)
            parent_stride[i] = parent_stride[i + 1] * padded_parent_shape[i + 1];
    }

    // Iterate through the gradient tensor
    size_t total_grad_elements = grad_output.num_elements();
    std::vector<int> grad_indices(max_dims);
    std::vector<int> parent_indices(max_dims);

    for (size_t i = 0; i < total_grad_elements; ++i)
    {
        // Convert linear index to multi-dimensional indices for grad_output
        size_t temp_grad_idx = i;
        for (int d = max_dims - 1; d >= 0; --d)
        {
            grad_indices[d] = temp_grad_idx / grad_stride[d];
            temp_grad_idx %= grad_stride[d];
        }

        // Calculate corresponding parent indices
        for (int d = 0; d < max_dims; ++d)
        {
            parent_indices[d] = (padded_parent_shape[d] == 1) ? 0 : grad_indices[d];
        }

        // Calculate linear index for parent gradient
        size_t parent_linear_idx = 0;
        for (size_t d = 0; d < max_dims; ++d)
        {
            parent_linear_idx += parent_indices[d] * parent_stride[d];
        }

        // Accumulate the gradient
        parent_grad.data_[parent_linear_idx] += grad_output.get_data()[i];
    }
}

// Backward methods for specific operations

void Tensor::backward_add(const Tensor &grad_output)
{
    /*
    Gradient of Z = A + B with respect to A is dZ/dA = 1.
    Gradient of Z = A + B with respect to B is dZ/dB = 1.
    Chain rule: dL/dA = dL/dZ * dZ/dA = grad_output * 1 = grad_output
    dL/dB = dL/dZ * dZ/dB = grad_output * 1 = grad_output
    */

    if (parents_.size() == 2)
    {
        const Tensor *parent_a = parents_[0];
        const Tensor *parent_b = parents_[1];

        Tensor grad_a_propagated;
        reduce_gradient(grad_output, grad_a_propagated, parent_a->get_shape());
        const_cast<Tensor *>(parent_a)->backward(grad_a_propagated);

        Tensor grad_b_propagated;
        reduce_gradient(grad_output, grad_b_propagated, parent_b->get_shape());
        const_cast<Tensor *>(parent_b)->backward(grad_b_propagated);
    }
    else
    {
        std::cerr << "Error: Add operation expected 2 parents, but found " << parents_.size() << std::endl;
    }
}

void Tensor::backward_sub(const Tensor &grad_output)
{
    /*
    Gradient of Z = A - B with respect to A is dZ/dA = 1.
    Gradient of Z = A - B with respect to B is dZ/dB = -1.
    Chain rule: dL/dA = dL/dZ * dZ/dA = grad_output * 1 = grad_output
    dL/dB = dL/dZ * dZ/dB = grad_output * -1 = -grad_output
    */

    if (parents_.size() == 2)
    {
        const Tensor *parent_a = parents_[0];
        const Tensor *parent_b = parents_[1];

        // Propagate gradient to parent A (dL/dA = grad_output)
        Tensor grad_a_propagated;
        reduce_gradient(grad_output, grad_a_propagated, parent_a->get_shape());
        const_cast<Tensor *>(parent_a)->backward(grad_a_propagated);

        // Propagate gradient to parent B (dL/dB = -grad_output)
        Tensor neg_grad_output(grad_output.get_shape());
        std::vector<float> neg_data(grad_output.num_elements());
        for (size_t i = 0; i < neg_data.size(); ++i)
        {
            neg_data[i] = -grad_output.get_data()[i];
        }
        neg_grad_output.set_data(neg_data);

        Tensor grad_b_propagated;
        reduce_gradient(neg_grad_output, grad_b_propagated, parent_b->get_shape());
        const_cast<Tensor *>(parent_b)->backward(grad_b_propagated);
    }
    else
    {
        std::cerr << "Error: Sub operation expected 2 parents, but found " << parents_.size() << std::endl;
    }
}

void Tensor::backward_mul(const Tensor &grad_output)
{
    /*
    Gradient of Z = A * B (element-wise) with respect to A is dZ/dA = B.
    Gradient of Z = A * B (element-wise) with respect to B is dZ/dB = A.
    Chain rule: dL/dA = dL/dZ * dZ/dA = grad_output * B
    dL/dB = dL/dZ * dZ/dB = grad_output * A
    */
    if (parents_.size() == 2)
    {
        const Tensor *parent_a = parents_[0];
        const Tensor *parent_b = parents_[1];

        // Calculate gradient for parent A: grad_output * parent_b
        Tensor grad_a_intermediate = grad_output * (*parent_b);

        // Reduce gradient for parent A
        Tensor grad_a_propagated;
        reduce_gradient(grad_a_intermediate, grad_a_propagated, parent_a->get_shape());
        const_cast<Tensor *>(parent_a)->backward(grad_a_propagated);

        // Calculate gradient for parent B: grad_output * parent_a
        Tensor grad_b_intermediate = grad_output * (*parent_a);

        // Reduce gradient for parent B
        Tensor grad_b_propagated;
        reduce_gradient(grad_b_intermediate, grad_b_propagated, parent_b->get_shape());
        const_cast<Tensor *>(parent_b)->backward(grad_b_propagated);
    }
    else
    {
        std::cerr << "Error: Mul operation expected 2 parents, but found " << parents_.size() << std::endl;
    }
}

void Tensor::backward_dot(const Tensor &grad_output)
{
    /*
    Backward pass for Z = A.dot(B):
    dL/dA = dL/dZ . B^T
    dL/dB = A^T . dL/dZ
    This requires implementing matrix multiplication for gradients and handling batch dimensions and broadcasting correctly.
    */

    if (parents_.size() == 2)
    {
        const Tensor *parent_a = parents_[0];
        const Tensor *parent_b = parents_[1];

        // Calculate dL/dA = dL/dZ . B^T
        // Need to transpose the last two dimensions of parent_b
        std::vector<int> b_transpose_perm(parent_b->get_shape().size());
        std::iota(b_transpose_perm.begin(), b_transpose_perm.end(), 0);
        std::swap(b_transpose_perm[b_transpose_perm.size() - 1], b_transpose_perm[b_transpose_perm.size() - 2]);
        Tensor parent_b_transposed = parent_b->transpose(b_transpose_perm);

        Tensor grad_a_intermediate = grad_output.dot(parent_b_transposed);

        // Reduce gradient for parent A
        Tensor grad_a_propagated;
        reduce_gradient(grad_a_intermediate, grad_a_propagated, parent_a->get_shape());
        const_cast<Tensor *>(parent_a)->backward(grad_a_propagated);

        // Calculate dL/dB = A^T . dL/dZ
        // Need to transpose the last two dimensions of parent_a
        std::vector<int> a_transpose_perm(parent_a->get_shape().size());
        std::iota(a_transpose_perm.begin(), a_transpose_perm.end(), 0);
        std::swap(a_transpose_perm[a_transpose_perm.size() - 1], a_transpose_perm[a_transpose_perm.size() - 2]);
        Tensor parent_a_transposed = parent_a->transpose(a_transpose_perm);

        Tensor grad_b_intermediate = parent_a_transposed.dot(grad_output);

        // Reduce gradient for parent B
        Tensor grad_b_propagated;
        reduce_gradient(grad_b_intermediate, grad_b_propagated, parent_b->get_shape());
        const_cast<Tensor *>(parent_b)->backward(grad_b_propagated);
    }
    else
    {
        std::cerr << "Error: Dot operation expected 2 parents, but found " << parents_.size() << std::endl;
    }
}

void Tensor::backward_transpose(const Tensor &grad_output)
{
    /*
    Backward pass for Y = X.transpose(p):
    dL/dX = dL/dY.transpose(p_inverse)
    We need the original permutation 'p' used in the forward pass to calculate 'p_inverse'.
    This requires storing the permutation in the Tensor or Operation.
    */

    if (parents_.size() == 1)
    {
        const Tensor *parent = parents_[0];
        if (grad_output.get_shape().size() == 2)
        {
            Tensor grad_input = grad_output.transpose({1, 0});
            grad_input.creator_op_ = OperationType::None;
            grad_input.parents_.clear();
            const_cast<Tensor *>(parent)->backward(grad_input);
            std::cerr << "Warning: Backward pass for 2D Transpose operation is implemented assuming {1,0} permutation." << std::endl;
        }
        else
        {
            std::cerr << "Warning: Backward pass for multi-dimensional Transpose operation is not implemented." << std::endl;
        }
    }
    else
    {
        std::cerr << "Error: Transpose operation expected 1 parent, but found " << parents_.size() << std::endl;
    }
}

void Tensor::backward_reshape(const Tensor &grad_output)
{
    /*
    Backward pass for Y = X.reshape(new_shape):
    dL/dX = dL/dY.reshape(original_shape_of_X)
    We need the original shape of the parent tensor before the reshape.
    This requires storing the original shape in the Tensor or Operation.
    */

    if (parents_.size() == 1)
    {
        const Tensor *parent = parents_[0];
        Tensor grad_input = grad_output.reshape(parent->get_shape());
        grad_input.creator_op_ = OperationType::None;
        grad_input.parents_.clear();
        const_cast<Tensor *>(parent)->backward(grad_input);
        std::cerr << "Warning: Backward pass for Reshape operation is implemented assuming parent's current shape is the original shape." << std::endl;
    }
    else
    {
        std::cerr << "Error: Reshape operation expected 1 parent, but found " << parents_.size() << std::endl;
    }
}
