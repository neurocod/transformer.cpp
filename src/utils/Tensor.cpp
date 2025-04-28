#include "utils/Tensor.h"
#include <stdexcept>
#include <cassert>
#include <iostream>
#include <numeric>
#include <algorithm>

std::vector<Tensor *> Tensor::optimizable_tensors_;

// Default constructor
Tensor::Tensor() : shape_{{}}, data_{{}}, grad_{{}}, creator_op_(OperationType::None), parents_{}, is_optimizable_{false} {}

// Constructor with shape
Tensor::Tensor(const std::vector<int> &shape, bool is_optimizable) : shape_{shape}, creator_op_(OperationType::None), parents_{}
{
    size_t total_elements = num_elements();
    data_.resize(total_elements, 0.0f);
    grad_.resize(total_elements, 0.0f);
    if (is_optimizable_)
    {
        optimizable_tensors_.push_back(this);
    }
}

// Constructor with shape and data
Tensor::Tensor(const std::vector<int> &shape, const std::vector<float> &data, bool is_optimizable) : shape_{shape}, data_{data}, creator_op_(OperationType::None), parents_{}
{
    size_t total_elements = num_elements();
    if (data_.size() != total_elements)
    {
        throw std::runtime_error("Data size does not match the specified shape in constructor.");
    }
    grad_.resize(total_elements, 0.0f);
    if (is_optimizable_)
    {
        optimizable_tensors_.push_back(this);
    }
}

// Destructor
Tensor::~Tensor()
{
    if (is_optimizable_)
    {
        optimizable_tensors_.erase(
            std::remove(optimizable_tensors_.begin(), optimizable_tensors_.end(), this),
            optimizable_tensors_.end());
    }
}

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

std::vector<Tensor *> &Tensor::get_optimizable_tensors()
{
    return optimizable_tensors_;
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
    result.parents_.push_back(const_cast<Tensor *>(this));
    result.parents_.push_back(const_cast<Tensor *>(&other));

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
    result.parents_.push_back(const_cast<Tensor *>(this));
    result.parents_.push_back(const_cast<Tensor *>(&other));

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
    result.parents_.push_back(const_cast<Tensor *>(this));
    result.parents_.push_back(const_cast<Tensor *>(&other));

    for (size_t i = 0; i < result.data_.size(); ++i)
    {
        result.data_[i] = broadcasted_a.data_[i] * broadcasted_b.data_[i];
    }
    return result;
}

Tensor Tensor::operator/(const Tensor &other) const
{
    std::vector<int> result_shape = calculate_broadcast_shape(shape_, other.shape_);
    Tensor broadcasted_a = broadcast_to(result_shape);
    Tensor broadcasted_b = other.broadcast_to(result_shape);

    Tensor result(result_shape);
    result.creator_op_ = OperationType::Div;
    result.parents_.push_back(const_cast<Tensor *>(this));   // numerator
    result.parents_.push_back(const_cast<Tensor *>(&other)); // denominator

    for (size_t i = 0; i < result.data_.size(); ++i)
    {
        if (broadcasted_b.data_[i] == 0.0f)
        {
            throw std::runtime_error("Division by zero encountered in Tensor division.");
        }
        result.data_[i] = broadcasted_a.data_[i] / broadcasted_b.data_[i];
    }
    return result;
}

Tensor Tensor::dot(const Tensor &other) const
{
    if (shape_.size() < 1 || other.shape_.size() < 1)
    {
        throw std::runtime_error("Dot product requires tensors with at least 1 dimension.");
    }
    // Vector dot product
    if (shape_.size() == 1 && other.shape_.size() == 1)
    {
        if (shape_[0] != other.shape_[0])
        {
            throw std::runtime_error("Vector dot product requires vectors of the same size.");
        }
        // Result is a scalar (1D tensor with size 1)
        Tensor result({1}, {0.0f}, false);
        result.creator_op_ = OperationType::Dot;
        result.parents_.push_back(const_cast<Tensor *>(this));
        result.parents_.push_back(const_cast<Tensor *>(&other));
        for (size_t i = 0; i < shape_[0]; ++i)
        {
            result.data_[0] += data_[i] * other.data_[i];
        }
        return result;
    }
    // Matrix-vector product
    else if (shape_.size() >= 2 && other.shape_.size() == 1)
    {
        int cols_a = shape_.back();
        int vec_size = other.shape_[0];
        if (cols_a != vec_size)
        {
            throw std::runtime_error("Matrix-vector product: Inner dimensions must match (" +
                                     std::to_string(cols_a) + " vs " + std::to_string(vec_size) + ").");
        }
        // Result shape is the matrix shape excluding the last dimension
        std::vector<int> result_shape(shape_.begin(), shape_.end() - 1);
        Tensor result(result_shape);
        result.creator_op_ = OperationType::Dot;
        result.parents_.push_back(const_cast<Tensor *>(this));
        result.parents_.push_back(const_cast<Tensor *>(&other));

        size_t outer_elements = result.num_elements();
        size_t matrix_stride = cols_a;

        for (size_t i = 0; i < outer_elements; ++i)
        {
            float sum = 0.0f;
            size_t matrix_row_start_idx = i * matrix_stride;
            for (int k = 0; k < cols_a; ++k)
            {
                sum += data_[matrix_row_start_idx + k] * other.data_[k];
            }
            result.data_[i] = sum;
        }
        return result;
    }
    // Matrix-matrix product (batched)
    else if (shape_.size() >= 2 && other.shape_.size() >= 2)
    {
        int rows_a = shape_[shape_.size() - 2];
        int cols_a = shape_[shape_.size() - 1];
        int rows_b = other.shape_[other.shape_.size() - 2];
        int cols_b = other.shape_[other.shape_.size() - 1];

        if (cols_a != rows_b)
        {
            throw std::runtime_error("Matrix multiplication: Inner dimensions must match (" +
                                     std::to_string(cols_a) + " vs " + std::to_string(rows_b) + ").");
        }

        // Batch Dimension Handling
        std::vector<int> batch_dims_a(shape_.begin(), shape_.end() - 2);
        std::vector<int> batch_dims_b(other.shape_.begin(), other.shape_.end() - 2);
        std::vector<int> batch_shape = calculate_broadcast_shape(batch_dims_a, batch_dims_b);

        std::vector<int> result_shape = batch_shape;
        result_shape.push_back(rows_a);
        result_shape.push_back(cols_b);

        Tensor result(result_shape);
        result.creator_op_ = OperationType::Dot;
        result.parents_.push_back(const_cast<Tensor *>(this));
        result.parents_.push_back(const_cast<Tensor *>(&other));

        std::vector<size_t> stride_a = calculate_strides(shape_);
        std::vector<size_t> stride_b = calculate_strides(other.shape_);
        std::vector<size_t> stride_result = calculate_strides(result_shape);

        // Batch Iteration
        size_t total_batches = batch_shape.empty() ? 1 : std::accumulate(batch_shape.begin(), batch_shape.end(), size_t{1}, std::multiplies<size_t>());
        std::vector<size_t> batch_strides_result = calculate_strides(batch_shape);

        for (size_t batch_linear_idx = 0; batch_linear_idx < total_batches; ++batch_linear_idx)
        {
            // Calculate multi-dimensional batch index for current batch
            std::vector<int> batch_idx(batch_shape.size());
            size_t temp_batch_linear = batch_linear_idx;
            for (size_t i = 0; i < batch_shape.size(); ++i)
            {
                if (batch_strides_result[i] == 0)
                    continue;
                batch_idx[i] = temp_batch_linear / batch_strides_result[i];
                temp_batch_linear %= batch_strides_result[i];
            }

            // Calculate linear offset for the start of the current batch in each tensor
            size_t offset_a = 0, offset_b = 0, offset_result = 0;
            int batch_offset_a = batch_shape.size() - batch_dims_a.size();
            int batch_offset_b = batch_shape.size() - batch_dims_b.size();

            for (size_t i = 0; i < batch_shape.size(); ++i)
            {
                // Handle broadcasting: if original batch dim was 1, use index 0, else use current batch_idx
                if (i >= batch_offset_a)
                {
                    offset_a += (batch_dims_a[i - batch_offset_a] == 1 ? 0 : batch_idx[i]) * stride_a[i - batch_offset_a];
                }
                if (i >= batch_offset_b)
                {
                    offset_b += (batch_dims_b[i - batch_offset_b] == 1 ? 0 : batch_idx[i]) * stride_b[i - batch_offset_b];
                }
                offset_result += batch_idx[i] * stride_result[i];
            }

            // Matrix Multiplication for the Current Batch
            for (int i = 0; i < rows_a; ++i)
            {
                for (int j = 0; j < cols_b; ++j)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < cols_a; ++k)
                    {
                        // Calculate linear indices within the current batch
                        size_t idx_a = offset_a + i * stride_a[shape_.size() - 2] + k * stride_a[shape_.size() - 1];
                        size_t idx_b = offset_b + k * stride_b[other.shape_.size() - 2] + j * stride_b[other.shape_.size() - 1];

                        // Bounds checking
                        if (idx_a >= data_.size() || idx_b >= other.data_.size())
                        {
                            throw std::runtime_error("Index out of bounds during dot product calculation.");
                        }
                        sum += data_[idx_a] * other.data_[idx_b];
                    }
                    // Calculate result linear index within the current batch
                    size_t idx_result = offset_result + i * stride_result[result_shape.size() - 2] + j * stride_result[result_shape.size() - 1];
                    if (idx_result >= result.data_.size())
                    {
                        throw std::runtime_error("Result index out of bounds during dot product calculation.");
                    }
                    result.data_[idx_result] = sum;
                }
            }
        }
        return result;
    }
    else
    {
        throw std::runtime_error("Unsupported shapes for dot product.");
    }
}

Tensor Tensor::transpose(const std::vector<int> &permutation) const
{
    if (permutation.size() != shape_.size())
    {
        throw std::runtime_error("Permutation size (" + std::to_string(permutation.size()) +
                                 ") must match tensor dimension (" + std::to_string(shape_.size()) + ").");
    }
    // Check if permutation is valid
    std::vector<int> sorted_permutation = permutation;
    std::sort(sorted_permutation.begin(), sorted_permutation.end());
    bool valid_perm = true;
    if (shape_.size() > 0)
    {
        for (size_t i = 0; i < sorted_permutation.size(); ++i)
        {
            if (sorted_permutation[i] != static_cast<int>(i))
            {
                valid_perm = false;
                break;
            }
        }
    }
    else if (!permutation.empty())
    {
        valid_perm = false;
    }

    if (!valid_perm)
    {
        std::string perm_str = "[";
        for (size_t k = 0; k < permutation.size(); ++k)
            perm_str += std::to_string(permutation[k]) + (k == permutation.size() - 1 ? "" : ", ");
        perm_str += "]";
        throw std::runtime_error("Invalid permutation " + perm_str + " for transpose.");
    }

    std::vector<int> new_shape(shape_.size());
    for (size_t i = 0; i < shape_.size(); ++i)
    {
        new_shape[i] = shape_[permutation[i]];
    }

    Tensor result(new_shape);
    result.creator_op_ = OperationType::Transpose;
    result.parents_.push_back(const_cast<Tensor *>(this));
    result.forward_permutation_ = permutation;

    // Efficient Transposition
    if (num_elements() == 0)
        return result;

    std::vector<size_t> old_strides(shape_.size());
    std::vector<size_t> new_strides(new_shape.size());
    old_strides.back() = 1;
    new_strides.back() = 1;
    for (int i = shape_.size() - 2; i >= 0; --i)
        old_strides[i] = old_strides[i + 1] * shape_[i + 1];
    for (int i = new_shape.size() - 2; i >= 0; --i)
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];

    std::vector<int> current_indices(shape_.size());
    size_t total_elements = num_elements();

    for (size_t i = 0; i < total_elements; ++i)
    {
        // Calculate original multi-dimensional indices from linear index i
        size_t temp_linear_idx = i;
        for (size_t d = 0; d < shape_.size(); ++d)
        {
            if (old_strides[d] == 0)
                continue;
            current_indices[d] = temp_linear_idx / old_strides[d];
            temp_linear_idx %= old_strides[d];
        }

        // Calculate new linear index based on permuted indices
        size_t new_linear_idx = 0;
        for (size_t d = 0; d < new_shape.size(); ++d)
        {
            new_linear_idx += current_indices[permutation[d]] * new_strides[d];
        }

        // Bounds checks
        if (i >= data_.size() || new_linear_idx >= result.data_.size())
        {
            throw std::runtime_error("Index out of bounds during transpose calculation.");
        }
        result.data_[new_linear_idx] = data_[i];
    }

    return result;
}

Tensor Tensor::reshape(const std::vector<int> &new_shape) const
{
    size_t current_elements = num_elements();
    size_t new_num_elements = 1;
    bool has_neg_one = false;
    int neg_one_idx = -1;
    for (size_t i = 0; i < new_shape.size(); ++i)
    {
        if (new_shape[i] == -1)
        {
            if (has_neg_one)
                throw std::runtime_error("Reshape can only have one dimension specified as -1.");
            has_neg_one = true;
            neg_one_idx = i;
        }
        else if (new_shape[i] <= 0)
        {
            throw std::runtime_error("Reshape dimensions must be positive or -1.");
        }
        else
        {
            new_num_elements *= new_shape[i];
        }
    }

    std::vector<int> actual_new_shape = new_shape;
    if (has_neg_one)
    {
        if (current_elements == 0 && new_num_elements > 0)
        {
            throw std::runtime_error("Cannot infer dimension for -1 when reshaping an empty tensor to a non-empty one.");
        }
        if (new_num_elements == 0)
        {
            throw std::runtime_error("Internal error: product of positive dimensions is zero in reshape.");
        }
        if (current_elements % new_num_elements != 0)
        {
            throw std::runtime_error("Cannot infer dimension for -1: total elements not divisible by product of other dimensions.");
        }
        actual_new_shape[neg_one_idx] = current_elements / new_num_elements;
        new_num_elements = current_elements;
    }

    if (current_elements != new_num_elements)
    {
        std::string old_shape_str = "[";
        for (size_t k = 0; k < shape_.size(); ++k)
            old_shape_str += std::to_string(shape_[k]) + (k == shape_.size() - 1 ? "" : ", ");
        old_shape_str += "]";
        std::string new_shape_str = "[";
        for (size_t k = 0; k < new_shape.size(); ++k)
            new_shape_str += std::to_string(new_shape[k]) + (k == new_shape.size() - 1 ? "" : ", ");
        new_shape_str += "]";
        throw std::runtime_error("Total number of elements must remain the same during reshape. Cannot reshape " +
                                 old_shape_str + " (" + std::to_string(current_elements) + " elements) to " +
                                 new_shape_str + " (" + std::to_string(new_num_elements) + " elements).");
    }

    Tensor result(actual_new_shape, data_);
    result.creator_op_ = OperationType::Reshape;
    result.parents_.push_back(const_cast<Tensor *>(this));
    result.original_shape_before_reshape_ = this->shape_;

    return result;
}

Tensor Tensor::sum() const
{
    Tensor result({1});
    result.creator_op_ = OperationType::Sum;
    result.parents_.push_back(const_cast<Tensor *>(this));

    float total_sum = 0.0f;
    for (float val : data_)
    {
        total_sum += val;
    }
    result.data_[0] = total_sum;

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
            std::cout << "Backward Add" << std::endl;
            backward_add(grad_output);
            break;
        case OperationType::Sub:
            std::cout << "Backward Sub" << std::endl;
            backward_sub(grad_output);
            break;
        case OperationType::Mul:
            std::cout << "Backward Mul" << std::endl;
            backward_mul(grad_output);
            break;
        case OperationType::Dot:
            std::cout << "Backward Dot" << std::endl;
            backward_dot(grad_output);
            break;
        case OperationType::Transpose:
            std::cout << "Backward Transpose" << std::endl;
            backward_transpose(grad_output);
            break;
        case OperationType::Reshape:
            std::cout << "Backward Reshape" << std::endl;
            backward_reshape(grad_output);
            break;
        case OperationType::Sum:
            std::cout << "Backward Sum" << std::endl;
            backward_sum(grad_output);
            break;
        case OperationType::Div:
            std::cout << "Backward Div" << std::endl;
            backward_div(grad_output);
            break;
        }
    }
}

// Helper to reduce gradient for broadcasting
void Tensor::reduce_gradient(const Tensor &grad_output, Tensor &parent_grad, const std::vector<int> &parent_shape)
{
    const std::vector<int> &grad_shape = grad_output.get_shape();

    // If shapes are identical, no reduction needed
    if (grad_shape == parent_shape)
    {
        parent_grad = Tensor(parent_shape);
        parent_grad.data_ = grad_output.get_data();
        return;
    }

    std::cout << "Parent shape: [";
    for (size_t i = 0; i < parent_shape.size(); ++i) {
        std::cout << parent_shape[i];
        if (i < parent_shape.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    // Initialize parent_grad with zeros and the correct shape
    parent_grad = Tensor(parent_shape);

    // Identify dimensions to sum over
    std::vector<int> dims_to_sum;
    int max_dims = std::max(grad_shape.size(), parent_shape.size());
    int grad_dim_offset = max_dims - grad_shape.size();
    int parent_dim_offset = max_dims - parent_shape.size();

    for (int i = 0; i < max_dims; ++i)
    {
        int grad_dim = (i < grad_dim_offset) ? 1 : grad_shape[i - grad_dim_offset];
        int parent_dim = (i < parent_dim_offset) ? 1 : parent_shape[i - parent_dim_offset];

        if (parent_dim == 1 && grad_dim > 1)
        {
            dims_to_sum.push_back(i);
        }
        else if (parent_dim != grad_dim && parent_dim != 1)
        {
            // Should have been caught by is_broadcastable earlier
            throw std::runtime_error("Inconsistent shapes found during gradient reduction.");
        }
    }

    // Perform Reduction
    size_t parent_total_elements = parent_grad.num_elements();
    if (parent_total_elements == 0)
        return;

    std::vector<size_t> parent_strides = calculate_strides(parent_shape);
    std::vector<size_t> grad_strides = calculate_strides(grad_shape);

    std::vector<int> parent_indices(parent_shape.size());
    std::vector<int> grad_indices_template(grad_shape.size());

    for (size_t p_idx = 0; p_idx < parent_total_elements; ++p_idx)
    {
        // Get multi-dim indices for parent_grad[p_idx]
        size_t temp_p_idx = p_idx;
        for (int d = parent_shape.size() - 1; d >= 0; --d)
        {
            parent_indices[d] = temp_p_idx % parent_shape[d];
            temp_p_idx /= parent_shape[d];
        }

        // Determine the corresponding range/indices in grad_output to sum
        float sum_val = 0.0f;
        size_t grad_total_elements = grad_output.num_elements();
        std::vector<int> current_grad_indices(grad_shape.size());

        // Iterate through all grad_output elements and check if they map to the current parent element
        for (size_t g_idx = 0; g_idx < grad_total_elements; ++g_idx)
        {
            // Get multi-dim indices for grad_output[g_idx]
            size_t temp_g_idx = g_idx;
            for (int d = grad_shape.size() - 1; d >= 0; --d)
            {
                current_grad_indices[d] = temp_g_idx % grad_shape[d];
                temp_g_idx /= grad_shape[d];
            }

            // Check if this grad element corresponds to the current parent element
            bool corresponds = true;
            int p_offset = grad_shape.size() - parent_shape.size();
            for (size_t d = 0; d < parent_shape.size(); ++d)
            {
                if (parent_shape[d] != 1 && parent_indices[d] != current_grad_indices[d + p_offset])
                {
                    corresponds = false;
                    break;
                }
            }

            if (corresponds)
            {
                sum_val += grad_output.data_[g_idx];
            }
        }

        // Assign the sum to parent_grad.data_
        parent_grad.data_[p_idx] = sum_val;
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
        Tensor *parent_a = parents_[0];
        Tensor *parent_b = parents_[1];

        Tensor grad_a_propagated;
        reduce_gradient(grad_output, grad_a_propagated, parent_a->get_shape());
        parent_a->backward(grad_a_propagated);

        Tensor grad_b_propagated;
        reduce_gradient(grad_output, grad_b_propagated, parent_b->get_shape());
        parent_b->backward(grad_b_propagated);
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
        Tensor *parent_a = parents_[0];
        Tensor *parent_b = parents_[1];

        // Propagate gradient to parent A (dL/dA = grad_output)
        Tensor grad_a_propagated;
        reduce_gradient(grad_output, grad_a_propagated, parent_a->get_shape());
        parent_a->backward(grad_a_propagated);

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
        parent_b->backward(grad_b_propagated);
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
        Tensor *parent_a = parents_[0];
        Tensor *parent_b = parents_[1];

        // Calculate gradient for parent A: grad_output * parent_b
        Tensor grad_a_intermediate = grad_output * (*parent_b);

        // Reduce gradient for parent A
        Tensor grad_a_propagated;
        reduce_gradient(grad_a_intermediate, grad_a_propagated, parent_a->get_shape());
        parent_a->backward(grad_a_propagated);

        // Calculate gradient for parent B: grad_output * parent_a
        Tensor grad_b_intermediate = grad_output * (*parent_a);

        // Reduce gradient for parent B
        Tensor grad_b_propagated;
        reduce_gradient(grad_b_intermediate, grad_b_propagated, parent_b->get_shape());
        parent_b->backward(grad_b_propagated);
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
        Tensor *parent_a = parents_[0];
        Tensor *parent_b = parents_[1];

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
        parent_a->backward(grad_a_propagated);

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
        parent_b->backward(grad_b_propagated);
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
    We use the stored forward permutation to calculate the inverse permutation.
    */

    if (parents_.size() == 1)
    {
        Tensor *parent = parents_[0];

        // Calculate the inverse permutation from the stored forward permutation.
        std::vector<int> inverse_permutation(forward_permutation_.size());
        for (size_t i = 0; i < forward_permutation_.size(); ++i)
        {
            inverse_permutation[forward_permutation_[i]] = i;
        }

        // Transpose the incoming gradient using the inverse permutation.
        Tensor grad_input = grad_output.transpose(inverse_permutation);

        // Propagate the gradient to the parent.
        parent->backward(Tensor(parent->get_shape(), grad_input.get_data()));
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
    We use the stored original shape before reshape to reshape the gradient.
    */

    if (parents_.size() == 1)
    {
        Tensor *parent = parents_[0];

        // Ensure the incoming gradient has the same number of elements as the parent's original shape
        size_t expected_elements = std::accumulate(original_shape_before_reshape_.begin(),
                                                   original_shape_before_reshape_.end(),
                                                   1, std::multiplies<size_t>());
        if (grad_output.num_elements() != expected_elements)
        {
            throw std::runtime_error("Gradient element count mismatch during reshape backward pass.");
        }
        // Reshape the incoming gradient to match the original shape of the parent
        Tensor grad_input_reshaped(original_shape_before_reshape_, grad_output.get_data());

        parent->backward(grad_input_reshaped);
    }
    else
    {
        std::cerr << "Error: Reshape operation expected 1 parent, but found " << parents_.size() << std::endl;
    }
}

void Tensor::backward_sum(const Tensor &grad_output)
{
    /*
    Backward pass for Y = sum(X):
    dL/dX = dL/dY * dY/dX
    Since Y = sum(X_i), dY/dX_i = 1 for all i.
    So, dL/dX_i = dL/dY * 1 = dL/dY.
    The gradient dL/dY is the incoming grad_output (which should be a scalar).
    */

    if (parents_.size() == 1)
    {
        Tensor *parent = parents_[0];

        // The incoming gradient should be a scalar (shape {1})
        if (grad_output.get_shape().size() != 1 || grad_output.get_shape()[0] != 1)
        {
            throw std::runtime_error("Gradient for sum operation must be a scalar.");
        }
        float grad_value = grad_output.get_data()[0];

        Tensor grad_input(parent->get_shape());
        std::fill(grad_input.data_.begin(), grad_input.data_.end(), grad_value);

        parent->backward(grad_input);
    }
    else
    {
        std::cerr << "Error: Sum operation expected 1 parent, but found " << parents_.size() << std::endl;
    }
}

void Tensor::backward_div(const Tensor &grad_output)
{
    /*
    Backward pass for Z = A / B (element-wise):
    dL/dA = dL/dZ * dZ/dA = grad_output * (1 / B)
    dL/dB = dL/dZ * dZ/dB = grad_output * (-A / B^2)
    */
    if (parents_.size() == 2)
    {
        Tensor *parent_a = parents_[0]; // Numerator
        Tensor *parent_b = parents_[1]; // Denominator

        const std::vector<float> &parent_b_data = parent_b->get_data();

        // Gradient for parent A: grad_output * (1 / parent_b)
        std::vector<int> result_shape_a = calculate_broadcast_shape(grad_output.get_shape(), parent_b->get_shape());
        Tensor broadcasted_grad_a = grad_output.broadcast_to(result_shape_a);
        Tensor broadcasted_parent_b = parent_b->broadcast_to(result_shape_a);

        Tensor grad_a_intermediate(result_shape_a);
        for (size_t i = 0; i < grad_a_intermediate.data_.size(); ++i)
        {
            if (broadcasted_parent_b.data_[i] == 0.0f)
            {
                grad_a_intermediate.data_[i] = 0.0f;
            }
            else
            {
                grad_a_intermediate.data_[i] = broadcasted_grad_a.data_[i] / broadcasted_parent_b.data_[i];
            }
        }

        // Reduce gradient for parent A
        Tensor grad_a_propagated;
        reduce_gradient(grad_a_intermediate, grad_a_propagated, parent_a->get_shape());
        parent_a->backward(grad_a_propagated);

        // Fradient for parent B: grad_output * (-parent_a / (parent_b * parent_b))
        std::vector<int> result_shape_b = calculate_broadcast_shape(calculate_broadcast_shape(grad_output.get_shape(), parent_a->get_shape()), parent_b->get_shape());
        Tensor broadcasted_grad_b = grad_output.broadcast_to(result_shape_b);
        Tensor broadcasted_parent_a = parent_a->broadcast_to(result_shape_b);
        Tensor broadcasted_parent_b_b = parent_b->broadcast_to(result_shape_b);

        Tensor grad_b_intermediate(result_shape_b);
        for (size_t i = 0; i < grad_b_intermediate.data_.size(); ++i)
        {
            if (broadcasted_parent_b_b.data_[i] == 0.0f)
            {
                grad_b_intermediate.data_[i] = 0.0f;
            }
            else
            {
                grad_b_intermediate.data_[i] = broadcasted_grad_b.data_[i] * (-broadcasted_parent_a.data_[i] / (broadcasted_parent_b_b.data_[i] * broadcasted_parent_b_b.data_[i]));
            }
        }

        // Reduce gradient for parent B
        Tensor grad_b_propagated;
        reduce_gradient(grad_b_intermediate, grad_b_propagated, parent_b->get_shape());
        parent_b->backward(grad_b_propagated);
    }
    else
    {
        std::cerr << "Error: Division operation expected 2 parents, but found " << parents_.size() << std::endl;
    }
}
