#ifndef TRANSFORMER_CPP_TENSOR_H
#define TRANSFORMER_CPP_TENSOR_H

#include <vector>
#include <stdexcept>
#include <numeric>

enum class OperationType
{
    None,
    Add,
    Sub,
    Mul,
    Dot,
    Transpose,
    Reshape
};

class Tensor
{
public:
    // Constructor
    Tensor();
    // Constructor with shape
    Tensor(const std::vector<int> &shape);
    // Constructor with shape and data (initializer list or vector)
    Tensor(const std::vector<int> &shape, const std::vector<float> &data);

    // Destructor
    ~Tensor();

    // Getters for shape, data, and gradient
    const std::vector<int> &get_shape() const;
    const std::vector<float> &get_data() const;
    const std::vector<float> &get_grad() const;

    // Setter for data
    void set_data(const std::vector<float> &data);

    // Get element by multi-dimensional index
    float get(const std::vector<int> &indices) const;
    // Set element by multi-dimensional index
    void set(const std::vector<int> &indices, float value);

    // Basic tensor operations
    Tensor operator+(const Tensor &other) const;
    Tensor operator-(const Tensor &other) const;
    Tensor operator*(const Tensor &other) const;                 // Element-wise multiplication
    Tensor dot(const Tensor &other) const;                       // Matrix multiplication
    Tensor transpose(const std::vector<int> &permutation) const; // Transpose with permutation
    Tensor reshape(const std::vector<int> &new_shape) const;

    // Gradient handling (simplified for now)
    void zero_grad();
    void backward(const Tensor &grad_output); // Placeholder

private:
    std::vector<int> shape_;
    std::vector<float> data_;
    std::vector<float> grad_;

    // Stores the type of operation that produced this tensor.
    OperationType creator_op_ = OperationType::None;

    // Stores pointers to the input tensors (parents) of the operation that created this tensor.
    std::vector<const Tensor *> parents_;

    // Helper to calculate the linear index from multi-dimensional indices
    size_t get_linear_index(const std::vector<int> &indices) const;

    // Helper to calculate the new shape for broadcasting
    static std::vector<int> calculate_broadcast_shape(const std::vector<int> &shape1, const std::vector<int> &shape2);
    // Helper to perform broadcasting
    Tensor broadcast_to(const std::vector<int> &new_shape) const;
    // Helper to check if two shapes are broadcastable
    static bool is_broadcastable(const std::vector<int> &shape1, const std::vector<int> &shape2);

    // Helper to calculate total number of elements from shape
    size_t num_elements() const
    {
        if (shape_.empty())
            return 0;
        return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
    }

    // Helper to check if shapes are compatible for element-wise operations
    bool are_shapes_compatible(const Tensor &other) const
    {
        return shape_ == other.shape_;
    }

    // These methods calculate and propagate gradients to the parent tensors.
    void reduce_gradient(const Tensor &grad_output, Tensor &parent_grad, const std::vector<int> &parent_shape);
    void backward_add(const Tensor &grad_output);
    void backward_sub(const Tensor &grad_output);
    void backward_mul(const Tensor &grad_output);
    void backward_dot(const Tensor &grad_output);
    void backward_transpose(const Tensor &grad_output);
    void backward_reshape(const Tensor &grad_output);
};

#endif // TRANSFORMER_CPP_TENSOR_H
