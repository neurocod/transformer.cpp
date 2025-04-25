#include "utils/Tensor.h"
#include <stdexcept>

// Default constructor
Tensor::Tensor() : shape_{{0}}, data_{Eigen::MatrixXd()}, grad_{Eigen::MatrixXd()} {}

// Constructor with shape
Tensor::Tensor(const std::vector<int>& shape) : shape_{shape} {
    if (shape.size() == 2) {
        data_ = Eigen::MatrixXd::Zero(shape[0], shape[1]);
        grad_ = Eigen::MatrixXd::Zero(shape[0], shape[1]);
    } else {
        // Handle other dimensions if needed, for now, only 2D is explicitly supported
        throw std::runtime_error("Tensor constructor only supports 2D shapes for now.");
    }
}

// Constructor with data
Tensor::Tensor(const Eigen::MatrixXd& data) : data_{data}, shape_{{(int)data.rows(), (int)data.cols()}} {
    grad_ = Eigen::MatrixXd::Zero(data.rows(), data.cols());
}

// Destructor
Tensor::~Tensor() {
    // Eigen manages memory, so no explicit deallocation needed here
}

// Getters
const std::vector<int>& Tensor::get_shape() const {
    return shape_;
}

const Eigen::MatrixXd& Tensor::get_data() const {
    return data_;
}

const Eigen::MatrixXd& Tensor::get_grad() const {
    return grad_;
}

// Setter for data
void Tensor::set_data(const Eigen::MatrixXd& data) {
    if (data.rows() != shape_[0] || data.cols() != shape_[1]) {
        throw std::runtime_error("Data shape mismatch in set_data.");
    }
    data_ = data;
    grad_ = Eigen::MatrixXd::Zero(shape_[0], shape_[1]); // Reset gradient on setting new data
}

// Basic tensor operations
Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes do not match for addition.");
    }
    return Tensor(data_ + other.data_);
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes do not match for subtraction.");
    }
    return Tensor(data_ - other.data_);
}

Tensor Tensor::operator*(const Tensor& other) const { // Element-wise multiplication
     if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes do not match for element-wise multiplication.");
    }
    return Tensor(data_.cwiseProduct(other.data_));
}

Tensor Tensor::dot(const Tensor& other) const {      // Matrix multiplication
    if (shape_[1] != other.shape_[0]) {
        throw std::runtime_error("Tensor shapes are not compatible for matrix multiplication.");
    }
    return Tensor(data_ * other.data_);
}

Tensor Tensor::transpose() const {
    return Tensor(data_.transpose());
}

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    if (new_shape.size() != 2) {
         throw std::runtime_error("Reshape only supports 2D shapes for now.");
    }
    if (shape_[0] * shape_[1] != new_shape[0] * new_shape[1]) {
        throw std::runtime_error("Total number of elements must remain the same during reshape.");
    }
    Eigen::Map<const Eigen::MatrixXd> temp_data(data_.data(), new_shape[0], new_shape[1]);
    return Tensor(temp_data);
}

// Gradient handling
void Tensor::zero_grad() {
    grad_.setZero();
}

void Tensor::backward(const Tensor& grad_output) {
    // This is a simplified backward pass.
    // A full implementation requires tracking the operation that created this tensor
    // and applying the chain rule.
    // For now, we'll just add the incoming gradient.
    if (shape_ != grad_output.shape_) {
         throw std::runtime_error("Gradient shape mismatch in backward.");
    }
    grad_ += grad_output.data_;
}