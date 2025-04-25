#ifndef TRANSFORMER_CPP_TENSOR_H
#define TRANSFORMER_CPP_TENSOR_H

#include <vector>
#include <Eigen/Dense>

class Tensor {
public:
    // Constructor
    Tensor();
    Tensor(const std::vector<int>& shape);
    Tensor(const Eigen::MatrixXd& data);

    // Destructor
    ~Tensor();

    // Getters for shape and data
    const std::vector<int>& get_shape() const;
    const Eigen::MatrixXd& get_data() const;
    const Eigen::MatrixXd& get_grad() const;

    // Setter for data
    void set_data(const Eigen::MatrixXd& data);

    // Basic tensor operations (declarations)
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const; // Element-wise multiplication
    Tensor dot(const Tensor& other) const;      // Matrix multiplication
    Tensor transpose() const;
    Tensor reshape(const std::vector<int>& new_shape) const;

    // Gradient handling
    void zero_grad();
    void backward(const Tensor& grad_output);

private:
    std::vector<int> shape_;
    Eigen::MatrixXd data_;
    Eigen::MatrixXd grad_;
};

#endif // TRANSFORMER_CPP_TENSOR_H