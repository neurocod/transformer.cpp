#include "layers/Linear.h"

#include <random>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <memory>
#include <iostream>

Linear::Linear(int input_dim, int output_dim)
    : input_dim_(input_dim), output_dim_(output_dim)
{
    // Weights shape: (input_dim, output_dim)
    weights_ = Tensor::create(std::vector<int>{input_dim, output_dim}, true);
    // Biases shape: (1, output_dim)
    biases_ = Tensor::create(std::vector<int>{1, output_dim_}, true);

    // Initialize weights and biases with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0, 0.01); // Mean 0, small standard deviation, using float

    // Initialize weights data
    std::shared_ptr<std::vector<float>> weights_data = std::make_shared<std::vector<float>>(output_dim_ * input_dim_);
    for (int i = 0; i < output_dim_ * input_dim_; ++i)
    {
        (*weights_data)[i] = d(gen);
    }
    weights_->set_data(weights_data);

    // Initialize biases data
    std::shared_ptr<std::vector<float>> biases_data = std::make_shared<std::vector<float>>(output_dim_);
    for (int i = 0; i < output_dim_; ++i)
    {
        (*biases_data)[i] = 0.0f;
    }
    biases_->set_data(biases_data);
}

std::shared_ptr<Tensor> Linear::forward(const Tensor &input)
{
    // Input shape is (..., input_dim)
    // Weights shape is (input_dim, output_dim)
    // output = input * weights + biases

    if (input.get_shape().empty() || input.get_shape().back() != input_dim_)
    {
        throw std::runtime_error("Input tensor's last dimension is incompatible with Linear layer input dimension.");
    }

    std::shared_ptr<Tensor> product = input.dot(weights_);
    std::shared_ptr<Tensor> output = *product + biases_;

    return output;
}

// Destructor
Linear::~Linear()
{
}

std::shared_ptr<Tensor> Linear::get_weights()
{
    return weights_;
}

std::shared_ptr<Tensor> Linear::get_biases()
{
    return biases_;
}
