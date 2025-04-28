#include "layers/Linear.h"

#include <random>
#include <stdexcept>
#include <vector>
#include <numeric>

Linear::Linear(int input_dim, int output_dim)
    : input_dim_(input_dim), output_dim_(output_dim)
{

    // Weights shape: (input_dim, output_dim)
    weights_ = Tensor({input_dim, output_dim});
    // Biases shape: (1, output_dim) - assuming broadcasting will handle batch dimension
    biases_ = Tensor({1, output_dim_});

    // Initialize weights and biases with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0, 0.01); // Mean 0, small standard deviation, using float

    // Initialize weights data
    std::vector<float> weights_data(output_dim_ * input_dim_);
    for (int i = 0; i < output_dim_ * input_dim_; ++i)
    {
        weights_data[i] = d(gen);
    }
    weights_.set_data(weights_data);

    // Initialize biases data
    std::vector<float> biases_data(output_dim_);
    for (int i = 0; i < output_dim_; ++i)
    {
        biases_data[i] = 0.0f;
    }
    biases_.set_data(biases_data);
}

Tensor Linear::forward(const Tensor &input)
{
    // Input shape is (..., input_dim)
    // Weights shape is (input_dim, output_dim)
    // output = input * weights + biases

    // Check input shape compatibility
    if (input.get_shape().empty() || input.get_shape().back() != input_dim_)
    {
        throw std::runtime_error("Input tensor's last dimension is incompatible with Linear layer input dimension.");
    }

    // Transpose weights from (output_dim, input_dim) to (input_dim, output_dim)
    std::vector<int> weights_permutation = {1, 0};
    Tensor transposed_weights = weights_.transpose(weights_permutation);

    Tensor output = input.dot(transposed_weights);
    output = output + biases_;

    return output;
}

// Destructor
Linear::~Linear() {}

Tensor &Linear::get_weights()
{
    return weights_;
}

Tensor &Linear::get_biases()
{
    return biases_;
}
