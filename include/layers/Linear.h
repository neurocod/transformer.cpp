#ifndef TRANSFORMER_CPP_LINEAR_H
#define TRANSFORMER_CPP_LINEAR_H

#include "utils/Tensor.h"
#include <vector>

class Linear
{
public:
    Linear(int input_dim, int output_dim);

    Tensor forward(const Tensor &input);

    // Destructor
    ~Linear();

    Tensor &get_weights();
    Tensor &get_biases();

private:
    Tensor weights_; // Weight matrix (output_dim, input_dim)
    Tensor biases_;  // Bias vector (1, output_dim)

    int input_dim_;
    int output_dim_;
};

#endif // TRANSFORMER_CPP_LINEAR_H
