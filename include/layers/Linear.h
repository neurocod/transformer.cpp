#pragma once

#include "../utils/Tensor.h"

class Linear {
public:
  Linear(int input_dim, int output_dim);

  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> &input);

  // Destructor
  ~Linear();

  std::shared_ptr<Tensor> get_weights();
  std::shared_ptr<Tensor> get_biases();

private:
  std::shared_ptr<Tensor> weights_; // Weight matrix (input_dim, output_dim)
  std::shared_ptr<Tensor> biases_;  // Bias vector (1, output_dim)

  int input_dim_;
  int output_dim_;
};
