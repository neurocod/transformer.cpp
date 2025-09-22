#pragma once
#include "../utils/Tensor.h"

class Linear {
public:
  Linear(int inputDim, int outputDim, const std::string& name);
  ~Linear() {}

  Tensor::Ptr forward(Tensor::Ptr &input);
  Tensor::Ptr weights();
  Tensor::Ptr biases();
private:
  Tensor::Ptr _weights; // Weight matrix (inputDim, outputDim)
  Tensor::Ptr _biases;  // Bias vector (1, outputDim)

  int _inputDim;
  int _outputDim;
};
