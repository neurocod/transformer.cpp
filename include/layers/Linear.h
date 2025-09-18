#pragma once
#include "../utils/Tensor.h"

class Linear {
public:
  Linear(int inputDim, int outputDim, const std::string& name);
  ~Linear() {}

  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> &input);
  std::shared_ptr<Tensor> weights();
  std::shared_ptr<Tensor> biases();
private:
  std::shared_ptr<Tensor> _weights; // Weight matrix (inputDim, outputDim)
  std::shared_ptr<Tensor> _biases;  // Bias vector (1, outputDim)

  int _inputDim;
  int _outputDim;
};
