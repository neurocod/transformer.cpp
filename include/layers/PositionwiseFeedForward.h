#pragma once
#include "Activations.h"
#include "Linear.h"

class PositionwiseFeedForward {
public:
  PositionwiseFeedForward(int inputDim, int hidden_dim);
  ~PositionwiseFeedForward() = default;

  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> &input);
private:
  Linear fc1_;
  Linear fc2_;
  ReLU activation_;

  int _inputDim;
  int hidden_dim_;
};
