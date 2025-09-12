#pragma once

#include "Activations.h"
#include "Linear.h"
#include <memory>
#include <vector>

class Tensor;

class PositionwiseFeedForward {
public:
  // Constructor
  PositionwiseFeedForward(int input_dim, int hidden_dim);

  // Forward pass
  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> &input);

  // Destructor
  ~PositionwiseFeedForward() = default;

private:
  Linear fc1_;
  Linear fc2_;
  ReLU activation_;

  int input_dim_;
  int hidden_dim_;
};
