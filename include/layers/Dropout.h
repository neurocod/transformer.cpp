#pragma once
#include "../utils/Tensor.h"
#include <random>

class Dropout {
public:
  Dropout(float rate);
  ~Dropout() = default;

  // Forward pass
  std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input,
                                  bool is_training);
private:
  float rate_; // Dropout rate

  // Random number generator and distribution for creating the mask
  std::mt19937 generator_;
  std::uniform_real_distribution<float> distribution_;
};
