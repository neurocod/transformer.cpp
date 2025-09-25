#pragma once
#include "../utils/Tensor.h"

class Dropout {
public:
  Dropout(float rate);
  ~Dropout() = default;

  // Forward pass
  Tensor::Ptr forward(const Tensor::Ptr &input,
                                  bool isTraining);
private:
  float rate_; // Dropout rate

  // Random number generator and distribution for creating the mask
  std::mt19937 generator_;
  std::uniform_real_distribution<float> distribution_;
  using Vec = Tensor::Vec;
};
