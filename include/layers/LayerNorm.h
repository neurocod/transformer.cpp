#pragma once
#include "../utils/Tensor.h"

class LayerNorm {
public:
  LayerNorm(const std::string& name, int normalized_shape, float epsilon = 1e-5f);
  ~LayerNorm() = default;
  using Vec = Tensor::Vec;

  // Forward pass
  Tensor::Ptr forward(const Tensor::Ptr &input);


  // Getters for learnable parameters
  Tensor::Ptr get_gamma();
  Tensor::Ptr get_beta();

private:
  Tensor::Ptr gamma_; // Learnable scale parameter (gain)
  Tensor::Ptr beta_;  // Learnable shift parameter (bias)
  float epsilon_;

  int normalized_shape_;

  Tensor::Ptr mean_;
  Tensor::Ptr inv_stddev_;
  Tensor::Ptr centered_input_;
};
