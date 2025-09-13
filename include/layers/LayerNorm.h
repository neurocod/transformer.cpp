#pragma once

#include "../utils/Tensor.h"
#include <memory>
#include <vector>

class LayerNorm {
public:
  LayerNorm(int normalized_shape, float epsilon = 1e-5f);
  ~LayerNorm() = default;

  // Forward pass
  std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input);


  // Getters for learnable parameters
  std::shared_ptr<Tensor> get_gamma();
  std::shared_ptr<Tensor> get_beta();

private:
  std::shared_ptr<Tensor> gamma_; // Learnable scale parameter (gain)
  std::shared_ptr<Tensor> beta_;  // Learnable shift parameter (bias)
  float epsilon_;

  int normalized_shape_;

  std::shared_ptr<Tensor> mean_;
  std::shared_ptr<Tensor> inv_stddev_;
  std::shared_ptr<Tensor> centered_input_;
};
