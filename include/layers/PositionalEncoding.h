#pragma once

#include "../utils/Tensor.h"
#include <cmath>
#include <memory>
#include <vector>

class PositionalEncoding {
public:
  PositionalEncoding(int max_sequence_length, int embed_dim);
  ~PositionalEncoding() = default;

  // Forward pass
  std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input);


private:
  std::shared_ptr<Tensor> positional_encodings_; // Pre-calculated

  int max_sequence_length_;
  int embed_dim_;
};
