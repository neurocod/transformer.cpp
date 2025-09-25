#pragma once
#include "../utils/Tensor.h"

class PositionalEncoding {
public:
  PositionalEncoding(int maxSequenceLength, int embedDim);
  ~PositionalEncoding() = default;

  // Forward pass
  Tensor::Ptr forward(const Tensor::Ptr &input);

private:
  Tensor::Ptr positional_encodings_; // Pre-calculated
  using Vec = Tensor::Vec;

  int _maxSequenceLength;
  int _embedDim;
};
