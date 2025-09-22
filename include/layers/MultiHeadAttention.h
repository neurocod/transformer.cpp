#pragma once
#include "Linear.h"

class MultiHeadAttention {
public:
  MultiHeadAttention(const std::string& name, int embedDim, int numHeads);
  ~MultiHeadAttention() = default;

  // Forward pass
  // query, key, and value can be the same tensor for self-attention, or
  // different for cross-attention
  Tensor::Ptr forward(Tensor::Ptr &query,
                                  Tensor::Ptr &key,
                                  Tensor::Ptr &value,
                                  Tensor::Ptr mask = nullptr);

private:
  Linear query_proj_;
  Linear key_proj_;
  Linear value_proj_;
  Linear output_proj_;

  int _embedDim;
  int _numHeads;
  int head_dim_;

  // Helper function to calculate scaled dot-product attention
  Tensor::Ptr scaled_dot_product_attention(
      Tensor::Ptr &q, Tensor::Ptr &k,
      Tensor::Ptr &v, Tensor::Ptr mask = nullptr);
};
