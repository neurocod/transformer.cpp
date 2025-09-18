#pragma once
#include "Linear.h"

class MultiHeadAttention {
public:
  MultiHeadAttention(const std::string& name, int embedDim, int numHeads);
  ~MultiHeadAttention() = default;

  // Forward pass
  // query, key, and value can be the same tensor for self-attention, or
  // different for cross-attention
  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> &query,
                                  std::shared_ptr<Tensor> &key,
                                  std::shared_ptr<Tensor> &value,
                                  std::shared_ptr<Tensor> mask = nullptr);

private:
  Linear query_proj_;
  Linear key_proj_;
  Linear value_proj_;
  Linear output_proj_;

  int _embedDim;
  int _numHeads;
  int head_dim_;

  // Helper function to calculate scaled dot-product attention
  std::shared_ptr<Tensor> scaled_dot_product_attention(
      std::shared_ptr<Tensor> &q, std::shared_ptr<Tensor> &k,
      std::shared_ptr<Tensor> &v, std::shared_ptr<Tensor> mask = nullptr);
};
