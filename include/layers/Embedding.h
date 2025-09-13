#pragma once
#include "../utils/Tensor.h"
#include <memory>
#include <random>
#include <vector>

class Embedding {
public:
  Embedding(int vocab_size, int embed_dim);
  ~Embedding() = default;

  // Input shape: (batch_size, sequence_length)
  // Output shape: (batch_size, sequence_length, embed_dim)
  std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input_ids);


  std::shared_ptr<Tensor> get_weights();
private:
  std::shared_ptr<Tensor> weights_;

  int vocab_size_;
  int embed_dim_;
};
