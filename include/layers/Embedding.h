#pragma once
#include "../utils/Tensor.h"

class Embedding {
public:
  Embedding(int vocab_size, int embedDim);
  ~Embedding() = default;

  // Input shape: (batchSize, sequence_length)
  // Output shape: (batchSize, sequence_length, embedDim)
  std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input_ids);


  std::shared_ptr<Tensor> get_weights();
private:
  std::shared_ptr<Tensor> weights_;

  int vocab_size_;
  int embed_dim_;
};
