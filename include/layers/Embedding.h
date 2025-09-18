#pragma once
#include "../utils/Tensor.h"

class Embedding {
public:
  Embedding(int vocabSize, int embedDim);
  ~Embedding() {}

  // Input shape: (batchSize, sequence_length)
  // Output shape: (batchSize, sequence_length, embedDim)
  std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input_ids);
  std::shared_ptr<Tensor> weights();
private:
  std::shared_ptr<Tensor> _weights;

  int _vocabSize;
  int _embedDim;
};
