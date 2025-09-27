#pragma once
#include "../utils/Tensor.h"

class Embedding {
public:
  Embedding(int vocabSize, int embedDim);
  ~Embedding() {}

  // Input shape: (batchSize, sequenceLength)
  // Output shape: (batchSize, sequenceLength, embedDim)
  Tensor::Ptr forward(const Tensor::Ptr &input_ids);
  Tensor::Ptr weights();
private:
  Tensor::Ptr _weights;
  using Vec = Tensor::Vec;

  int _vocabSize;
  int _embedDim;
};
