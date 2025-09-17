#pragma once
#include "EncoderLayer.h"

class Encoder {
public:
  Encoder(int numLayers, int embedDim, int numHeads, int ffHiddenDim,
          float dropoutRate);
  ~Encoder() = default;

  // Forward pass
  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> &input,
                                  std::shared_ptr<Tensor> &padding_mask,
                                  bool is_training);


private:
  std::vector<EncoderLayer> layers_;

  int num_layers_;
  int embed_dim_;
  int num_heads_;
  int ff_hidden_dim_;
  float dropout_rate_;
};
