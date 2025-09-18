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
                                  bool isTraining);


private:
  std::vector<EncoderLayer> layers_;

  int _numLayers;
  int _embedDim;
  int _numHeads;
  int _ffHiddenDim;
  float _dropoutRate;
};
