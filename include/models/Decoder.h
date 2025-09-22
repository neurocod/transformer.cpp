#pragma once
#include "DecoderLayer.h"

class Decoder {
public:
  Decoder(int numLayers, int embedDim, int numHeads, int ffHiddenDim,
          float dropoutRate);
  ~Decoder() = default;

  // Forward pass
  Tensor::Ptr forward(Tensor::Ptr &target_input,
                                  Tensor::Ptr &encoder_output,
                                  Tensor::Ptr &look_ahead_mask,
                                  Tensor::Ptr &padding_mask,
                                  bool isTraining);

private:
  std::vector<DecoderLayer> layers_;

  int _numLayers;
  int _embedDim;
  int _numHeads;
  int _ffHiddenDim;
  float _dropoutRate;
};
