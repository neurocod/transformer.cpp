#include "models/Encoder.h"

Encoder::Encoder(int numLayers, int embedDim, int numHeads,
                 int ffHiddenDim, float dropoutRate)
    : _numLayers(numLayers), _embedDim(embedDim), _numHeads(numHeads),
      _ffHiddenDim(ffHiddenDim), _dropoutRate(dropoutRate) {
  for (int i = 0; i < _numLayers; ++i) {
    layers_.emplace_back(_embedDim, _numHeads, _ffHiddenDim, _dropoutRate);
  }
}

Tensor::Ptr Encoder::forward(Tensor::Ptr &input,
                                         Tensor::Ptr &padding_mask,
                                         bool isTraining) {
  // Input shape: (batchSize, sequenceLength, embedDim)

  Tensor::Ptr output = input;

  for (size_t i = 0; i < layers_.size(); ++i) {
    output = layers_[i].forward(output, padding_mask, isTraining);
  }

  return output;
}