#include "models/Decoder.h"

Decoder::Decoder(int numLayers, int embedDim, int numHeads,
                 int ffHiddenDim, float dropoutRate)
    : _numLayers(numLayers), _embedDim(embedDim), _numHeads(numHeads),
      _ffHiddenDim(ffHiddenDim), _dropoutRate(dropoutRate) {
  for (int i = 0; i < _numLayers; ++i) {
    layers_.emplace_back(_embedDim, _numHeads, _ffHiddenDim, _dropoutRate);
  }
}

Tensor::Ptr Decoder::forward(Tensor::Ptr &target_input,
                 Tensor::Ptr &encoder_output,
                 Tensor::Ptr &look_ahead_mask,
                 Tensor::Ptr &padding_mask, bool isTraining) {
  Tensor::Ptr output = target_input;

  for (size_t i = 0; i < layers_.size(); ++i) {
    output = layers_[i].forward(output, encoder_output, look_ahead_mask,
                                padding_mask, isTraining);
  }

  return output;
}