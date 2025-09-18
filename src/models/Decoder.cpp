#include "models/Decoder.h"

Decoder::Decoder(int numLayers, int embedDim, int numHeads,
                 int ffHiddenDim, float dropoutRate)
    : _numLayers(numLayers), _embedDim(embedDim), _numHeads(numHeads),
      _ffHiddenDim(ffHiddenDim), _dropoutRate(dropoutRate) {
  for (int i = 0; i < _numLayers; ++i) {
    layers_.emplace_back(_embedDim, _numHeads, _ffHiddenDim, _dropoutRate);
  }
}

std::shared_ptr<Tensor>
Decoder::forward(std::shared_ptr<Tensor> &target_input,
                 std::shared_ptr<Tensor> &encoder_output,
                 std::shared_ptr<Tensor> &look_ahead_mask,
                 std::shared_ptr<Tensor> &padding_mask, bool isTraining) {
  std::shared_ptr<Tensor> output = target_input;

  for (size_t i = 0; i < layers_.size(); ++i) {
    output = layers_[i].forward(output, encoder_output, look_ahead_mask,
                                padding_mask, isTraining);
  }

  return output;
}