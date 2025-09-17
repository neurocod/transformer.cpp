#include "models/Decoder.h"

Decoder::Decoder(int numLayers, int embedDim, int numHeads,
                 int ffHiddenDim, float dropoutRate)
    : num_layers_(numLayers), embed_dim_(embedDim), num_heads_(numHeads),
      ff_hidden_dim_(ffHiddenDim), dropout_rate_(dropoutRate) {
  for (int i = 0; i < num_layers_; ++i) {
    layers_.emplace_back(embed_dim_, num_heads_, ff_hidden_dim_, dropout_rate_);
  }
}

std::shared_ptr<Tensor>
Decoder::forward(std::shared_ptr<Tensor> &target_input,
                 std::shared_ptr<Tensor> &encoder_output,
                 std::shared_ptr<Tensor> &look_ahead_mask,
                 std::shared_ptr<Tensor> &padding_mask, bool is_training) {
  std::shared_ptr<Tensor> output = target_input;

  for (size_t i = 0; i < layers_.size(); ++i) {
    output = layers_[i].forward(output, encoder_output, look_ahead_mask,
                                padding_mask, is_training);
  }

  return output;
}