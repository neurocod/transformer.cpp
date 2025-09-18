#pragma once
#include "../layers/Dropout.h"
#include "../layers/LayerNorm.h"
#include "../layers/MultiHeadAttention.h"
#include "../layers/PositionwiseFeedForward.h"

class DecoderLayer {
public:
  DecoderLayer(int embedDim, int numHeads, int ffHiddenDim,
               float dropoutRate = 0.1f);
  ~DecoderLayer() = default;

  // Forward pass
  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> &target_input,
                                  std::shared_ptr<Tensor> &encoder_output,
                                  std::shared_ptr<Tensor> &look_ahead_mask,
                                  std::shared_ptr<Tensor> &padding_mask,
                                  bool isTraining);

private:
  MultiHeadAttention masked_self_attention_;
  LayerNorm layernorm1_;
  Dropout dropout1_;
  MultiHeadAttention cross_attention_;
  LayerNorm layernorm2_;
  Dropout dropout2_;
  PositionwiseFeedForward feed_forward_;
  LayerNorm layernorm3_;
  Dropout dropout3_;

  float _dropoutRate;
};
