#pragma once
#include "../layers/Dropout.h"
#include "../layers/LayerNorm.h"
#include "../layers/MultiHeadAttention.h"
#include "../layers/PositionwiseFeedForward.h"

class EncoderLayer {
public:
  EncoderLayer(int embedDim, int numHeads, int ffHiddenDim,
               float dropoutRate = 0.1f);
  ~EncoderLayer() = default;

  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> &input,
                                  std::shared_ptr<Tensor> &padding_mask,
                                  bool isTraining);

private:
  MultiHeadAttention self_attention_;
  LayerNorm layernorm1_;
  Dropout dropout1_;
  PositionwiseFeedForward feed_forward_;
  LayerNorm layernorm2_;
  Dropout dropout2_;

  float _dropoutRate;
};
