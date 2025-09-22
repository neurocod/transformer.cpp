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

  Tensor::Ptr forward(Tensor::Ptr &input,
                                  Tensor::Ptr &padding_mask,
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
