#ifndef TRANSFORMER_CPP_ENCODERLAYER_H
#define TRANSFORMER_CPP_ENCODERLAYER_H

#include "utils/Tensor.h"
#include "layers/MultiHeadAttention.h"
#include "layers/PositionwiseFeedForward.h"
#include "layers/LayerNorm.h"
#include "layers/Dropout.h"
#include <memory>
#include <vector>

class Tensor;
class MultiHeadAttention;
class PositionwiseFeedForward;
class LayerNorm;
class Dropout;

class EncoderLayer {
public:
    // Constructor
    EncoderLayer(int embed_dim, int num_heads, int ff_hidden_dim, float dropout_rate = 0.1f);

    // Forward pass
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor>& input, bool is_training);

    // Destructor
    ~EncoderLayer() = default;

private:
    MultiHeadAttention self_attention_;
    LayerNorm layernorm1_;
    Dropout dropout1_;
    PositionwiseFeedForward feed_forward_;
    LayerNorm layernorm2_;
    Dropout dropout2_;

    float dropout_rate_;
};

#endif // TRANSFORMER_CPP_ENCODERLAYER_H
