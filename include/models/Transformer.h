#pragma once
#include "../layers/Embedding.h"
#include "../layers/Linear.h"
#include "../layers/PositionalEncoding.h"
#include "../utils/Masking.h"
#include "Decoder.h"
#include "Encoder.h"

class Transformer {
public:
  Transformer(int input_vocab_size, int target_vocab_size, int embedDim,
              int maxSequenceLength, int numLayers, int numHeads,
              int ffHiddenDim, float dropoutRate, float padTokenId = 0.0f);

  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &encoder_input_ids,
          const std::shared_ptr<Tensor> &decoder_input_ids, bool is_training);

  void save_weights(const std::string &filename) const;
  void load_weights(const std::string &filename);

  ~Transformer() = default;

private:
  Embedding encoder_embedding_;
  PositionalEncoding encoder_positional_encoding_;
  Encoder encoder_;

  Embedding decoder_embedding_;
  PositionalEncoding decoder_positional_encoding_;
  Decoder decoder_;

  Linear final_linear_;

  int input_vocab_size_;
  int target_vocab_size_;
  int embed_dim_;
  int max_sequence_length_;
  int num_layers_;
  int num_heads_;
  int ff_hidden_dim_;
  float dropout_rate_;
  float pad_token_id_;

  std::shared_ptr<Tensor>
  create_encoder_padding_mask(const std::shared_ptr<Tensor> &encoder_input_ids);
  std::shared_ptr<Tensor> create_decoder_self_attention_mask(
      const std::shared_ptr<Tensor> &decoder_input_ids);
  std::shared_ptr<Tensor> create_decoder_cross_attention_mask(
      const std::shared_ptr<Tensor> &encoder_input_ids);
};
