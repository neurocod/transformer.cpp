#include "models/Transformer.h"
#include <iostream>
#include <stdexcept>
#include "utils/Helpers.h"

Transformer::Transformer(int input_vocab_size, int target_vocab_size, int embed_dim,
                         int max_sequence_length, int num_layers, int num_heads,
                         int ff_hidden_dim, float dropout_rate, float pad_token_id)
    : input_vocab_size_(input_vocab_size),
      target_vocab_size_(target_vocab_size),
      embed_dim_(embed_dim),
      max_sequence_length_(max_sequence_length),
      num_layers_(num_layers),
      num_heads_(num_heads),
      ff_hidden_dim_(ff_hidden_dim),
      dropout_rate_(dropout_rate),
      pad_token_id_(pad_token_id),
      encoder_embedding_(input_vocab_size, embed_dim),
      encoder_positional_encoding_(max_sequence_length, embed_dim),
      encoder_(num_layers, embed_dim, num_heads, ff_hidden_dim, dropout_rate),
      decoder_embedding_(target_vocab_size, embed_dim),
      decoder_positional_encoding_(max_sequence_length, embed_dim),
      decoder_(num_layers, embed_dim, num_heads, ff_hidden_dim, dropout_rate),
      final_linear_(embed_dim, target_vocab_size)
{
    if (embed_dim_ % num_heads_ != 0)
    {
        throw std::runtime_error("Embedding dimension must be divisible by the number of heads.");
    }
}

std::shared_ptr<Tensor> Transformer::create_encoder_padding_mask(const std::shared_ptr<Tensor> &encoder_input_ids)
{
    return create_padding_mask(encoder_input_ids, pad_token_id_);
}

std::shared_ptr<Tensor> Transformer::create_decoder_self_attention_mask(const std::shared_ptr<Tensor> &decoder_input_ids)
{
    int sequence_length = decoder_input_ids->get_shape()[1];
    std::shared_ptr<Tensor> look_ahead = create_look_ahead_mask(sequence_length);
    std::shared_ptr<Tensor> padding = create_padding_mask(decoder_input_ids, pad_token_id_);
    return *look_ahead + padding;
}

std::shared_ptr<Tensor> Transformer::create_decoder_cross_attention_mask(const std::shared_ptr<Tensor> &encoder_input_ids)
{
    // Cross-attention uses the padding mask from the encoder input
    return create_padding_mask(encoder_input_ids, pad_token_id_);
}

std::shared_ptr<Tensor> Transformer::forward(const std::shared_ptr<Tensor> &encoder_input_ids,
                                             const std::shared_ptr<Tensor> &decoder_input_ids,
                                             bool is_training)
{
    const std::vector<int> &enc_input_shape = encoder_input_ids->get_shape();
    const std::vector<int> &dec_input_shape = decoder_input_ids->get_shape();

    if (enc_input_shape.size() != 2 || dec_input_shape.size() != 2)
    {
        throw std::runtime_error("Encoder and decoder input IDs must be 2D tensors (batch_size, sequence_length).");
    }

    std::shared_ptr<Tensor> encoder_padding_mask = create_encoder_padding_mask(encoder_input_ids);
    std::shared_ptr<Tensor> decoder_self_attention_mask = create_decoder_self_attention_mask(decoder_input_ids);
    std::shared_ptr<Tensor> decoder_cross_attention_mask = create_decoder_cross_attention_mask(encoder_input_ids);

    std::shared_ptr<Tensor> encoder_input_embeddings = encoder_embedding_.forward(encoder_input_ids);
    std::shared_ptr<Tensor> encoder_input_with_pos = encoder_positional_encoding_.forward(encoder_input_embeddings);

    // Placeholder for dropout on embeddings
    std::shared_ptr<Tensor> encoder_input_dropped = encoder_input_with_pos;

    std::shared_ptr<Tensor> encoder_output = encoder_.forward(encoder_input_dropped, encoder_padding_mask, is_training);

    std::shared_ptr<Tensor> decoder_input_embeddings = decoder_embedding_.forward(decoder_input_ids);
    std::shared_ptr<Tensor> decoder_input_with_pos = decoder_positional_encoding_.forward(decoder_input_embeddings);

    // Placeholder for dropout on embeddings if needed
    std::shared_ptr<Tensor> decoder_input_dropped = decoder_input_with_pos;

    std::shared_ptr<Tensor> decoder_output = decoder_.forward(decoder_input_dropped, encoder_output, decoder_self_attention_mask, decoder_cross_attention_mask, is_training);

    std::shared_ptr<Tensor> logits = final_linear_.forward(decoder_output);

    return logits;
}