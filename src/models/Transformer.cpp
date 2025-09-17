#include "models/Transformer.h"
#include "utils/BinaryWriter.h"
#include "utils/BinaryReader.h"

Transformer::Transformer(int input_vocab_size, int target_vocab_size,
                         int embed_dim, int max_sequence_length, int num_layers,
                         int num_heads, int ff_hidden_dim, float dropout_rate,
                         float pad_token_id)
    : input_vocab_size_(input_vocab_size),
      target_vocab_size_(target_vocab_size), embed_dim_(embed_dim),
      max_sequence_length_(max_sequence_length), num_layers_(num_layers),
      num_heads_(num_heads), ff_hidden_dim_(ff_hidden_dim),
      dropout_rate_(dropout_rate), pad_token_id_(pad_token_id),
      encoder_embedding_(input_vocab_size, embed_dim),
      encoder_positional_encoding_(max_sequence_length, embed_dim),
      encoder_(num_layers, embed_dim, num_heads, ff_hidden_dim, dropout_rate),
      decoder_embedding_(target_vocab_size, embed_dim),
      decoder_positional_encoding_(max_sequence_length, embed_dim),
      decoder_(num_layers, embed_dim, num_heads, ff_hidden_dim, dropout_rate),
      final_linear_(embed_dim, target_vocab_size, "final") {
  if (embed_dim_ % num_heads_ != 0) {
    throw std::runtime_error(
        "Embedding dimension must be divisible by the number of heads.");
  }
}

std::shared_ptr<Tensor> Transformer::create_encoder_padding_mask(
    const std::shared_ptr<Tensor> &encoder_input_ids) {
  return create_padding_mask(encoder_input_ids, pad_token_id_);
}

std::shared_ptr<Tensor> Transformer::create_decoder_self_attention_mask(
    const std::shared_ptr<Tensor> &decoder_input_ids) {
  const auto &decoder_shape = decoder_input_ids->get_shape();
  if (decoder_shape.size() != 2) {
    throw std::runtime_error(
        "Decoder input for self-attention mask must be 2D (batch, seq_len).");
  }
  int batch_size = decoder_shape[0];
  int sequence_length = decoder_shape[1];

  std::shared_ptr<Tensor> look_ahead_mask =
      create_look_ahead_mask(sequence_length);
  std::shared_ptr<Tensor> padding_mask =
      create_padding_mask(decoder_input_ids, pad_token_id_);

  const auto &look_ahead_shape = look_ahead_mask->get_shape();
  const auto &padding_shape = padding_mask->get_shape();

  if (look_ahead_shape !=
          std::vector<int>{1, 1, sequence_length, sequence_length} ||
      padding_shape != std::vector<int>{batch_size, 1, 1, sequence_length}) {
    throw std::runtime_error("Unexpected shapes for look-ahead or padding "
                             "masks during combination.");
  }

  std::vector<int> combined_mask_shape = {batch_size, 1, sequence_length,
                                          sequence_length};
  std::shared_ptr<Tensor> combined_mask = Tensor::create(combined_mask_shape);
  std::vector<float> &combined_mask_data = combined_mask->data_ref();

  const std::vector<float> &look_ahead_data = look_ahead_mask->get_data();
  const std::vector<float> &padding_data = padding_mask->get_data();

  for (int b = 0; b < batch_size; ++b) {
    for (int tq = 0; tq < sequence_length; ++tq) {
      for (int tk = 0; tk < sequence_length; ++tk) {
        size_t look_ahead_idx = tq * sequence_length + tk;
        size_t padding_idx = b * sequence_length + tk;
        size_t combined_idx =
            b * sequence_length * sequence_length + tq * sequence_length + tk;
        combined_mask_data[combined_idx] = std::max(
            look_ahead_data[look_ahead_idx], padding_data[padding_idx]);
      }
    }
  }

  return combined_mask;
}

std::shared_ptr<Tensor> Transformer::create_decoder_cross_attention_mask(
    const std::shared_ptr<Tensor> &encoder_input_ids) {
  return create_padding_mask(encoder_input_ids, pad_token_id_);
}

std::shared_ptr<Tensor>
Transformer::forward(const std::shared_ptr<Tensor> &encoder_input_ids,
                     const std::shared_ptr<Tensor> &decoder_input_ids,
                     bool is_training) {
  const std::vector<int> &enc_input_shape = encoder_input_ids->get_shape();
  const std::vector<int> &dec_input_shape = decoder_input_ids->get_shape();

  if (enc_input_shape.size() != 2 || dec_input_shape.size() != 2) {
    throw std::runtime_error("Encoder and decoder input IDs must be 2D tensors "
                             "(batch_size, sequence_length).");
  }

  std::shared_ptr<Tensor> encoder_padding_mask =
      create_encoder_padding_mask(encoder_input_ids);
  std::shared_ptr<Tensor> decoder_self_attention_mask =
      create_decoder_self_attention_mask(decoder_input_ids);
  std::shared_ptr<Tensor> decoder_cross_attention_mask =
      create_decoder_cross_attention_mask(encoder_input_ids);

  std::shared_ptr<Tensor> encoder_input_embeddings =
      encoder_embedding_.forward(encoder_input_ids);
  std::shared_ptr<Tensor> encoder_input_with_pos =
      encoder_positional_encoding_.forward(encoder_input_embeddings);

  std::shared_ptr<Tensor> encoder_input_dropped = encoder_input_with_pos;

  std::shared_ptr<Tensor> encoder_output = encoder_.forward(
      encoder_input_dropped, encoder_padding_mask, is_training);

  std::shared_ptr<Tensor> decoder_input_embeddings =
      decoder_embedding_.forward(decoder_input_ids);
  std::shared_ptr<Tensor> decoder_input_with_pos =
      decoder_positional_encoding_.forward(decoder_input_embeddings);

  std::shared_ptr<Tensor> decoder_input_dropped = decoder_input_with_pos;

  std::shared_ptr<Tensor> decoder_output = decoder_.forward(
      decoder_input_dropped, encoder_output, decoder_self_attention_mask,
      decoder_cross_attention_mask, is_training);

  std::shared_ptr<Tensor> logits = final_linear_.forward(decoder_output);

  return logits;
}

void Transformer::save_weights(const std::string& filename) const {
  std::ofstream outfile(filename, std::ios::binary | std::ios::trunc);
  if (!outfile.is_open()) {
    throw std::runtime_error("Failed to open file for saving weights: " + filename);
  }

  BinaryWriter writer(outfile);

  const auto& optimizable_tensors = Tensor::get_optimizable_tensors();
  const uint32_t num_tensors = optimizable_tensors.size();

  writer.write(num_tensors);

  for (const auto& tensor_ptr : optimizable_tensors) {
    if (!tensor_ptr) {
      std::cerr << "Warning: Encountered null tensor pointer while saving weights.\n";
      continue;
    }

    tensor_ptr->write(writer);

    if (!writer.good()) {
      throw std::ios_base::failure("Error writing tensor data to file: " + filename);
    }
  }
}

void Transformer::load_weights(const std::string& filename) {
  std::ifstream infile(filename, std::ios::binary);
  if (!infile.is_open()) {
    throw std::runtime_error("Failed to open file for loading weights: " + filename);
  }

  BinaryReader reader(infile);

  uint32_t num_tensors_in_file = reader.read<uint32_t>();
  if (!reader.good()) {
    throw std::runtime_error("Failed to read number of tensors from file: " + filename);
  }

  auto& optimizable_tensors = Tensor::get_optimizable_tensors();
  if (num_tensors_in_file != optimizable_tensors.size()) {
    throw std::runtime_error(std::format(
      "Mismatch between number of tensors in file ({}) and model ({}). Model architecture may have changed.",
      num_tensors_in_file, optimizable_tensors.size()));
  }

  std::cout << std::format("Loading {} optimizable tensors from {}...\n", num_tensors_in_file, filename);

  for (size_t i = 0; i < num_tensors_in_file; ++i) {
    std::shared_ptr<Tensor>& tensor_ptr = optimizable_tensors[i];

    if (!tensor_ptr) {
      throw std::runtime_error(std::format("Warning: Encountered null tensor pointer in model while "
        "loading weights for index {}\n", i));
    }
    tensor_ptr->read(reader);
  }
}