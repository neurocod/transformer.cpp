#include "models/Transformer.h"
#include "utils/BinaryWriter.h"
#include "utils/BinaryReader.h"

Transformer::Transformer(int input_vocab_size, int target_vocab_size,
                         int embedDim, int maxSequenceLength, int numLayers,
                         int numHeads, int ffHiddenDim, float dropoutRate,
                         float padTokenId)
    : _inputVocabSize(input_vocab_size),
      _targetVocabSize(target_vocab_size), _embedDim(embedDim),
      _maxSequenceLength(maxSequenceLength), _numLayers(numLayers),
      _numHeads(numHeads), _ffHiddenDim(ffHiddenDim),
      _dropoutRate(dropoutRate), _padTokenId(padTokenId),
      _encoderEmbedding(input_vocab_size, embedDim),
      _encoderPositionalEncoding(maxSequenceLength, embedDim),
      _encoder(numLayers, embedDim, numHeads, ffHiddenDim, dropoutRate),
      _decoderEmbedding(target_vocab_size, embedDim),
      _decoderPositionalEncoding(maxSequenceLength, embedDim),
      _decoder(numLayers, embedDim, numHeads, ffHiddenDim, dropoutRate),
      _linearFinal(embedDim, target_vocab_size, "final") {
  if (_embedDim % _numHeads != 0) {
    throw std::runtime_error(
        "Embedding dimension must be divisible by the number of heads.");
  }
}

std::shared_ptr<Tensor> Transformer::createEncoderPaddingMask(const std::shared_ptr<Tensor> &encoderInputIds) {
  return create_padding_mask(encoderInputIds, _padTokenId);
}

std::shared_ptr<Tensor> Transformer::createDecoderSelfAttentionMask(const std::shared_ptr<Tensor> &decoderInputIds) {
  const auto &decoder_shape = decoderInputIds->get_shape();
  if (decoder_shape.size() != 2) {
    throw std::runtime_error(
        "Decoder input for self-attention mask must be 2D (batch, seq_len).");
  }
  int batchSize = decoder_shape[0];
  int sequence_length = decoder_shape[1];

  std::shared_ptr<Tensor> look_ahead_mask =
      create_look_ahead_mask(sequence_length);
  std::shared_ptr<Tensor> padding_mask =
      create_padding_mask(decoderInputIds, _padTokenId);

  const auto &look_ahead_shape = look_ahead_mask->get_shape();
  const auto &padding_shape = padding_mask->get_shape();

  if (look_ahead_shape !=
          std::vector<int>{1, 1, sequence_length, sequence_length} ||
      padding_shape != std::vector<int>{batchSize, 1, 1, sequence_length}) {
    throw std::runtime_error("Unexpected shapes for look-ahead or padding "
                             "masks during combination.");
  }

  std::vector<int> combined_mask_shape = {batchSize, 1, sequence_length,
                                          sequence_length};
  std::shared_ptr<Tensor> combined_mask = Tensor::create(combined_mask_shape);
  std::vector<float> &combined_mask_data = combined_mask->data_ref();

  const std::vector<float> &look_ahead_data = look_ahead_mask->get_data();
  const std::vector<float> &padding_data = padding_mask->get_data();

  for (int b = 0; b < batchSize; ++b) {
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
    const std::shared_ptr<Tensor> &encoderInputIds) {
  return create_padding_mask(encoderInputIds, _padTokenId);
}

std::shared_ptr<Tensor>
Transformer::forward(const std::shared_ptr<Tensor> &encoderInputIds,
                     const std::shared_ptr<Tensor> &decoderInputIds,
                     bool isTraining) {
  const std::vector<int> &enc_input_shape = encoderInputIds->get_shape();
  const std::vector<int> &dec_input_shape = decoderInputIds->get_shape();

  if (enc_input_shape.size() != 2 || dec_input_shape.size() != 2) {
    throw std::runtime_error("Encoder and decoder input IDs must be 2D tensors "
                             "(batchSize, sequence_length).");
  }

  std::shared_ptr<Tensor> encoder_padding_mask =
      createEncoderPaddingMask(encoderInputIds);
  std::shared_ptr<Tensor> decoder_self_attention_mask =
      createDecoderSelfAttentionMask(decoderInputIds);
  std::shared_ptr<Tensor> decoder_cross_attention_mask =
      create_decoder_cross_attention_mask(encoderInputIds);

  std::shared_ptr<Tensor> encoder_input_embeddings =
      _encoderEmbedding.forward(encoderInputIds);
  std::shared_ptr<Tensor> encoder_input_with_pos =
      _encoderPositionalEncoding.forward(encoder_input_embeddings);

  std::shared_ptr<Tensor> encoder_input_dropped = encoder_input_with_pos;

  std::shared_ptr<Tensor> encoder_output = _encoder.forward(
      encoder_input_dropped, encoder_padding_mask, isTraining);

  std::shared_ptr<Tensor> decoder_input_embeddings =
      _decoderEmbedding.forward(decoderInputIds);
  std::shared_ptr<Tensor> decoder_input_with_pos =
      _decoderPositionalEncoding.forward(decoder_input_embeddings);

  std::shared_ptr<Tensor> decoder_input_dropped = decoder_input_with_pos;

  std::shared_ptr<Tensor> decoder_output = _decoder.forward(
      decoder_input_dropped, encoder_output, decoder_self_attention_mask,
      decoder_cross_attention_mask, isTraining);

  std::shared_ptr<Tensor> logits = _linearFinal.forward(decoder_output);

  return logits;
}

void Transformer::saveWeights(const std::string& filename) const {
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

void Transformer::loadWeights(const std::string& filename) {
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