#include "models/Transformer.h"
#include "utils/BinaryWriter.h"
#include "utils/BinaryReader.h"
#include "utils/ConfigParser.h"

Transformer::Transformer(const TransformerConfig& cf): _(cf),
      _encoderEmbedding(cf.inputVocabSize, cf.embedDim),
      _encoderPositionalEncoding(cf.maxSequenceLength, cf.embedDim),
      _encoder(cf.numLayers, cf.embedDim, cf.numHeads, cf.ffHiddenDim, cf.dropoutRate),
      _decoderEmbedding(cf.targetVocabSize, cf.embedDim),
      _decoderPositionalEncoding(cf.maxSequenceLength, cf.embedDim),
      _decoder(cf.numLayers, cf.embedDim, cf.numHeads, cf.ffHiddenDim, cf.dropoutRate),
      _linearFinal(cf.embedDim, cf.targetVocabSize, "final") {
  if (cf.embedDim % cf.numHeads != 0) {
    throw std::runtime_error(
        "Embedding dimension must be divisible by the number of heads.");
  }
}

std::shared_ptr<Tensor> Transformer::createEncoderPaddingMask(const std::shared_ptr<Tensor> &encoderInputIds) {
  return create_padding_mask(encoderInputIds, _.padTokenId);
}

std::shared_ptr<Tensor> Transformer::createDecoderSelfAttentionMask(const std::shared_ptr<Tensor> &decoderInputIds) {
  const auto &decoder_shape = decoderInputIds->shape();
  if (decoder_shape.size() != 2) {
    throw std::runtime_error(
        "Decoder input for self-attention mask must be 2D (batch, seq_len).");
  }
  int batchSize = decoder_shape[0];
  int sequence_length = decoder_shape[1];

  std::shared_ptr<Tensor> look_ahead_mask =
      create_look_ahead_mask(sequence_length);
  std::shared_ptr<Tensor> padding_mask = create_padding_mask(decoderInputIds, _.padTokenId);

  const auto &look_ahead_shape = look_ahead_mask->shape();
  const auto &padding_shape = padding_mask->shape();

  if (look_ahead_shape !=
          std::vector<int>{1, 1, sequence_length, sequence_length} ||
      padding_shape != std::vector<int>{batchSize, 1, 1, sequence_length}) {
    throw std::runtime_error("Unexpected shapes for look-ahead or padding "
                             "masks during combination.");
  }

  std::vector<int> combined_mask_shape = {batchSize, 1, sequence_length,
                                          sequence_length};
  std::shared_ptr<Tensor> combined_mask = Tensor::create(combined_mask_shape);
  std::vector<float> &combined_mask_data = combined_mask->dataRef();

  const std::vector<float> &look_ahead_data = look_ahead_mask->data();
  const std::vector<float> &padding_data = padding_mask->data();

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
  return create_padding_mask(encoderInputIds, _.padTokenId);
}

std::shared_ptr<Tensor>
Transformer::forward(const std::shared_ptr<Tensor> &encoderInputIds,
                     const std::shared_ptr<Tensor> &decoderInputIds,
                     bool isTraining) {
  const std::vector<int> &enc_input_shape = encoderInputIds->shape();
  const std::vector<int> &dec_input_shape = decoderInputIds->shape();

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

static const std::string binFileTag("TensorConfig");

void Transformer::saveToFile(const std::string &filename) const {
  std::ofstream outfile(filename, std::ios::binary | std::ios::trunc);
  if (!outfile.is_open()) {
    throw std::runtime_error("Failed to open file for saving weights: " + filename);
  }

  BinaryWriter writer(outfile);

  writer.write(binFileTag);
  writer.write(_.toString());

  const auto& optimizable_tensors = Tensor::get_optimizable_tensors();
  const uint32_t num_tensors = optimizable_tensors.size();

  writer.write(num_tensors);

  for (const auto& tensor_ptr : optimizable_tensors) {
    if (!tensor_ptr) {
      spdlog::error("Warning: Encountered null tensor pointer while saving weights.");
      continue;
    }

    tensor_ptr->write(writer);

    if (!writer.ok()) {
      throw std::ios_base::failure("Error writing tensor data to file: " + filename);
    }
  }
}

std::shared_ptr<Transformer> Transformer::loadFromFile(const std::string &filename, int inputVocabSize,
                                                       int targetVocabSize) {
  std::ifstream infile(filename, std::ios::binary);
  if (!infile.is_open()) {
    spdlog::error("Can't open file {}", filename);
    return 0;
  }

  BinaryReader reader(infile);
  auto fileTag = reader.readString();
  if (!reader.ok() || fileTag != binFileTag) {
    spdlog::error("Incorrect Transformer file format in {}", filename);
    return 0;
  }
  std::string configIni = reader.readString();
  if (!reader.ok() || configIni.empty()) {
    spdlog::error("Incorrect Transformer config in {}", filename);
    return 0;
  }

  ConfigParser parser;
  parser.loadIniValues(configIni);

  TransformerConfig config;
  config.init(parser);
  config.inputVocabSize = inputVocabSize;
  config.targetVocabSize = targetVocabSize;

  auto ret = std::make_shared<Transformer>(config);

  uint32_t num_tensors_in_file = reader.read<uint32_t>();
  if (!reader.ok()) {
    throw std::runtime_error("Failed to read number of tensors from file: " + filename);
  }

  auto& optimizable_tensors = Tensor::get_optimizable_tensors();
  if (num_tensors_in_file != optimizable_tensors.size()) {
    throw std::runtime_error(std::format(
      "Mismatch between number of tensors in file ({}) and model ({}). Model architecture may have changed.",
      num_tensors_in_file, optimizable_tensors.size()));
  }

  spdlog::info("Loading {} optimizable tensors from {}...", num_tensors_in_file, filename);

  for (size_t i = 0; i < num_tensors_in_file; ++i) {
    std::shared_ptr<Tensor>& tensor_ptr = optimizable_tensors[i];

    if (!tensor_ptr) {
      throw std::runtime_error(std::format("Warning: Encountered null tensor pointer in model while "
        "loading weights for index {}", i));
    }
    tensor_ptr->read(reader);
  }
  return ret;
}