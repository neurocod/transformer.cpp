#pragma once
#include "../layers/Embedding.h"
#include "../layers/Linear.h"
#include "../layers/PositionalEncoding.h"
#include "../utils/Masking.h"
#include "utils/TransformerConfig.h"
#include "Decoder.h"
#include "Encoder.h"
class TransformerConfig;

class Transformer {
public:
  Transformer(const TransformerConfig& config);
  virtual ~Transformer() {}

	std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& encoderInputIds,
		const std::shared_ptr<Tensor>& decoderInputIds, bool isTraining);

  void saveToFile(const std::string &filename) const;
  static std::shared_ptr<Transformer> loadFromFile(const std::string &filename, int inputVocabSize, int targetVocabSize);

protected:
  TransformerConfig _;
  Embedding _encoderEmbedding;
  PositionalEncoding _encoderPositionalEncoding;
  Encoder _encoder;

  Embedding _decoderEmbedding;
  PositionalEncoding _decoderPositionalEncoding;
  Decoder _decoder;

  Linear _linearFinal;

  std::shared_ptr<Tensor> createEncoderPaddingMask(const std::shared_ptr<Tensor> &encoderInputIds);
  std::shared_ptr<Tensor> createDecoderSelfAttentionMask(const std::shared_ptr<Tensor> &decoderInputIds);
  std::shared_ptr<Tensor> create_decoder_cross_attention_mask(const std::shared_ptr<Tensor> &encoderInputIds);
};
