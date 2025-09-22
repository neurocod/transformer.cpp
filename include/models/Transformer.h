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

	Tensor::Ptr forward(const Tensor::Ptr& encoderInputIds,
		const Tensor::Ptr& decoderInputIds, bool isTraining);

  bool saveToFile(const std::string &filename) const;
  static std::shared_ptr<Transformer> loadFromFile(const std::string &filename, int inputVocabSize, int targetVocabSize);

protected:
  TransformerConfig _cfg;
  Embedding _encoderEmbedding;
  PositionalEncoding _encoderPositionalEncoding;
  Encoder _encoder;

  Embedding _decoderEmbedding;
  PositionalEncoding _decoderPositionalEncoding;
  Decoder _decoder;

  Linear _linearFinal;

  Tensor::Ptr createEncoderPaddingMask(const Tensor::Ptr &encoderInputIds);
  Tensor::Ptr createDecoderSelfAttentionMask(const Tensor::Ptr &decoderInputIds);
  Tensor::Ptr create_decoder_cross_attention_mask(const Tensor::Ptr &encoderInputIds);
};
