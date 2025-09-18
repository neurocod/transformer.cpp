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
  virtual ~Transformer() {}

	std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& encoderInputIds,
		const std::shared_ptr<Tensor>& decoderInputIds, bool isTraining);

  void saveWeights(const std::string &filename) const;
  void loadWeights(const std::string &filename);
protected:
  Embedding _encoderEmbedding;
  PositionalEncoding _encoderPositionalEncoding;
  Encoder _encoder;

  Embedding _decoderEmbedding;
  PositionalEncoding _decoderPositionalEncoding;
  Decoder _decoder;

  Linear _linearFinal;

  const int _inputVocabSize;
  const int _targetVocabSize;
  const int _embedDim;
  const int _maxSequenceLength;
  const int _numLayers;
  const int _numHeads;
  const int _ffHiddenDim;
  float _dropoutRate;
  float _padTokenId;

  std::shared_ptr<Tensor> createEncoderPaddingMask(const std::shared_ptr<Tensor> &encoderInputIds);
  std::shared_ptr<Tensor> createDecoderSelfAttentionMask(const std::shared_ptr<Tensor> &decoderInputIds);
  std::shared_ptr<Tensor> create_decoder_cross_attention_mask(const std::shared_ptr<Tensor> &encoderInputIds);
};
