#include "layers/Embedding.h"

Embedding::Embedding(int vocabSize, int embedDim)
    : _vocabSize(vocabSize), _embedDim(embedDim) {
  // Embedding matrix shape: (vocabSize, embedDim)
  _weights = Tensor::create(std::vector<int>{_vocabSize, _embedDim}, "embed.w");

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

  std::shared_ptr<Vec> weights_data =
      std::make_shared<Vec>(_vocabSize * _embedDim);
  for (size_t i = 0; i < weights_data->size(); ++i) {
    (*weights_data)[i] = dist(gen);
  }
  _weights->set_data(weights_data);
}

Tensor::Ptr Embedding::forward(const Tensor::Ptr &input_ids) {
  // input_ids shape: (batchSize, sequenceLength)
  // Output shape: (batchSize, sequenceLength, embedDim)

  const std::vector<int> &input_shape = input_ids->shape();
  if (input_shape.size() != 2) {
    throw std::runtime_error("Embedding layer input must be a 2D tensor "
                             "(batchSize, sequenceLength).");
  }

  size_t batchSize = input_shape[0];
  size_t sequenceLength = input_shape[1];

  Tensor::Ptr output = Tensor::create({(int)batchSize, (int)sequenceLength, _embedDim});
  Vec &output_data = output->dataRef();

  const Vec &input_ids_data = input_ids->data();
  const Vec &weights_data = _weights->data();

  for (size_t i = 0; i < batchSize * sequenceLength; ++i) {
    float token_id_float = input_ids_data[i];
    if (token_id_float < 0 || token_id_float >= _vocabSize ||
        std::fmod(token_id_float, 1.0f) != 0.0f) {
      throw std::runtime_error("Input token ID out of vocabulary bounds or not "
                               "an integer in Embedding forward.");
    }
    int token_id = static_cast<int>(token_id_float);

    size_t output_start_idx = i * _embedDim;
    size_t weights_start_idx = token_id * _embedDim;

    for (size_t j = 0; j < _embedDim; ++j) {
      output_data[output_start_idx + j] = weights_data[weights_start_idx + j];
    }
  }

  output->set_creator_op(OperationType::EmbeddingLookup);
  output->set_parents({_weights});
  output->embedding_indices_ = input_ids;

  return output;
}

Tensor::Ptr Embedding::weights() { return _weights; }
