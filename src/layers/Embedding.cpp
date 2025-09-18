#include "layers/Embedding.h"

Embedding::Embedding(int vocabSize, int embedDim)
    : _vocabSize(vocabSize), _embedDim(embedDim) {
  // Embedding matrix shape: (vocabSize, embedDim)
  _weights = Tensor::create(std::vector<int>{_vocabSize, _embedDim}, "embed.w");

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

  std::shared_ptr<std::vector<float>> weights_data =
      std::make_shared<std::vector<float>>(_vocabSize * _embedDim);
  for (size_t i = 0; i < weights_data->size(); ++i) {
    (*weights_data)[i] = dist(gen);
  }
  _weights->set_data(weights_data);
}

std::shared_ptr<Tensor>
Embedding::forward(const std::shared_ptr<Tensor> &input_ids) {
  // input_ids shape: (batchSize, sequence_length)
  // Output shape: (batchSize, sequence_length, embedDim)

  const std::vector<int> &input_shape = input_ids->get_shape();
  if (input_shape.size() != 2) {
    throw std::runtime_error("Embedding layer input must be a 2D tensor "
                             "(batchSize, sequence_length).");
  }

  size_t batchSize = input_shape[0];
  size_t sequence_length = input_shape[1];

  std::shared_ptr<Tensor> output =
      Tensor::create({(int)batchSize, (int)sequence_length, _embedDim});
  std::vector<float> &output_data = output->data_ref();

  const std::vector<float> &input_ids_data = input_ids->get_data();
  const std::vector<float> &weights_data = _weights->get_data();

  for (size_t i = 0; i < batchSize * sequence_length; ++i) {
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

std::shared_ptr<Tensor> Embedding::weights() { return _weights; }
