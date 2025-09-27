#include "layers/PositionalEncoding.h"

PositionalEncoding::PositionalEncoding(int maxSequenceLength, int embedDim)
    : _maxSequenceLength(maxSequenceLength), _embedDim(embedDim) {
  // Pre-calculate the positional encodings matrix
  positional_encodings_ = Tensor::create({_maxSequenceLength, _embedDim});
  Vec &encodings_data = positional_encodings_->dataRef();

  for (int pos = 0; pos < _maxSequenceLength; ++pos) {
    for (int i = 0; i < _embedDim; ++i) {
      if (i % 2 == 0) {
        // Sine for even dimensions
        encodings_data[pos * _embedDim + i] = std::sin(
            pos / std::pow(10000.0f, static_cast<float>(i) / _embedDim));
      } else {
        // Cosine for odd dimensions
        encodings_data[pos * _embedDim + i] = std::cos(
            pos / std::pow(10000.0f, static_cast<float>(i - 1) / _embedDim));
      }
    }
  }
}

Tensor::Ptr PositionalEncoding::forward(const Tensor::Ptr &input) {
  const std::vector<int> &input_shape = input->shape();
  if (input_shape.size() != 3) {
    throw std::runtime_error("Positional Encoding input must be a 3D tensor "
                             "(batchSize, sequenceLength, embedDim).");
  }

  size_t batchSize = input_shape[0];
  size_t sequenceLength = input_shape[1];
  size_t embedDim = input_shape[2];

  if (embedDim != _embedDim) {
    throw std::runtime_error(
        "Input embedding dimension mismatch in Positional Encoding.");
  }

  if (sequenceLength > _maxSequenceLength) {
    throw std::runtime_error("Input sequence length exceeds maxSequenceLength in Positional Encoding");
  }

  // Add the pre-calculated positional encodings to the input tensor.
  Tensor::Ptr output = Tensor::create(input_shape);
  const Vec &input_data = input->data();
  Vec &output_data = output->dataRef();
  const Vec &encoding_data = positional_encodings_->data();

  for (size_t b = 0; b < batchSize; ++b) {
    for (size_t s = 0; s < sequenceLength; ++s) {
      size_t input_start_idx = (b * sequenceLength + s) * embedDim;
      size_t encoding_start_idx = s * embedDim;

      for (size_t d = 0; d < embedDim; ++d) {
        output_data[input_start_idx + d] =
            input_data[input_start_idx + d] +
            encoding_data[encoding_start_idx + d];
      }
    }
  }
  output->set_creator_op(
      OperationType::Add); // Doesn't rly matter what this is cause you don't
                           // train the positional encodings
  output->set_parents({input, positional_encodings_});

  return output;
}