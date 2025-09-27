#include "utils/Masking.h"
#include <algorithm>
#include <iostream>

using Vec = Tensor::Vec;
Tensor::Ptr create_look_ahead_mask(int sequenceLength) {
  Tensor::Ptr mask = Tensor::create({sequenceLength, sequenceLength});
  Vec &mask_data = mask->dataRef();

  // Create a lower triangular mask with -inf in the upper triangle
  float negative_infinity = -std::numeric_limits<float>::infinity();

  for (int i = 0; i < sequenceLength; ++i) {
    for (int j = 0; j < sequenceLength; ++j) {
      if (j > i) {
        mask_data[i * sequenceLength + j] = negative_infinity;
      } else {
        mask_data[i * sequenceLength + j] = 0.0f;
      }
    }
  }

  mask = mask->reshape({1, 1, sequenceLength, sequenceLength});

  return mask;
}

Tensor::Ptr create_padding_mask(const Tensor::Ptr &input_ids,
                    float padTokenId) {
  const std::vector<int> &input_shape = input_ids->shape();
  if (input_shape.size() != 2) {
    throw std::runtime_error("Input IDs for padding mask must be a 2D tensor "
                             "(batchSize, sequenceLength).");
  }

  size_t batchSize = input_shape[0];
  size_t sequenceLength = input_shape[1];

  Tensor::Ptr mask = Tensor::create({(int)batchSize, (int)sequenceLength});
  Vec &mask_data = mask->dataRef();
  const Vec &input_ids_data = input_ids->data();

  for (size_t i = 0; i < batchSize * sequenceLength; ++i) {
    if (input_ids_data[i] == padTokenId) {
      mask_data[i] = -std::numeric_limits<float>::infinity();
    } else {
      mask_data[i] = 0.0f;
    }
  }

  // Create a mask of shape (batchSize, 1, 1, sequenceLength) and rely on
  // broadcasting rules to expand it over the query sequence length dimension.
  Tensor::Ptr final_mask = mask->reshape({(int)batchSize, 1, 1, (int)sequenceLength});

  return final_mask;
}