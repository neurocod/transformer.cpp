#pragma once
#include "Tensor.h"

// Generates a look-ahead mask to prevent attention to future tokens in the
// decoder.
Tensor::Ptr create_look_ahead_mask(int sequence_length);

// Generates a padding mask to ignore padded elements in sequences.
Tensor::Ptr create_padding_mask(const Tensor::Ptr &input_ids,
                    float padTokenId = 0.0f);
