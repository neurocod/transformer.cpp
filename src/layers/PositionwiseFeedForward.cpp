#include "layers/PositionwiseFeedForward.h"

PositionwiseFeedForward::PositionwiseFeedForward(int inputDim, int hidden_dim):
  fc1_(inputDim, hidden_dim, "pos.fc1"),
  fc2_(hidden_dim, inputDim, "pos.fc2"),
  _inputDim(inputDim),
  hidden_dim_(hidden_dim) {}

Tensor::Ptr PositionwiseFeedForward::forward(Tensor::Ptr &input) {
  // The feed-forward network is applied to the last dimension (inputDim)
  // independently for each position.

  // First linear transformation
  Tensor::Ptr output_fc1 = fc1_.forward(input);

  // Apply activation function
  Tensor::Ptr activated_output = activation_.forward(output_fc1);

  // Second linear transformation
  Tensor::Ptr output_fc2 = fc2_.forward(activated_output);

  return output_fc2;
}