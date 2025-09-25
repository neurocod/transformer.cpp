#include "layers/Activations.h"

Tensor::Ptr ReLU::forward(const Tensor::Ptr &input) {
  Tensor::Ptr output = Tensor::create(input->shape());
  const Vec &input_data = input->data();
  Vec &output_data = output->dataRef();

  for (size_t i = 0; i < input_data.size(); ++i) {
    output_data[i] = std::max(0.0f, input_data[i]);
  }

  output->set_creator_op(OperationType::ReLU);
  output->set_parents({input});

  return output;
}

Tensor::Ptr GELU::forward(const Tensor::Ptr &input) {
  Tensor::Ptr output = Tensor::create(input->shape());
  const Vec &input_data = input->data();
  Vec &output_data = output->dataRef();

  // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2 / PI) * (x + 0.044715 *
  // x^3)))
  const float M_SQRT2_OVER_PI = 0.7978845608028654f; // sqrt(2 / PI)
  const float GELU_CONSTANT = 0.044715f;

  for (size_t i = 0; i < input_data.size(); ++i) {
    float x = input_data[i];
    float tanh_arg = M_SQRT2_OVER_PI * (x + GELU_CONSTANT * std::pow(x, 3));
    output_data[i] = 0.5f * x * (1.0f + std::tanh(tanh_arg));
  }

  output->set_creator_op(OperationType::GELU);
  output->set_parents({input});

  return output;
}

Tensor::Ptr Sigmoid::forward(const Tensor::Ptr &input) {
  Tensor::Ptr output = Tensor::create(input->shape());
  const Vec &input_data = input->data();
  Vec &output_data = output->dataRef();

  for (size_t i = 0; i < input_data.size(); ++i) {
    output_data[i] = 1.0f / (1.0f + std::exp(-input_data[i]));
  }

  output->set_creator_op(OperationType::Sigmoid);
  output->set_parents({input});

  return output;
}

Tensor::Ptr Tanh::forward(const Tensor::Ptr &input) {
  Tensor::Ptr output = Tensor::create(input->shape());
  const Vec &input_data = input->data();
  Vec &output_data = output->dataRef();

  for (size_t i = 0; i < input_data.size(); ++i) {
    output_data[i] = std::tanh(input_data[i]);
  }

  output->set_creator_op(OperationType::Tanh);
  output->set_parents({input});

  return output;
}
