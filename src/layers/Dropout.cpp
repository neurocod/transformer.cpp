#include "layers/Dropout.h"

Dropout::Dropout(float rate) : rate_(rate),
  generator_(std::random_device{}()),
  distribution_(0.0f, 1.0f)
{
  if (rate_ < 0.0f || rate_ >= 1.0f) {
    throw std::runtime_error("Dropout rate must be between 0 and less than 1.");
  }
}

Tensor::Ptr Dropout::forward(const Tensor::Ptr &input,
                                         bool isTraining) {
  if (!isTraining)
    return input;

  Tensor::Ptr output = Tensor::create(input->shape());
  Tensor::Ptr mask = Tensor::create(input->shape());

  const Vec &input_data = input->data();
  Vec &output_data = output->dataRef();
  Vec &mask_data = mask->dataRef();

  float scale = 1.0f / (1.0f - rate_);

  for (size_t i = 0; i < input_data.size(); ++i) {
    if (distribution_(generator_) > rate_) {
      // Keep the element
      output_data[i] = input_data[i] * scale;
      mask_data[i] = 1.0f;
    } else {
      // Drop the element
      output_data[i] = 0.0f;
      mask_data[i] = 0.0f;
    }
  }

  output->set_creator_op(OperationType::Dropout);
  output->set_parents({input});
  output->dropout_mask_ = mask;
  output->dropout_scale_ = scale;

  return output;
}