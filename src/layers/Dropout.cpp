#include "layers/Dropout.h"

Dropout::Dropout(float rate) : rate_(rate),
  generator_(std::random_device{}()),
  distribution_(0.0f, 1.0f)
{
  if (rate_ < 0.0f || rate_ >= 1.0f) {
    throw std::runtime_error("Dropout rate must be between 0 and less than 1.");
  }
}

std::shared_ptr<Tensor> Dropout::forward(const std::shared_ptr<Tensor> &input,
                                         bool isTraining) {
  if (!isTraining) {
    return input;
  }

  std::shared_ptr<Tensor> output = Tensor::create(input->get_shape());
  std::shared_ptr<Tensor> mask = Tensor::create(input->get_shape());

  const std::vector<float> &input_data = input->get_data();
  std::vector<float> &output_data = output->data_ref();
  std::vector<float> &mask_data = mask->data_ref();

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