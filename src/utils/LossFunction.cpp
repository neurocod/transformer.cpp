#include <utils/LossFunction.h>

using Vec = Tensor::Vec;
void LossFunction::backward(Tensor::Ptr &loss) {
  // The gradient of the loss with respect to itself is 1.0.
  if (loss->shape().size() != 1 || loss->shape()[0] != 1) {
    throw std::runtime_error(
        "Backward pass for loss must start from a scalar loss tensor.");
  }
  Tensor::Ptr initial_grad_for_loss = Tensor::create(
      {1}, std::make_shared<Vec>(Vec{{1.0f}}));
  loss->backward(initial_grad_for_loss);
}

Tensor::Ptr MeanSquaredErrorLoss::computeLoss(Tensor::Ptr &predictions,
                                   Tensor::Ptr &targets) {
  if (predictions->shape() != targets->shape()) {
    throw std::runtime_error(
        "Prediction and target shapes mismatch in MeanSquaredErrorLoss.");
  }

  Tensor::Ptr diff = *predictions - targets;
  Tensor::Ptr squared_diff = *diff * diff;
  Tensor::Ptr sum_squared_diff = squared_diff->sum();

  size_t num_elements = predictions->num_elements();
  if (num_elements == 0) {
    return Tensor::create(
        std::vector<int>{1},
        std::make_shared<Vec>(Vec{0.0f}));
  }

  Tensor::Ptr num_elements_tensor = Tensor::create(
      {1}, std::make_shared<Vec>(
               Vec{{static_cast<float>(num_elements)}}));
  Tensor::Ptr mean_loss = (*sum_squared_diff) / num_elements_tensor;

  return mean_loss;
}

// Helper function for LogSoftmax
Tensor::Ptr log_softmax(const Tensor::Ptr &input) {
  Tensor::Ptr output = Tensor::create(input->shape());
  const Vec &input_data = input->data();
  Vec &output_data = output->dataRef();
  const std::vector<int> &shape = input->shape();
  size_t last_dim_size = shape.empty() ? 0 : shape.back();
  size_t num_elements = input->num_elements();

  if (last_dim_size == 0 || num_elements == 0) {
    return output;
  }

  size_t outer_dims_elements = num_elements / last_dim_size;

  for (size_t i = 0; i < outer_dims_elements; ++i) {
    size_t start_idx = i * last_dim_size;
    float max_val = -std::numeric_limits<float>::infinity();

    // Find the maximum value for numerical stability
    for (size_t j = 0; j < last_dim_size; ++j) {
      if (input_data[start_idx + j] > max_val) {
        max_val = input_data[start_idx + j];
      }
    }

    float log_sum_exp = 0.0f;
    for (size_t j = 0; j < last_dim_size; ++j) {
      log_sum_exp += std::exp(input_data[start_idx + j] - max_val);
    }
    log_sum_exp = max_val + std::log(log_sum_exp);

    // Calculate log softmax
    for (size_t j = 0; j < last_dim_size; ++j) {
      output_data[start_idx + j] = input_data[start_idx + j] - log_sum_exp;
    }
  }

  output->set_creator_op(OperationType::LogSoftmax);
  output->set_parents({input});

  return output;
}

// Helper function for Negative Log Likelihood Loss
Tensor::Ptr nll_loss(const Tensor::Ptr &log_probabilities,
         const Tensor::Ptr &targets) {
  Tensor::Ptr loss = Tensor::create(std::vector<int>{1},
      std::make_shared<Vec>(Vec{0.0f}));
  const Vec &log_prob_data = log_probabilities->data();
  const Vec &target_data = targets->data();
  float total_loss = 0.0f;

  const std::vector<int> &log_prob_shape = log_probabilities->shape();
  size_t last_dim_size = log_prob_shape.empty() ? 0 : log_prob_shape.back();
  size_t num_elements_log_prob = log_probabilities->num_elements();

  if (last_dim_size == 0 || num_elements_log_prob == 0) {
    return loss;
  }

  if (log_prob_shape.size() < 1) {
    throw std::runtime_error(
        "Log probabilities tensor must have at least 1 dimension for NLLLoss.");
  }

  size_t outer_dims_elements = num_elements_log_prob / last_dim_size;

  if (targets->num_elements() != outer_dims_elements) {
    throw std::runtime_error("Target data size mismatch with log probabilities "
                             "outer dimensions in NLLLoss.");
  }

  for (size_t i = 0; i < outer_dims_elements; ++i) {
    size_t log_prob_start_idx = i * last_dim_size;
    size_t target_idx = i;

    float target_val = target_data[target_idx];
    if (target_val < 0 || target_val >= static_cast<float>(last_dim_size) ||
        std::fmod(target_val, 1.0f) != 0.0f) {
      throw std::runtime_error(
          "Target class index is out of bounds or not an integer in NLLLoss.");
    }
    int target_class = static_cast<int>(target_val);

    total_loss += -log_prob_data[log_prob_start_idx + target_class];
  }

  // Mean loss
  if (outer_dims_elements > 0) {
    loss->set(std::vector<int>{0}, total_loss / outer_dims_elements);
  } else {
    loss->set(std::vector<int>{0}, 0.0f);
  }

  loss->set_creator_op(OperationType::NegativeLogLikelihood);
  loss->set_parents({log_probabilities, targets});

  return loss;
}

Tensor::Ptr CrossEntropyLoss::computeLoss(Tensor::Ptr &predictions,
                               Tensor::Ptr &targets) {
  // Apply LogSoftmax to predictions
  Tensor::Ptr log_probs = log_softmax(predictions);

  // Compute Negative Log Likelihood Loss
  Tensor::Ptr loss = nll_loss(log_probs, targets);

  return loss;
}
