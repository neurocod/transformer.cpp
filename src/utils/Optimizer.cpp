#include "utils/Optimizer.h"
#include "utils/Helpers.h"
#include <cmath>
#include <iostream>
#include <limits>

void SGD::step() {
  // Iterate through all parameters
  for (const Tensor::Ptr &param : parameters_) {
    if (param) {
      // Could make a copy of the param_data and then set it again, but this
      // qould be inefficient because you have an extra copy
      std::vector<float> &param_data = param->dataRef();
      const std::vector<float> &param_grad = param->grad();

      if (param_data.size() != param_grad.size()) {
        throw std::runtime_error(
            "Parameter data and gradient size mismatch in SGD step.");
      }

      // param_data = param_data - learningRate * param_grad
      for (size_t i = 0; i < param_data.size(); ++i) {
        param_data[i] -= learning_rate_ * param_grad[i];
      }
    }
  }
}

Adam::Adam(float learningRate, float beta1, float beta2, float epsilon)
    : learning_rate_(learningRate), beta1_(beta1), beta2_(beta2),
      epsilon_(epsilon), t_(0) {
  // Initialize moments m and v for each optimizable parameter
  for (const auto &param : parameters_) {
    m_.push_back(Tensor::create(param->shape()));
    v_.push_back(Tensor::create(param->shape()));
  }
}

void Adam::step() {
  t_++;

  float beta1_t = std::pow(beta1_, (float)t_);
  float beta2_t = std::pow(beta2_, (float)t_);

  // Gradient Clipping
  float total_norm_sq = 0.0f;
  for (size_t i = 0; i < parameters_.size(); ++i) {
    Tensor::Ptr param = parameters_[i];
    if (!param || param->grad().empty())
      continue;

    const std::vector<float> &param_grad = param->grad();
    for (float grad_val : param_grad) {
      total_norm_sq += grad_val * grad_val;
    }
  }
  float total_norm = std::sqrt(total_norm_sq);
  float clip_threshold = 1.0f;
  float clip_coef = clip_threshold / (total_norm + epsilon_);

  // Clip gradients if the total norm exceeds the threshold
  if (clip_coef < 1.0f) {
    for (size_t i = 0; i < parameters_.size(); ++i) {
      Tensor::Ptr param = parameters_[i];
      if (!param || param->grad().empty())
        continue;

      std::vector<float> &param_grad_mutable = param->gradRef();
      for (float &grad_val : param_grad_mutable) {
        grad_val *= clip_coef;
      }
    }
  }

  for (size_t i = 0; i < parameters_.size(); ++i) {
    Tensor::Ptr param = parameters_[i];
    if (!param || param->data().empty() || param->grad().empty())
      continue;

    std::vector<float> &param_data = param->dataRef();
    const std::vector<float> &param_grad = param->grad();

    std::vector<float> &m_data = m_[i]->dataRef();
    std::vector<float> &v_data = v_[i]->dataRef();

    if (param_data.size() != param_grad.size() ||
        param_data.size() != m_data.size() ||
        param_data.size() != v_data.size()) {
      throw std::runtime_error(
          "Parameter, gradient, and moment size mismatch in Adam step.");
    }

    for (size_t j = 0; j < param_data.size(); ++j) {
      // Update biased first and second moment estimates
      m_data[j] = beta1_ * m_data[j] + (1.0f - beta1_) * param_grad[j];
      v_data[j] =
          beta2_ * v_data[j] + (1.0f - beta2_) * param_grad[j] * param_grad[j];

      // Compute bias-corrected first and second moment estimates
      float m_hat = m_data[j] / (1.0f - beta1_t);
      float v_hat = v_data[j] / (1.0f - beta2_t);

      // Update parameters
      param_data[j] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
    }
  }
}