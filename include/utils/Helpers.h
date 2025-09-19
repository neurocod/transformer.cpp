#pragma once
#include "Tensor.h"

inline void print_tensor(const std::shared_ptr<Tensor> &t,
                         const std::string &name) {
  if (!t)
    return;
  std::stringstream out;
  out << "--- Tensor: " << name << " ---" << std::endl;
  out << "Shape: [";
  for (size_t i = 0; i < t->shape().size(); ++i) {
    out << t->shape()[i]
              << (i == t->shape().size() - 1 ? "" : ", ");
  }
  out << "]" << std::endl;

  out << "Data: [";
  const auto &data = t->data();
  size_t print_limit = std::min((size_t)20, data.size());
  for (size_t i = 0; i < print_limit; ++i) {
    out << data[i] << (i == print_limit - 1 ? "" : ", ");
  }
  if (data.size() > print_limit) {
    out << ", ...";
  }
  out << "]" << std::endl;

  out << "Grad: [";
  const auto &grad = t->grad();
  print_limit = std::min((size_t)20, grad.size());
  for (size_t i = 0; i < print_limit; ++i) {
    out << grad[i] << (i == print_limit - 1 ? "" : ", ");
  }
  if (grad.size() > print_limit) {
    out << ", ...";
  }
  out << "]" << std::endl;
  out << "--------------------------";
  spdlog::info(out.str());
}

inline bool are_tensors_equal(const std::shared_ptr<Tensor> &t1,
                              const std::shared_ptr<Tensor> &t2,
                              float tolerance = 1e-9) {
  if (!t1 || !t2)
    return false;
  if (t1->shape() != t2->shape()) {
    spdlog::error("Shape mismatch!");
    return false;
  }
  const auto &data1 = t1->data();
  const auto &data2 = t2->data();
  if (data1.size() != data2.size()) {
    spdlog::error("Data size mismatch!");
    return false;
  }

  for (size_t i = 0; i < data1.size(); ++i) {
    if (std::abs(data1[i] - data2[i]) > tolerance) {
      spdlog::error("Data mismatch at index {}: {} vs {}", i, data1[i], data2[i]);
      return false;
    }
  }
  return true;
}

template <typename T>
inline std::string vector_to_string(const std::vector<T> &vec) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    oss << vec[i];
    if (i != vec.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}
