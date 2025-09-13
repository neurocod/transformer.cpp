#pragma once
#include "../utils/Tensor.h"

// Base class for activation functions
class Activation {
public:
  virtual ~Activation() = default;

  virtual std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) = 0;
};

// ReLU Activation
class ReLU : public Activation {
public:
  ~ReLU() override = default;
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;
};

// GELU Activation
class GELU : public Activation {
public:
  ~GELU() override = default;
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;
};

// Sigmoid Activation
class Sigmoid : public Activation {
public:
  ~Sigmoid() override = default;
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;
};

// Tahn Activation
class Tanh : public Activation {
public:
  ~Tanh() override = default;
  std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) override;
};
