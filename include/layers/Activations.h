#pragma once
#include "../utils/Tensor.h"

// Base class for activation functions
class Activation {
public:
  virtual ~Activation() = default;

  virtual Tensor::Ptr forward(const Tensor::Ptr &input) = 0;
  using Vec = Tensor::Vec;
};

// ReLU Activation
class ReLU : public Activation {
public:
  ~ReLU() override = default;
  Tensor::Ptr forward(const Tensor::Ptr &input) override;
};

// GELU Activation
class GELU : public Activation {
public:
  ~GELU() override = default;
  Tensor::Ptr forward(const Tensor::Ptr &input) override;
};

// Sigmoid Activation
class Sigmoid : public Activation {
public:
  ~Sigmoid() override = default;
  Tensor::Ptr forward(const Tensor::Ptr &input) override;
};

// Tahn Activation
class Tanh : public Activation {
public:
  ~Tanh() override = default;
  Tensor::Ptr forward(const Tensor::Ptr &input) override;
};
