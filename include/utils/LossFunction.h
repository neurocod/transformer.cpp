#pragma once
#include "Tensor.h"

class LossFunction {
public:
  virtual ~LossFunction() = default;

  virtual Tensor::Ptr
  computeLoss(Tensor::Ptr &predictions,
               Tensor::Ptr &targets) = 0;

  // This assumes computeLoss returns a shared_ptr to a Tensor that has its
  // computation graph set up.
  void backward(Tensor::Ptr &loss);
};

class MeanSquaredErrorLoss : public LossFunction {
public:
  MeanSquaredErrorLoss() = default;

  ~MeanSquaredErrorLoss() override = default;

  // MSE = mean((predictions - targets)^2)
  Tensor::Ptr computeLoss(Tensor::Ptr &predictions,
               Tensor::Ptr &targets) override;
};

class CrossEntropyLoss : public LossFunction {
public:
  CrossEntropyLoss() = default;
  ~CrossEntropyLoss() override = default;

  // Computes the Cross-Entropy Loss
  Tensor::Ptr computeLoss(Tensor::Ptr &predictions,
               Tensor::Ptr &targets) override;
};
