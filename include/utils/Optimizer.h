#pragma once
#include "Tensor.h"

// Base class for all optimizers
class Optimizer {
public:
  Optimizer() {}
  virtual ~Optimizer() = default;

  // Derived classes must implement this to update parameters based on gradients
  virtual void step() = 0;

  // Zeros the gradients of all parameters managed by this optimizer
  void zeroGrad() {
    for (Tensor::Ptr &param : parameters_) {
      if (param) {
        param->zeroGrad();
      }
    }
  }

protected:
  // List of shared pointers to the parameters to update
  std::vector<Tensor::Ptr> &parameters_ = Tensor::get_optimizable_tensors();
};

class SGD : public Optimizer {
public:
  SGD(float learningRate) : learning_rate_(learningRate) {}
  ~SGD() override = default;

  // Implements the SGD optimization step
  void step() override;

private:
  float learning_rate_;
};

class Adam : public Optimizer {
public:
  Adam(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
       float epsilon = 1e-8f);
  ~Adam() override = default;

  // Implements the Adam optimization step
  void step() override;
private:
  float learning_rate_;
  float beta1_;
  float beta2_;
  float epsilon_;
  int t_; // Timestep

  std::vector<Tensor::Ptr> m_; // First moment
  std::vector<Tensor::Ptr> v_; // Second moment
};
