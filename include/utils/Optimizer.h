#ifndef TRANSFORMER_CPP_OPTIMIZER_H
#define TRANSFORMER_CPP_OPTIMIZER_H

#include "utils/Tensor.h"
#include <vector>
#include <numeric>

class Tensor;

// Base class for all optimizers
class Optimizer {
public:
    // Constructor
    Optimizer() {}

    virtual ~Optimizer() = default;

    // Derived classes must implement this to update parameters based on gradients
    virtual void step() = 0;

    // Zeros the gradients of all parameters managed by this optimizer
    void zero_grad() {
        for (auto param : parameters_) {
            if (param) {
                param->zero_grad();
            }
        }
    }

protected:
    // List of pointers to the parameters to update
    std::vector<Tensor*>& parameters_ = Tensor::get_optimizable_tensors();
};

#endif // TRANSFORMER_CPP_OPTIMIZER_H
