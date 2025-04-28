#ifndef TRANSFORMER_CPP_OPTIMIZER_H
#define TRANSFORMER_CPP_OPTIMIZER_H

#include "utils/Tensor.h"
#include <vector>
#include <numeric>

class Tensor;

// Base class for all optimizers
class Optimizer
{
public:
    // Constructor
    Optimizer() {}

    virtual ~Optimizer() = default;

    // Derived classes must implement this to update parameters based on gradients
    virtual void step() = 0;

    // Zeros the gradients of all parameters managed by this optimizer
    void zero_grad()
    {
        for (Tensor *param : parameters_)
        {
            if (param)
            {
                param->zero_grad();
            }
        }
    }

protected:
    // List of pointers to the parameters to update
    std::vector<Tensor *> &parameters_ = Tensor::get_optimizable_tensors();
};

class SGD : public Optimizer
{
public:
    // Constructor takes a learning rate
    SGD(float learning_rate)
        : learning_rate_(learning_rate) {}

    // Destructor
    ~SGD() override = default;

    // Implements the SGD optimization step
    void step() override
    {
        // Iterate through all parameters
        for (Tensor *param : parameters_)
        {
            if (param)
            {
                // Could make a copy of the param_data and then set it again, but this qould be inefficient because you have an extra copy
                std::vector<float> &param_data = const_cast<std::vector<float> &>(param->get_data());
                const std::vector<float> &param_grad = param->get_grad();

                if (param_data.size() != param_grad.size())
                {
                    throw std::runtime_error("Parameter data and gradient size mismatch in SGD step.");
                }

                // param_data = param_data - learning_rate * param_grad
                for (size_t i = 0; i < param_data.size(); ++i)
                {
                    param_data[i] -= learning_rate_ * param_grad[i];
                }
            }
        }
    }

private:
    float learning_rate_;
};

#endif // TRANSFORMER_CPP_OPTIMIZER_H
