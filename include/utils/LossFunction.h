#ifndef TRANSFORMER_CPP_LOSSFUNCTION_H
#define TRANSFORMER_CPP_LOSSFUNCTION_H

#include "utils/Tensor.h"

class LossFunction
{
public:
    virtual ~LossFunction() = default;

    virtual Tensor compute_loss(const Tensor &predictions, const Tensor &targets) = 0;

    // This assumes compute_loss returns a Tensor that has its computation graph set up.
    void backward(Tensor &loss)
    {
        // The gradient of the loss with respect to itself is 1.0.
        if (loss.get_shape().size() != 1 || loss.get_shape()[0] != 1)
        {
            throw std::runtime_error("Backward pass for loss must start from a scalar loss tensor.");
        }
        Tensor initial_grad_for_loss({1}, {{1.0f}});
        loss.backward(initial_grad_for_loss);
    }
};

class MeanSquaredErrorLoss : public LossFunction
{
public:
    MeanSquaredErrorLoss() = default;

    ~MeanSquaredErrorLoss() override = default;

    // MSE = mean((predictions - targets)^2)
    Tensor compute_loss(const Tensor &predictions, const Tensor &targets) override
    {
        if (predictions.get_shape() != targets.get_shape())
        {
            throw std::runtime_error("Prediction and target shapes mismatch in MeanSquaredErrorLoss.");
        }

        Tensor diff = predictions - targets;
        Tensor squared_diff = diff * diff;
        Tensor sum_squared_diff = squared_diff.sum();
        size_t num_elements = predictions.num_elements();
        if (num_elements == 0)
        {
            return Tensor({1}, {{0.0f}});
        }

        Tensor num_elements_tensor({1}, {{static_cast<float>(num_elements)}});
        Tensor mean_loss = sum_squared_diff / num_elements_tensor;

        return mean_loss;
    }
};

#endif // TRANSFORMER_CPP_LOSSFUNCTION_H
