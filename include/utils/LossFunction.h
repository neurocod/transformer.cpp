#ifndef TRANSFORMER_CPP_LOSSFUNCTION_H
#define TRANSFORMER_CPP_LOSSFUNCTION_H

#include "utils/Tensor.h"
#include <memory>

class Tensor;

class LossFunction
{
public:
    virtual ~LossFunction() = default;

    virtual std::shared_ptr<Tensor> compute_loss(std::shared_ptr<Tensor> &predictions, std::shared_ptr<Tensor> &targets) = 0;

    // This assumes compute_loss returns a shared_ptr to a Tensor that has its computation graph set up.
    void backward(std::shared_ptr<Tensor> &loss)
    {
        // The gradient of the loss with respect to itself is 1.0.
        if (loss->get_shape().size() != 1 || loss->get_shape()[0] != 1)
        {
            throw std::runtime_error("Backward pass for loss must start from a scalar loss tensor.");
        }
        std::shared_ptr<Tensor> initial_grad_for_loss = Tensor::create({1}, std::make_shared<std::vector<float>>(std::vector<float>{{1.0f}}));
        loss->backward(initial_grad_for_loss);
    }
};

class MeanSquaredErrorLoss : public LossFunction
{
public:
    MeanSquaredErrorLoss() = default;

    ~MeanSquaredErrorLoss() override = default;

    // MSE = mean((predictions - targets)^2)
    std::shared_ptr<Tensor> compute_loss(std::shared_ptr<Tensor> &predictions, std::shared_ptr<Tensor> &targets) override
    {
        if (predictions->get_shape() != targets->get_shape())
        {
            throw std::runtime_error("Prediction and target shapes mismatch in MeanSquaredErrorLoss.");
        }

        std::shared_ptr<Tensor> diff = *predictions - targets;
        std::shared_ptr<Tensor> squared_diff = *diff * diff;
        std::shared_ptr<Tensor> sum_squared_diff = squared_diff->sum();

        size_t num_elements = predictions->num_elements();
        if (num_elements == 0)
        {
            return Tensor::create(std::vector<int>{1}, std::make_shared<std::vector<float>>(std::vector<float>{0.0f}));
        }

        std::shared_ptr<Tensor> num_elements_tensor = Tensor::create({1}, std::make_shared<std::vector<float>>(std::vector<float>{{static_cast<float>(num_elements)}}));
        std::shared_ptr<Tensor> mean_loss = (*sum_squared_diff) / num_elements_tensor;

        return mean_loss;
    }
};

#endif // TRANSFORMER_CPP_LOSSFUNCTION_H
