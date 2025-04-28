#ifndef SGD_OPTIMIZER_H
#define SGD_OPTIMIZER_H

#include "utils/Tensor.h"
#include "utils/Optimizer.h"
#include <vector>
#include <numeric>

class SGD : public Optimizer {
    public:
        // Constructor takes parameters and a learning rate
        SGD(std::vector<Tensor*>& parameters, float learning_rate)
            : Optimizer(parameters), learning_rate_(learning_rate) {}
    
        // Destructor
        ~SGD() override = default;
    
        // Implements the SGD optimization step
        void step() override {
            // Iterate through all parameters
            for (Tensor* param : parameters_) {
                if (param) {
                    // Could make a copy of the param_data and then set it again, but this qould be inefficient because you have an extra copy
                    std::vector<float>& param_data = const_cast<std::vector<float>&>(param->get_data());
                    const std::vector<float>& param_grad = param->get_grad();
    
                    if (param_data.size() != param_grad.size()) {
                        throw std::runtime_error("Parameter data and gradient size mismatch in SGD step.");
                    }
    
                    // param_data = param_data - learning_rate * param_grad
                    for (size_t i = 0; i < param_data.size(); ++i) {
                        param_data[i] -= learning_rate_ * param_grad[i];
                    }
                }
            }
        }
    
    private:
        float learning_rate_;
    };

#endif // SGD_OPTIMIZER_H
