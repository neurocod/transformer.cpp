#include "layers/Linear.h"
#include "utils/Optimizer.h"
#include "utils/Tensor.h"
#include "utils/LossFunction.h"
#include <vector>
#include <iostream>
#include <sstream>

template<typename T>
std::string vector_to_string(const std::vector<T>& vec) {
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

void print_parents(const Tensor& tensor, int depth = 0) {
    std::string indent(depth * 2, ' ');
    std::cout << indent << "Tensor data: " << vector_to_string(tensor.get_data()) << std::endl;
    
    const auto& parents = tensor.get_parents();
    for (const auto& parent : parents) {
        std::cout << indent << "Parent:" << std::endl;
        print_parents(*parent, depth + 1);
    }
}

class SimpleModel {
public:
    SimpleModel(int input_dim, int output_dim) : linear_layer_(input_dim, output_dim) {}

    Tensor forward(const Tensor& input) {
        return linear_layer_.forward(input);
    }

private:
    Linear linear_layer_;
};

int main() {
    int input_dim = 5;
    int output_dim = 2;

    SimpleModel model(input_dim, output_dim);

    MeanSquaredErrorLoss criterion; 

    float learning_rate = 0.01f;
    SGD optimizer(learning_rate);

    // Training loop
    int num_epochs = 100;

    // Example input and target data
    Tensor input_data({1, input_dim}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor target_data({1, output_dim}, {0.0f, 1.0f});

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        optimizer.zero_grad();

        std::cout << "Epoch: " << epoch << std::endl;

        Tensor predictions = model.forward(input_data);
        print_parents(predictions);

        std::cout << "Computing loss..." << std::endl;

        Tensor loss = criterion.compute_loss(predictions, target_data);

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Loss: " << vector_to_string(loss.get_data()) << std::endl;
        std::cout << "\nTensor computation graph:" << std::endl;

        std::cout << "Backward pass..." << std::endl;

        criterion.backward(loss);

        optimizer.step();
    }

    std::cout << "\nTraining finished." << std::endl;

    return 0;
}
