#include "layers/Linear.h"
#include "layers/Activations.h"
#include "utils/Optimizer.h"
#include "utils/Tensor.h"
#include "utils/LossFunction.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <memory>

template <typename T>
std::string vector_to_string(const std::vector<T> &vec)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        oss << vec[i];
        if (i != vec.size() - 1)
        {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

void print_parents(const std::shared_ptr<Tensor> &tensor, int depth = 0)
{
    if (!tensor)
    {
        return;
    }
    std::string indent(depth * 2, ' ');
    std::cout << indent << "Tensor data: " << vector_to_string(tensor->get_data()) << std::endl;

    const auto &parents = tensor->get_parents();
    for (const auto &parent : parents)
    {
        std::cout << indent << "Parent:" << std::endl;
        print_parents(parent, depth + 1);
    }
}

class SimpleModel
{
public:
    SimpleModel(int input_dim, int output_dim) : linear_layer_(input_dim, output_dim) {}

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> &input)
    {
        std::shared_ptr<Tensor> L1 = linear_layer_.forward(input);
        std::shared_ptr<Tensor> activation = ReLU().forward(L1);
        return activation;
    }

private:
    Linear linear_layer_;
};

int main()
{
    int input_dim = 5;
    int output_dim = 2;

    SimpleModel model(input_dim, output_dim);

    CrossEntropyLoss criterion;

    float learning_rate = 0.01f;
    SGD optimizer(learning_rate);

    // Training loop
    int num_epochs = 100;

    // Example input and target data
    std::shared_ptr<Tensor> input_data = Tensor::create(std::vector<int>{1, input_dim}, std::make_shared<std::vector<float>>(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}));
    std::shared_ptr<Tensor> target_data = Tensor::create(std::vector<int>{1}, std::make_shared<std::vector<float>>(std::vector<float>{1.0f}));

    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        optimizer.zero_grad();

        std::shared_ptr<Tensor> predictions = model.forward(input_data);

        std::shared_ptr<Tensor> loss = criterion.compute_loss(predictions, target_data);

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Loss: " << vector_to_string(loss->get_data()) << std::endl;

        criterion.backward(loss);

        optimizer.step();
    }

    std::cout << "\nTraining finished." << std::endl;

    return 0;
}
