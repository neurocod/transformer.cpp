#include "layers/Linear.h"
#include "layers/Activations.h"
#include "layers/LayerNorm.h"
#include "utils/Optimizer.h"
#include "utils/Tensor.h"
#include "utils/LossFunction.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <memory>
#include <random>

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

class SimpleModel
{
public:
    SimpleModel(int input_dim, int hidden_dim, int output_dim)
        : linear1_(input_dim, hidden_dim),
          layernorm_(hidden_dim),
          linear2_(hidden_dim, output_dim),
          relu_() {}

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> &input)
    {
        // Input -> Linear1 -> LayerNorm -> ReLU -> Linear2 -> Output
        std::shared_ptr<Tensor> L1_output = linear1_.forward(input);
        std::shared_ptr<Tensor> layernorm_output = layernorm_.forward(L1_output);
        std::shared_ptr<Tensor> activation_output = relu_.forward(layernorm_output);
        std::shared_ptr<Tensor> L2_output = linear2_.forward(activation_output);

        return L2_output;
    }

private:
    Linear linear1_;
    LayerNorm layernorm_;
    Linear linear2_;
    ReLU relu_;
};

int main()
{
    // Model dimensions
    int input_dim = 10;
    int hidden_dim = 20;
    int output_dim = 3;

    SimpleModel model(input_dim, hidden_dim, output_dim);

    CrossEntropyLoss criterion;

    float learning_rate = 0.001f;
    Adam optimizer(learning_rate);

    // Training loop parameters
    int num_epochs = 500;
    int batch_size = 4;

    // Generating random input data and corresponding random target class indices
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> input_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> target_dist(0, output_dim - 1);

    std::vector<float> input_data_vec(batch_size * input_dim);
    std::vector<float> target_data_vec(batch_size);

    for (size_t i = 0; i < batch_size * input_dim; ++i)
    {
        input_data_vec[i] = input_dist(gen);
    }
    for (size_t i = 0; i < batch_size; ++i)
    {
        target_data_vec[i] = static_cast<float>(target_dist(gen));
    }

    std::shared_ptr<Tensor> input_data = Tensor::create({batch_size, input_dim}, std::make_shared<std::vector<float>>(input_data_vec));
    std::shared_ptr<Tensor> target_data = Tensor::create({batch_size}, std::make_shared<std::vector<float>>(target_data_vec));

    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        optimizer.zero_grad();

        std::shared_ptr<Tensor> predictions = model.forward(input_data);

        std::shared_ptr<Tensor> loss = criterion.compute_loss(predictions, target_data);

        if ((epoch + 1) % 10 == 0 || epoch == 0)
        {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Loss: " << vector_to_string(loss->get_data()) << std::endl;
        }

        criterion.backward(loss);

        optimizer.step();
    }

    std::cout << "\nTraining finished." << std::endl;

    return 0;
}