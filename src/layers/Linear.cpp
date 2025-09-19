#include "layers/Linear.h"

Linear::Linear(int inputDim, int outputDim, const std::string& name)
    : _inputDim(inputDim), _outputDim(outputDim) {
  // Weights shape: (inputDim, outputDim)
  _weights = Tensor::create(std::vector<int>{inputDim, outputDim}, name + ".Linear::k");
  // Biases shape: (outputDim)
  _biases = Tensor::create(std::vector<int>{_outputDim}, name + ".Linear::b");
  if (_outputDim == 32)
    int t = 3;

  // Initialize weights and biases with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  float std_dev = std::sqrt(2.0f / _inputDim);
  std::normal_distribution<float> d(0, std_dev);

  // Initialize weights data
  std::shared_ptr<std::vector<float>> weights_data =
      std::make_shared<std::vector<float>>(_inputDim * _outputDim);
  for (size_t i = 0; i < weights_data->size(); ++i) {
    (*weights_data)[i] = d(gen);
  }
  _weights->set_data(weights_data);

  std::shared_ptr<std::vector<float>> biases_data =
      std::make_shared<std::vector<float>>(_outputDim);
  std::fill(biases_data->begin(), biases_data->end(), 0.0f);
  _biases->set_data(biases_data);
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> &input) {
  // Input shape is (..., inputDim)
  // Weights shape is (inputDim, outputDim)
  // output = input * weights + biases

  if (input->shape().empty() || input->shape().back() != _inputDim) {
    throw std::runtime_error("Input tensor's last dimension is incompatible "
                             "with Linear layer input dimension.");
  }

  std::shared_ptr<Tensor> product = input->dot(_weights);
  std::shared_ptr<Tensor> output = *product + _biases;

  return output;
}

std::shared_ptr<Tensor> Linear::weights() { return _weights; }

std::shared_ptr<Tensor> Linear::biases() { return _biases; }
