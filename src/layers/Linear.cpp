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
  std::shared_ptr<Vec> weights_data =
      std::make_shared<Vec>(_inputDim * _outputDim);
  for (size_t i = 0; i < weights_data->size(); ++i) {
    (*weights_data)[i] = d(gen);
  }
  _weights->set_data(weights_data);

  std::shared_ptr<Vec> biases_data =
      std::make_shared<Vec>(_outputDim);
  std::fill(biases_data->begin(), biases_data->end(), 0.0f);
  _biases->set_data(biases_data);
}

Tensor::Ptr Linear::forward(Tensor::Ptr &input) {
  // Input shape is (..., inputDim)
  // Weights shape is (inputDim, outputDim)
  // output = input * weights + biases

  if (input->shape().empty() || input->shape().back() != _inputDim) {
    throw std::runtime_error("Input tensor's last dimension is incompatible "
                             "with Linear layer input dimension.");
  }

  Tensor::Ptr product = input->dot(_weights);
  Tensor::Ptr output = *product + _biases;

  return output;
}

Tensor::Ptr Linear::weights() { return _weights; }

Tensor::Ptr Linear::biases() { return _biases; }
