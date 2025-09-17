#include "utils/TransformerConfig.h"
#include "utils/ConfigParser.h"

TransformerConfig& TransformerConfig::mutableInstance() {
  static TransformerConfig instance;
  return instance;
}
const TransformerConfig& TransformerConfig::instance() {
  return mutableInstance();
}

void TransformerConfig::init(const std::string& filename) {
  ConfigParser& config = ConfigParser::instance(filename);
  mutableInstance().init(config);
}

void TransformerConfig::init(const ConfigParser& config) {
  num_threads = config.value<int>("num_threads");
  inference_mode = config.value<bool>("inference_mode");
  weights_filename = config.value<std::string>("weights_filename");
  data_filename = config.value<std::string>("data_filename");

  // Model Architecture
  embed_dim = config.value<int>("embed_dim");
  max_sequence_length = config.value<int>("max_sequence_length");
  num_layers = config.value<int>("num_layers");
  num_heads = config.value<int>("num_heads");
  ff_hidden_dim = config.value<int>("ff_hidden_dim");
  dropout_rate = config.value<float>("dropout_rate");
  pad_token_id = config.value<float>("pad_token_id");

  // Training Parameters
  learning_rate = config.value<float>("learning_rate");
  num_epochs = config.value<int>("num_epochs");
  batch_size = config.value<int>("batch_size");
  input_seq_length = config.value<int>("input_seq_length");
  decoder_seq_length = config.value<int>("decoder_seq_length");

  // Inference Parameters
  max_generate_length = config.value<int>("max_generate_length");
  initial_prompt = config.value<std::string>("initial_prompt");

  if (weights_filename == "auto") {
    weights_filename = std::format("weigth-{}-{}-{}.bin", num_layers, ff_hidden_dim, num_heads);
  }
}

std::string TransformerConfig::modelFileNameByParameters() const {
  return std::format("{}", 0);
}