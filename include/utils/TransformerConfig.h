#pragma once
class ConfigParser;

class TransformerConfig {
public:
  static const TransformerConfig& instance();
  static void init(const std::string& filename);

  std::string modelFileNameByParameters() const;

  bool inference_mode;
  bool load_existing_weights;
  std::string weights_filename;
  std::string data_filename;

  // Model Architecture
  int embed_dim;
  int max_sequence_length;
  int num_layers;
  int num_heads;
  int ff_hidden_dim;
  float dropout_rate;
  float pad_token_id;

  // Training Parameters
  float learning_rate;
  int num_epochs;
  int batch_size;
  int input_seq_length;
  int decoder_seq_length;

  // Inference Parameters
  int max_generate_length;
  std::string initial_prompt;
private:
  static TransformerConfig& mutableInstance();
  void init(const ConfigParser& config);
};