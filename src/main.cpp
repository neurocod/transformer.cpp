#include "models/Transformer.h"
#include "utils/TransformerConfig.h"
#include "utils/DataLoader.h"
#include "utils/Helpers.h"
#include "utils/LossFunction.h"
#include "utils/Optimizer.h"
#include "utils/Tensor.h"

// Function to convert string to tensor of IDs
std::shared_ptr<Tensor> string_to_tensor(const std::string &text,
                                         const DataLoader &loader,
                                         int seq_len) {
  const auto &char_to_id = loader.get_char_to_id_map();
  std::vector<float> ids;
  ids.reserve(seq_len);
  for (char c : text) {
    auto it = char_to_id.find(c);
    if (it != char_to_id.end()) {
      ids.push_back(static_cast<float>(it->second));
    } else {
      ids.push_back(0.0f);
    }
  }
  // Pad
  while (ids.size() < seq_len) {
    ids.push_back(0.0f);
  }
  if (ids.size() > seq_len) {
    ids.resize(seq_len);
  }

  // Create tensor with shape (1, seq_len) for batch size 1
  return Tensor::create({1, seq_len},
                        std::make_shared<std::vector<float>>(ids));
}

void pressEnterToContinue() {
  std::cout << "\nPress Enter twice to continue...";
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Clear the input buffer
  std::cin.get(); // Wait for a character (Enter key)
}

void trainModel(Transformer& model, const TransformerConfig& cf, DataLoader& dataLoader) {
  Adam optimizer(cf.learningRate);
  CrossEntropyLoss criterion;

  std::cout << "Starting Transformer training..." << std::endl;
  using clock = std::chrono::high_resolution_clock;

  auto t_train_start = clock::now();

  for (int epoch = 0; epoch < cf.numEpochs; ++epoch) {
    auto epoch_start = std::chrono::high_resolution_clock::now();

    optimizer.zero_grad();

    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> batch_data = dataLoader.randBatch();
    std::shared_ptr<Tensor> encoder_input_ids = batch_data.first;
    std::shared_ptr<Tensor> target_output_ids = batch_data.second;
    std::shared_ptr<Tensor> decoder_input_ids = encoder_input_ids;

    // Forward pass
    std::shared_ptr<Tensor> logits =
      model.forward(encoder_input_ids, decoder_input_ids, true);

    // Reshape logits
    std::shared_ptr<Tensor> reshaped_logits =
      logits->reshape({ cf.batchSize * cf.decoderSeqLength, dataLoader.get_vocab_size() });

    // Compute loss
    std::shared_ptr<Tensor> loss =
      criterion.computeLoss(reshaped_logits, target_output_ids);

    std::cout << std::format("Epoch [{}/{}], Loss: {}\n", (epoch + 1), cf.numEpochs, vector_to_string(loss->get_data()));

    // Backward pass
    criterion.backward(loss);
    optimizer.step();

    auto epoch_end = std::chrono::high_resolution_clock::now();
    auto epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start).count();
    std::cout << std::format("[LOG] Epoch {} took {} ms.\n", (epoch + 1), epoch_ms);
  }

  auto t_train_end = clock::now();
  auto train_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_train_end - t_train_start).count();
  std::cout << std::format("[LOG] Model training took {} ms.\n", train_ms);

  std::cout << "\nTransformer training finished." << std::endl;

  try {
    std::cout << std::format("Saving final weights to {}\n", cf.weightsFilename);
    model.save_weights(cf.weightsFilename);
    std::cout << "Weights saved successfully." << std::endl;
  }
  catch (const std::exception& e) {
    std::cerr << "Failed to save weights: " << e.what() << std::endl;
  }
}

void inferenceMode(Transformer& model, const TransformerConfig& cf, DataLoader& dataLoader) {
  std::cout << "\n--- Running Inference ---\n";
  std::cout << std::format("Initial prompt: \"{}\"\n", cf.initialPrompt);

  int current_seq_len = cf.inputSeqLength;

  // Tokenize the initial prompt
  std::shared_ptr<Tensor> current_input_ids =
    string_to_tensor(cf.initialPrompt, dataLoader, current_seq_len);
  std::vector<float> generated_ids = current_input_ids->get_data();

  // Remove padding from initial prompt display if any was added by
  // string_to_tensor
  while (!generated_ids.empty() && generated_ids.back() == 0.0f) {
    generated_ids.pop_back();
  }
  // Get actual length of the prompt after removing padding
  int current_total_len = generated_ids.size();

  std::cout << "Generating..." << std::endl;
  for (float id_float : generated_ids) {
    std::cout << dataLoader.get_char_from_id(static_cast<int>(id_float));
  }
  std::cout << std::flush;

  for (int i = 0; i < cf.maxGenerateLength; ++i) {
    std::vector<float> step_input_vec;
    int start_idx = std::max(0, current_total_len - current_seq_len);
    for (int k = start_idx; k < current_total_len; ++k) {
      step_input_vec.push_back(generated_ids[k]);
    }
    // Pad if the current sequence is shorter than model's input length
    while (step_input_vec.size() < current_seq_len) {
      step_input_vec.push_back(cf.padTokenId);
    }

    // Create tensor for this step (batch size 1)
    std::shared_ptr<Tensor> step_input_tensor =
      Tensor::create({ 1, current_seq_len },
        std::make_shared<std::vector<float>>(step_input_vec));

    // Encoder input and Decoder input are the same for autoregressive
    // generation
    std::shared_ptr<Tensor> encoder_input = step_input_tensor;
    std::shared_ptr<Tensor> decoder_input = step_input_tensor;

    // Forward pass (is_training = false)
    std::shared_ptr<Tensor> logits =
      model.forward(encoder_input, decoder_input, false);
    const int target_vocab_size = dataLoader.get_vocab_size();
    // Logits shape: (1, current_seq_len, target_vocab_size)

    int last_token_index = std::min(current_total_len, current_seq_len) - 1;
    if (last_token_index < 0) {
      std::cerr << std::format("\nError: last_token_index is negative ({}). current_total_len={}. Breaking.\n", last_token_index, current_total_len);
      break;
    }

    const auto& logits_data = logits->get_data();
    size_t logits_offset = last_token_index * target_vocab_size;

    // Find the token with the highest logit
    float max_logit = -std::numeric_limits<float>::infinity();
    int predicted_id = -1;
    if (logits_offset + target_vocab_size > logits_data.size()) {
      std::cerr << std::format("\nError: Logits offset ({}) + vocab_size ({}) exceeds logits_data size ({}). last_token_index={}. Breaking.\n",
        logits_offset, target_vocab_size, logits_data.size(), last_token_index);
      break;
    }

    for (int j = 0; j < target_vocab_size; ++j) {
      if (logits_data[logits_offset + j] > max_logit) {
        max_logit = logits_data[logits_offset + j];
        predicted_id = j;
      }
    }

    if (predicted_id == -1) {
      std::cerr << "\nError: No valid predicted_id found (argmax failed?). "
        "Breaking."
        << std::endl;
      break;
    }

    // Append the predicted ID to our sequence
    generated_ids.push_back(static_cast<float>(predicted_id));
    current_total_len++;

    // Convert the predicted ID to a character and print it
    char next_char = dataLoader.get_char_from_id(predicted_id);

    std::cout << next_char << std::flush;
  }

  std::cout << "\n--- Generation Complete ---\n";
}
int mainExcept() {
#ifdef NDEBUG
  std::cout << "[LOG] Build type: RELEASE" << std::endl;
#else
  std::cout << "[LOG] Build type: DEBUG" << std::endl;
#endif
#ifdef _WIN32
  namespace fs = std::filesystem;
  fs::path cwd = fs::current_path();
  fs::path newCwd = cwd.parent_path().parent_path();
  fs::current_path(newCwd);
#endif
  using clock = std::chrono::high_resolution_clock;

  auto t_init_start = clock::now();

  TransformerConfig::init("../config.ini");
  const TransformerConfig& cf = TransformerConfig::instance();

  DataLoader dataLoader(cf.inputSeqLength, cf.batchSize);
  dataLoader.readFile(cf.dataFilename);

  const int input_vocab_size = dataLoader.get_vocab_size();
  const int target_vocab_size = dataLoader.get_vocab_size();

  Transformer model(input_vocab_size, target_vocab_size, cf.embedDim,
    cf.maxSequenceLength, cf.numLayers, cf.numHeads, cf.ffHiddenDim,
    cf.dropoutRate, cf.padTokenId);

  auto t_init_end = clock::now();
  auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_init_end - t_init_start).count();
  std::cout << std::format("[LOG] Model initialization took {} ms.\n", init_ms);

  // Print model parameters count
  size_t total_params = 0;
  const auto &parameters = Tensor::get_optimizable_tensors();
  for (const auto &param : parameters) {
    if (param) {
      total_params += param->num_elements();
    }
  }
  std::cout << std::format("Total optimizable parameters: {}\n", total_params);

  if (cf.inferenceMode) {
    if (!std::filesystem::exists(cf.weightsFilename)) {
      std::cerr << std::format("Weights file '{}' not found. Cannot run inference. Exiting.\n", cf.weightsFilename);
      return 1;
    }
    try {
      std::cout << std::format("Attempting to load weights from {}...\n", cf.weightsFilename);
      model.load_weights(cf.weightsFilename);
      std::cout << "Weights loaded successfully." << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "Failed to load weights: " << e.what() << std::endl;
      if (cf.inferenceMode) {
        std::cerr << "Cannot run inference without weights. Exiting."
          << std::endl;
        return 1;
      }
    }
    inferenceMode(model, cf, dataLoader);
  } else {
    trainModel(model, cf, dataLoader);
  }

  return 0;
}

int main() {
  try {
    return mainExcept();
  }
  catch(const std::exception& ex) {
    std::cerr << "\n\n===\nUncaught exception: " << ex.what();
    return 1;
  }
}