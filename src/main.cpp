#include "models/Transformer.h"
#include "utils/TransformerConfig.h"
#include "utils/DataLoader.h"
#include "utils/Helpers.h"
#include "utils/LossFunction.h"
#include "utils/Optimizer.h"
#include "utils/Tensor.h"

using Vec = Tensor::Vec;
// Function to convert string to tensor of IDs
Tensor::Ptr string_to_tensor(const std::string &text,
                                         const DataLoader &loader,
                                         int seq_len) {
  const auto &char_to_id = loader.charToIdMap();
  Vec ids;
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
                        std::make_shared<Vec>(ids));
}

void trainModel(const TransformerConfig& cf, DataLoader& dataLoader) {
  using clock = std::chrono::high_resolution_clock;
  auto t_init_start = clock::now();

  Transformer model(cf);

  auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t_init_start).count();
  spdlog::info("Model initialization took {} ms", init_ms);

  // Print model parameters count
  size_t total_params = 0;
  const auto &parameters = Tensor::get_optimizable_tensors();
  for (const auto &param : parameters) {
    if (param) {
      total_params += param->num_elements();
    }
  }
  spdlog::info("Total optimizable parameters: {}", total_params);


  Adam optimizer(cf.learningRate);
  CrossEntropyLoss criterion;

  spdlog::info("Starting Transformer training...");

  auto t_train_start = clock::now();

  for (int epoch = 0; epoch < cf.numEpochs; ++epoch) {
    auto epoch_start = std::chrono::high_resolution_clock::now();

    optimizer.zeroGrad();

    std::pair<Tensor::Ptr, Tensor::Ptr> batch_data = dataLoader.randBatch();
    Tensor::Ptr encoderInputIds = batch_data.first;
    Tensor::Ptr target_output_ids = batch_data.second;
    Tensor::Ptr decoderInputIds = encoderInputIds;

    // Forward pass
    Tensor::Ptr logits = model.forward(encoderInputIds, decoderInputIds, true);

    // Reshape logits
    Tensor::Ptr reshaped_logits = logits->reshape({ cf.batchSize * cf.decoderSeqLength, dataLoader.vocabSize() });

    // Compute loss
    Tensor::Ptr loss = criterion.computeLoss(reshaped_logits, target_output_ids);

    spdlog::info("Epoch [{}/{}], Loss: {}", (epoch + 1), cf.numEpochs, vector_to_string(loss->data()));

    // Backward pass
    criterion.backward(loss);
    optimizer.step();

    auto epoch_end = std::chrono::high_resolution_clock::now();
    auto epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start).count();
    spdlog::info("Epoch {} took {} ms", (epoch + 1), epoch_ms);
  }

  auto t_train_end = clock::now();
  auto train_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_train_end - t_train_start).count();
  spdlog::info("Model training took {} ms", train_ms);
  spdlog::info("Transformer training finished.");

  spdlog::info("Saving final weights to {}", cf.weightsFilename);
  if (model.saveToFile(cf.weightsFilename))
    spdlog::info("Weights saved successfully.");
}

int inferenceMode(const TransformerConfig& cf, DataLoader& dataLoader) {
  if (!std::filesystem::exists(cf.weightsFilename)) {
    spdlog::error("Weights file '{}' not found. Cannot run inference. Exiting.", cf.weightsFilename);
    return 1;
  }
  std::shared_ptr<Transformer> model; 
  try {
    spdlog::info("Attempting to load weights from {}...", cf.weightsFilename);
    model = Transformer::loadFromFile(cf.weightsFilename, dataLoader.vocabSize(), dataLoader.vocabSize());
  } catch (const std::exception &e) {
    spdlog::error("Failed to load weights: {}", e.what());
    model = nullptr;
  }
  
  if (!model) {
    spdlog::error("Cannot run inference without weights. Exiting.");
    return 1;
  }

  spdlog::info("Weights loaded successfully.\n--- Running Inference ---");
  spdlog::info("Initial prompt: \"{}\"", cf.initialPrompt);

  int current_seq_len = cf.inputSeqLength;

  // Tokenize the initial prompt
  Tensor::Ptr current_input_ids = string_to_tensor(cf.initialPrompt, dataLoader, current_seq_len);
  Vec generated_ids = current_input_ids->data();

  // Remove padding from initial prompt display if any was added by
  // string_to_tensor
  while (!generated_ids.empty() && generated_ids.back() == 0.0f) {
    generated_ids.pop_back();
  }
  // Get actual length of the prompt after removing padding
  int current_total_len = generated_ids.size();

  spdlog::info("Generating...");
  for (float id_float : generated_ids) {
    std::cout << dataLoader.charFromId(static_cast<int>(id_float));
  }

  for (int i = 0; i < cf.maxGenerateLength; ++i) {
    Vec step_input_vec;
    int start_idx = std::max(0, current_total_len - current_seq_len);
    for (int k = start_idx; k < current_total_len; ++k) {
      step_input_vec.push_back(generated_ids[k]);
    }
    // Pad if the current sequence is shorter than model's input length
    while (step_input_vec.size() < current_seq_len) {
      step_input_vec.push_back(cf.padTokenId);
    }

    // Create tensor for this step (batch size 1)
    Tensor::Ptr step_input_tensor = Tensor::create({ 1, current_seq_len },
        std::make_shared<Vec>(step_input_vec));

    // Encoder input and Decoder input are the same for autoregressive
    // generation
    Tensor::Ptr encoder_input = step_input_tensor;
    Tensor::Ptr decoder_input = step_input_tensor;

    // Forward pass (isTraining = false)
    Tensor::Ptr logits = model->forward(encoder_input, decoder_input, false);
    const int targetVocabSize = dataLoader.vocabSize();
    // Logits shape: (1, current_seq_len, targetVocabSize)

    int last_token_index = std::min(current_total_len, current_seq_len) - 1;
    if (last_token_index < 0) {
      spdlog::error("\nError: last_token_index is negative ({}). current_total_len={}. Breaking.", last_token_index, current_total_len);
      break;
    }

    const auto& logits_data = logits->data();
    size_t logits_offset = last_token_index * targetVocabSize;

    // Find the token with the highest logit
    float max_logit = -std::numeric_limits<float>::infinity();
    int predicted_id = -1;
    if (logits_offset + targetVocabSize > logits_data.size()) {
      spdlog::error("\nError: Logits offset ({}) + vocabSize ({}) exceeds logits_data size ({}). last_token_index={}. Breaking.",
        logits_offset, targetVocabSize, logits_data.size(), last_token_index);
      break;
    }

    for (int j = 0; j < targetVocabSize; ++j) {
      if (logits_data[logits_offset + j] > max_logit) {
        max_logit = logits_data[logits_offset + j];
        predicted_id = j;
      }
    }

    if (predicted_id == -1) {
      spdlog::error("\nError: No valid predicted_id found (argmax failed?). Breaking.");
      break;
    }

    // Append the predicted ID to our sequence
    generated_ids.push_back(static_cast<float>(predicted_id));
    current_total_len++;

    // Convert the predicted ID to a character and print it
    char next_char = dataLoader.charFromId(predicted_id);

    std::cout << next_char << std::flush;
  }
  std::cout << "\n";
  spdlog::info("--- Generation Complete ---");
  return 0;
}

int mainExcept() {
#ifdef NDEBUG
  spdlog::info("Build type: RELEASE");
#else
  spdlog::info("Build type: DEBUG");
#endif
#ifdef _WIN32
  namespace fs = std::filesystem;
  fs::path cwd = fs::current_path();
  fs::path newCwd = cwd.parent_path().parent_path();
  fs::current_path(newCwd);
#endif
  TransformerConfig cf;
  cf.init("../config.ini");

  DataLoader dataLoader(cf.inputSeqLength, cf.batchSize);
  dataLoader.readFile(cf.dataFilename);

  cf.inputVocabSize = dataLoader.vocabSize();
  cf.targetVocabSize = dataLoader.vocabSize();
  TransformerConfig::instance() = cf;

  if (cf.inferenceMode)
    return inferenceMode(cf, dataLoader);

  trainModel(cf, dataLoader);
  return 0;
}

int main() {
  try {
    return mainExcept();
  } catch(const std::exception& ex) {
    spdlog::error("\n===\nUncaught exception: {}", ex.what());
    return 1;
  }
}