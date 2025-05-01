#include "layers/Linear.h"
#include "layers/Activations.h"
#include "layers/LayerNorm.h"
#include "layers/Embedding.h"
#include "layers/PositionalEncoding.h"
#include "models/Encoder.h"
#include "models/Decoder.h"
#include "models/Transformer.h"
#include "utils/Optimizer.h"
#include "utils/Tensor.h"
#include "utils/LossFunction.h"
#include "utils/Masking.h"
#include "utils/Helpers.h"
#include "utils/DataLoader.h"

#include <vector>
#include <iostream>
#include <sstream>
#include <memory>
#include <random>
#include <chrono>
#include <filesystem>

std::mt19937 global_rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());

int main()
{
    int embed_dim = 128;
    int max_sequence_length = 100;
    int num_layers = 2;
    int num_heads = 8;
    int ff_hidden_dim = 512;
    float dropout_rate = 0.1f;
    float pad_token_id = 0.0f;

    // Training Parameters
    float learning_rate = 0.001f;
    int num_epochs = 10;
    int batch_size = 8;
    int input_seq_length = 10;
    int decoder_seq_length = 10;

    std::string data_filename = "../data/tiny_shakespeare.txt";
    std::string weights_filename = "transformer_weights.bin";
    bool load_existing_weights = true;

    DataLoader data_loader(data_filename, input_seq_length, batch_size);
    data_loader.load_data();

    int input_vocab_size = data_loader.get_vocab_size();
    int target_vocab_size = data_loader.get_vocab_size();

    Transformer model(input_vocab_size, target_vocab_size, embed_dim,
                      max_sequence_length, num_layers, num_heads,
                      ff_hidden_dim, dropout_rate, pad_token_id);

    // Load weights if they exist
    if (load_existing_weights && std::filesystem::exists(weights_filename)) {
        try {
            std::cout << "Attempting to load weights from " << weights_filename << "..." << std::endl;
            model.load_weights(weights_filename);
            std::cout << "Weights loaded successfully." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load weights: " << e.what() << std::endl;
            std::cerr << "Starting training from scratch." << std::endl;
        }
    } else {
         std::cout << "No existing weights file found or loading disabled. Starting training from scratch." << std::endl;
    }

    Adam optimizer(learning_rate);

    CrossEntropyLoss criterion;

    // Synthetic Data Generation
    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> input_dist(1, input_vocab_size - 1);
    std::uniform_int_distribution<int> target_dist(1, target_vocab_size - 1);

    std::cout << "Starting Transformer training test run..." << std::endl;

    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        optimizer.zero_grad();

        std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> batch_data = data_loader.get_batch();
        std::shared_ptr<Tensor> encoder_input_ids = batch_data.first;
        std::shared_ptr<Tensor> target_output_ids = batch_data.second;
        std::shared_ptr<Tensor> decoder_input_ids = encoder_input_ids;

        // Forward pass (is_training = true)
        std::shared_ptr<Tensor> logits = model.forward(encoder_input_ids, decoder_input_ids, true);

        // Reshape logits to (batch_size * decoder_seq_length, target_vocab_size) for CrossEntropyLoss
        std::shared_ptr<Tensor> reshaped_logits = logits->reshape({batch_size * decoder_seq_length, target_vocab_size});

        // Compute loss. CrossEntropyLoss expects predictions of shape (N, C) and targets of shape (N).
        std::shared_ptr<Tensor> loss = criterion.compute_loss(reshaped_logits, target_output_ids);

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Loss: " << vector_to_string(loss->get_data()) << std::endl;

        // Backward pass
        criterion.backward(loss);

        optimizer.step();
    }

    std::cout << "\nTransformer training test run finished." << std::endl;

    // Save weights after training
    try {
        std::cout << "Saving final weights to " << weights_filename << "..." << std::endl;
        model.save_weights(weights_filename);
        std::cout << "Weights saved successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to save weights: " << e.what() << std::endl;
    }

    return 0;
}