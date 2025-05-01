#include "utils/DataLoader.h"
#include <iostream>
#include <set>
#include <chrono>

DataLoader::DataLoader(const std::string& filename, int sequence_length, int batch_size)
    : filename_(filename), sequence_length_(sequence_length), batch_size_(batch_size), num_batches_(0) {}

void DataLoader::load_data() {
    std::cout << "Loading data from: " << filename_ << std::endl;
    std::ifstream file(filename_);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open data file: " + filename_);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();
    file.close();

    // Build Vocabulary
    std::set<char> unique_chars(text.begin(), text.end());
    chars_.assign(unique_chars.begin(), unique_chars.end());
    std::sort(chars_.begin(), chars_.end());

    for (size_t i = 0; i < chars_.size(); ++i) {
        char_to_id_[chars_[i]] = i;
        id_to_char_[i] = chars_[i];
    }

    std::cout << "Vocabulary size: " << chars_.size() << std::endl;

    // Numericalize the data
    data_.reserve(text.length());
    for (char c : text) {
        data_.push_back(char_to_id_[c]);
    }

    std::cout << "Dataset size (characters): " << data_.size() << std::endl;

    // Calculate number of possible batches
    size_t total_sequence_batches = data_.size() / (sequence_length_);
    num_batches_ = total_sequence_batches / batch_size_;

     if (num_batches_ == 0) {
        throw std::runtime_error("Dataset too small to form even one batch with current sequence length and batch size.");
    }

    std::cout << "Number of full batches available per epoch: " << num_batches_ << std::endl;
}

int DataLoader::get_vocab_size() const {
    return chars_.size();
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> DataLoader::get_batch() {
    if (data_.empty()) {
         throw std::runtime_error("Data not loaded. Call load_data() first.");
    }

    // Pick a random starting batch index
    std::uniform_int_distribution<size_t> distribution(0, num_batches_ - 1);
    size_t batch_index = distribution(global_rng);

    size_t start_index = batch_index * batch_size_ * sequence_length_;

    std::vector<float> input_batch_vec(batch_size_ * sequence_length_);
    std::vector<float> target_batch_vec(batch_size_ * sequence_length_);

    // Extract data for the batch
    for (int i = 0; i < batch_size_; ++i) {
        size_t sequence_start = start_index + i * sequence_length_;
        for (int j = 0; j < sequence_length_; ++j) {
            // Input is the current character
            input_batch_vec[i * sequence_length_ + j] = static_cast<float>(data_[sequence_start + j]);

            // Target is the next character (shifted by one)
            // Handle the case where the target is beyond the end of the data_ vector
            if (sequence_start + j + 1 < data_.size()) {
                 target_batch_vec[i * sequence_length_ + j] = static_cast<float>(data_[sequence_start + j + 1]);
            } else {
                 target_batch_vec[i * sequence_length_ + j] = static_cast<float>(data_[sequence_start + j]);
            }
        }
    }

    std::shared_ptr<Tensor> input_tensor = Tensor::create({batch_size_, sequence_length_}, std::make_shared<std::vector<float>>(input_batch_vec));
    std::shared_ptr<Tensor> target_tensor = Tensor::create({batch_size_ * sequence_length_}, std::make_shared<std::vector<float>>(target_batch_vec));


    return {input_tensor, target_tensor};
}

char DataLoader::get_char_from_id(int id) const {
    if (id < 0 || id >= chars_.size()) {
        throw std::out_of_range("ID out of vocabulary range.");
    }
    return id_to_char_.at(id);
}

int DataLoader::get_id_from_char(char character) const {
     if (char_to_id_.find(character) == char_to_id_.end()) {
        throw std::runtime_error("Character not found in vocabulary.");
    }
    return char_to_id_.at(character);
}