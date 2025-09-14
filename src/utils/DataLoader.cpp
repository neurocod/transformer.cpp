#include "utils/DataLoader.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <unordered_set>
#include <random>

static std::mt19937 g_randBatch(
  std::chrono::high_resolution_clock::now().time_since_epoch().count());

DataLoader::DataLoader(const std::string &filename, int sequence_length,
                       int batch_size)
    : filename_(filename), sequence_length_(sequence_length),
      batch_size_(batch_size), num_batches_(0) {}

void DataLoader::load_data() {
  std::cout << "Loading data from: " << filename_ << std::endl;
  std::ifstream file(filename_, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Could not open data file: " + filename_);
  }

  std::string text;
  text.resize(static_cast<std::size_t>(file.tellg()));
  file.seekg(0);
  file.read(text.data(), text.size());

  // Build Vocabulary
  std::unordered_set<char> unique_chars(text.begin(), text.end());
  chars_.assign(unique_chars.begin(), unique_chars.end());

  for (size_t i = 0; i < chars_.size(); ++i) {
    char_to_id_[chars_[i]] = i;
    id_to_char_[i] = chars_[i];
  }
  std::cout << "char_to_id: " << std::endl;
  for (auto &pair : char_to_id_) {
    std::cout << pair.first << " -> " << pair.second << std::endl;
  }
  std::cout << "id_to_char: " << std::endl;
  for (auto &pair : id_to_char_) {
    std::cout << pair.first << " -> " << pair.second << std::endl;
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
    throw std::runtime_error("Dataset too small to form even one batch with "
                             "current sequence length and batch size.");
  }

  std::cout << "Number of full batches available per epoch: " << num_batches_
            << std::endl;
}

int DataLoader::get_vocab_size() const { return chars_.size(); }

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> DataLoader::randBatch() {
  if (data_.empty()) {
    throw std::runtime_error("Data not loaded. Call load_data() first.");
  }

  // Pick a random starting batch index
  std::uniform_int_distribution<size_t> randBatch(0, num_batches_ - 1);
  size_t batch_index = randBatch(g_randBatch);

  size_t start_index = batch_index * batch_size_ * sequence_length_;

  std::vector<float> input_batch_vec(batch_size_ * sequence_length_);
  std::vector<float> target_batch_vec(batch_size_ * sequence_length_);

  // Extract data for the batch
  for (int nBatch = 0; nBatch < batch_size_; ++nBatch) {
    size_t sequence_start = nBatch * sequence_length_ + start_index;
    for (int i = 0; i < sequence_length_; ++i) {
      const int pos = nBatch * sequence_length_ + i;
      // Input is the current character
      input_batch_vec[pos] = static_cast<float>(data_[sequence_start + i]);

      // Target is the next character (shifted by one)
      // Handle the case where the target is beyond the end of the data_ vector
      if (sequence_start + i + 1 < data_.size()) {
        target_batch_vec[pos] = static_cast<float>(data_[sequence_start + i + 1]);
      } else {
        target_batch_vec[pos] = static_cast<float>(data_[sequence_start + i]);
      }
    }
  }

  std::shared_ptr<Tensor> input_tensor =
      Tensor::create({batch_size_, sequence_length_},
                     std::make_shared<std::vector<float>>(input_batch_vec));
  std::shared_ptr<Tensor> target_tensor =
      Tensor::create({batch_size_ * sequence_length_},
                     std::make_shared<std::vector<float>>(target_batch_vec));

  return {input_tensor, target_tensor};
}

char DataLoader::get_char_from_id(int id) const {
  auto it = id_to_char_.find(id);
  if (it == id_to_char_.end()) {
    // Return a default character or throw an error for unknown ID
    it = id_to_char_.find(0); // Assuming 0 might be padding or unknown
    if (it != id_to_char_.end())
      return it->second;
    return '?';
  }
  return it->second;
}

int DataLoader::get_id_from_char(char ch) const {
  auto it = char_to_id_.find(ch);
  if (it == char_to_id_.end()) {
    it = char_to_id_.find('\0');
    if (it != char_to_id_.end())
      return it->second;
    if (char_to_id_.count(0))
      return char_to_id_.at(0);
    return -1;
  }
  return it->second;
}

const std::unordered_map<char, int> &DataLoader::get_char_to_id_map() const {
  return char_to_id_;
}

const std::unordered_map<int, char> &DataLoader::get_id_to_char_map() const {
  return id_to_char_;
}