#pragma once
#include "Tensor.h"
#include <string>
#include <unordered_map>

class DataLoader {
public:
  DataLoader(const std::string &filename, int sequence_length, int batch_size);

  // Load and preprocess the data
  void load_data();

  // Get vocabulary size
  int get_vocab_size() const;

  // Get a random batch of data
  std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> randBatch();

  // Get character from ID
  char get_char_from_id(int id) const;

  // Get ID from character
  int get_id_from_char(char character) const;

  // Getter for the char_to_id map
  const std::unordered_map<char, int> &get_char_to_id_map() const;

  // Getter for the id_to_char map
  const std::unordered_map<int, char> &get_id_to_char_map() const;

private:
  std::string filename_;
  int sequence_length_;
  int batch_size_;

  std::vector<char> chars_;
  std::unordered_map<char, int> char_to_id_;
  std::unordered_map<int, char> id_to_char_;
  std::vector<int> data_;

  size_t num_batches_;
};
