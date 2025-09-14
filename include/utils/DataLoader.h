#pragma once
#include "Tensor.h"

class DataLoader {
public:
  DataLoader(int sequence_length, int batch_size);

  void loadData(const std::string& filename);
  std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> randBatch();

  char get_char_from_id(int id) const;
  int get_id_from_char(char character) const;

  int get_vocab_size() const;
  const std::unordered_map<char, int> &get_char_to_id_map() const;
  const std::unordered_map<int, char> &get_id_to_char_map() const;
private:
  int sequence_length_;
  int batch_size_;

  std::vector<char> chars_;
  std::unordered_map<char, int> char_to_id_;
  std::unordered_map<int, char> id_to_char_;
  std::vector<int> data_;

  size_t num_batches_ = 0;
};
