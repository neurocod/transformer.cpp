#pragma once
#include "Tensor.h"
#include "CharToIdGenerator.h"

class DataLoader {
public:
  DataLoader(int sequence_length, int batchSize);

  void readFile(const std::string& filename);
  std::pair<Tensor::Ptr, Tensor::Ptr> randBatch();

  char get_char_from_id(int id) const;
  int get_id_from_char(char character) const;

  int get_vocab_size() const;
  const std::unordered_map<char, int> &get_char_to_id_map() const;
  const std::unordered_map<int, char> &get_id_to_char_map() const;
  void printStatistics() const;
  void fillGeneratedText();
private:
  const int _sequenceLength;
  const int _batchSize;
  CharToIdTextGenerator _textGenerator;

  std::vector<char> _chars;
  std::unordered_map<char, int> _charToId;
  std::unordered_map<int, char> _idToChar;
  std::vector<int> _data;

  size_t _numBatches = 0;
};
