#pragma once
#include "Tensor.h"
#include "CharToIdGenerator.h"
class BinaryReader;
class BinaryWriter;

class Tokenizer {
protected:
  std::vector<char> _chars;
  std::unordered_map<char, int> _charToId;
  std::unordered_map<int, char> _idToChar;
public:
  void write(BinaryWriter &writer) const;
  bool read(BinaryReader &reader);

  const std::unordered_map<char, int> &charToIdMap() const { return _charToId; }
  const std::unordered_map<int, char> &idToCharMap() const { return _idToChar; }

  char charFromId(int id) const;
  int idFromChar(char character) const;
  int vocabSize() const { return _chars.size(); }
};

class DataLoader: public Tokenizer {
public:
  DataLoader(int sequence_length, int batchSize);

  void readFile(const std::string& filename);
  std::pair<Tensor::Ptr, Tensor::Ptr> randBatch();

  void printStatistics() const;
  void fillGeneratedText();
private:
  const int _sequenceLength;
  const int _batchSize;
  CharToIdGenerator _textGenerator;
  using Vec = Tensor::Vec;

  std::vector<int> _data;

  size_t _numBatches = 0;
};
