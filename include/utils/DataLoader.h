#pragma once
#include "Tensor.h"
#include "CharToIdGenerator.h"
#include "Tokenizer.h"

class DataLoader: public Tokenizer {
public:
  using Ptr = std::shared_ptr<DataLoader>;
  DataLoader(int sequenceLength, int batchSize);

  void readFile(const std::string& filename);
  std::pair<Tensor::Ptr, Tensor::Ptr> randBatch();

  void printStatistics() const;
private:
  const int _sequenceLength;
  const int _batchSize;
  CharToIdGenerator _textGenerator;
  using Vec = Tensor::Vec;

  std::vector<int> _data;

  size_t _numBatches = 0;
};
