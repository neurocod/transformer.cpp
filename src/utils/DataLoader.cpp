#include "utils/DataLoader.h"

static std::mt19937 g_randBatch(
  std::chrono::high_resolution_clock::now().time_since_epoch().count());

DataLoader::DataLoader(int sequence_length, int batchSize):
  _sequenceLength(sequence_length),
  _batchSize(batchSize)
{
}

void printEscapedChar(char c) {
  switch (c) {
  case ' ': std::cout << "' '"; break;
  case '\0': std::cout << "\\0"; break;
  case '\n': std::cout << "\\n"; break;
  case '\r': std::cout << "\\r"; break;
  case '\t': std::cout << "\\t"; break;
  case '\\': std::cout << "\\\\"; break;
  case '\"': std::cout << "\\\""; break;
  case '\'': std::cout << "\\'"; break;
  default: std::cout << c; break;
  }
}
void DataLoader::printStatistics() const {
  std::cout << "char -> id: " << std::endl;
  // print sorted
  std::vector<std::pair<char, int>> charPairs(_charToId.begin(), _charToId.end());
  std::sort(charPairs.begin(), charPairs.end());
  for (const auto& pair : charPairs) {
    printEscapedChar(pair.first);
    std::cout << " -> " << pair.second << std::endl;
  }

  std::cout << "id -> char: " << std::endl;
  std::vector<std::pair<int, char>> idPairs(_idToChar.begin(), _idToChar.end());
  std::sort(idPairs.begin(), idPairs.end());
  for (const auto& pair : idPairs) {
    std::cout << pair.first << " -> ";
    printEscapedChar(pair.second);
    std::cout << std::endl;
  }

  std::cout << "Vocabulary size: " << _chars.size() << std::endl;
}
void DataLoader::readFile(const std::string& filename) {
  if (!_data.empty())
    throw std::runtime_error("file already read");

  std::string text;
  const bool generate = filename == "generate";
  if (generate) {
    const size_t target = _sequenceLength * _batchSize;
    while (text.size() < target)
      text += _textGenerator.generateSimple();
  } else {
    spdlog::info("Loading data from {}", filename);
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file)
      throw std::runtime_error("Could not open data file: " + filename);

    text.resize(static_cast<std::size_t>(file.tellg()));
    file.seekg(0);
    file.read(text.data(), text.size());
  }

  // Build Vocabulary
  std::unordered_set<char> unique_chars(text.begin(), text.end());
  _chars.assign(unique_chars.begin(), unique_chars.end());

  for (size_t i = 0; i < _chars.size(); ++i) {
    _charToId[_chars[i]] = i;
    _idToChar[i] = _chars[i];
  }
  printStatistics();

  // Numericalize the data
  _data.reserve(text.length());
  for (char c : text) {
    _data.push_back(_charToId[c]);
  }

  spdlog::info("Dataset size (characters): {}", _data.size());

  // Calculate number of possible batches
  size_t total_sequence_batches = _data.size() / (_sequenceLength);
  _numBatches = total_sequence_batches / _batchSize;

  if (_numBatches == 0) {
    throw std::runtime_error("Dataset too small to form even one batch with "
                             "current sequence length and batch size.");
  }

  spdlog::info("Number of full batches available per epoch: {}", _numBatches);
}

int DataLoader::get_vocab_size() const { return _chars.size(); }

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> DataLoader::randBatch() {
  if (_data.empty())
    throw std::runtime_error("Data not loaded. Call readFile() first.");

  // Pick a random starting batch index
  std::uniform_int_distribution<size_t> randBatch(0, _numBatches - 1);
  const size_t batchIndex = randBatch(g_randBatch);
  const size_t startIndex = batchIndex * _batchSize * _sequenceLength;

  std::vector<float> inputBatchVec(_batchSize * _sequenceLength);
  std::vector<float> targetBatchVec(_batchSize * _sequenceLength);

  // Extract data for the batch
  for (int j = 0; j < _batchSize; ++j) {
    const size_t sequenceStart = j * _sequenceLength + startIndex;
    for (int i = 0; i < _sequenceLength; ++i) {
      const int pos = j * _sequenceLength + i;
      // Input is the current character
      inputBatchVec[pos] = static_cast<float>(_data[sequenceStart + i]);

      // Target is the next character (shifted by one)
      // Handle the case where the target is beyond the end of the _data vector
      if (sequenceStart + i + 1 < _data.size()) {
        targetBatchVec[pos] = static_cast<float>(_data[sequenceStart + i + 1]);
      } else {
        targetBatchVec[pos] = static_cast<float>(_data[sequenceStart + i]);
      }
    }
  }

  std::shared_ptr<Tensor> input_tensor =
      Tensor::create({_batchSize, _sequenceLength},
                     std::make_shared<std::vector<float>>(inputBatchVec));
  std::shared_ptr<Tensor> target_tensor =
      Tensor::create({_batchSize * _sequenceLength},
                     std::make_shared<std::vector<float>>(targetBatchVec));

  return {input_tensor, target_tensor};
}

char DataLoader::get_char_from_id(int id) const {
  auto it = _idToChar.find(id);
  if (it == _idToChar.end()) {
    // Return a default character or throw an error for unknown ID
    it = _idToChar.find(0); // Assuming 0 might be padding or unknown
    if (it != _idToChar.end())
      return it->second;
    return '?';
  }
  return it->second;
}

int DataLoader::get_id_from_char(char ch) const {
  auto it = _charToId.find(ch);
  if (it == _charToId.end()) {
    it = _charToId.find('\0');
    if (it != _charToId.end())
      return it->second;
    if (_charToId.count(0))
      return _charToId.at(0);
    return -1;
  }
  return it->second;
}

const std::unordered_map<char, int> &DataLoader::get_char_to_id_map() const {
  return _charToId;
}

const std::unordered_map<int, char> &DataLoader::get_id_to_char_map() const {
  return _idToChar;
}