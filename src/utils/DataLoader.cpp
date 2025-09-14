#include "utils/DataLoader.h"

static std::mt19937 g_randBatch(
  std::chrono::high_resolution_clock::now().time_since_epoch().count());

DataLoader::DataLoader(int sequence_length, int batch_size):
  sequence_length_(sequence_length),
  _batchSize(batch_size)
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
void DataLoader::readFile(const std::string& filename) {
  if (!data_.empty())
    throw std::runtime_error("file already read");

  std::cout << "Loading data from: " << filename << std::endl;
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Could not open data file: " + filename);
  }

  std::string text;
  text.resize(static_cast<std::size_t>(file.tellg()));
  file.seekg(0);
  file.read(text.data(), text.size());

  // Build Vocabulary
  std::unordered_set<char> unique_chars(text.begin(), text.end());
  chars_.assign(unique_chars.begin(), unique_chars.end());

  for (size_t i = 0; i < chars_.size(); ++i) {
    _charToId[chars_[i]] = i;
    _idToChar[i] = chars_[i];
  }
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

  std::cout << "Vocabulary size: " << chars_.size() << std::endl;

  // Numericalize the data
  data_.reserve(text.length());
  for (char c : text) {
    data_.push_back(_charToId[c]);
  }

  std::cout << "Dataset size (characters): " << data_.size() << std::endl;

  // Calculate number of possible batches
  size_t total_sequence_batches = data_.size() / (sequence_length_);
  num_batches_ = total_sequence_batches / _batchSize;

  if (num_batches_ == 0) {
    throw std::runtime_error("Dataset too small to form even one batch with "
                             "current sequence length and batch size.");
  }

  std::cout << "Number of full batches available per epoch: " << num_batches_
            << std::endl;
}

int DataLoader::get_vocab_size() const { return chars_.size(); }

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> DataLoader::randBatch() {
  if (data_.empty())
    throw std::runtime_error("Data not loaded. Call readFile() first.");

  // Pick a random starting batch index
  std::uniform_int_distribution<size_t> randBatch(0, num_batches_ - 1);
  const size_t batchIndex = randBatch(g_randBatch);

  const size_t start_index = batchIndex * _batchSize * sequence_length_;

  std::vector<float> input_batch_vec(_batchSize * sequence_length_);
  std::vector<float> target_batch_vec(_batchSize * sequence_length_);

  // Extract data for the batch
  for (int nBatch = 0; nBatch < _batchSize; ++nBatch) {
    const size_t sequence_start = nBatch * sequence_length_ + start_index;
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
      Tensor::create({_batchSize, sequence_length_},
                     std::make_shared<std::vector<float>>(input_batch_vec));
  std::shared_ptr<Tensor> target_tensor =
      Tensor::create({_batchSize * sequence_length_},
                     std::make_shared<std::vector<float>>(target_batch_vec));

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