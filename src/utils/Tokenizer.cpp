#include "utils/Tokenizer.h"
#include "utils/BinaryWriter.h"
#include "utils/BinaryReader.h"

void Tokenizer::write(BinaryWriter &writer) const {
    const uint32_t vocabSize = static_cast<uint32_t>(_chars.size());
    writer.write(vocabSize);

    if (! _chars.empty()) {
        writer.writeVector(_chars);
    }
}

bool Tokenizer::read(BinaryReader &reader) {
    const uint32_t vocabSize = reader.read<uint32_t>();
    if (!reader.ok())
        return false;

    // Read the characters
    std::vector<char> chars;
    reader.readVector(chars, vocabSize);
    if (!reader.ok())
        return false;

    _chars = std::move(chars);
    // Rebuild the maps
    _charToId.clear();
    _idToChar.clear();

    for (size_t i = 0; i < _chars.size(); ++i) {
        _charToId[_chars[i]] = static_cast<int>(i);
        _idToChar[static_cast<int>(i)] = _chars[i];
    }
    return true;
}

char Tokenizer::charFromId(int id) const {
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

int Tokenizer::idFromChar(char ch) const {
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