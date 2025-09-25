#pragma once
class BinaryReader;
class BinaryWriter;

class Tokenizer : public std::enable_shared_from_this<Tokenizer> {
protected:
  std::vector<char> _chars;
  std::unordered_map<char, int> _charToId;
  std::unordered_map<int, char> _idToChar;
public:
  using Ptr = std::shared_ptr<Tokenizer>;
  virtual ~Tokenizer() {}

  void write(BinaryWriter &writer) const;
  bool read(BinaryReader &reader);

  const std::unordered_map<char, int> &charToIdMap() const { return _charToId; }
  const std::unordered_map<int, char> &idToCharMap() const { return _idToChar; }

  char charFromId(int id) const;
  int idFromChar(char character) const;
  int vocabSize() const { return _chars.size(); }
  int inputVocabSize() const { return _chars.size(); }
  int targetVocabSize() const { return _chars.size(); }
};
