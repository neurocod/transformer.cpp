#pragma once
class ConfigParser;

class TransformerConfig {
public:
  static TransformerConfig& instance();
  void init(const std::string& filename);
  void init(const ConfigParser &config);
  std::string toString() const;
  static bool unitTest();

  std::string modelFileNameByParameters() const;

  size_t numThreads = 4;
  bool inferenceMode = true;
  std::string weightsFilename;
  std::string dataFilename;

  // Model Architecture
  int embedDim;
  int maxSequenceLength;
  int numLayers;
  int numHeads;
  int ffHiddenDim;
  float dropoutRate;
  float padTokenId = 0;

  // Training Parameters
  float learningRate;
  int numEpochs;
  int batchSize;
  int inputSeqLength;
  int decoderSeqLength;

  // Inference Parameters
  int maxGenerateLength;
  std::string initialPrompt;
};