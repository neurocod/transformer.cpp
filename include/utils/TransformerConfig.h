#pragma once
class ConfigParser;

class TransformerConfig {
public:
  static const TransformerConfig& instance();
  static void init(const std::string& filename);

  std::string modelFileNameByParameters() const;

  size_t numThreads;
  bool inferenceMode;
  std::string weightsFilename;
  std::string dataFilename;

  // Model Architecture
  int embedDim;
  int maxSequenceLength;
  int numLayers;
  int numHeads;
  int ffHiddenDim;
  float dropoutRate;
  float padTokenId;

  // Training Parameters
  float learningRate;
  int numEpochs;
  int batchSize;
  int inputSeqLength;
  int decoderSeqLength;

  // Inference Parameters
  int maxGenerateLength;
  std::string initialPrompt;
private:
  static TransformerConfig& mutableInstance();
  void init(const ConfigParser& config);
};