#include "utils/TransformerConfig.h"
#include "utils/ConfigParser.h"

TransformerConfig& TransformerConfig::mutableInstance() {
  static TransformerConfig instance;
  return instance;
}
const TransformerConfig& TransformerConfig::instance() {
  return mutableInstance();
}

void TransformerConfig::init(const std::string& filename) {
  ConfigParser& config = ConfigParser::instance(filename);
  mutableInstance().init(config);
}

void TransformerConfig::init(const ConfigParser& config) {
  numThreads = config.value<int>("numThreads");
  inferenceMode = config.value<bool>("inferenceMode");
  weightsFilename = config.value<std::string>("weightsFilename");
  dataFilename = config.value<std::string>("dataFilename");

  // Model Architecture
  embedDim = config.value<int>("embedDim");
  maxSequenceLength = config.value<int>("maxSequenceLength");
  numLayers = config.value<int>("numLayers");
  numHeads = config.value<int>("numHeads");
  ffHiddenDim = config.value<int>("ffHiddenDim");
  dropoutRate = config.value<float>("dropoutRate");
  padTokenId = config.value<float>("padTokenId");

  // Training Parameters
  learningRate = config.value<float>("learningRate");
  numEpochs = config.value<int>("numEpochs");
  batchSize = config.value<int>("batchSize");
  inputSeqLength = config.value<int>("inputSeqLength");
  decoderSeqLength = config.value<int>("decoderSeqLength");

  // Inference Parameters
  maxGenerateLength = config.value<int>("maxGenerateLength");
  initialPrompt = config.value<std::string>("initialPrompt");

  if (weightsFilename == "auto") {
    weightsFilename = std::format("weigth-{}-{}-{}.bin", numLayers, ffHiddenDim, numHeads);
  }
}

std::string TransformerConfig::modelFileNameByParameters() const {
  return std::format("{}", 0);
}