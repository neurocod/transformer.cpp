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

std::string TransformerConfig::toString() const {
  std::ostringstream oss;
  oss << "numThreads=" << numThreads << "\n";
  oss << "inferenceMode=" << (inferenceMode ? "true" : "false") << "\n";
  oss << "weightsFilename=" << weightsFilename << "\n";
  oss << "dataFilename=" << dataFilename << "\n";
  oss << "embedDim=" << embedDim << "\n";
  oss << "maxSequenceLength=" << maxSequenceLength << "\n";
  oss << "numLayers=" << numLayers << "\n";
  oss << "numHeads=" << numHeads << "\n";
  oss << "ffHiddenDim=" << ffHiddenDim << "\n";
  oss << "dropoutRate=" << dropoutRate << "\n";
  oss << "padTokenId=" << padTokenId << "\n";
  oss << "learningRate=" << learningRate << "\n";
  oss << "numEpochs=" << numEpochs << "\n";
  oss << "batchSize=" << batchSize << "\n";
  oss << "inputSeqLength=" << inputSeqLength << "\n";
  oss << "decoderSeqLength=" << decoderSeqLength << "\n";
  oss << "maxGenerateLength=" << maxGenerateLength << "\n";
  oss << "initialPrompt=" << initialPrompt << "\n";
  return oss.str();
}
/*
bool TransformerConfig::unitTest() {
    std::string parsed = R"ini(inferenceMode = false
weightsFilename = auto
dataFilename = ../data/tiny_shakespeare.txt

embedDim = 256
maxSequenceLength = 100
numLayers = 32
numHeads = 16
ffHiddenDim = 1024
dropoutRate = 0.1
padTokenId = 0.0

learningRate = 0.0005
numEpochs = 100
batchSize = 128
inputSeqLength = 10
decoderSeqLength = 10

maxGenerateLength = 100
initialPrompt = ROMEO:

numThreads = 500
 )ini";
 
    ConfigParser parser;
    parser.loadFile(testConfigFile);

    TransformerConfig cfg;
    cfg.init(parser);

    std::string expected = "numThreads=2\n"
                           "inferenceMode=false\n"
                           "weightsFilename=test_weights.bin\n"
                           "dataFilename=test_data.txt\n"
                           "embedDim=128\n"
                           "maxSequenceLength=64\n"
                           "numLayers=4\n"
                           "numHeads=8\n"
                           "ffHiddenDim=256\n"
                           "dropoutRate=0.1\n"
                           "padTokenId=0\n"
                           "learningRate=0.001\n"
                           "numEpochs=10\n"
                           "batchSize=16\n"
                           "inputSeqLength=32\n"
                           "decoderSeqLength=32\n"
                           "maxGenerateLength=50\n"
                           "initialPrompt=Hello\n";

    // 5. Сравниваем результат
    assert(cfg.toString() == expected && "TransformerConfig::toString() failed unit test");

    // (опционально) удалить временный файл
    std::remove(testConfigFile.c_str());
}*/