#include "utils/ConfigParser.h"

template <> std::string ConfigParser::value<std::string>(const std::string &key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto                        it = config_map_.find(key);
  if (it == config_map_.end()) {
    throw std::runtime_error("Config key not found: " + key);
  }
  return it->second;
}

ConfigParser &ConfigParser::instance(const std::string &filename) {
  static ConfigParser instance(filename);
  return instance;
}

void ConfigParser::loadFile(const std::string &filename) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_map_.clear();

  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open config file: " + filename);
  }

  std::string line;
  while (std::getline(file, line)) {
    // Remove leading/trailing whitespace
    line.erase(line.begin(),
               std::find_if(line.begin(), line.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    line.erase(std::find_if(line.rbegin(), line.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
               line.end());

    // Skip empty lines and comments
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::size_t equals_pos = line.find('=');
    if (equals_pos != std::string::npos) {
      std::string key   = line.substr(0, equals_pos);
      std::string value = line.substr(equals_pos + 1);

      // Trim whitespace from key and value
      key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
      value.erase(value.begin(),
                  std::find_if(value.begin(), value.end(), [](unsigned char ch) { return !std::isspace(ch); }));
      value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
                  value.end());

      config_map_[key] = value;
    }
  }
}

std::string ConfigParser::toString() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::ostringstream oss;
  for (const auto &[key, value] : config_map_) {
    oss << key << "=" << value << "\n";
  }
  return oss.str();
}