#pragma once

class ConfigParser {
public:
  static ConfigParser& instance(const std::string& filename = "");

  // Delete copy constructor and assignment operator to prevent cloning
  ConfigParser(const ConfigParser &) = delete;
  ConfigParser &operator=(const ConfigParser &) = delete;

  void loadFile(const std::string &filename);
  void loadIniValues(const std::string &values);

  template <typename T> T value(const std::string &key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = config_map_.find(key);
    if (it == config_map_.end()) {
      throw std::runtime_error("Config key not found: " + key);
    }

    if constexpr (std::is_same_v<T, bool>) {
      std::string value = it->second;
      std::transform(value.begin(), value.end(), value.begin(), ::tolower);
      if (value == "true")
        return true;
      if (value == "false")
        return false;
      throw std::runtime_error("Failed to parse boolean value for key: " + key);
    } else {
      std::stringstream ss(it->second);
      T value;
      ss >> value;
      if (ss.fail() || !ss.eof()) {
        throw std::runtime_error("Failed to parse config value for key: " +
                                 key);
      }
      return value;
    }
  }

  virtual std::string toString() const;

private:
  ConfigParser(const std::string &filename = "") {
    if (!filename.empty()) {
      loadFile(filename);
    }
  }

  std::unordered_map<std::string, std::string> config_map_;
  mutable std::mutex mutex_;
};

template <>
std::string ConfigParser::value<std::string>(const std::string &key) const;
