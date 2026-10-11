#pragma once
#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// Only return storage after synchronous graphExecute completes and the host
// consumes outputs. Resident contexts must rebind before their next execute.
// Never hand the same storage to two in-flight executions.
class GraphIoPool {
 public:
  void take(const std::string& role, size_t index, size_t size, std::vector<uint8_t>& target) {
    auto& bytes = storage_[key(role, index, size)];
    target.swap(bytes);
    target.resize(size);
    std::fill(target.begin(), target.end(), 0);
  }
  void put(const std::string& role, size_t index, std::vector<uint8_t>& bytes) {
    if (!bytes.empty()) bytes.swap(storage_[key(role, index, bytes.size())]);
    bytes.clear();
  }
 private:
  static std::string key(const std::string& role, size_t index, size_t size) {
    return role + ":" + std::to_string(index) + ":" + std::to_string(size);
  }
  std::map<std::string, std::vector<uint8_t>> storage_;
};
