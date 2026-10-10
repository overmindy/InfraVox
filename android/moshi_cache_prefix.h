#pragma once
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cstdint>

namespace MoshiCachePrefix {
inline void validate(const std::vector<uint8_t>& full, size_t heads,
                     size_t capacity, size_t width, size_t slots) {
  if (!heads || !capacity || !width || slots > capacity ||
      heads > SIZE_MAX / capacity || heads * capacity > SIZE_MAX / width ||
      full.size() != heads * capacity * width)
    throw std::runtime_error("Invalid Temporal cache dimensions");
}
inline void restore(std::vector<uint8_t>& full, const std::vector<uint8_t>& prefix,
                    size_t heads, size_t capacity, size_t width, size_t slots) {
  validate(full, heads, capacity, width, slots);
  const size_t row = slots * width;
  if (prefix.size() != heads * row) throw std::runtime_error("Temporal cache position mismatch");
  std::fill(full.begin(), full.end(), 0);
  for (size_t head = 0; head < heads && row; ++head)
    std::memcpy(full.data() + head * capacity * width, prefix.data() + head * row, row);
}
inline void pack(const std::vector<uint8_t>& full, std::vector<uint8_t>& prefix,
                 size_t heads, size_t capacity, size_t width, size_t slots) {
  validate(full, heads, capacity, width, slots);
  const size_t row = slots * width;
  prefix.resize(heads * row);
  for (size_t head = 0; head < heads && row; ++head)
    std::memcpy(prefix.data() + head * row, full.data() + head * capacity * width, row);
}
}  // namespace MoshiCachePrefix
