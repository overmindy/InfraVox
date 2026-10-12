#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

// Host-owned [heads, capacity, widthBytes] cache. Append writes one slot per
// head; it never grows or shifts the history. This does NOT by itself make a
// QNN RAW input zero-copy, nor authorize input/output tensor aliasing.
class MoshiKvRing {
 public:
  MoshiKvRing(size_t heads, size_t capacity, size_t widthBytes)
      : heads_(heads), capacity_(capacity), width_(widthBytes) {
    if (!heads || !capacity || !widthBytes || heads > SIZE_MAX / capacity ||
        heads * capacity > SIZE_MAX / widthBytes)
      throw std::invalid_argument("Invalid KV ring shape");
    bytes_.resize(heads * capacity * widthBytes, 0);
  }
  void reset() {
    std::fill(bytes_.begin(), bytes_.end(), 0);
    next_ = 0;
  }
  void append(uint64_t absolutePosition, const void* delta, size_t length) {
    if (absolutePosition != next_ || !delta || length != heads_ * width_ ||
        next_ == std::numeric_limits<uint64_t>::max())
      throw std::invalid_argument("KV append position or delta size mismatch");
    const auto* source = static_cast<const uint8_t*>(delta);
    const size_t slot = absolutePosition % capacity_;
    for (size_t head = 0; head < heads_; ++head)
      std::memcpy(bytes_.data() + (head * capacity_ + slot) * width_,
                  source + head * width_, width_);
    ++next_;
  }
  const std::vector<uint8_t>& bytes() const { return bytes_; }
  size_t capacity() const { return capacity_; }
  uint64_t nextPosition() const { return next_; }
  size_t validSlots() const { return static_cast<size_t>(std::min<uint64_t>(next_, capacity_)); }
 private:
  size_t heads_, capacity_, width_;
  uint64_t next_ = 0;
  std::vector<uint8_t> bytes_;
};
