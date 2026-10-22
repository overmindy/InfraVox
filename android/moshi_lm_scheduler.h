#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>

// Pinned Kyutai LMGen._step contract, batch=1, greedy, no conditioning/CFG.
// Audio streams 1..8 are generated; streams 9..16 come from streaming Mimi.
class MoshiDelayRing {
 public:
  static constexpr std::array<int, 17> delays =
      {0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1};
  void reset() {
    offset_ = 0;
    pending_ = false;
    for (auto& stream : ring_) stream.fill(-2);
  }
  MoshiDelayRing() { reset(); }
  std::array<int32_t, 17> begin(const std::array<int32_t, 8>& user) {
    if (pending_) throw std::runtime_error("Delay ring frame already pending");
    for (size_t k = 0; k < 8; ++k) {
      if (user[k] < 0 || user[k] >= 2048) throw std::runtime_error("Invalid Mimi token");
      ring_[k + 9][(offset_ + delays[k + 9]) % 3] = user[k];
    }
    std::array<int32_t, 17> input{};
    for (size_t k = 0; k < 17; ++k) {
      input[k] = offset_ <= delays[k] ? (k == 0 ? 32000 : 2048) : ring_[k][offset_ % 3];
      if (input[k] < 0) throw std::runtime_error("Ungenerated temporal input");
    }
    pending_ = true;
    return input;
  }
  std::optional<std::array<int32_t, 9>> finish(int32_t text,
                                              const std::array<int32_t, 8>& audio) {
    if (!pending_) throw std::runtime_error("Delay ring has no pending frame");
    if (text < 0 || text >= 32000) throw std::runtime_error("Invalid text token");
    for (auto token : audio)
      if (token < 0 || token >= 2048) throw std::runtime_error("Invalid generated audio token");
    ++offset_;
    ring_[0][offset_ % 3] = text;
    for (size_t k = 0; k < 8; ++k) ring_[k + 1][offset_ % 3] = audio[k];
    pending_ = false;
    if (offset_ <= 1) return std::nullopt;
    std::array<int32_t, 9> out{};
    for (size_t k = 0; k < 9; ++k)
      out[k] = ring_[k][(offset_ - 1 + delays[k]) % 3];
    return out;
  }
 private:
  int64_t offset_ = 0;
  bool pending_ = false;
  std::array<std::array<int32_t, 3>, 17> ring_{};
};
