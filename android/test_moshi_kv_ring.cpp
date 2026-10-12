#include "moshi_kv_ring.h"
#include <cassert>
#include <iostream>

int main() {
  MoshiKvRing ring(2, 3, sizeof(uint32_t));
  const auto* address = ring.bytes().data();
  for (uint64_t position = 0; position < 8; ++position) {
    const auto before = ring.bytes();
    uint32_t delta[2] = {static_cast<uint32_t>(position + 1),
                         static_cast<uint32_t>(position + 101)};
    ring.append(position, delta, sizeof(delta));
    assert(ring.bytes().data() == address);
    assert(ring.nextPosition() == position + 1);
    assert(ring.validSlots() == std::min<uint64_t>(position + 1, 3));
    for (size_t head = 0; head < 2; ++head) {
      for (size_t slot = 0; slot < 3; ++slot) {
        const size_t offset = (head * 3 + slot) * sizeof(uint32_t);
        if (slot != position % 3)
          assert(std::memcmp(before.data() + offset, ring.bytes().data() + offset, sizeof(uint32_t)) == 0);
        else {
          uint32_t value;
          std::memcpy(&value, ring.bytes().data() + offset, sizeof(value));
          assert(value == delta[head]);
        }
      }
    }
  }
  bool rejected = false;
  uint32_t delta[2] = {};
  try { ring.append(0, delta, sizeof(delta)); } catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
  ring.reset();
  assert(ring.bytes().data() == address && ring.nextPosition() == 0 && ring.validSlots() == 0);
  for (auto byte : ring.bytes()) assert(byte == 0);
  ring.append(0, delta, sizeof(delta));
  std::cout << "PASS: preallocation, append-only slot writes, wrap, absolute position, reset\n";
}
