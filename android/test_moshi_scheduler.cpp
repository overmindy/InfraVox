#include "moshi_lm_scheduler.h"
#include "moshi_cache_prefix.h"
#include <cassert>

int main() {
  std::vector<uint8_t> full(2 * 4 * 2, 0);
  full[0] = 11; full[1] = 12; full[8] = 21; full[9] = 22;
  std::vector<uint8_t> prefix;
  MoshiCachePrefix::pack(full, prefix, 2, 4, 2, 1);
  assert((prefix == std::vector<uint8_t>{11, 12, 21, 22}));
  std::vector<uint8_t> restored(full.size(), 255);
  MoshiCachePrefix::restore(restored, prefix, 2, 4, 2, 1);
  assert(restored == full);
  MoshiCachePrefix::pack(full, prefix, 2, 4, 2, 4);
  MoshiCachePrefix::restore(restored, prefix, 2, 4, 2, 4);
  assert(restored == full);
  bool rejected = false;
  try { MoshiCachePrefix::restore(restored, prefix, 2, 4, 2, 5); }
  catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);
  MoshiDelayRing ring;
  std::array<int32_t, 8> user{};
  std::array<int32_t, 8> previousAudio{};
  for (int frame = 0; frame < 10; ++frame) {
    for (int k = 0; k < 8; ++k) user[k] = 100 + frame * 8 + k;
    auto input = ring.begin(user);
    if (frame == 0) {
      assert(input[0] == 32000);
      for (int k = 1; k < 17; ++k) assert(input[k] == 2048);
    } else {
      assert(input[0] == 19 + frame);
      assert(input[1] == previousAudio[0]);
      assert(input[9] == user[0]);
      for (int k = 1; k < 8; ++k) {
        assert(input[k + 9] == (frame == 1 ? 2048 : 100 + (frame - 1) * 8 + k));
        assert(input[k + 1] == (frame == 1 ? 2048 : previousAudio[k]));
      }
    }
    std::array<int32_t, 8> generated{};
    for (int k = 0; k < 8; ++k) generated[k] = frame * 8 + k;
    auto output = ring.finish(20 + frame, generated);
    if (frame == 0) assert(!output);
    else {
      assert(output);
      assert((*output)[0] == 19 + frame);
      assert((*output)[1] == previousAudio[0]);
      for (int k = 1; k < 8; ++k) assert((*output)[k + 1] == generated[k]);
    }
    previousAudio = generated;
  }
  ring.reset();
  auto initial = ring.begin(user);
  assert(initial[0] == 32000);
  ring.reset();  // Also cancel a pending frame when a clip is reset.
  assert(ring.begin(user)[0] == 32000);
}
