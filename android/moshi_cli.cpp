// Command-line Moshi session on the phone: steps a 24 kHz mono PCM16 WAV through the native runner as
// fast as it goes and writes the reply, per-step times and tokens. Same code path as the app's JNI.
//   moshi_cli <qnn lib dir> <model dir> <input.wav> <output dir> [seed] [text temperature]

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

void* moshiCreate(const std::string& libraryDir, const std::string& modelDirectory, bool codecOnly);
void moshiBegin(void* session, bool highPerformance, float textTemperature, uint32_t seed);
bool moshiStep(void* session, const int16_t* input, int16_t* pcm, int32_t* tokens);
void moshiEnd(void* session, const std::string& performanceCsv);
void moshiDestroy(void* session);

namespace {

constexpr size_t kFrameSamples = 1920;
constexpr size_t kFlushFrames = 3;  // silence that pushes the last real frames through the pipeline

uint32_t le32(const char* d) {
  return uint8_t(d[0]) | uint8_t(d[1]) << 8 | uint8_t(d[2]) << 16 | uint32_t(uint8_t(d[3])) << 24;
}

std::vector<int16_t> readWav(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) || std::memcmp(bytes.data() + 8, "WAVE", 4))
    throw std::runtime_error("Not a WAV file: " + path);
  bool format = false;
  for (size_t at = 12; at + 8 <= bytes.size();) {
    const uint32_t size = le32(bytes.data() + at + 4);
    const char* body = bytes.data() + at + 8;
    if (!std::memcmp(bytes.data() + at, "fmt ", 4)) {
      const uint16_t channels = uint8_t(body[2]) | uint8_t(body[3]) << 8, bits = uint8_t(body[14]) | uint8_t(body[15]) << 8;
      if (le32(body + 4) != 24000 || channels != 1 || bits != 16) throw std::runtime_error("Expected 24 kHz mono PCM16");
      format = true;
    } else if (!std::memcmp(bytes.data() + at, "data", 4) && format) {
      std::vector<int16_t> samples(std::min<size_t>(size, bytes.size() - at - 8) / 2);
      std::memcpy(samples.data(), body, samples.size() * 2);
      return samples;
    }
    at += 8 + size + (size & 1);
  }
  throw std::runtime_error("No PCM data in " + path);
}

void writeWav(const std::string& path, const std::vector<int16_t>& samples) {
  std::ofstream out(path, std::ios::binary);
  auto u32 = [&](uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
  auto u16 = [&](uint16_t v) { out.write(reinterpret_cast<const char*>(&v), 2); };
  const uint32_t data = samples.size() * 2;
  out.write("RIFF", 4); u32(36 + data); out.write("WAVEfmt ", 8);
  u32(16); u16(1); u16(1); u32(24000); u32(48000); u16(2); u16(16);
  out.write("data", 4); u32(data);
  out.write(reinterpret_cast<const char*>(samples.data()), data);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: moshi_cli <qnn lib dir> <model dir> <input.wav> <output dir> [seed] [text temperature]\n";
    return 2;
  }
  try {
    const std::string libraries = argv[1], model = argv[2], output = argv[4];
    const uint32_t seed = argc > 5 ? std::strtoul(argv[5], nullptr, 10) : 1;
    const float temperature = argc > 6 ? std::strtof(argv[6], nullptr) : 0.7f;
    setenv("ADSP_LIBRARY_PATH", libraries.c_str(), 1);
    std::vector<int16_t> input = readWav(argv[3]);
    input.resize((input.size() / kFrameSamples + kFlushFrames) * kFrameSamples, 0);

    void* session = moshiCreate(libraries, model, false);
    moshiBegin(session, true, temperature, seed);
    std::vector<int16_t> reply;
    std::ofstream steps(output + "/steps.csv"), tokens(output + "/tokens.csv");
    steps << "step,ms\n";
    tokens << "frame,text,codes\n";
    std::array<int16_t, kFrameSamples> pcm{};
    std::array<int32_t, 9> out{};
    const size_t frames = input.size() / kFrameSamples;
    size_t emitted = 0;
    for (size_t i = 0; i < frames; ++i) {
      const auto start = std::chrono::steady_clock::now();
      const bool ready = moshiStep(session, input.data() + i * kFrameSamples, pcm.data(), out.data());
      steps << i << ',' << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() << '\n';
      if (!ready) continue;
      reply.insert(reply.end(), pcm.begin(), pcm.end());
      tokens << emitted++ << ',' << out[0] << ',';
      for (size_t c = 1; c < out.size(); ++c) tokens << out[c] << (c + 1 < out.size() ? " " : "\n");
      if (i % 250 == 0) std::cerr << "step " << i << "/" << frames << "\n";
    }
    moshiEnd(session, output + "/performance.csv");
    moshiDestroy(session);
    writeWav(output + "/reply.wav", reply);
    std::cerr << "done: " << frames << " steps, " << emitted << " frames out\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
