// Command-line MiniCPM-o session on the phone: optionally shows the model a photo (raw interleaved RGB,
// attached before the call like a photo picked in the app), then steps a 16 kHz mono PCM16 WAV through
// the native runner as fast as it goes and writes the reply text, 24 kHz audio and per-graph timings.
//   minicpmo_cli <qnn lib dir> <model dir> <input.wav> <output dir> [image.rgb width height] [seed]

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

void* minicpmoCreate(const std::string& libraryDir, const std::string& modelDirectory);
void minicpmoBegin(void* session, bool highPerformance, float temperature, uint64_t seed);
bool minicpmoStep(void* session, const int16_t* input, size_t count, std::vector<int16_t>& pcm, std::vector<uint8_t>& text);
void minicpmoAttachImage(void* session, const uint8_t* rgb, size_t width, size_t height);
void minicpmoEnd(void* session, const std::string& performanceCsv);
void minicpmoDestroy(void* session);

namespace {

constexpr size_t kFrameSamples = 1280;

uint32_t le32(const char* d) {
  return uint8_t(d[0]) | uint8_t(d[1]) << 8 | uint8_t(d[2]) << 16 | uint32_t(uint8_t(d[3])) << 24;
}

std::vector<char> readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("Cannot read " + path);
  return std::vector<char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::vector<int16_t> readWav(const std::string& path) {
  const auto bytes = readFile(path);
  if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) || std::memcmp(bytes.data() + 8, "WAVE", 4))
    throw std::runtime_error("Not a WAV file: " + path);
  bool format = false;
  for (size_t at = 12; at + 8 <= bytes.size();) {
    const uint32_t size = le32(bytes.data() + at + 4);
    const char* body = bytes.data() + at + 8;
    if (!std::memcmp(bytes.data() + at, "fmt ", 4)) {
      const uint16_t channels = uint8_t(body[2]) | uint8_t(body[3]) << 8, bits = uint8_t(body[14]) | uint8_t(body[15]) << 8;
      if (le32(body + 4) != 16000 || channels != 1 || bits != 16) throw std::runtime_error("Expected 16 kHz mono PCM16");
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
  if (argc != 5 && argc != 6 && argc != 8 && argc != 9) {
    std::cerr << "usage: minicpmo_cli <qnn lib dir> <model dir> <input.wav> <output dir> [image.rgb width height] [seed]\n";
    return 2;
  }
  try {
    const std::string libraries = argv[1], model = argv[2], output = argv[4];
    const bool image = argc >= 8;
    const uint64_t seed = argc == 6 || argc == 9 ? std::strtoull(argv[argc - 1], nullptr, 10) : 1;
    setenv("ADSP_LIBRARY_PATH", libraries.c_str(), 1);
    const auto input = readWav(argv[3]);
    void* session = minicpmoCreate(libraries, model);
    if (image) {
      const auto rgb = readFile(argv[5]);
      const size_t width = std::strtoul(argv[6], nullptr, 10), height = std::strtoul(argv[7], nullptr, 10);
      if (rgb.size() != width * height * 3) throw std::runtime_error("RGB file does not match the image size");
      minicpmoAttachImage(session, reinterpret_cast<const uint8_t*>(rgb.data()), width, height);
    }
    minicpmoBegin(session, true, 0.7f, seed);
    std::vector<int16_t> reply, pcm;
    std::vector<uint8_t> text;
    std::string transcript;
    size_t unit = 0;
    for (size_t at = 0; at + kFrameSamples <= input.size(); at += kFrameSamples) {
      text.clear();
      if (!minicpmoStep(session, input.data() + at, kFrameSamples, pcm, text)) continue;
      reply.insert(reply.end(), pcm.begin(), pcm.end());
      const std::string piece(text.begin(), text.end());
      transcript += piece;
      std::cout << unit++ << (piece.empty() ? "" : " " + piece) << "\n" << std::flush;
    }
    minicpmoEnd(session, output + "/performance.csv");
    minicpmoDestroy(session);
    writeWav(output + "/reply.wav", reply);
    std::ofstream(output + "/transcript.txt") << transcript << "\n";
    std::cout << "reply: " << transcript << "\n";
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
