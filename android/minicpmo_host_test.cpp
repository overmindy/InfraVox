// Host-side checks of minicpmo_runner.cpp's CPU math against vectors from the server scripts
// (scripts/minicpmo_hift_check.py and a captured duplex session's Whisper input). Builds and
// runs on a desktop; no QNN device or ONNX Runtime library is needed.
//   clang++ -std=c++17 -O2 -I <qnn include> -I <ort headers> android/minicpmo_host_test.cpp -o host_test
//   ./host_test <dir with audio.f32 mel.f32 mel_filters.f32 f0.f32 ... wave.f32>

#define ORT_API_MANUAL_INIT
#include "minicpmo_runner.cpp"

#include <iostream>

namespace {

std::vector<float> load(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("Cannot read " + path);
  std::vector<float> data(static_cast<size_t>(file.tellg()) / sizeof(float));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
  return data;
}

double maxAbs(const float* a, const float* b, size_t n) {
  double worst = 0;
  for (size_t i = 0; i < n; ++i) worst = std::max(worst, static_cast<double>(std::fabs(a[i] - b[i])));
  return worst;
}

bool report(const char* name, double error, double tolerance) {
  std::cout << name << ": max abs error " << error << (error <= tolerance ? "  ok" : "  FAIL") << "\n";
  return error <= tolerance;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) { std::cerr << "usage: host_test DIR\n"; return 2; }
  const std::string dir = argv[1];
  bool ok = true;

  // Streaming mel: replay the duplex unit boundaries (first unit 560 zeros + 16480 samples, then 16000).
  const auto audio = load(dir + "/audio.f32");
  const auto mel = load(dir + "/mel.f32");
  const auto filters = load(dir + "/mel_filters.f32");
  StreamingMel streaming(filters.data());
  std::vector<float> buffer(560, 0.0f);
  buffer.insert(buffer.end(), audio.begin(), audio.end());
  size_t at = 0;
  double worst = 0;
  const size_t units = mel.size() / (80 * 104);
  for (size_t u = 0; u < units; ++u) {
    const size_t need = u == 0 ? 16480 : 16000;
    if (at + need > buffer.size()) break;
    size_t frames = 0;
    const auto out = streaming.process(buffer.data() + at, need, frames);
    at += need;
    const float* reference = mel.data() + u * 80 * 104;
    for (size_t m = 0; m < 80; ++m)
      worst = std::max(worst, maxAbs(out.data() + m * frames, reference + m * 104 + (104 - frames), frames));
  }
  ok &= report("streaming mel", worst, 1e-3);

  // HiFT CPU tail: NSF source (noise off), STFT, iSTFT.
  const auto f0 = load(dir + "/f0.f32");
  const auto source = load(dir + "/source.f32");
  const auto weight = load(dir + "/weight.f32");
  const auto bias = load(dir + "/bias.f32");
  std::mt19937_64 random(0);
  const auto quiet = nsfSource(f0, random, weight.data(), bias[0], false);
  ok &= report("nsf source (noise off)", maxAbs(quiet.data(), source.data(), source.size()), 1e-4);

  const auto signal = load(dir + "/signal.f32");
  const auto stft = load(dir + "/stft.f32");
  const size_t frames = signal.size() / 4 + 1;
  std::vector<float> spectrum(18 * frames);
  stft16(signal, spectrum.data(), frames);
  ok &= report("stft", maxAbs(spectrum.data(), stft.data(), stft.size()), 1e-4);

  const auto head = load(dir + "/head.f32");
  const auto wave = load(dir + "/wave.f32");
  const auto samples = istft16(head.data(), frames, frames);
  ok &= report("istft", maxAbs(samples.data(), wave.data(), wave.size()), 1e-4);
  return ok ? 0 : 1;
}
