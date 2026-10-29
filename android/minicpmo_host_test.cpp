// Host-side checks of minicpmo_runner.cpp's CPU math against vectors from the server scripts
// (scripts/minicpmo_hift_check.py, scripts/minicpmo_tts_cpu.py with its tts_cpu.bin, a captured
// duplex session's Whisper input, and minicpmo_vision_reference.py hosttest in DIR/vision). Builds and
// runs on a desktop; no QNN device or ONNX Runtime library is needed.
//   clang++ -std=c++17 -O2 -I <qnn include> -I <ort headers> android/minicpmo_host_test.cpp -o host_test
//   ./host_test <dir with audio.f32 mel.f32 mel_filters.f32 f0.f32 ... wave.f32>

#define ORT_API_MANUAL_INIT
#include "minicpmo_runner.cpp"

#include <iostream>

namespace {

template <typename T>
std::vector<T> loadAs(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("Cannot read " + path);
  std::vector<T> data(static_cast<size_t>(file.tellg()) / sizeof(T));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
  return data;
}

std::vector<float> load(const std::string& path) { return loadAs<float>(path); }

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

  // CPU TTS: teacher-forced steps against scripts/minicpmo_tts_cpu.py (tts_cpu.bin from the server).
  const auto embeds = load(dir + "/tts_embeds.f32");
  const auto expected = load(dir + "/tts_logits.f32");
  CpuTts tts(dir + "/tts_cpu.bin");
  tts.awake(true);
  double ttsFirst = 0, ttsWorst = 0;
  const size_t steps = embeds.size() / kTtsDim;
  const auto start = Clock::now();
  for (size_t s = 0; s < steps; ++s) {
    const auto logits = tts.step(embeds.data() + s * kTtsDim);
    double error = 0, norm = 0;
    for (size_t c = 0; c < static_cast<size_t>(kCodes); ++c) {
      const double reference = expected[s * kCodes + c];
      error += (logits[c] - reference) * (logits[c] - reference);
      norm += reference * reference;
    }
    if (s == 0) ttsFirst = std::sqrt(error / norm);
    ttsWorst = std::max(ttsWorst, std::sqrt(error / norm));
  }
  tts.awake(false);
  std::cout << "cpu tts: " << elapsedMs(start) / steps << " ms per step on this host\n";
  // Same arithmetic, so step 0 matches exactly; later steps drift by int8 rounding ties that flip
  // differently in float, which the KV cache carries forward (noise of the quantization's size).
  ok &= report("cpu tts step 0 (relative)", ttsFirst, 1e-5);
  ok &= report("cpu tts all steps (relative)", ttsWorst, 2e-2);

  // Image preparation against the official MiniCPMVImageProcessor and NaViT / resampler positions
  // (scripts/minicpmo_vision_reference.py hosttest into DIR/vision, vision_assets.bin next to them).
  if (std::ifstream(dir + "/vision/0_dims.i32").good()) {
    const Assets vision(dir + "/vision/vision_assets.bin");
    constexpr size_t kSlots = 1056, kPatch = VisionInputs::kPatch;
    for (int i = 0; std::ifstream(dir + "/vision/" + std::to_string(i) + "_dims.i32").good(); ++i) {
      const std::string prefix = dir + "/vision/" + std::to_string(i) + "_";
      const auto dims = loadAs<int32_t>(prefix + "dims.i32");
      const auto rgb = loadAs<uint8_t>(prefix + "rgb.u8");
      const VisionInputs in(rgb.data(), dims[0], dims[1], kSlots, vision.i32("vision_buckets"), vision.f16("vision_positions"));
      const size_t count = in.rows * in.cols;
      const std::string name = "image " + std::to_string(i) + " ";
      ok &= report((name + "grid (rows + cols mismatch)").c_str(), std::fabs(double(in.rows) - dims[3]) + std::fabs(double(in.cols) - dims[2]), 0);
      if (in.rows != static_cast<size_t>(dims[3]) || in.cols != static_cast<size_t>(dims[2])) continue;
      // pixel_values [3, 14, count * 14]: pixel (c, ky, kx) of patch p at [c, ky, p * 14 + kx]. The resize
      // reproduces Pillow's fixed-point arithmetic, so the pixels match exactly.
      const auto pixels = load(prefix + "pixels.f32");
      double worst = 0;
      for (size_t c = 0; c < 3; ++c)
        for (size_t ky = 0; ky < kPatch; ++ky)
          for (size_t p = 0; p < count; ++p)
            for (size_t kx = 0; kx < kPatch; ++kx) {
              const float ours = in.patches[((c * kPatch + ky) * kPatch + kx) * kSlots + p];
              worst = std::max(worst, double(std::fabs(ours - pixels[(c * kPatch + ky) * count * kPatch + p * kPatch + kx])));
            }
      ok &= report((name + "pixels").c_str(), worst, 1e-6);
      const auto ids = loadAs<int32_t>(prefix + "ids.i32");
      size_t wrong = 0;
      for (size_t p = 0; p < count; ++p) wrong += in.positionId(vision.i32("vision_buckets"), p) != static_cast<size_t>(ids[p]);
      ok &= report((name + "position ids (wrong count)").c_str(), double(wrong), 0);
      const auto keys = load(prefix + "keys.f32");
      worst = 0;
      for (size_t p = 0; p < count; ++p)
        for (size_t k = 0; k < static_cast<size_t>(kDim); ++k)
          worst = std::max(worst, double(std::fabs(in.keys[k * kSlots + p] - keys[p * kDim + k])));
      ok &= report((name + "resampler positions").c_str(), worst, 1e-2);  // the official table is bf16
    }
  }
  return ok ? 0 : 1;
}
