// Run Temporal v2 parts in sequence on a captured layer-0 stream and save the final hidden states.
// INPUT.f32 / OUTPUT.f32 are raw float32 [frames, 4096]; the server decodes OUTPUT into tokens.
#define MIMI_CODEC_LIBRARY
#include "mimi_qnn_runner.cpp"

int main(int argc, char** argv) {
  try {
    if (argc < 5) {
      std::cerr << "Usage: moshi_temporal_chain_probe LIB_DIR INPUT.f32 OUTPUT.f32 PART0.bin [PART1.bin ...]\n";
      return 2;
    }
    Runtime runtime(argv[1]);
    std::ifstream input(argv[2], std::ios::binary | std::ios::ate);
    const size_t bytes = static_cast<size_t>(input.tellg());
    if (!bytes || bytes % (TemporalPartV2::kDim * sizeof(float))) throw std::runtime_error("INPUT must be float32 [frames, 4096]");
    std::vector<float> frames(bytes / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(frames.data()), static_cast<std::streamsize>(bytes));
    const size_t count = frames.size() / TemporalPartV2::kDim;

    std::vector<std::unique_ptr<TemporalPartV2>> parts;
    for (int i = 4; i < argc; ++i) {
      const auto start = Clock::now();
      parts.push_back(std::make_unique<TemporalPartV2>(runtime, argv[i]));
      std::cout << "loaded " << parts.back()->label() << " shared_kv=" << parts.back()->sharedInputs()
                << " ms=" << elapsedMs(start) << '\n';
    }
    runtime.beginPerformanceVote();
    std::vector<float> output(frames.size()), a(TemporalPartV2::kDim), b(TemporalPartV2::kDim);
    std::vector<double> frameMs;
    for (size_t t = 0; t < count; ++t) {
      const auto start = Clock::now();
      std::memcpy(a.data(), frames.data() + t * TemporalPartV2::kDim, a.size() * sizeof(float));
      for (auto& part : parts) {
        part->step(static_cast<int32_t>(t), a.data(), b.data());
        a.swap(b);
      }
      frameMs.push_back(elapsedMs(start));
      std::memcpy(output.data() + t * TemporalPartV2::kDim, a.data(), a.size() * sizeof(float));
    }
    runtime.endPerformanceVote();
    std::ofstream out(argv[3], std::ios::binary);
    out.write(reinterpret_cast<const char*>(output.data()), static_cast<std::streamsize>(output.size() * sizeof(float)));
    if (!out) throw std::runtime_error("Cannot write OUTPUT");
    std::vector<double> sorted(frameMs.begin() + std::min<size_t>(2, frameMs.size() - 1), frameMs.end());
    std::sort(sorted.begin(), sorted.end());
    std::cout << "frames=" << count << " temporal_step_median_ms=" << sorted[sorted.size() / 2]
              << " min_ms=" << sorted.front() << " max_ms=" << sorted.back() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "moshi_temporal_chain_probe error: " << error.what() << '\n';
    return 1;
  }
}
