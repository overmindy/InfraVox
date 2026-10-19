// Measure the App's native reply path without changing its models or settings.
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

void* createConfiguredCodecSession(const std::string&, const std::string&,
    const std::string&, const std::string&, const std::string&, bool);
size_t runMimiCodecSessionStreaming(void*, const std::string&, const std::string&, size_t,
    const std::function<void(size_t, size_t, double, const std::vector<int16_t>&)>&, bool);
void destroyMimiCodecSession(void*);

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

int main(int argc, char** argv) {
  try {
    if (argc != 9) throw std::runtime_error(
        "Usage: probe LIB ENCODER DECODER LM CACHE INPUT OUTPUT_PREFIX FRAMES; use 0 for the full input");
    const std::string prefix = argv[7];
    const auto frameCount = std::stoul(argv[8]);
    if (frameCount == 1 || frameCount > 3000) throw std::runtime_error("Use 0 or 2..3000 frames");
    setenv("ADSP_LIBRARY_PATH", argv[1], 1);
    std::ofstream report(prefix + ".latency.json");
    if (!report) throw std::runtime_error("Cannot create the latency report");
    const auto processStart = Clock::now();
    std::unique_ptr<void, decltype(&destroyMimiCodecSession)> session(
        createConfiguredCodecSession(argv[1], argv[2], argv[3], argv[4], argv[5], true),
        &destroyMimiCodecSession);
    const double prepare = elapsed(processStart);
    report << std::fixed << std::setprecision(3)
           << "{\"format\":\"moshi-reply-latency-v1\",\"prepare_ms\":" << prepare
           << ",\"measurement\":\"PCM callback readiness; excludes Java UI and speaker latency\",\"runs\":[";
    std::cout << "LATENCY_PREPARE ms=" << prepare << std::endl;
    for (int repeat = 0; repeat < 2; ++repeat) {
      const std::string name = repeat ? "warm" : "first";
      const std::string output = prefix + "." + name + ".wav";
      std::ofstream timeline(prefix + "." + name + ".timeline.csv");
      if (!timeline) throw std::runtime_error("Cannot create the frame timeline");
      timeline << "consumed_frames,emitted_frames,elapsed_ms,frame_compute_ms,pcm_samples\n";
      double firstPcm = -1, lastPcm = -1;
      size_t callbacks = 0, samples = 0;
      const auto runStart = Clock::now();
      const size_t emitted = runMimiCodecSessionStreaming(session.get(), argv[6], output, frameCount,
          [&](size_t consumed, size_t produced, double frameMs, const std::vector<int16_t>& pcm) {
            const double now = elapsed(runStart);
            ++callbacks; samples += pcm.size();
            if (!pcm.empty()) {
              if (firstPcm < 0) firstPcm = now;
              lastPcm = now;
            }
            timeline << consumed << ',' << produced << ',' << now << ',' << frameMs << ',' << pcm.size() << '\n';
            timeline.flush();
            std::cout << "LATENCY_FRAME run=" << name << " consumed=" << consumed
                      << " emitted=" << produced << " elapsed_ms=" << now
                      << " pcm_samples=" << pcm.size() << std::endl;
          }, true);
      const double total = elapsed(runStart);
      if (!timeline || !emitted || firstPcm < 0 || samples != emitted * 1920)
        throw std::runtime_error("The run did not produce the expected PCM frames");
      if (repeat) report << ',';
      report << "{\"name\":\"" << name << "\",\"input_frames\":" << callbacks
             << ",\"output_frames\":" << emitted << ",\"first_pcm_ms\":" << firstPcm
             << ",\"last_pcm_ms\":" << lastPcm << ",\"total_ms\":" << total
             << ",\"output_seconds\":" << samples / 24000.0 << '}';
      report.flush();
      std::cout << "LATENCY_RESULT run=" << name << " first_pcm_ms=" << firstPcm
                << " total_ms=" << total << " output_frames=" << emitted << std::endl;
    }
    report << "]}\n";
    if (!report) throw std::runtime_error("Cannot write the latency report");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "LATENCY_FAILED: " << error.what() << '\n';
    return 1;
  }
}
