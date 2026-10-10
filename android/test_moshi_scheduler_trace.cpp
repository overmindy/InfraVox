#include "moshi_lm_scheduler.h"
#include <fstream>
#include <iostream>
#include <string>

// Fixture layout: user[8], raw text, raw audio[8], temporal input[17], emitted[9].
int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("Usage: test_moshi_scheduler_trace FIXTURE");
    std::ifstream fixture(argv[1]);
    int frames = 0;
    if (!(fixture >> frames) || frames < 2) throw std::runtime_error("Invalid frame count");
    MoshiDelayRing ring;
    auto read = [&](int32_t& value) {
      if (!(fixture >> value)) throw std::runtime_error("Truncated fixture");
    };
    for (int frame = 0; frame < frames; ++frame) {
      std::array<int32_t, 8> user{}, audio{};
      std::array<int32_t, 17> expectedInput{};
      std::array<int32_t, 9> expectedOutput{};
      int32_t text;
      for (auto& value : user) read(value);
      read(text);
      for (auto& value : audio) read(value);
      for (auto& value : expectedInput) read(value);
      for (auto& value : expectedOutput) read(value);
      auto input = ring.begin(user);
      if (input != expectedInput)
        throw std::runtime_error("Temporal input mismatch at frame " + std::to_string(frame));
      auto output = ring.finish(text, audio);
      if (frame == 0) {
        if (output) throw std::runtime_error("Unexpected warmup output");
        for (auto value : expectedOutput)
          if (value != -1) throw std::runtime_error("Invalid warmup fixture");
      } else if (!output || *output != expectedOutput) {
        throw std::runtime_error("Decoder token alignment mismatch at frame " + std::to_string(frame));
      }
    }
    std::string extra;
    if (fixture >> extra) throw std::runtime_error("Extra fixture data");
    std::cout << "PASS: " << frames << " captured Temporal inputs and " << frames - 1
              << " delay-aligned decoder outputs match LMGen exactly\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
