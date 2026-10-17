// Compare codec precision using identical generated audio tokens and state resets.
#define MIMI_CODEC_LIBRARY
#include "mimi_qnn_runner.cpp"

int main(int argc, char** argv) {
  try {
    if (argc != 5) throw std::runtime_error("Usage: decoder_probe LIB MODEL.bin CODES.csv OUTPUT.wav");
    Runtime runtime(argv[1]);
    Graph decoder(runtime, argv[2]);
    decoder.resetState();
    std::ifstream source(argv[3]);
    if (!source) throw std::runtime_error("Cannot read generated codes");
    std::vector<int16_t> pcm;
    std::string line;
    size_t frame = 0;
    while (std::getline(source, line)) {
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream row(line);
      size_t index;
      std::array<int32_t, 8> codes{};
      if (!(row >> index) || index != frame) throw std::runtime_error("Nonconsecutive code rows");
      for (auto& code : codes)
        if (!(row >> code) || code < 0 || code >= 2048) throw std::runtime_error("Invalid audio code");
      std::string extra;
      if (row >> extra) throw std::runtime_error("Unexpected code column");
      auto& input = decoder.input("codes");
      expect(input, QNN_DATATYPE_INT_32, sizeof(codes));
      std::memcpy(input.bytes.data(), codes.data(), sizeof(codes));
      decoder.execute();
      auto& output = decoder.output("audio");
      expect(output, QNN_DATATYPE_FLOAT_32, kFrameSamples*sizeof(float));
      for (size_t i = 0; i < kFrameSamples; ++i) {
        float value;
        std::memcpy(&value, output.bytes.data()+i*sizeof(float), sizeof(float));
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite decoded audio");
        pcm.push_back(static_cast<int16_t>(std::lround(std::clamp(value, -1.0f, 32767.0f/32768)*32768)));
      }
      decoder.advanceState();
      ++frame;
    }
    if (!frame) throw std::runtime_error("Empty code sequence");
    writeWav(argv[4], pcm);
    runtime.writeTimings(std::string(argv[4])+".performance.csv");
    std::cout << "DECODED_FRAMES=" << frame << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "DECODER_PROBE_FAILED: " << error.what() << '\n';
    return 1;
  }
}
