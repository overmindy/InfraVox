// Time one single-graph QNN context binary on device with zero inputs.
// Inputs of 1 MiB or more are bound to registered shared memory, as the runner does for KV.
#define MIMI_CODEC_LIBRARY
#include "mimi_qnn_runner.cpp"

int main(int argc, char** argv) {
  try {
    if (argc < 3 || argc > 5) {
      std::cerr << "Usage: moshi_graph_bench LIB_DIR MODEL.bin [ITERATIONS=20] [TIMINGS.csv]\n";
      return 2;
    }
    const int iterations = argc >= 4 ? std::atoi(argv[3]) : 20;
    Runtime runtime(argv[1]);
    if (environmentLevel("MOSHI_HTP_PERFORMANCE", 1)) runtime.beginPerformanceVote();
    const auto load = Clock::now();
    Graph graph(runtime, argv[2]);
    std::cout << "load_ms=" << elapsedMs(load) << '\n';

    std::vector<std::string> large;
    for (const auto& input : graph.inputs()) {
      std::cout << "input " << input.name << " dtype=0x" << std::hex << tensorType(input.tensor) << std::dec
                << " bytes=" << input.bytes.size() << '\n';
      if (input.bytes.size() >= (1u << 20)) large.push_back(input.name);
    }
    for (const auto& output : graph.outputs())
      std::cout << "output " << output.name << " dtype=0x" << std::hex << tensorType(output.tensor) << std::dec
                << " bytes=" << output.bytes.size() << '\n';
    std::vector<std::unique_ptr<SharedBuffer>> shared;
    for (const auto& name : large) {
      size_t bytes = 0;
      for (const auto& input : graph.inputs())
        if (input.name == name) bytes = input.bytes.size();
      shared.push_back(std::make_unique<SharedBuffer>(runtime.sharedMemory, bytes));
      graph.bindInput(name, *shared.back());
    }
    std::cout << "shared_inputs=" << graph.sharedInputs() << '/' << large.size() << '\n';

    std::vector<double> times;
    for (int i = 0; i < iterations; ++i) {
      const auto start = Clock::now();
      graph.execute();
      times.push_back(elapsedMs(start));
    }
    runtime.endPerformanceVote();
    if (argc == 5) runtime.writeTimings(argv[4]);
    std::vector<double> sorted(times.begin() + std::min<size_t>(2, times.size() - 1), times.end());
    std::sort(sorted.begin(), sorted.end());
    std::cout << "execute_ms";
    for (double t : times) std::cout << ' ' << t;
    std::cout << "\nmedian_ms=" << sorted[sorted.size() / 2] << " min_ms=" << sorted.front()
              << " (first two iterations excluded)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "moshi_graph_bench error: " << error.what() << '\n';
    return 1;
  }
}
