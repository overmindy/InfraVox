#include "moshi_graph_io_pool.h"
#include <cassert>
#include <iostream>

int main() {
  GraphIoPool pool;
  std::vector<uint8_t> first, second, output;
  pool.take("input", 0, 1024, first);
  auto* original = first.data();
  std::fill(first.begin(), first.end(), 42);
  pool.put("input", 0, first);
  assert(first.empty());
  pool.take("input", 0, 1024, second);
  assert(second.data() == original);
  for (auto value : second) assert(value == 0);
  pool.take("output", 0, 1024, output);
  assert(output.data() != second.data());
  pool.put("input", 0, second);
  pool.take("input", 0, 2048, first);
  assert(first.size() == 2048); // Different ABI cannot accidentally reuse 1024 bytes.
  pool.put("input", 0, first);
  pool.put("input", 0, first); // Failure cleanup/destructor is idempotent.
  std::cout << "PASS: reused pointer, zero reset, separate input/output, ABI sizes, repeated cleanup\n";
}
