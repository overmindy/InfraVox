// Minimal on-device Mimi DLC loopback: PCM16 WAV -> QNN encoder -> QNN decoder -> WAV.
// Uses the QAIRT 2.45 public QNN interfaces. Each graph owns its streaming state.

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "moshi_lm_scheduler.h"
#include "moshi_cache_prefix.h"

namespace {

constexpr size_t kFrameSamples = 1920;
constexpr uint32_t kSampleRate = 24000;

void check(Qnn_ErrorHandle_t status, const std::string& operation) {
  if (status != QNN_SUCCESS) {
    throw std::runtime_error(operation + " failed, QNN status " +
                             std::to_string(static_cast<unsigned long long>(status)));
  }
}

void* openLibrary(const std::string& path) {
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) throw std::runtime_error("dlopen " + path + ": " + dlerror());
  return handle;
}

template <typename Function>
Function symbol(void* handle, const char* name) {
  void* result = dlsym(handle, name);
  if (!result) throw std::runtime_error(std::string("dlsym ") + name + ": " + dlerror());
  return reinterpret_cast<Function>(result);
}

const char* tensorName(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_1) return tensor.v1.name;
  if (tensor.version == QNN_TENSOR_VERSION_2) return tensor.v2.name;
  throw std::runtime_error("Unsupported QNN tensor descriptor version");
}

Qnn_DataType_t tensorType(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_1) return tensor.v1.dataType;
  if (tensor.version == QNN_TENSOR_VERSION_2) return tensor.v2.dataType;
  throw std::runtime_error("Unsupported QNN tensor descriptor version");
}

uint32_t tensorRank(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_1) return tensor.v1.rank;
  if (tensor.version == QNN_TENSOR_VERSION_2) return tensor.v2.rank;
  throw std::runtime_error("Unsupported QNN tensor descriptor version");
}

const uint32_t* tensorDimensions(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_1) return tensor.v1.dimensions;
  if (tensor.version == QNN_TENSOR_VERSION_2) return tensor.v2.dimensions;
  throw std::runtime_error("Unsupported QNN tensor descriptor version");
}

void setClientBuffer(Qnn_Tensor_t& tensor, void* data, uint32_t length) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    tensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    tensor.v1.clientBuf = {data, length};
  } else if (tensor.version == QNN_TENSOR_VERSION_2) {
    tensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
    tensor.v2.clientBuf = {data, length};
  } else {
    throw std::runtime_error("Unsupported QNN tensor descriptor version");
  }
}

size_t tensorBytes(const Qnn_Tensor_t& tensor) {
  size_t width;
  switch (tensorType(tensor)) {
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
      width = 4;
      break;
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64:
      width = 8;
      break;
    default:
      throw std::runtime_error(std::string("Unsupported tensor datatype for ") +
                               tensorName(tensor) + ": " +
                               std::to_string(tensorType(tensor)));
  }
  size_t elements = 1;
  for (uint32_t i = 0; i < tensorRank(tensor); ++i) {
    const uint32_t dimension = tensorDimensions(tensor)[i];
    if (dimension == 0 || elements > SIZE_MAX / dimension) {
      throw std::runtime_error(std::string("Invalid tensor dimensions for ") +
                               tensorName(tensor));
    }
    elements *= dimension;
  }
  if (elements > SIZE_MAX / width) throw std::runtime_error("Tensor byte size overflow");
  return elements * width;
}

struct Buffer {
  Qnn_Tensor_t tensor{};
  std::vector<uint8_t> bytes;
  std::string name;
};

class Runtime {
 public:
  explicit Runtime(const std::string& libraryDir) {
    backendLibrary_ = openLibrary(libraryDir + "/libQnnHtp.so");
    systemLibrary_ = openLibrary(libraryDir + "/libQnnSystem.so");

    auto getQnn = symbol<decltype(&QnnInterface_getProviders)>(
        backendLibrary_, "QnnInterface_getProviders");
    const QnnInterface_t** providers = nullptr;
    uint32_t count = 0;
    check(getQnn(&providers, &count), "QnnInterface_getProviders");
    for (uint32_t i = 0; i < count; ++i) {
      const auto& version = providers[i]->apiVersion.coreApiVersion;
      if (version.major == QNN_API_VERSION_MAJOR &&
          version.minor >= QNN_API_VERSION_MINOR) {
        qnnProvider_ = providers[i];
        break;
      }
    }
    if (!qnnProvider_) throw std::runtime_error("No compatible QNN interface");
    qnn = qnnProvider_->QNN_INTERFACE_VER_NAME;

    auto getSystem = symbol<decltype(&QnnSystemInterface_getProviders)>(
        systemLibrary_, "QnnSystemInterface_getProviders");
    const QnnSystemInterface_t** systemProviders = nullptr;
    count = 0;
    check(getSystem(&systemProviders, &count), "QnnSystemInterface_getProviders");
    for (uint32_t i = 0; i < count; ++i) {
      const auto& version = systemProviders[i]->systemApiVersion;
      if (version.major == QNN_SYSTEM_API_VERSION_MAJOR &&
          version.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
        system = systemProviders[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
        break;
      }
    }
    if (!system.systemDlcCreateFromFile || !system.systemDlcComposeGraphs) {
      throw std::runtime_error("No compatible QNN System DLC interface");
    }
    check(qnn.backendCreate(nullptr, nullptr, &backend), "backendCreate");
    const auto deviceStatus = qnn.deviceCreate(nullptr, nullptr, &device);
    if (deviceStatus != QNN_SUCCESS &&
        deviceStatus != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
      check(deviceStatus, "deviceCreate");
    }
    std::cerr << "QNN HTP and System DLC interfaces ready\n";
  }

  ~Runtime() {
    if (device && qnn.deviceFree) qnn.deviceFree(device);
    if (backend && qnn.backendFree) qnn.backendFree(backend);
    // The HTP backend can retain process-wide cleanup hooks, so leave its SO loaded.
  }

  const QnnInterface_t& provider() const { return *qnnProvider_; }
  QNN_INTERFACE_VER_TYPE qnn{};
  QNN_SYSTEM_INTERFACE_VER_TYPE system{};
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;

 private:
  void* backendLibrary_ = nullptr;
  void* systemLibrary_ = nullptr;
  const QnnInterface_t* qnnProvider_ = nullptr;
};

class Graph {
 public:
  Graph(Runtime& runtime, const std::string& dlcPath) : runtime_(runtime) {
    try {
    check(runtime_.qnn.contextCreate(runtime_.backend, runtime_.device, nullptr, &context_),
          "contextCreate " + dlcPath);
    const auto fileStatus = runtime_.system.systemDlcCreateFromFile(nullptr, dlcPath.c_str(), &dlc_);
    if (fileStatus == QNN_COMMON_ERROR_MEM_ALLOC && runtime_.system.systemDlcCreateFromBinary) {
      // Retry a read-only mapping for SDK file-loader failures on external storage.
      // The binary buffer must stay alive until the DLC handle has been freed.
      std::cerr << "DLC file loader returned " << fileStatus
                << "; retrying read-only mapped binary: " << dlcPath << '\n';
      if (dlc_) { runtime_.system.systemDlcFree(dlc_); dlc_ = nullptr; }
      const int fd = open(dlcPath.c_str(), O_RDONLY | O_CLOEXEC);
      if (fd < 0) throw std::runtime_error("Cannot open DLC: " + dlcPath + ": " + std::strerror(errno));
      struct stat info{};
      if (fstat(fd, &info) != 0 || info.st_size <= 0 ||
          static_cast<uint64_t>(info.st_size) > SIZE_MAX) {
        close(fd); throw std::runtime_error("Invalid DLC size: " + dlcPath);
      }
      mappedSize_ = static_cast<size_t>(info.st_size);
      mappedDlc_ = mmap(nullptr, mappedSize_, PROT_READ, MAP_PRIVATE, fd, 0);
      const int mapError = errno;
      close(fd);
      if (mappedDlc_ == MAP_FAILED) {
        mappedDlc_ = nullptr;
        throw std::runtime_error("Cannot map DLC: " + dlcPath + ": " + std::strerror(mapError));
      }
      check(runtime_.system.systemDlcCreateFromBinary(nullptr,
                static_cast<const uint8_t*>(mappedDlc_), mappedSize_, &dlc_),
            "systemDlcCreateFromBinary " + dlcPath);
    } else {
      check(fileStatus, "systemDlcCreateFromFile " + dlcPath);
    }
    uint32_t count = 0;
    check(runtime_.system.systemDlcComposeGraphs(
              dlc_, nullptr, 0, runtime_.backend, context_, runtime_.provider(),
              QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1, &graphs_, &count),
          "systemDlcComposeGraphs " + dlcPath);
    if (count != 1 || !graphs_ || graphs_[0].version !=
                                  QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
      throw std::runtime_error("Expected one DLC graph with v1 graph metadata");
    }
    const auto& graphInfo = graphs_[0].graphInfoV1;
    name_ = graphInfo.graphName;
    check(runtime_.qnn.graphRetrieve(context_, name_.c_str(), &graph_),
          "graphRetrieve " + name_);
    check(runtime_.qnn.graphFinalize(graph_, nullptr, nullptr), "graphFinalize " + name_);
    prepare(graphInfo.graphInputs, graphInfo.numGraphInputs, inputs_, inputTensors_);
    prepare(graphInfo.graphOutputs, graphInfo.numGraphOutputs, outputs_, outputTensors_);
    std::cerr << dlcPath << " graph=" << name_ << '\n';
    printIO("input", inputs_);
    printIO("output", outputs_);
    } catch (...) {
      release();
      throw;
    }
  }

  ~Graph() {
    release();
  }

 private:
  void release() {
    if (context_ && runtime_.qnn.contextFree) runtime_.qnn.contextFree(context_, nullptr);
    if (graphs_) std::free(graphs_);
    if (dlc_ && runtime_.system.systemDlcFree) runtime_.system.systemDlcFree(dlc_);
    if (mappedDlc_) munmap(mappedDlc_, mappedSize_);
  }

 public:

  Buffer& input(const std::string& name) { return find(inputs_, name); }
  Buffer& output(const std::string& name) {
    for (auto& buffer : outputs_) {
      if (buffer.name == name) return buffer;
    }
    // AI Hub DLCs preserve input names, but export outputs as output_0..3.
    if (name == "codes" || name == "audio") return find(outputs_, "output_0");
    if (name == "position_out") return find(outputs_, "output_1");
    if (name == "kv_cache_out") return find(outputs_, "output_2");
    if (name == "conv_state_out") return find(outputs_, "output_3");
    throw std::runtime_error("Missing QNN graph output: " + name);
  }

  double execute() {
    const auto start = std::chrono::steady_clock::now();
    check(runtime_.qnn.graphExecute(graph_, inputTensors_.data(), inputTensors_.size(),
                                    outputTensors_.data(), outputTensors_.size(), nullptr,
                                    nullptr),
          "graphExecute " + name_);
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  }

  void advanceState() {
    for (const auto& name : {"position", "kv_cache", "conv_state"}) {
      auto& source = output(std::string(name) + "_out");
      auto& target = input(name);
      copy(source, target);
    }
  }

  void resetState() {
    for (const auto& name : {"position", "kv_cache", "conv_state"}) {
      auto& state = input(name);
      std::fill(state.bytes.begin(), state.bytes.end(), 0);
    }
  }

  static void copy(const Buffer& source, Buffer& target) {
    if (tensorType(source.tensor) != tensorType(target.tensor) ||
        source.bytes.size() != target.bytes.size()) {
      throw std::runtime_error("Incompatible tensor copy: " + source.name + " -> " + target.name);
    }
    std::memcpy(target.bytes.data(), source.bytes.data(), source.bytes.size());
  }

 private:
  static Buffer& find(std::vector<Buffer>& buffers, const std::string& name) {
    for (auto& buffer : buffers) {
      if (buffer.name == name) return buffer;
    }
    throw std::runtime_error("Missing QNN graph tensor: " + name);
  }

  static void prepare(Qnn_Tensor_t* metadata, uint32_t count,
                      std::vector<Buffer>& buffers, std::vector<Qnn_Tensor_t>& tensors) {
    if (count == 0) throw std::runtime_error("Graph has no tensors");
    buffers.resize(count);
    tensors.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
      auto& buffer = buffers[i];
      buffer.tensor = metadata[i];
      buffer.name = tensorName(metadata[i]);
      buffer.bytes.resize(tensorBytes(metadata[i]));
      if (buffer.bytes.size() > UINT32_MAX) throw std::runtime_error("QNN client buffer exceeds uint32");
      setClientBuffer(buffer.tensor, buffer.bytes.data(),
                      static_cast<uint32_t>(buffer.bytes.size()));
      tensors[i] = buffer.tensor;
    }
  }

  static void printIO(const char* kind, const std::vector<Buffer>& buffers) {
    for (const auto& buffer : buffers) {
      std::cerr << "  " << kind << ' ' << buffer.name << " dtype=" <<
          tensorType(buffer.tensor) << " version=" << buffer.tensor.version
                << " shape=[";
      for (uint32_t i = 0; i < tensorRank(buffer.tensor); ++i) {
        if (i) std::cerr << ',';
        std::cerr << tensorDimensions(buffer.tensor)[i];
      }
      std::cerr << "] bytes=" << buffer.bytes.size() << '\n';
    }
  }

  Runtime& runtime_;
  Qnn_ContextHandle_t context_ = nullptr;
  QnnSystemDlc_Handle_t dlc_ = nullptr;
  void* mappedDlc_ = nullptr;
  size_t mappedSize_ = 0;
  QnnSystemContext_GraphInfo_t* graphs_ = nullptr;
  Qnn_GraphHandle_t graph_ = nullptr;
  std::string name_;
  std::vector<Buffer> inputs_;
  std::vector<Buffer> outputs_;
  std::vector<Qnn_Tensor_t> inputTensors_;
  std::vector<Qnn_Tensor_t> outputTensors_;
};

uint16_t read16(const uint8_t* data) {
  return uint16_t(data[0]) | uint16_t(data[1]) << 8;
}

uint32_t read32(const uint8_t* data) {
  return uint32_t(data[0]) | uint32_t(data[1]) << 8 |
         uint32_t(data[2]) << 16 | uint32_t(data[3]) << 24;
}

void write16(std::ostream& output, uint16_t value) {
  output.put(char(value & 255));
  output.put(char((value >> 8) & 255));
}

void write32(std::ostream& output, uint32_t value) {
  write16(output, uint16_t(value & 65535));
  write16(output, uint16_t(value >> 16));
}

std::vector<int16_t> readWav(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open input WAV: " + path);
  std::vector<uint8_t> file((std::istreambuf_iterator<char>(input)), {});
  if (file.size() < 44 || std::memcmp(file.data(), "RIFF", 4) ||
      std::memcmp(file.data() + 8, "WAVE", 4)) {
    throw std::runtime_error("Expected RIFF/WAVE input");
  }
  bool formatSeen = false;
  std::vector<int16_t> samples;
  for (size_t cursor = 12; cursor + 8 <= file.size();) {
    const uint32_t length = read32(file.data() + cursor + 4);
    const size_t start = cursor + 8;
    if (length > file.size() - start) throw std::runtime_error("Malformed WAV chunk");
    if (!std::memcmp(file.data() + cursor, "fmt ", 4)) {
      if (length < 16 || read16(file.data() + start) != 1 ||
          read16(file.data() + start + 2) != 1 ||
          read32(file.data() + start + 4) != kSampleRate ||
          read16(file.data() + start + 14) != 16) {
        throw std::runtime_error("Expected 24 kHz mono PCM16 WAV");
      }
      formatSeen = true;
    } else if (!std::memcmp(file.data() + cursor, "data", 4)) {
      if (length % 2) throw std::runtime_error("Odd PCM16 data size");
      samples.resize(length / 2);
      for (size_t i = 0; i < samples.size(); ++i) {
        samples[i] = static_cast<int16_t>(read16(file.data() + start + 2 * i));
      }
    }
    cursor = start + length + (length & 1);
  }
  if (!formatSeen || samples.empty()) throw std::runtime_error("Missing WAV fmt or data chunk");
  return samples;
}

void writeWav(const std::string& path, const std::vector<int16_t>& samples) {
  const uint32_t dataBytes = uint32_t(samples.size() * 2);
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("Cannot create output WAV: " + path);
  output.write("RIFF", 4);
  write32(output, 36 + dataBytes);
  output.write("WAVEfmt ", 8);
  write32(output, 16);
  write16(output, 1);
  write16(output, 1);
  write32(output, kSampleRate);
  write32(output, kSampleRate * 2);
  write16(output, 2);
  write16(output, 16);
  output.write("data", 4);
  write32(output, dataBytes);
  for (int16_t sample : samples) write16(output, static_cast<uint16_t>(sample));
  if (!output) throw std::runtime_error("Failed writing output WAV");
}

void expect(Buffer& buffer, Qnn_DataType_t type, size_t bytes) {
  if (tensorType(buffer.tensor) != type || buffer.bytes.size() != bytes) {
    throw std::runtime_error("Unexpected tensor contract: " + buffer.name);
  }
}

}  // namespace

size_t runCodecFrames(Graph& encoder, Graph& decoder, const std::string& inputWav,
                      const std::string& outputWav, size_t requestedFrames) {
    const auto samples = readWav(inputWav);
    // Zero means every complete 80 ms frame in the WAV. Mimi's upstream KV
    // cache is a ring: its 250 slots wrap while position keeps advancing.
    if (requestedFrames == 0) {
      requestedFrames = samples.size() / kFrameSamples;
    }
    if (requestedFrames == 0 || requestedFrames > samples.size() / kFrameSamples ||
        requestedFrames > static_cast<size_t>(INT32_MAX / 2)) {
      throw std::runtime_error("Requested frames exceed available complete 80 ms WAV frames");
    }
    encoder.resetState();
    decoder.resetState();
    auto& audioInput = encoder.input("audio");
    auto& codesOutput = encoder.output("codes");
    auto& codesInput = decoder.input("codes");
    auto& audioOutput = decoder.output("audio");
    expect(audioInput, QNN_DATATYPE_FLOAT_32, kFrameSamples * sizeof(float));
    expect(codesOutput, QNN_DATATYPE_INT_32, 8 * sizeof(int32_t));
    expect(codesInput, QNN_DATATYPE_INT_32, 8 * sizeof(int32_t));
    expect(audioOutput, QNN_DATATYPE_FLOAT_32, kFrameSamples * sizeof(float));
    std::vector<int16_t> reconstructed;
    reconstructed.reserve(requestedFrames * kFrameSamples);
    std::ofstream codesFile(outputWav + ".codes.txt");
    if (!codesFile) throw std::runtime_error("Cannot create codes file");
    for (size_t frame = 0; frame < requestedFrames; ++frame) {
      std::array<float, kFrameSamples> pcm{};
      for (size_t i = 0; i < kFrameSamples; ++i) {
        pcm[i] = float(samples[frame * kFrameSamples + i]) / 32768.0f;
      }
      std::memcpy(audioInput.bytes.data(), pcm.data(), audioInput.bytes.size());
      const double encoderMs = encoder.execute();
      Graph::copy(codesOutput, codesInput);
      int32_t encoderPosition;
      std::memcpy(&encoderPosition, encoder.output("position_out").bytes.data(),
                  sizeof(encoderPosition));
      encoder.advanceState();
      const double decoderMs = decoder.execute();
      int32_t decoderPosition;
      std::memcpy(&decoderPosition, decoder.output("position_out").bytes.data(),
                  sizeof(decoderPosition));
      decoder.advanceState();
      const int32_t expectedPosition = static_cast<int32_t>(2 * (frame + 1));
      if (encoderPosition != expectedPosition || decoderPosition != expectedPosition) {
        throw std::runtime_error("Mimi streaming position did not advance by two");
      }
      for (size_t i = 0; i < kFrameSamples; ++i) {
        float value;
        std::memcpy(&value, audioOutput.bytes.data() + i * sizeof(float), sizeof(float));
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite decoder audio");
        reconstructed.push_back(static_cast<int16_t>(std::lrintf(
            std::clamp(value, -1.0f, 32767.0f / 32768.0f) * 32768.0f)));
      }
      codesFile << frame;
      for (size_t i = 0; i < 8; ++i) {
        int32_t code;
        std::memcpy(&code, codesOutput.bytes.data() + i * sizeof(int32_t), sizeof(code));
        codesFile << ',' << code;
      }
      codesFile << '\n';
      std::cerr << "frame=" << frame << " encoder_ms=" << encoderMs
                << " decoder_ms=" << decoderMs << '\n';
    }
    if (!codesFile) throw std::runtime_error("Failed writing codes file");
    writeWav(outputWav, reconstructed);
    std::cerr << "Wrote " << reconstructed.size() << " samples to " << outputWav << '\n';
    return requestedFrames;
}

// Keep one LM graph/context resident at a time. Full resident weights plus
// 64 full FP32 Temporal caches exceed many phone memory budgets. This is an offline
// correctness path, not a real-time implementation.
class PagedMoshiLm {
 public:
  PagedMoshiLm(Runtime& runtime, std::string directory)
      : runtime_(runtime), directory_(std::move(directory)) {
    for (const auto& name : {"frontend", "head"}) requireFile(name);
    for (int i = 0; i < 32; i += 2) requireFile(shardName(i));
    for (int i = 0; i < 8; ++i) requireFile("depformer_codebook_" + std::to_string(i));
  }

  void reset() {
    ring_.reset();
    for (auto& cache : temporalCache_) std::vector<uint8_t>().swap(cache);
  }

  std::optional<std::array<int32_t, 9>> step(const std::array<int32_t, 8>& user,
                                           int32_t position) {
    auto sequence = ring_.begin(user);
    std::vector<uint8_t> hidden;
    {
      Graph graph(runtime_, path("frontend"));
      set(graph.input("sequence"), sequence.data(), sizeof(sequence), QNN_DATATYPE_INT_32);
      graph.execute();
      finite(graph.output("hidden"));
      hidden = graph.output("hidden").bytes;
    }
    for (int layer = 0; layer < 32; layer += 2) {
      Graph graph(runtime_, path(shardName(layer)));
      set(graph.input("hidden"), hidden.data(), hidden.size(), QNN_DATATYPE_FLOAT_32);
      set(graph.input("position"), &position, sizeof(position), QNN_DATATYPE_INT_32);
      for (int j = 0; j < 4; ++j) {
        const std::string name = cacheName(layer + j / 2, j % 2);
        auto& input = graph.input(name);
        expect(input, QNN_DATATYPE_FLOAT_32, 1ULL * 32 * 3000 * 128 * sizeof(float));
        auto& cache = temporalCache_[2 * layer + j];
        // No wrap in this <=3000-frame host. Store only slots already written;
        // restore each head's prefix into a fresh zero-filled full ABI tensor.
        MoshiCachePrefix::restore(input.bytes, cache, 32, 3000, 128 * sizeof(float), position);
      }
      graph.execute();
      finite(graph.output("output_hidden"));
      hidden = graph.output("output_hidden").bytes;
      for (int j = 0; j < 4; ++j) {
        auto& output = graph.output("output_" + cacheName(layer + j / 2, j % 2));
        expect(output, QNN_DATATYPE_FLOAT_32, 1ULL * 32 * 3000 * 128 * sizeof(float));
        auto& cache = temporalCache_[2 * layer + j];
        MoshiCachePrefix::pack(output.bytes, cache, 32, 3000, 128 * sizeof(float), position + 1);
      }
    }
    std::vector<uint8_t> temporal;
    int32_t previous;
    {
      Graph graph(runtime_, path("head"));
      set(graph.input("hidden"), hidden.data(), hidden.size(), QNN_DATATYPE_FLOAT_32);
      graph.execute();
      finite(graph.output("temporal"));
      temporal = graph.output("temporal").bytes;
      auto& logits = graph.output("text_logits");
      expect(logits, QNN_DATATYPE_FLOAT_32, 32000 * sizeof(float));
      finite(logits);
      float best = -INFINITY;
      previous = 0;
      for (int32_t i = 0; i < 32000; ++i) {
        float value;
        std::memcpy(&value, logits.bytes.data() + i * sizeof(float), sizeof(value));
        if (value > best) { best = value; previous = i; }
      }
    }
    const int32_t text = previous;
    std::array<int32_t, 8> audio{};
    // Depth cache is reset EACH audio frame, but shared across all 8 codebooks.
    std::array<std::vector<uint8_t>, 12> depthCache;
    for (int book = 0; book < 8; ++book) {
      Graph graph(runtime_, path("depformer_codebook_" + std::to_string(book)));
      set(graph.input("previous_token"), &previous, sizeof(previous), QNN_DATATYPE_INT_32);
      set(graph.input("temporal"), temporal.data(), temporal.size(), QNN_DATATYPE_FLOAT_32);
      for (int j = 0; j < 12; ++j) {
        auto& input = graph.input(cacheName(j / 2, j % 2));
        expect(input, QNN_DATATYPE_FLOAT_32, 16 * 8 * 64 * sizeof(float));
        if (!depthCache[j].empty()) set(input, depthCache[j].data(), depthCache[j].size(), QNN_DATATYPE_FLOAT_32);
      }
      graph.execute();
      finite(graph.output("audio_logits"));
      auto& token = graph.output("audio_token");
      expect(token, QNN_DATATYPE_INT_32, sizeof(int32_t));
      std::memcpy(&previous, token.bytes.data(), sizeof(previous));
      audio[book] = previous;
      for (int j = 0; j < 12; ++j) {
        auto& output = graph.output("output_" + cacheName(j / 2, j % 2));
        expect(output, QNN_DATATYPE_FLOAT_32, 16 * 8 * 64 * sizeof(float));
        depthCache[j] = output.bytes;
      }
    }
    return ring_.finish(text, audio);
  }

 private:
  static void set(Buffer& buffer, const void* data, size_t size, Qnn_DataType_t type) {
    expect(buffer, type, size);
    std::memcpy(buffer.bytes.data(), data, size);
  }
  static void finite(const Buffer& buffer) {
    if (tensorType(buffer.tensor) != QNN_DATATYPE_FLOAT_32 || buffer.bytes.size() % sizeof(float))
      throw std::runtime_error("Expected FP32 LM output: " + buffer.name);
    for (size_t i = 0; i < buffer.bytes.size(); i += sizeof(float)) {
      float value;
      std::memcpy(&value, buffer.bytes.data() + i, sizeof(value));
      if (!std::isfinite(value)) throw std::runtime_error("Nonfinite LM output: " + buffer.name);
    }
  }
  static std::string cacheName(int layer, int value) {
    return "layer_" + std::to_string(layer) + (value ? "_value" : "_key");
  }
  static std::string shardName(int layer) {
    return "temporal_layers_" + std::to_string(layer) + "_" + std::to_string(layer + 1);
  }
  std::string path(const std::string& name) const { return directory_ + "/dlc/" + name + ".dlc"; }
  void requireFile(const std::string& name) const {
    std::ifstream file(path(name), std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0) throw std::runtime_error("Missing LM DLC: " + path(name));
  }
  Runtime& runtime_;
  std::string directory_;
  MoshiDelayRing ring_;
  std::array<std::vector<uint8_t>, 64> temporalCache_;
};

size_t runMoshiFrames(Graph& encoder, Graph& decoder, PagedMoshiLm& lm,
                      const std::string& inputWav, const std::string& outputWav,
                      size_t requestedFrames) {
  const auto samples = readWav(inputWav);
  if (!requestedFrames) requestedFrames = samples.size() / kFrameSamples;
  if (requestedFrames < 2 || requestedFrames > samples.size() / kFrameSamples || requestedFrames > 3000)
    throw std::runtime_error("Moshi needs 2..3000 complete frames (one warm-up frame)");
  encoder.resetState(); decoder.resetState(); lm.reset();
  std::vector<int16_t> result;
  result.reserve((requestedFrames - 1) * kFrameSamples);
  std::ofstream codes(outputWav + ".codes.txt");
  std::ofstream text(outputWav + ".text_tokens.txt");
  if (!codes || !text) throw std::runtime_error("Cannot write Moshi token files");
  size_t emitted = 0;
  for (size_t frame = 0; frame < requestedFrames; ++frame) {
    auto start = std::chrono::steady_clock::now();
    auto& input = encoder.input("audio");
    expect(input, QNN_DATATYPE_FLOAT_32, kFrameSamples * sizeof(float));
    for (size_t i = 0; i < kFrameSamples; ++i) {
      const float value = samples[frame * kFrameSamples + i] / 32768.0f;
      std::memcpy(input.bytes.data() + i * sizeof(float), &value, sizeof(value));
    }
    encoder.execute();
    std::array<int32_t, 8> user{};
    auto& encoded = encoder.output("codes");
    expect(encoded, QNN_DATATYPE_INT_32, sizeof(user));
    std::memcpy(user.data(), encoded.bytes.data(), sizeof(user));
    int32_t encoderPosition;
    auto& encoderPositionOutput = encoder.output("position_out");
    expect(encoderPositionOutput, QNN_DATATYPE_INT_32, sizeof(encoderPosition));
    std::memcpy(&encoderPosition, encoderPositionOutput.bytes.data(), sizeof(encoderPosition));
    if (encoderPosition != 2 * static_cast<int32_t>(frame + 1))
      throw std::runtime_error("Moshi encoder streaming position did not advance by two");
    encoder.advanceState();
    auto generated = lm.step(user, static_cast<int32_t>(frame));
    if (generated) {
      auto& decoderCodes = decoder.input("codes");
      expect(decoderCodes, QNN_DATATYPE_INT_32, 8 * sizeof(int32_t));
      std::memcpy(decoderCodes.bytes.data(), generated->data() + 1, 8 * sizeof(int32_t));
      decoder.execute();
      int32_t decoderPosition;
      auto& decoderPositionOutput = decoder.output("position_out");
      expect(decoderPositionOutput, QNN_DATATYPE_INT_32, sizeof(decoderPosition));
      std::memcpy(&decoderPosition, decoderPositionOutput.bytes.data(), sizeof(decoderPosition));
      if (decoderPosition != 2 * static_cast<int32_t>(emitted + 1))
        throw std::runtime_error("Moshi decoder streaming position did not advance by two");
      decoder.advanceState();
      auto& wave = decoder.output("audio");
      expect(wave, QNN_DATATYPE_FLOAT_32, kFrameSamples * sizeof(float));
      for (size_t i = 0; i < kFrameSamples; ++i) {
        float value;
        std::memcpy(&value, wave.bytes.data() + i * sizeof(float), sizeof(value));
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite Moshi decoder audio");
        result.push_back(static_cast<int16_t>(std::lrintf(std::clamp(value, -1.0f, 32767.0f / 32768.0f) * 32768)));
      }
      codes << emitted;
      for (size_t i = 1; i < 9; ++i) codes << ',' << (*generated)[i];
      codes << '\n'; text << emitted << ',' << (*generated)[0] << '\n';
      ++emitted;
    }
    std::cerr << "moshi_frame=" << frame << " emitted=" << emitted << " total_ms="
              << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() << '\n';
  }
  if (!codes || !text) throw std::runtime_error("Failed writing Moshi tokens");
  writeWav(outputWav, result);
  return emitted;
}

class PreparedMimiCodec {
 public:
  PreparedMimiCodec(const std::string& libraryDir, const std::string& encoderDlc,
                    const std::string& decoderDlc, const std::string& lmDirectory = "")
      : runtime_(libraryDir), encoder_(runtime_, encoderDlc),
        decoder_(runtime_, decoderDlc) {
    if (!lmDirectory.empty()) lm_ = std::make_unique<PagedMoshiLm>(runtime_, lmDirectory);
  }

  size_t run(const std::string& inputWav, const std::string& outputWav,
             size_t requestedFrames) {
    if (lm_) return runMoshiFrames(encoder_, decoder_, *lm_, inputWav, outputWav, requestedFrames);
    return runCodecFrames(encoder_, decoder_, inputWav, outputWav, requestedFrames);
  }

 private:
  Runtime runtime_;
  Graph encoder_;
  Graph decoder_;
  std::unique_ptr<PagedMoshiLm> lm_;
};

void* createMimiCodecSession(const std::string& libraryDir,
                             const std::string& encoderDlc,
                             const std::string& decoderDlc) {
  return new PreparedMimiCodec(libraryDir, encoderDlc, decoderDlc);
}

void* createMoshiCodecSession(const std::string& libraryDir,
                              const std::string& encoderDlc,
                              const std::string& decoderDlc,
                              const std::string& lmDirectory) {
  return new PreparedMimiCodec(libraryDir, encoderDlc, decoderDlc, lmDirectory);
}

size_t runMimiCodecSession(void* session, const std::string& inputWav,
                           const std::string& outputWav, size_t requestedFrames) {
  if (!session) throw std::runtime_error("Mimi DLCs have not been loaded");
  return static_cast<PreparedMimiCodec*>(session)->run(inputWav, outputWav,
                                                       requestedFrames);
}

void destroyMimiCodecSession(void* session) {
  delete static_cast<PreparedMimiCodec*>(session);
}

size_t runMimiCodec(const std::string& libraryDir, const std::string& encoderDlc,
                    const std::string& decoderDlc, const std::string& inputWav,
                    const std::string& outputWav, size_t requestedFrames) {
  PreparedMimiCodec session(libraryDir, encoderDlc, decoderDlc);
  return session.run(inputWav, outputWav, requestedFrames);
}

#ifndef MIMI_CODEC_LIBRARY
int main(int argc, char** argv) {
  try {
    if (argc != 7 && argc != 8) {
      std::cerr << "Usage: mimi_qnn_runner LIB_DIR ENCODER.dlc DECODER.dlc "
                   "INPUT.wav OUTPUT.wav FRAMES [LM_BUNDLE_DIR] (0 = all complete frames)\n";
      return 2;
    }
    if (argc == 8) {
      PreparedMimiCodec session(argv[1], argv[2], argv[3], argv[7]);
      session.run(argv[4], argv[5], std::stoul(argv[6]));
    } else {
      runMimiCodec(argv[1], argv[2], argv[3], argv[4], argv[5], std::stoul(argv[6]));
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Mimi QNN runner error: " << error.what() << '\n';
    return 1;
  }
}
#endif
