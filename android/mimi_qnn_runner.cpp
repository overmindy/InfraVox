// On-device Moshi / Mimi runner over QNN HTP context binaries (QAIRT 2.45 APIs).
// Every model is a device-specific .bin context; all graphs stay resident for the session.

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpSystemContext.h"
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "moshi_lm_scheduler.h"

namespace {

constexpr size_t kFrameSamples = 1920;  // 80 ms at 24 kHz

using Clock = std::chrono::steady_clock;
double elapsedMs(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void check(Qnn_ErrorHandle_t status, const std::string& operation) {
  if (status != QNN_SUCCESS)
    throw std::runtime_error(operation + " failed, QNN status " + std::to_string(status));
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

// QNN tensors are v1 or v2; both share the fields used here.
#define TENSOR_FIELD(tensor, field) \
  ((tensor).version == QNN_TENSOR_VERSION_2 ? (tensor).v2.field : (tensor).v1.field)

const char* tensorName(const Qnn_Tensor_t& t) { return TENSOR_FIELD(t, name); }
Qnn_DataType_t tensorType(const Qnn_Tensor_t& t) { return TENSOR_FIELD(t, dataType); }
uint32_t tensorRank(const Qnn_Tensor_t& t) { return TENSOR_FIELD(t, rank); }
const uint32_t* tensorDimensions(const Qnn_Tensor_t& t) { return TENSOR_FIELD(t, dimensions); }

void setClientBuffer(Qnn_Tensor_t& t, void* data, size_t length) {
  if (length > UINT32_MAX) throw std::runtime_error("QNN client buffer exceeds uint32");
  const Qnn_ClientBuffer_t buffer = {data, static_cast<uint32_t>(length)};
  if (t.version == QNN_TENSOR_VERSION_2) { t.v2.memType = QNN_TENSORMEMTYPE_RAW; t.v2.clientBuf = buffer; }
  else { t.v1.memType = QNN_TENSORMEMTYPE_RAW; t.v1.clientBuf = buffer; }
}

void setMemHandle(Qnn_Tensor_t& t, Qnn_MemHandle_t handle) {
  if (t.version == QNN_TENSOR_VERSION_2) { t.v2.memType = QNN_TENSORMEMTYPE_MEMHANDLE; t.v2.memHandle = handle; }
  else { t.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE; t.v1.memHandle = handle; }
}

size_t tensorBytes(const Qnn_Tensor_t& t) {
  size_t width;
  switch (tensorType(t)) {
    case QNN_DATATYPE_UFIXED_POINT_8: case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UINT_8: case QNN_DATATYPE_INT_8: case QNN_DATATYPE_BOOL_8: width = 1; break;
    case QNN_DATATYPE_UFIXED_POINT_16: case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UINT_16: case QNN_DATATYPE_INT_16: case QNN_DATATYPE_FLOAT_16: width = 2; break;
    case QNN_DATATYPE_FLOAT_32: case QNN_DATATYPE_INT_32: case QNN_DATATYPE_UINT_32: width = 4; break;
    case QNN_DATATYPE_INT_64: case QNN_DATATYPE_UINT_64: width = 8; break;
    default: throw std::runtime_error(std::string("Unsupported tensor datatype for ") + tensorName(t));
  }
  size_t elements = 1;
  for (uint32_t i = 0; i < tensorRank(t); ++i) elements *= tensorDimensions(t)[i];
  return elements * width;
}

struct MappedFile {
  void* data = MAP_FAILED;
  size_t size = 0;
  explicit MappedFile(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    struct stat info{};
    if (fd < 0 || fstat(fd, &info) || info.st_size <= 0) {
      if (fd >= 0) close(fd);
      throw std::runtime_error("Missing or unreadable model file (check 644 permissions): " + path);
    }
    size = static_cast<size_t>(info.st_size);
    data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
  }
  ~MappedFile() { if (data != MAP_FAILED) munmap(data, size); }
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
};

// Single-graph metadata from a context binary, independent of the metadata struct version.
struct BinaryGraphInfo {
  std::string name;
  Qnn_Tensor_t* inputs = nullptr; uint32_t numInputs = 0;
  Qnn_Tensor_t* outputs = nullptr; uint32_t numOutputs = 0;
  uint64_t spillFillBytes = 0;
};

BinaryGraphInfo readGraphInfo(const QnnSystemContext_BinaryInfo_t* info, const std::string& path) {
  QnnSystemContext_GraphInfo_t* graphs = nullptr;
  uint32_t count = 0;
  switch (info->version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: graphs = info->contextBinaryInfoV1.graphs; count = info->contextBinaryInfoV1.numGraphs; break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: graphs = info->contextBinaryInfoV2.graphs; count = info->contextBinaryInfoV2.numGraphs; break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: graphs = info->contextBinaryInfoV3.graphs; count = info->contextBinaryInfoV3.numGraphs; break;
    default: break;
  }
  if (count != 1 || !graphs) throw std::runtime_error("Expected exactly one graph in " + path);
  BinaryGraphInfo out;
  const auto fill = [&](const auto& g) {
    out.name = g.graphName;
    out.inputs = g.graphInputs; out.numInputs = g.numGraphInputs;
    out.outputs = g.graphOutputs; out.numOutputs = g.numGraphOutputs;
  };
  switch (graphs[0].version) {
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1: fill(graphs[0].graphInfoV1); break;
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2: fill(graphs[0].graphInfoV2); break;
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3: {
      const auto& g = graphs[0].graphInfoV3;
      fill(g);
      if (g.graphBlobInfo && g.graphBlobInfoSize >= sizeof(QnnHtpSystemContext_GraphBlobInfo_t)) {
        const auto* htp = static_cast<const QnnHtpSystemContext_GraphBlobInfo_t*>(g.graphBlobInfo);
        if (htp->version == QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1)
          out.spillFillBytes = htp->contextBinaryGraphBlobInfoV1.spillFillBufferSize;
      }
      break;
    }
    default: throw std::runtime_error("Unsupported graph metadata in " + path);
  }
  return out;
}

// rpcmem (ION / DMA-BUF) lets HTP read large inputs without a per-execute FastRPC copy.
class SharedMemory {
 public:
  SharedMemory() {
    library_ = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (!library_) return;
    alloc_ = reinterpret_cast<Alloc>(dlsym(library_, "rpcmem_alloc"));
    free_ = reinterpret_cast<Free>(dlsym(library_, "rpcmem_free"));
    toFd_ = reinterpret_cast<ToFd>(dlsym(library_, "rpcmem_to_fd"));
  }
  bool available() const { return alloc_ && free_ && toFd_; }
  void* allocate(size_t bytes, int& fd) {
    fd = -1;
    if (!available() || bytes > INT32_MAX) return nullptr;
    void* data = alloc_(kHeapSystem, kDefaultFlags, static_cast<int>(bytes));
    if (data) fd = toFd_(data);
    if (data && fd < 0) { free_(data); data = nullptr; }
    return data;
  }
  void release(void* data) { if (data && free_) free_(data); }

 private:
  static constexpr int kHeapSystem = 25;
  static constexpr uint32_t kDefaultFlags = 1;
  using Alloc = void* (*)(int, uint32_t, int);
  using Free = void (*)(void*);
  using ToFd = int (*)(void*);
  void* library_ = nullptr;
  Alloc alloc_ = nullptr;
  Free free_ = nullptr;
  ToFd toFd_ = nullptr;
};

// Zeroed host buffer in rpcmem when available, so HTP reads it without a per-execute copy.
class SharedBuffer {
 public:
  SharedBuffer(SharedMemory& memory, size_t bytes) : memory_(memory), bytes_(bytes) {
    data_ = static_cast<uint8_t*>(memory_.allocate(bytes_, fd_));
    if (!data_) { fallback_.resize(bytes_); data_ = fallback_.data(); }
    std::memset(data_, 0, bytes_);
  }
  ~SharedBuffer() { if (fd_ >= 0) memory_.release(data_); }
  SharedBuffer(const SharedBuffer&) = delete;
  SharedBuffer& operator=(const SharedBuffer&) = delete;
  uint8_t* data() const { return data_; }
  size_t bytes() const { return bytes_; }
  int fd() const { return fd_; }

 private:
  SharedMemory& memory_;
  size_t bytes_;
  int fd_ = -1;
  uint8_t* data_ = nullptr;
  std::vector<uint8_t> fallback_;
};

class Runtime {
 public:
  explicit Runtime(const std::string& libraryDir) {
    backendLibrary_ = openLibrary(libraryDir + "/libQnnHtp.so");
    systemLibrary_ = openLibrary(libraryDir + "/libQnnSystem.so");
    const QnnInterface_t** providers = nullptr;
    uint32_t count = 0;
    check(symbol<decltype(&QnnInterface_getProviders)>(backendLibrary_, "QnnInterface_getProviders")(&providers, &count),
          "QnnInterface_getProviders");
    for (uint32_t i = 0; i < count && !provider_; ++i) {
      const auto& v = providers[i]->apiVersion.coreApiVersion;
      if (v.major == QNN_API_VERSION_MAJOR && v.minor >= QNN_API_VERSION_MINOR) provider_ = providers[i];
    }
    if (!provider_) throw std::runtime_error("No compatible QNN interface");
    qnn = provider_->QNN_INTERFACE_VER_NAME;

    const QnnSystemInterface_t** systemProviders = nullptr;
    count = 0;
    check(symbol<decltype(&QnnSystemInterface_getProviders)>(systemLibrary_, "QnnSystemInterface_getProviders")(
              &systemProviders, &count), "QnnSystemInterface_getProviders");
    for (uint32_t i = 0; i < count; ++i) {
      const auto& v = systemProviders[i]->systemApiVersion;
      if (v.major == QNN_SYSTEM_API_VERSION_MAJOR && v.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
        system = systemProviders[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
        break;
      }
    }
    if (!system.systemContextCreate || !system.systemContextGetBinaryInfo || !qnn.contextCreateFromBinary)
      throw std::runtime_error("No compatible QNN System context interface");
    check(qnn.backendCreate(nullptr, nullptr, &backend), "backendCreate");
    const auto status = qnn.deviceCreate(nullptr, nullptr, &device);
    if (status != QNN_SUCCESS && status != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) check(status, "deviceCreate");
  }

  ~Runtime() {
    endPerformanceVote();
    if (device && qnn.deviceFree) qnn.deviceFree(device);
    if (backend && qnn.backendFree) qnn.backendFree(backend);
    // The HTP backend can retain process-wide cleanup hooks, so its library stays loaded.
  }

  // Burst profile as in qnn-net-run: fixed max corners, DCVS off, short sleep latency,
  // and RPC polling. Thermal mitigation stays with the firmware.
  void beginPerformanceVote() {
    if (hasPowerConfig_) return;
    QnnDevice_Infrastructure_t infrastructure = nullptr;
    if (!qnn.deviceGetInfrastructure || qnn.deviceGetInfrastructure(&infrastructure) != QNN_SUCCESS ||
        !infrastructure || infrastructure->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) return;
    perf_ = infrastructure->perfInfra;
    if (!perf_.createPowerConfigId || !perf_.setPowerConfig ||
        perf_.createPowerConfigId(0, 0, &powerConfigId_) != QNN_SUCCESS) return;
    hasPowerConfig_ = true;

    QnnHtpPerfInfrastructure_PowerConfig_t dcvs{};
    dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    auto& v = dcvs.dcvsV3Config;
    v.contextId = powerConfigId_;
    v.setDcvsEnable = 1; v.dcvsEnable = 0;
    v.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    v.setSleepLatency = 1; v.sleepLatency = 40;
    v.setSleepDisable = 1; v.sleepDisable = 1;
    v.setBusParams = 1; v.setCoreParams = 1;
    v.busVoltageCornerMin = v.busVoltageCornerTarget = v.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    v.coreVoltageCornerMin = v.coreVoltageCornerTarget = v.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    QnnHtpPerfInfrastructure_PowerConfig_t latency{};
    latency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    latency.rpcControlLatencyConfig = 100;
    QnnHtpPerfInfrastructure_PowerConfig_t polling{};
    polling.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    polling.rpcPollingTimeConfig = 9999;

    const QnnHtpPerfInfrastructure_PowerConfig_t* dcvsConfigs[] = {&dcvs, nullptr};
    const QnnHtpPerfInfrastructure_PowerConfig_t* rpcConfigs[] = {&latency, &polling, nullptr};
    const bool accepted = perf_.setPowerConfig(powerConfigId_, dcvsConfigs) == QNN_SUCCESS;
    const bool rpc = perf_.setPowerConfig(powerConfigId_, rpcConfigs) == QNN_SUCCESS;
    timing("HTP", accepted ? (rpc ? "burst_vote_accepted" : "burst_vote_accepted_no_rpc") : "burst_vote_rejected", 0);
    if (!accepted) endPerformanceVote();
  }

  void endPerformanceVote() {
    if (hasPowerConfig_ && perf_.destroyPowerConfigId) perf_.destroyPowerConfigId(powerConfigId_);
    hasPowerConfig_ = false;
  }

  // A streaming session adds ~25 rows per frame; the cap keeps hours-long sessions bounded.
  void timing(const std::string& graph, const char* phase, double ms) {
    if (timings_.size() < kMaxTimings) timings_.push_back({graph, phase, ms});
  }

  void clearTimings() { timings_.clear(); }

  void writeTimings(const std::string& path) const {
    std::ofstream file(path);
    file << "graph,phase,milliseconds\n";
    for (const auto& row : timings_) file << row.graph << ',' << row.phase << ',' << row.ms << '\n';
    if (!file) throw std::runtime_error("Cannot write performance CSV: " + path);
  }

  QNN_INTERFACE_VER_TYPE qnn{};
  QNN_SYSTEM_INTERFACE_VER_TYPE system{};
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  // Sequentially executed LM contexts share one spill-fill buffer through an HTP group.
  Qnn_ContextHandle_t groupLeader = nullptr;
  uint64_t groupSpillBytes = 0;
  SharedMemory sharedMemory;

 private:
  static constexpr size_t kMaxTimings = 1 << 20;
  struct Timing { std::string graph; const char* phase; double ms; };
  std::vector<Timing> timings_;
  void* backendLibrary_ = nullptr;
  void* systemLibrary_ = nullptr;
  const QnnInterface_t* provider_ = nullptr;
  QnnHtpDevice_PerfInfrastructure_t perf_{};
  uint32_t powerConfigId_ = 0;
  bool hasPowerConfig_ = false;
};

struct Buffer {
  Qnn_Tensor_t tensor{};
  std::string name;
  std::vector<uint8_t> bytes;
  uint8_t* data() { return bytes.data(); }
};

class Graph {
 public:
  Graph(Runtime& runtime, const std::string& path, bool grouped = false)
      : runtime_(runtime), binary_(path) {
    label_ = path.substr(path.find_last_of('/') + 1);
    if (label_.size() < 4 || label_.compare(label_.size() - 4, 4, ".bin"))
      throw std::runtime_error("Expected a QNN context binary (.bin): " + path);
    label_.resize(label_.size() - 4);
    const auto start = Clock::now();
    check(runtime_.system.systemContextCreate(&systemContext_), "systemContextCreate");
    const QnnSystemContext_BinaryInfo_t* info = nullptr;
    Qnn_ContextBinarySize_t metadataSize = 0;
    check(runtime_.system.systemContextGetBinaryInfo(systemContext_, binary_.data, binary_.size, &info, &metadataSize),
          "context metadata " + label_);
    const auto graph = readGraphInfo(info, path);

    QnnHtpContext_CustomConfig_t acceleration{};
    acceleration.option = QNN_HTP_CONTEXT_CONFIG_OPTION_INIT_ACCELERATION;
    acceleration.initAcceleration = true;
    QnnHtpContext_CustomConfig_t group{};
    group.option = QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
    group.groupRegistration.firstGroupHandle = runtime_.groupLeader;
    group.groupRegistration.maxSpillFillBuffer = runtime_.groupSpillBytes;
    QnnContext_Config_t accelerationConfig{}, groupConfig{};
    accelerationConfig.option = groupConfig.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    accelerationConfig.customConfig = &acceleration;
    groupConfig.customConfig = &group;
    std::vector<const QnnContext_Config_t*> configs = {&accelerationConfig};
    if (grouped && runtime_.groupSpillBytes) configs.push_back(&groupConfig);
    configs.push_back(nullptr);
    check(runtime_.qnn.contextCreateFromBinary(runtime_.backend, runtime_.device, configs.data(), binary_.data,
                                               binary_.size, &context_, nullptr),
          "contextCreateFromBinary " + label_ + " (rebuild the context for this device/runtime)");
    if (grouped && runtime_.groupSpillBytes && !runtime_.groupLeader) runtime_.groupLeader = context_;
    check(runtime_.qnn.graphRetrieve(context_, graph.name.c_str(), &graph_), "graphRetrieve " + graph.name);
    prepare(graph.inputs, graph.numInputs, inputs_, inputTensors_);
    prepare(graph.outputs, graph.numOutputs, outputs_, outputTensors_);
    runtime_.timing(label_, "restore_context", elapsedMs(start));
  }

  ~Graph() {
    for (auto handle : memHandles_) runtime_.qnn.memDeRegister(&handle, 1);
    if (context_ && runtime_.qnn.contextFree) runtime_.qnn.contextFree(context_, nullptr);
    if (systemContext_ && runtime_.system.systemContextFree) runtime_.system.systemContextFree(systemContext_);
  }
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

  const std::string& label() const { return label_; }
  Buffer& input(const std::string& name) { return find(inputs_, name); }
  Buffer& output(const std::string& name) { return find(outputs_, name); }
  const std::vector<Buffer>& inputs() const { return inputs_; }

  // Binds a large input to externally owned storage; registered shared memory when possible.
  void bindInput(const std::string& name, const SharedBuffer& ring) {
    for (size_t i = 0; i < inputs_.size(); ++i) {
      auto& buffer = inputs_[i];
      if (buffer.name != name) continue;
      if (tensorBytes(buffer.tensor) != ring.bytes())
        throw std::runtime_error("Shared storage size does not match graph input " + name);
      std::vector<uint8_t>().swap(buffer.bytes);
      Qnn_MemHandle_t handle = nullptr;
      if (ring.fd() >= 0 && runtime_.qnn.memRegister) {
        Qnn_MemDescriptor_t descriptor = QNN_MEM_DESCRIPTOR_INIT;
        descriptor.memShape = {tensorRank(buffer.tensor), const_cast<uint32_t*>(tensorDimensions(buffer.tensor)), nullptr};
        descriptor.dataType = tensorType(buffer.tensor);
        descriptor.memType = QNN_MEM_TYPE_ION;
        descriptor.ionInfo.fd = ring.fd();
        if (runtime_.qnn.memRegister(context_, &descriptor, 1, &handle) != QNN_SUCCESS) handle = nullptr;
      }
      if (handle) {
        memHandles_.push_back(handle);
        setMemHandle(buffer.tensor, handle);
      } else {
        setClientBuffer(buffer.tensor, ring.data(), ring.bytes());
      }
      inputTensors_[i] = buffer.tensor;
      return;
    }
    throw std::runtime_error("Missing graph input: " + name);
  }

  void execute() {
    const auto start = Clock::now();
    check(runtime_.qnn.graphExecute(graph_, inputTensors_.data(), inputTensors_.size(), outputTensors_.data(),
                                    outputTensors_.size(), nullptr, nullptr),
          "graphExecute " + label_);
    runtime_.timing(label_, "execute", elapsedMs(start));
  }

  // Mimi streaming state: swap the output into the next input instead of copying megabytes.
  void advanceState() {
    for (const auto* name : {"position", "kv_cache", "conv_state"}) {
      auto& source = output(std::string(name) + "_out");
      auto& target = input(name);
      if (tensorType(source.tensor) != tensorType(target.tensor) || source.bytes.size() != target.bytes.size())
        throw std::runtime_error("Incompatible state tensors: " + source.name + " -> " + target.name);
      source.bytes.swap(target.bytes);
      rebind(source);
      rebind(target);
    }
  }

  void resetState() {
    for (const auto* name : {"position", "kv_cache", "conv_state"}) {
      auto& state = input(name);
      std::fill(state.bytes.begin(), state.bytes.end(), 0);
    }
  }

 private:
  Buffer& find(std::vector<Buffer>& buffers, const std::string& name) {
    for (auto& buffer : buffers)
      if (buffer.name == name) return buffer;
    throw std::runtime_error("Missing QNN graph tensor: " + name + " in " + label_);
  }

  void rebind(Buffer& buffer) {
    setClientBuffer(buffer.tensor, buffer.data(), buffer.bytes.size());
    for (auto* tensors : {&inputTensors_, &outputTensors_})
      for (auto& tensor : *tensors)
        if (!std::strcmp(tensorName(tensor), buffer.name.c_str())) tensor = buffer.tensor;
  }

  static void prepare(const Qnn_Tensor_t* metadata, uint32_t count, std::vector<Buffer>& buffers,
                      std::vector<Qnn_Tensor_t>& tensors) {
    if (!count) throw std::runtime_error("Graph has no tensors");
    buffers.resize(count);
    tensors.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
      auto& buffer = buffers[i];
      buffer.tensor = metadata[i];
      buffer.name = tensorName(metadata[i]);
      buffer.bytes.assign(tensorBytes(metadata[i]), 0);
      setClientBuffer(buffer.tensor, buffer.data(), buffer.bytes.size());
      tensors[i] = buffer.tensor;
    }
  }

  Runtime& runtime_;
  MappedFile binary_;  // Kept mapped until contextFree.
  std::string label_;
  QnnSystemContext_Handle_t systemContext_ = nullptr;
  Qnn_ContextHandle_t context_ = nullptr;
  Qnn_GraphHandle_t graph_ = nullptr;
  std::vector<Buffer> inputs_, outputs_;
  std::vector<Qnn_Tensor_t> inputTensors_, outputTensors_;
  std::vector<Qnn_MemHandle_t> memHandles_;
};

void expect(const Buffer& buffer, Qnn_DataType_t type, size_t bytes) {
  if (tensorType(buffer.tensor) != type || buffer.bytes.size() != bytes)
    throw std::runtime_error("Unexpected tensor contract: " + buffer.name);
}

void set(Buffer& buffer, const void* data, size_t size, Qnn_DataType_t type) {
  expect(buffer, type, size);
  std::memcpy(buffer.data(), data, size);
}

int16_t toPcm16(float value) {
  if (!std::isfinite(value)) throw std::runtime_error("Nonfinite decoder audio");
  return static_cast<int16_t>(std::lrintf(std::clamp(value, -1.0f, 32767.0f / 32768.0f) * 32768.0f));
}

uint32_t read32(const uint8_t* d) { return d[0] | d[1] << 8 | d[2] << 16 | uint32_t(d[3]) << 24; }

// Fixed-point I/O of a QNN tensor: real = scale * (q + offset). FLOAT_32 tensors pass through.
struct Affine {
  Qnn_DataType_t type = QNN_DATATYPE_FLOAT_32;
  float scale = 1.0f;
  int32_t offset = 0;
  bool operator==(const Affine& o) const { return type == o.type && scale == o.scale && offset == o.offset; }
};

Affine affineOf(const Qnn_Tensor_t& tensor) {
  Affine a;
  a.type = tensorType(tensor);
  if (a.type == QNN_DATATYPE_FLOAT_32) return a;
  if (a.type != QNN_DATATYPE_UFIXED_POINT_8 && a.type != QNN_DATATYPE_UFIXED_POINT_16)
    throw std::runtime_error(std::string("Unsupported fixed-point I/O type for ") + tensorName(tensor));
  const auto& q = TENSOR_FIELD(tensor, quantizeParams);
  if (q.encodingDefinition != QNN_DEFINITION_DEFINED || q.quantizationEncoding != QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
    throw std::runtime_error(std::string("Expected per-tensor scale/offset for ") + tensorName(tensor));
  a.scale = q.scaleOffsetEncoding.scale;
  a.offset = q.scaleOffsetEncoding.offset;
  return a;
}

size_t affineWidth(const Affine& a) {
  return a.type == QNN_DATATYPE_UFIXED_POINT_8 ? 1 : a.type == QNN_DATATYPE_UFIXED_POINT_16 ? 2 : 4;
}

void quantizeTo(const Affine& a, const float* src, size_t count, uint8_t* dst) {
  if (a.type == QNN_DATATYPE_FLOAT_32) { std::memcpy(dst, src, count * sizeof(float)); return; }
  const float top = a.type == QNN_DATATYPE_UFIXED_POINT_8 ? 255.0f : 65535.0f;
  for (size_t i = 0; i < count; ++i) {
    const float q = std::clamp(std::nearbyint(src[i] / a.scale) - static_cast<float>(a.offset), 0.0f, top);
    if (a.type == QNN_DATATYPE_UFIXED_POINT_8) dst[i] = static_cast<uint8_t>(q);
    else reinterpret_cast<uint16_t*>(dst)[i] = static_cast<uint16_t>(q);
  }
}

void dequantizeFrom(const Affine& a, const uint8_t* src, size_t count, float* dst) {
  if (a.type == QNN_DATATYPE_FLOAT_32) { std::memcpy(dst, src, count * sizeof(float)); return; }
  for (size_t i = 0; i < count; ++i) {
    const float q = a.type == QNN_DATATYPE_UFIXED_POINT_8 ? src[i] : reinterpret_cast<const uint16_t*>(src)[i];
    dst[i] = a.scale * (q + static_cast<float>(a.offset));
  }
}

// Real values into / out of a FP32 or fixed-point graph tensor.
void setReal(Buffer& buffer, const float* src, size_t count) {
  const auto a = affineOf(buffer.tensor);
  if (buffer.bytes.size() != count * affineWidth(a)) throw std::runtime_error("Unexpected tensor size: " + buffer.name);
  quantizeTo(a, src, count, buffer.data());
}

void getReal(const Buffer& buffer, float* dst, size_t count) {
  const auto a = affineOf(buffer.tensor);
  if (buffer.bytes.size() != count * affineWidth(a)) throw std::runtime_error("Unexpected tensor size: " + buffer.name);
  dequantizeFrom(a, buffer.bytes.data(), count, dst);
}

// One Qualcomm-LLM-layout Temporal part (scripts/moshi_temporal.py): host attention mask and
// RoPE cos/sin, fixed-point KV rings (key [H,1,D,P], value [H,1,P,D]), new K/V outputs only.
class TemporalPart {
  struct Ring { Affine affine; std::unique_ptr<SharedBuffer> storage; };

 public:
  static constexpr size_t kHeads = 32, kHeadDim = 128, kDim = 4096;
  static constexpr float kMaskMin = -50.0f;

  TemporalPart(Runtime& runtime, const std::string& path, bool grouped = false)
      : runtime_(runtime), rings_(), graph_(runtime, path, grouped) {
    for (const auto& input : graph_.inputs()) {
      int layer = -1;
      if (std::sscanf(input.name.c_str(), "past_key_%d_in", &layer) == 1) layers_.push_back(layer);
    }
    std::sort(layers_.begin(), layers_.end());
    if (layers_.empty()) throw std::runtime_error("No past_key inputs in " + path);
    const auto& firstKey = graph_.input(cacheName(layers_[0], true, true));
    past_ = tensorDimensions(firstKey.tensor)[3];
    hidden_ = affineOf(graph_.input("hidden").tensor);
    hiddenOut_ = affineOf(graph_.output("hidden_out").tensor);
    mask_ = affineOf(graph_.input("attention_mask").tensor);
    cos_ = affineOf(graph_.input("position_cos").tensor);
    sin_ = affineOf(graph_.input("position_sin").tensor);
    for (int layer : layers_) {
      for (bool key : {true, false}) {
        const auto& in = graph_.input(cacheName(layer, key, true));
        Ring ring{affineOf(in.tensor), nullptr};
        if (affineWidth(ring.affine) != 1 || !(ring.affine == affineOf(graph_.output(cacheName(layer, key, false)).tensor)))
          throw std::runtime_error("KV must be 8-bit with tied input/output encodings: " + in.name);
        ring.storage = std::make_unique<SharedBuffer>(runtime_.sharedMemory, in.bytes.size());
        graph_.bindInput(in.name, *ring.storage);
        rings_.push_back(std::move(ring));
      }
    }
    reset();
  }

  const std::string& label() const { return graph_.label(); }

  // Empty slots hold the quantized value of 0.0, as in the float graph; masked or not,
  // byte 0 would be -128*scale for a symmetric 8-bit encoding.
  void reset() {
    for (auto& ring : rings_) {
      const float zero = 0.0f;
      uint8_t code;
      quantizeTo(ring.affine, &zero, 1, &code);
      std::memset(ring.storage->data(), code, ring.storage->bytes());
    }
  }

  void step(int32_t position, const float* hidden, float* out) {
    quantizeTo(hidden_, hidden, kDim, graph_.input("hidden").data());
    std::vector<float> mask(past_ + 1, kMaskMin);
    mask[past_] = 0.0f;
    for (int32_t back = 1; back <= std::min<int32_t>(position, static_cast<int32_t>(past_)); ++back)
      mask[(position - back) % past_] = 0.0f;
    quantizeTo(mask_, mask.data(), mask.size(), graph_.input("attention_mask").data());
    float cos[kHeadDim / 2], sin[kHeadDim / 2];
    for (size_t i = 0; i < kHeadDim / 2; ++i) {
      const double angle = position * std::exp(-std::log(10000.0) * 2.0 * i / kHeadDim);
      cos[i] = static_cast<float>(std::cos(angle));
      sin[i] = static_cast<float>(std::sin(angle));
    }
    quantizeTo(cos_, cos, kHeadDim / 2, graph_.input("position_cos").data());
    quantizeTo(sin_, sin, kHeadDim / 2, graph_.input("position_sin").data());
    graph_.execute();
    dequantizeFrom(hiddenOut_, graph_.output("hidden_out").data(), kDim, out);
    const size_t slot = position % past_;
    for (size_t r = 0; r < rings_.size(); ++r) {
      const bool key = r % 2 == 0;
      writeSlot(rings_[r], graph_.output(cacheName(layers_[r / 2], key, false)).data(), slot, key);
    }
  }

 private:
  static std::string cacheName(int layer, bool key, bool input) {
    return std::string(key ? "past_key_" : "past_value_") + std::to_string(layer) + (input ? "_in" : "_out");
  }

  // Key out [H,1,D,1] -> ring [H,1,D,P] column `slot`; value out [H,1,1,D] -> ring [H,1,P,D] row `slot`.
  void writeSlot(Ring& ring, const uint8_t* source, size_t slot, bool key) {
    uint8_t* base = ring.storage->data();
    for (size_t h = 0; h < kHeads; ++h) {
      if (!key) {
        std::memcpy(base + (h * past_ + slot) * kHeadDim, source + h * kHeadDim, kHeadDim);
        continue;
      }
      for (size_t d = 0; d < kHeadDim; ++d) base[(h * kHeadDim + d) * past_ + slot] = source[h * kHeadDim + d];
    }
  }

  Runtime& runtime_;
  std::vector<Ring> rings_;  // Declared before graph_: deregister memory before freeing it.
  Graph graph_;
  std::vector<int> layers_;
  size_t past_ = 0;
  Affine hidden_, hiddenOut_, mask_, cos_, sin_;
};

// Mimi encoder/decoder pair; each graph carries its own streaming state (position advances by 2 per frame).
class Codec {
 public:
  Codec(Runtime& runtime, const std::string& encoder, const std::string& decoder)
      : encoder_(runtime, encoder), decoder_(runtime, decoder) {
    expect(encoder_.input("audio"), QNN_DATATYPE_FLOAT_32, kFrameSamples * sizeof(float));
    expect(encoder_.output("codes"), QNN_DATATYPE_INT_32, 8 * sizeof(int32_t));
    expect(decoder_.input("codes"), QNN_DATATYPE_INT_32, 8 * sizeof(int32_t));
    expect(decoder_.output("audio"), QNN_DATATYPE_FLOAT_32, kFrameSamples * sizeof(float));
  }

  void reset() { encoder_.resetState(); decoder_.resetState(); encoded_ = decoded_ = 0; }

  std::array<int32_t, 8> encode(const int16_t* pcm) {
    auto* audio = reinterpret_cast<float*>(encoder_.input("audio").data());
    for (size_t i = 0; i < kFrameSamples; ++i) audio[i] = pcm[i] / 32768.0f;
    encoder_.execute();
    std::array<int32_t, 8> codes{};
    std::memcpy(codes.data(), encoder_.output("codes").data(), sizeof(codes));
    checkPosition(encoder_, ++encoded_);
    encoder_.advanceState();
    return codes;
  }

  void decode(const int32_t* codes, int16_t* pcm) {
    set(decoder_.input("codes"), codes, 8 * sizeof(int32_t), QNN_DATATYPE_INT_32);
    decoder_.execute();
    const auto* audio = reinterpret_cast<const float*>(decoder_.output("audio").data());
    for (size_t i = 0; i < kFrameSamples; ++i) pcm[i] = toPcm16(audio[i]);
    checkPosition(decoder_, ++decoded_);
    decoder_.advanceState();
  }

 private:
  static void checkPosition(Graph& graph, size_t frames) {
    auto& position = graph.output("position_out");
    expect(position, QNN_DATATYPE_INT_32, sizeof(int32_t));
    int32_t value;
    std::memcpy(&value, position.data(), sizeof(value));
    if (value != static_cast<int32_t>(2 * frames))
      throw std::runtime_error(graph.label() + " streaming position did not advance by two");
  }
  Graph encoder_, decoder_;
  size_t encoded_ = 0, decoded_ = 0;
};

// moshi_export_frontend_table.py: the text table and 16 audio tables in BF16, one row per token.
// The Temporal input is the sum of one row per sequence slot, done on the CPU.
class EmbeddingTable {
 public:
  explicit EmbeddingTable(const std::string& path) : file_(path) {
    const auto* bytes = static_cast<const uint8_t*>(file_.data);
    if (file_.size < 20 || std::memcmp(bytes, "MOSHIEMB", 8) || read32(bytes + 8) != 1)
      throw std::runtime_error("Not a Moshi embedding table: " + path);
    dim_ = read32(bytes + 12);
    const uint32_t count = read32(bytes + 16);
    size_t offset = 20 + 4 * size_t(count);
    for (uint32_t t = 0; t < count && offset <= file_.size; ++t) {
      const uint32_t rows = read32(bytes + 20 + 4 * t);
      tables_.push_back({reinterpret_cast<const uint16_t*>(bytes + offset), rows});
      offset += size_t(rows) * dim_ * sizeof(uint16_t);
    }
    if (offset != file_.size) throw std::runtime_error("Truncated Moshi embedding table: " + path);
  }

  size_t dim() const { return dim_; }

  void sum(const int32_t* tokens, size_t count, float* out) const {
    if (count != tables_.size()) throw std::runtime_error("Sequence does not match the embedding tables");
    std::fill(out, out + dim_, 0.0f);
    for (size_t t = 0; t < count; ++t) {
      if (tokens[t] < 0 || uint32_t(tokens[t]) >= tables_[t].rows) throw std::runtime_error("Embedding token out of range");
      const uint16_t* row = tables_[t].data + size_t(tokens[t]) * dim_;
      for (size_t i = 0; i < dim_; ++i) {
        const uint32_t bits = uint32_t(row[i]) << 16;
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        out[i] += value;
      }
    }
  }

 private:
  struct Table { const uint16_t* data; uint32_t rows; };
  MappedFile file_;
  uint32_t dim_ = 0;
  std::vector<Table> tables_;
};

// Moshiko LM: CPU embedding sum -> 4 Temporal parts (8 layers each, scripts/moshi_temporal.py layout)
// -> head -> 8 sequential DepFormer codebooks.
class MoshiLm {
 public:
  static constexpr int kParts = 4;

  MoshiLm(Runtime& runtime, const std::string& directory)
      : runtime_(runtime), embeddings_(directory + "/frontend_embeddings.bin") {
    if (embeddings_.dim() != TemporalPart::kDim) throw std::runtime_error("Embedding width does not match Temporal");
    std::vector<std::string> names = {"head"};
    for (int book = 0; book < 8; ++book) names.push_back("depformer_codebook_" + std::to_string(book));
    std::vector<std::string> parts;
    for (int p = 0; p < kParts; ++p) parts.push_back(directory + "/dlc/temporal_part_" + std::to_string(p) + ".bin");

    uint64_t spill = 0;
    for (const auto& name : names) spill = std::max(spill, spillFillBytes(directory + "/dlc/" + name + ".bin"));
    for (const auto& path : parts) spill = std::max(spill, spillFillBytes(path));
    runtime_.groupSpillBytes = spill;
    for (const auto& name : names) graphs_[name] = std::make_unique<Graph>(runtime_, directory + "/dlc/" + name + ".bin", true);
    for (const auto& path : parts) temporal_.push_back(std::make_unique<TemporalPart>(runtime_, path, true));
  }

  ~MoshiLm() {
    // Every grouped context goes before the group leader (head, created first).
    temporal_.clear();
    auto leader = std::move(graphs_.at("head"));
    graphs_.clear();
    leader.reset();
    runtime_.groupLeader = nullptr;
  }

  void reset() {
    delay_.reset();
    for (auto& part : temporal_) part->reset();
    position_ = 0;
  }

  std::optional<std::array<int32_t, 9>> step(const std::array<int32_t, 8>& user) {
    const auto sequence = delay_.begin(user);
    std::vector<float> hidden(TemporalPart::kDim), next(TemporalPart::kDim);
    const auto start = Clock::now();
    embeddings_.sum(sequence.data(), sequence.size(), hidden.data());
    runtime_.timing("frontend_cpu", "execute", elapsedMs(start));
    for (auto& part : temporal_) {
      part->step(position_, hidden.data(), next.data());
      hidden.swap(next);
    }

    Graph& head = *graphs_.at("head");
    setReal(head.input("hidden"), hidden.data(), hidden.size());
    head.execute();
    getReal(head.output("temporal"), next.data(), next.size());
    getReal(head.output("text_logits"), logits_.data(), logits_.size());
    const int32_t text = static_cast<int32_t>(std::max_element(logits_.begin(), logits_.end()) - logits_.begin());

    // The depth cache restarts every frame and is handed from codebook to codebook; each graph
    // has its own 16-bit encodings, so the handoff requantizes unless they happen to match.
    constexpr size_t kCache = 16 * 8 * 64;
    std::array<int32_t, 8> audio{};
    int32_t previous = text;
    Graph* last = nullptr;
    std::vector<float> values(kCache);
    for (int book = 0; book < 8; ++book) {
      Graph& graph = *graphs_.at("depformer_codebook_" + std::to_string(book));
      set(graph.input("previous_token"), &previous, sizeof(previous), QNN_DATATYPE_INT_32);
      setReal(graph.input("temporal"), next.data(), next.size());
      for (int j = 0; j < 12; ++j) {
        auto& input = graph.input(cacheName(j / 2, j % 2));
        if (!last) {
          std::fill(values.begin(), values.end(), 0.0f);
          setReal(input, values.data(), kCache);
          continue;
        }
        const auto& source = last->output("output_" + input.name);
        if (affineOf(source.tensor) == affineOf(input.tensor) && source.bytes.size() == input.bytes.size()) {
          std::memcpy(input.data(), source.bytes.data(), input.bytes.size());
        } else {
          getReal(source, values.data(), kCache);
          setReal(input, values.data(), kCache);
        }
      }
      graph.execute();
      auto& token = graph.output("audio_token");
      expect(token, QNN_DATATYPE_INT_32, sizeof(int32_t));
      std::memcpy(&previous, token.data(), sizeof(previous));
      audio[book] = previous;
      last = &graph;
    }
    ++position_;
    return delay_.finish(text, audio);
  }

 private:
  uint64_t spillFillBytes(const std::string& path) {
    MappedFile binary(path);
    QnnSystemContext_Handle_t context = nullptr;
    check(runtime_.system.systemContextCreate(&context), "systemContextCreate");
    const QnnSystemContext_BinaryInfo_t* info = nullptr;
    Qnn_ContextBinarySize_t size = 0;
    const auto status = runtime_.system.systemContextGetBinaryInfo(context, binary.data, binary.size, &info, &size);
    const uint64_t bytes = status == QNN_SUCCESS ? readGraphInfo(info, path).spillFillBytes : 0;
    runtime_.system.systemContextFree(context);
    check(status, "context metadata " + path);
    return bytes;
  }
  // DepFormer depth-cache tensor names.
  static std::string cacheName(int layer, int value) {
    return "layer_" + std::to_string(layer) + (value ? "_value" : "_key");
  }

  Runtime& runtime_;
  EmbeddingTable embeddings_;
  std::map<std::string, std::unique_ptr<Graph>> graphs_;
  std::vector<std::unique_ptr<TemporalPart>> temporal_;
  std::vector<float> logits_ = std::vector<float>(32000);
  MoshiDelayRing delay_;
  int32_t position_ = 0;
};

// Frame-by-frame duplex session: the app feeds one 80 ms microphone frame per step and plays
// whatever comes back. All calls for one session come from a single thread.
class Session {
 public:
  Session(const std::string& libraryDir, const std::string& modelDirectory, bool codecOnly)
      : runtime_(libraryDir) {
    if (!codecOnly) lm_ = std::make_unique<MoshiLm>(runtime_, modelDirectory);
    codec_ = std::make_unique<Codec>(runtime_, modelDirectory + "/mimi_encoder.bin",
                                     modelDirectory + "/mimi_decoder.bin");
  }

  ~Session() { runtime_.endPerformanceVote(); codec_.reset(); lm_.reset(); }

  void begin(bool highPerformance) {
    runtime_.endPerformanceVote();
    runtime_.clearTimings();
    codec_->reset();
    if (lm_) lm_->reset();
    started_ = Clock::now();
    if (highPerformance) runtime_.beginPerformanceVote();
  }

  // Returns whether `pcm` holds a new frame. `tokens` gets the text token (-1 in codec mode)
  // then 8 audio codes. Moshi's first frame only primes the delay ring and returns false.
  bool step(const int16_t* input, int16_t* pcm, int32_t* tokens) {
    const auto start = Clock::now();
    const auto codes = codec_->encode(input);
    std::array<int32_t, 9> out{};
    if (lm_) {
      const auto generated = lm_->step(codes);
      if (!generated) { runtime_.timing("frame", "total", elapsedMs(start)); return false; }
      out = *generated;
    } else {
      out[0] = -1;
      std::copy(codes.begin(), codes.end(), out.begin() + 1);
    }
    codec_->decode(out.data() + 1, pcm);
    std::copy(out.begin(), out.end(), tokens);
    runtime_.timing("frame", "total", elapsedMs(start));
    return true;
  }

  void end(const std::string& performanceCsv) {
    runtime_.timing("pipeline", "wall_total", elapsedMs(started_));
    runtime_.endPerformanceVote();
    if (!performanceCsv.empty()) runtime_.writeTimings(performanceCsv);
  }

 private:
  Runtime runtime_;
  std::unique_ptr<MoshiLm> lm_;
  std::unique_ptr<Codec> codec_;
  Clock::time_point started_ = Clock::now();
};

}  // namespace

void* moshiCreate(const std::string& libraryDir, const std::string& modelDirectory, bool codecOnly) {
  return new Session(libraryDir, modelDirectory, codecOnly);
}

void moshiBegin(void* session, bool highPerformance) { static_cast<Session*>(session)->begin(highPerformance); }

bool moshiStep(void* session, const int16_t* input, int16_t* pcm, int32_t* tokens) {
  return static_cast<Session*>(session)->step(input, pcm, tokens);
}

void moshiEnd(void* session, const std::string& performanceCsv) { static_cast<Session*>(session)->end(performanceCsv); }

void moshiDestroy(void* session) { delete static_cast<Session*>(session); }
