// Shared QNN HTP plumbing for the on-device runners (QAIRT 2.45 APIs): library loading, context
// binaries with one or more graphs, rpcmem-backed inputs, fixed-point I/O and timing.
#pragma once

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpSystemContext.h"
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

namespace qnnrt {

using Clock = std::chrono::steady_clock;
inline double elapsedMs(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

inline void check(Qnn_ErrorHandle_t status, const std::string& operation) {
  if (status != QNN_SUCCESS)
    throw std::runtime_error(operation + " failed, QNN status " + std::to_string(status));
}

inline void* openLibrary(const std::string& path) {
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

inline const char* tensorName(const Qnn_Tensor_t& t) { return TENSOR_FIELD(t, name); }
inline Qnn_DataType_t tensorType(const Qnn_Tensor_t& t) { return TENSOR_FIELD(t, dataType); }
inline uint32_t tensorRank(const Qnn_Tensor_t& t) { return TENSOR_FIELD(t, rank); }
inline const uint32_t* tensorDimensions(const Qnn_Tensor_t& t) { return TENSOR_FIELD(t, dimensions); }

inline void setClientBuffer(Qnn_Tensor_t& t, void* data, size_t length) {
  if (length > UINT32_MAX) throw std::runtime_error("QNN client buffer exceeds uint32");
  const Qnn_ClientBuffer_t buffer = {data, static_cast<uint32_t>(length)};
  if (t.version == QNN_TENSOR_VERSION_2) { t.v2.memType = QNN_TENSORMEMTYPE_RAW; t.v2.clientBuf = buffer; }
  else { t.v1.memType = QNN_TENSORMEMTYPE_RAW; t.v1.clientBuf = buffer; }
}

inline void setMemHandle(Qnn_Tensor_t& t, Qnn_MemHandle_t handle) {
  if (t.version == QNN_TENSOR_VERSION_2) { t.v2.memType = QNN_TENSORMEMTYPE_MEMHANDLE; t.v2.memHandle = handle; }
  else { t.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE; t.v1.memHandle = handle; }
}

inline size_t tensorElements(const Qnn_Tensor_t& t) {
  size_t elements = 1;
  for (uint32_t i = 0; i < tensorRank(t); ++i) elements *= tensorDimensions(t)[i];
  return elements;
}

inline size_t tensorBytes(const Qnn_Tensor_t& t) {
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
  return tensorElements(t) * width;
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

// Graph metadata from a context binary, independent of the metadata struct version.
struct BinaryGraphInfo {
  std::string name;
  Qnn_Tensor_t* inputs = nullptr; uint32_t numInputs = 0;
  Qnn_Tensor_t* outputs = nullptr; uint32_t numOutputs = 0;
  uint64_t spillFillBytes = 0;
};

inline std::vector<BinaryGraphInfo> readGraphInfos(const QnnSystemContext_BinaryInfo_t* info, const std::string& path) {
  QnnSystemContext_GraphInfo_t* graphs = nullptr;
  uint32_t count = 0;
  switch (info->version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: graphs = info->contextBinaryInfoV1.graphs; count = info->contextBinaryInfoV1.numGraphs; break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: graphs = info->contextBinaryInfoV2.graphs; count = info->contextBinaryInfoV2.numGraphs; break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: graphs = info->contextBinaryInfoV3.graphs; count = info->contextBinaryInfoV3.numGraphs; break;
    default: break;
  }
  if (!count || !graphs) throw std::runtime_error("No graphs in " + path);
  std::vector<BinaryGraphInfo> result;
  for (uint32_t i = 0; i < count; ++i) {
    BinaryGraphInfo out;
    const auto fill = [&](const auto& g) {
      out.name = g.graphName;
      out.inputs = g.graphInputs; out.numInputs = g.numGraphInputs;
      out.outputs = g.graphOutputs; out.numOutputs = g.numGraphOutputs;
    };
    switch (graphs[i].version) {
      case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1: fill(graphs[i].graphInfoV1); break;
      case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2: fill(graphs[i].graphInfoV2); break;
      case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3: {
        const auto& g = graphs[i].graphInfoV3;
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
    result.push_back(out);
  }
  return result;
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

  // A streaming session adds a few dozen rows per step; the cap keeps hours-long sessions bounded.
  void timing(const std::string& graph, const char* phase, double ms) {
    std::lock_guard<std::mutex> lock(timingMutex_);
    if (timings_.size() < kMaxTimings) timings_.push_back({graph, phase, ms});
  }

  void clearTimings() { std::lock_guard<std::mutex> lock(timingMutex_); timings_.clear(); }

  void writeTimings(const std::string& path) {
    std::lock_guard<std::mutex> lock(timingMutex_);
    std::ofstream file(path);
    file << "graph,phase,milliseconds\n";
    for (const auto& row : timings_) file << row.graph << ',' << row.phase << ',' << row.ms << '\n';
    if (!file) throw std::runtime_error("Cannot write performance CSV: " + path);
  }

  // Spill-fill size of every graph in a context binary, read without restoring it.
  uint64_t spillFillBytes(const std::string& path) {
    MappedFile binary(path);
    QnnSystemContext_Handle_t context = nullptr;
    check(system.systemContextCreate(&context), "systemContextCreate");
    const QnnSystemContext_BinaryInfo_t* info = nullptr;
    Qnn_ContextBinarySize_t size = 0;
    const auto status = system.systemContextGetBinaryInfo(context, binary.data, binary.size, &info, &size);
    uint64_t bytes = 0;
    if (status == QNN_SUCCESS)
      for (const auto& graph : readGraphInfos(info, path)) bytes = std::max(bytes, graph.spillFillBytes);
    system.systemContextFree(context);
    check(status, "context metadata " + path);
    return bytes;
  }

  QNN_INTERFACE_VER_TYPE qnn{};
  QNN_SYSTEM_INTERFACE_VER_TYPE system{};
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  // Sequentially executed contexts share one spill-fill buffer through an HTP group.
  Qnn_ContextHandle_t groupLeader = nullptr;
  uint64_t groupSpillBytes = 0;
  SharedMemory sharedMemory;
  // Graphs may be driven from several threads, but the group's shared spill-fill buffer allows one at a time.
  std::mutex executeMutex;

 private:
  static constexpr size_t kMaxTimings = 1 << 20;
  struct Timing { std::string graph; const char* phase; double ms; };
  std::mutex timingMutex_;
  std::vector<Timing> timings_;
  void* backendLibrary_ = nullptr;
  void* systemLibrary_ = nullptr;
  const QnnInterface_t* provider_ = nullptr;
  QnnHtpDevice_PerfInfrastructure_t perf_{};
  uint32_t powerConfigId_ = 0;
  bool hasPowerConfig_ = false;
};

// One restored context binary; its graphs (several when they share weights) are retrieved by name.
class Context {
 public:
  Context(Runtime& runtime, const std::string& path, bool grouped)
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
    graphs_ = readGraphInfos(info, path);

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
    runtime_.timing(label_, "restore_context", elapsedMs(start));
  }

  ~Context() {
    if (context_ && runtime_.qnn.contextFree) runtime_.qnn.contextFree(context_, nullptr);
    if (systemContext_ && runtime_.system.systemContextFree) runtime_.system.systemContextFree(systemContext_);
  }
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  const std::string& label() const { return label_; }
  Qnn_ContextHandle_t handle() const { return context_; }
  bool isLeader() const { return context_ == runtime_.groupLeader; }

  const BinaryGraphInfo& graph(const std::string& name) const {
    if (name.empty()) {
      if (graphs_.size() != 1) throw std::runtime_error("Expected exactly one graph in " + label_);
      return graphs_[0];
    }
    for (const auto& g : graphs_) if (g.name == name) return g;
    throw std::runtime_error("Missing graph " + name + " in " + label_);
  }

 private:
  Runtime& runtime_;
  MappedFile binary_;  // Kept mapped until contextFree.
  std::string label_;
  QnnSystemContext_Handle_t systemContext_ = nullptr;
  Qnn_ContextHandle_t context_ = nullptr;
  std::vector<BinaryGraphInfo> graphs_;
};

struct Buffer {
  Qnn_Tensor_t tensor{};
  std::string name;
  std::vector<uint8_t> bytes;
  uint8_t* data() { return bytes.data(); }
};

// A graph with host buffers for every tensor. Binding sets are alternative input tensor lists
// that differ only in which shared buffers back some inputs (e.g. one KV set per Euler step).
class Graph {
 public:
  Graph(Runtime& runtime, const std::string& path, bool grouped = false)
      : Graph(runtime, std::make_shared<Context>(runtime, path, grouped), "") {}

  Graph(Runtime& runtime, std::shared_ptr<Context> context, const std::string& graphName)
      : runtime_(runtime), context_(std::move(context)) {
    const auto& graph = context_->graph(graphName);
    label_ = graphName.empty() ? context_->label() : graph.name;
    check(runtime_.qnn.graphRetrieve(context_->handle(), graph.name.c_str(), &graph_), "graphRetrieve " + graph.name);
    prepare(graph.inputs, graph.numInputs, inputs_, inputTensors_);
    prepare(graph.outputs, graph.numOutputs, outputs_, outputTensors_);
    sets_.push_back(inputTensors_);
  }

  ~Graph() {
    for (auto handle : memHandles_) runtime_.qnn.memDeRegister(&handle, 1);
  }
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

  const std::string& label() const { return label_; }
  Buffer& input(const std::string& name) { return find(inputs_, name); }
  Buffer& output(const std::string& name) { return find(outputs_, name); }
  const std::vector<Buffer>& inputs() const { return inputs_; }
  const std::vector<Buffer>& outputs() const { return outputs_; }

  size_t addBindingSet() { sets_.push_back(sets_[0]); return sets_.size() - 1; }

  // Binds a large input to externally owned storage; registered shared memory when possible.
  void bindInput(const std::string& name, const SharedBuffer& storage) { bindInput(0, name, storage); }

  void bindInput(size_t set, const std::string& name, const SharedBuffer& storage) {
    for (size_t i = 0; i < inputs_.size(); ++i) {
      auto& buffer = inputs_[i];
      if (buffer.name != name) continue;
      if (tensorBytes(buffer.tensor) != storage.bytes())
        throw std::runtime_error("Shared storage size does not match graph input " + name);
      if (set == 0) std::vector<uint8_t>().swap(buffer.bytes);
      Qnn_Tensor_t tensor = buffer.tensor;
      Qnn_MemHandle_t handle = nullptr;
      if (storage.fd() >= 0 && runtime_.qnn.memRegister) {
        Qnn_MemDescriptor_t descriptor = QNN_MEM_DESCRIPTOR_INIT;
        descriptor.memShape = {tensorRank(tensor), const_cast<uint32_t*>(tensorDimensions(tensor)), nullptr};
        descriptor.dataType = tensorType(tensor);
        descriptor.memType = QNN_MEM_TYPE_ION;
        descriptor.ionInfo.fd = storage.fd();
        if (runtime_.qnn.memRegister(context_->handle(), &descriptor, 1, &handle) != QNN_SUCCESS) handle = nullptr;
      }
      if (handle) {
        memHandles_.push_back(handle);
        setMemHandle(tensor, handle);
      } else {
        setClientBuffer(tensor, storage.data(), storage.bytes());
      }
      sets_[set][i] = tensor;
      if (set == 0) { buffer.tensor = tensor; inputTensors_[i] = tensor; }
      return;
    }
    throw std::runtime_error("Missing graph input: " + name);
  }

  void execute(size_t set = 0) {
    auto& inputs = set == 0 ? inputTensors_ : sets_[set];
    std::lock_guard<std::mutex> lock(runtime_.executeMutex);
    const auto start = Clock::now();
    check(runtime_.qnn.graphExecute(graph_, inputs.data(), inputs.size(), outputTensors_.data(),
                                    outputTensors_.size(), nullptr, nullptr),
          "graphExecute " + label_);
    runtime_.timing(label_, "execute", elapsedMs(start));
  }

  // Swaps an output's host buffer into the matching input instead of copying megabytes.
  void swapState(const std::string& outputName, const std::string& inputName) {
    auto& source = output(outputName);
    auto& target = input(inputName);
    if (tensorType(source.tensor) != tensorType(target.tensor) || source.bytes.size() != target.bytes.size())
      throw std::runtime_error("Incompatible state tensors: " + source.name + " -> " + target.name);
    source.bytes.swap(target.bytes);
    rebind(source);
    rebind(target);
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
    for (auto& set : sets_)
      for (auto& tensor : set)
        if (!std::strcmp(tensorName(tensor), buffer.name.c_str()) && TENSOR_FIELD(tensor, memType) == QNN_TENSORMEMTYPE_RAW)
          tensor = buffer.tensor;
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
  std::shared_ptr<Context> context_;  // Outlives memDeRegister in ~Graph.
  std::string label_;
  Qnn_GraphHandle_t graph_ = nullptr;
  std::vector<Buffer> inputs_, outputs_;
  std::vector<Qnn_Tensor_t> inputTensors_, outputTensors_;
  std::vector<std::vector<Qnn_Tensor_t>> sets_;
  std::vector<Qnn_MemHandle_t> memHandles_;
};

inline void expect(const Buffer& buffer, Qnn_DataType_t type, size_t bytes) {
  if (tensorType(buffer.tensor) != type || buffer.bytes.size() != bytes)
    throw std::runtime_error("Unexpected tensor contract: " + buffer.name);
}

inline void set(Buffer& buffer, const void* data, size_t size, Qnn_DataType_t type) {
  expect(buffer, type, size);
  std::memcpy(buffer.data(), data, size);
}

// Fixed-point I/O of a QNN tensor: real = scale * (q + offset). FLOAT_32 tensors pass through.
struct Affine {
  Qnn_DataType_t type = QNN_DATATYPE_FLOAT_32;
  float scale = 1.0f;
  int32_t offset = 0;
  bool operator==(const Affine& o) const { return type == o.type && scale == o.scale && offset == o.offset; }
};

inline Affine affineOf(const Qnn_Tensor_t& tensor) {
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

inline size_t affineWidth(const Affine& a) {
  return a.type == QNN_DATATYPE_UFIXED_POINT_8 ? 1 : a.type == QNN_DATATYPE_UFIXED_POINT_16 ? 2 : 4;
}

inline void quantizeTo(const Affine& a, const float* src, size_t count, uint8_t* dst) {
  if (a.type == QNN_DATATYPE_FLOAT_32) { std::memcpy(dst, src, count * sizeof(float)); return; }
  const float top = a.type == QNN_DATATYPE_UFIXED_POINT_8 ? 255.0f : 65535.0f;
  for (size_t i = 0; i < count; ++i) {
    const float q = std::clamp(std::nearbyint(src[i] / a.scale) - static_cast<float>(a.offset), 0.0f, top);
    if (a.type == QNN_DATATYPE_UFIXED_POINT_8) dst[i] = static_cast<uint8_t>(q);
    else reinterpret_cast<uint16_t*>(dst)[i] = static_cast<uint16_t>(q);
  }
}

inline void dequantizeFrom(const Affine& a, const uint8_t* src, size_t count, float* dst) {
  if (a.type == QNN_DATATYPE_FLOAT_32) { std::memcpy(dst, src, count * sizeof(float)); return; }
  for (size_t i = 0; i < count; ++i) {
    const float q = a.type == QNN_DATATYPE_UFIXED_POINT_8 ? src[i] : reinterpret_cast<const uint16_t*>(src)[i];
    dst[i] = a.scale * (q + static_cast<float>(a.offset));
  }
}

// Real values into / out of a FP32 or fixed-point graph tensor.
inline void setReal(Buffer& buffer, const float* src, size_t count) {
  const auto a = affineOf(buffer.tensor);
  if (buffer.bytes.size() != count * affineWidth(a)) throw std::runtime_error("Unexpected tensor size: " + buffer.name);
  quantizeTo(a, src, count, buffer.data());
}

inline void getReal(const Buffer& buffer, float* dst, size_t count) {
  const auto a = affineOf(buffer.tensor);
  if (buffer.bytes.size() != count * affineWidth(a)) throw std::runtime_error("Unexpected tensor size: " + buffer.name);
  dequantizeFrom(a, buffer.bytes.data(), count, dst);
}

}  // namespace qnnrt
