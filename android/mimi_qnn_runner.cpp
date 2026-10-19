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

constexpr size_t kFrameSamples = 1920;
constexpr uint32_t kSampleRate = 24000;
constexpr size_t kTemporalHeads = 32, kHeadBytes = 128 * sizeof(float);

using Clock = std::chrono::steady_clock;
double elapsedMs(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

int environmentLevel(const char* name, int fallback) {
  const char* value = std::getenv(name);
  return value && *value ? std::atoi(value) : fallback;
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
      throw std::runtime_error("Missing or unreadable context binary (check 644 permissions): " + path);
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

// Host-owned [heads, capacity, 128 floats] Temporal cache, laid out exactly like the graph input.
class KvRing {
 public:
  KvRing(SharedMemory& memory, size_t capacity) : memory_(memory), capacity_(capacity) {
    bytes_ = kTemporalHeads * capacity * kHeadBytes;
    data_ = static_cast<uint8_t*>(memory_.allocate(bytes_, fd_));
    if (!data_) { fallback_.resize(bytes_); data_ = fallback_.data(); }
    reset();
  }
  ~KvRing() { if (fd_ >= 0) memory_.release(data_); }
  KvRing(const KvRing&) = delete;
  KvRing& operator=(const KvRing&) = delete;

  // Masked slots must be finite zeros: softmax weights are zero but NaN * 0 is still NaN.
  void reset() { std::memset(data_, 0, bytes_); }
  void write(size_t position, const uint8_t* source, size_t sourceCapacity, size_t sourceSlot) {
    const size_t slot = position % capacity_;
    for (size_t head = 0; head < kTemporalHeads; ++head) {
      const auto* from = source + (head * sourceCapacity + sourceSlot) * kHeadBytes;
      const auto* values = reinterpret_cast<const float*>(from);
      for (size_t i = 0; i < kHeadBytes / sizeof(float); ++i)
        if (!std::isfinite(values[i])) throw std::runtime_error("Nonfinite new Temporal K/V");
      std::memcpy(data_ + (head * capacity_ + slot) * kHeadBytes, from, kHeadBytes);
    }
  }
  uint8_t* data() const { return data_; }
  size_t bytes() const { return bytes_; }
  size_t capacity() const { return capacity_; }
  int fd() const { return fd_; }

 private:
  SharedMemory& memory_;
  size_t capacity_, bytes_;
  int fd_ = -1;
  uint8_t* data_ = nullptr;
  std::vector<uint8_t> fallback_;
};

class Runtime {
 public:
  explicit Runtime(const std::string& libraryDir) : profileLevel_(environmentLevel("MOSHI_QNN_PROFILE", 0)) {
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
    if (profileLevel_ && qnn.profileCreate)
      check(qnn.profileCreate(backend, profileLevel_ > 1 ? QNN_PROFILE_LEVEL_DETAILED : QNN_PROFILE_LEVEL_BASIC, &profile),
            "profileCreate");
  }

  ~Runtime() {
    endPerformanceVote();
    if (profile && qnn.profileFree) qnn.profileFree(profile);
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

  void timing(const std::string& graph, const char* phase, double ms) { timings_.push_back({graph, phase, ms}); }

  void collectProfile(const std::string& graph) {
    if (!profile || !qnn.profileGetEvents || !qnn.profileGetEventData) return;
    const QnnProfile_EventId_t* events = nullptr;
    uint32_t count = 0;
    if (qnn.profileGetEvents(profile, &events, &count) != QNN_SUCCESS) return;
    std::function<void(QnnProfile_EventId_t, int)> visit = [&](QnnProfile_EventId_t id, int depth) {
      QnnProfile_EventData_t data{};
      if (qnn.profileGetEventData(id, &data) == QNN_SUCCESS)
        profileRows_.push_back({graph, data.type, data.unit, data.value, data.identifier ? data.identifier : ""});
      const QnnProfile_EventId_t* children = nullptr;
      uint32_t number = 0;
      if (depth < 4 && qnn.profileGetSubEvents && qnn.profileGetSubEvents(id, &children, &number) == QNN_SUCCESS)
        for (uint32_t i = 0; i < number; ++i) visit(children[i], depth + 1);
    };
    for (uint32_t i = 0; i < count; ++i) visit(events[i], 0);
  }

  void clearTimings() { timings_.clear(); profileRows_.clear(); }

  void writeTimings(const std::string& path) const {
    std::ofstream file(path);
    file << "graph,phase,milliseconds\n";
    for (const auto& row : timings_) file << row.graph << ',' << row.phase << ',' << row.ms << '\n';
    if (!file) throw std::runtime_error("Cannot write performance CSV: " + path);
    if (profileRows_.empty()) return;
    std::ofstream events(path + ".qnn.csv");
    events << "graph,event_type,event_unit,event_value,identifier\n";
    for (const auto& row : profileRows_) {
      std::string label;
      for (char c : row.identifier) { if (c == '"') label += '"'; label += c; }
      events << row.graph << ',' << row.type << ',' << row.unit << ',' << row.value << ",\"" << label << "\"\n";
    }
  }

  QNN_INTERFACE_VER_TYPE qnn{};
  QNN_SYSTEM_INTERFACE_VER_TYPE system{};
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  Qnn_ProfileHandle_t profile = nullptr;
  // Sequentially executed LM contexts share one spill-fill buffer through an HTP group.
  Qnn_ContextHandle_t groupLeader = nullptr;
  uint64_t groupSpillBytes = 0;
  SharedMemory sharedMemory;

 private:
  struct Timing { std::string graph; const char* phase; double ms; };
  struct ProfileRow { std::string graph; uint32_t type, unit; uint64_t value; std::string identifier; };
  int profileLevel_;
  std::vector<Timing> timings_;
  std::vector<ProfileRow> profileRows_;
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
  bool hasOutput(const std::string& name) const {
    return std::any_of(outputs_.begin(), outputs_.end(), [&](const Buffer& b) { return b.name == name; });
  }
  Buffer& output(const std::string& name) {
    if (hasOutput(name)) return find(outputs_, name);
    // AI Hub-compiled Mimi graphs keep input names but number their outputs.
    static const std::map<std::string, std::string> numbered = {
        {"codes", "output_0"}, {"audio", "output_0"}, {"position_out", "output_1"},
        {"kv_cache_out", "output_2"}, {"conv_state_out", "output_3"}};
    const auto alias = numbered.find(name);
    if (alias == numbered.end()) throw std::runtime_error("Missing QNN graph output: " + name);
    return find(outputs_, alias->second);
  }

  // Binds a large input to externally owned storage; registered shared memory when possible.
  void bindInput(const std::string& name, const KvRing& ring) {
    for (size_t i = 0; i < inputs_.size(); ++i) {
      auto& buffer = inputs_[i];
      if (buffer.name != name) continue;
      if (tensorBytes(buffer.tensor) != ring.bytes() || tensorType(buffer.tensor) != QNN_DATATYPE_FLOAT_32)
        throw std::runtime_error("KV storage does not match graph input " + name);
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
      sharedInputs_ += handle != nullptr;
      return;
    }
    throw std::runtime_error("Missing graph input: " + name);
  }
  size_t sharedInputs() const { return sharedInputs_; }

  void execute() {
    const auto start = Clock::now();
    check(runtime_.qnn.graphExecute(graph_, inputTensors_.data(), inputTensors_.size(), outputTensors_.data(),
                                    outputTensors_.size(), runtime_.profile, nullptr),
          "graphExecute " + label_);
    runtime_.timing(label_, "execute", elapsedMs(start));
    runtime_.collectProfile(label_);
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
  size_t sharedInputs_ = 0;
};

void expect(const Buffer& buffer, Qnn_DataType_t type, size_t bytes) {
  if (tensorType(buffer.tensor) != type || buffer.bytes.size() != bytes)
    throw std::runtime_error("Unexpected tensor contract: " + buffer.name);
}

void set(Buffer& buffer, const void* data, size_t size, Qnn_DataType_t type) {
  expect(buffer, type, size);
  std::memcpy(buffer.data(), data, size);
}

void requireFinite(const Buffer& buffer) {
  if (tensorType(buffer.tensor) != QNN_DATATYPE_FLOAT_32) throw std::runtime_error("Expected FP32: " + buffer.name);
  const auto* values = reinterpret_cast<const float*>(buffer.bytes.data());
  for (size_t i = 0; i < buffer.bytes.size() / sizeof(float); ++i)
    if (!std::isfinite(values[i])) throw std::runtime_error("Nonfinite output: " + buffer.name);
}

int16_t toPcm16(float value) {
  if (!std::isfinite(value)) throw std::runtime_error("Nonfinite decoder audio");
  return static_cast<int16_t>(std::lrintf(std::clamp(value, -1.0f, 32767.0f / 32768.0f) * 32768.0f));
}

uint32_t read32(const uint8_t* d) { return d[0] | d[1] << 8 | d[2] << 16 | uint32_t(d[3]) << 24; }
uint16_t read16(const uint8_t* d) { return uint16_t(d[0] | d[1] << 8); }

std::vector<int16_t> readWav(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open input WAV: " + path);
  std::vector<uint8_t> file((std::istreambuf_iterator<char>(input)), {});
  if (file.size() < 44 || std::memcmp(file.data(), "RIFF", 4) || std::memcmp(file.data() + 8, "WAVE", 4))
    throw std::runtime_error("Expected RIFF/WAVE input");
  bool formatSeen = false;
  std::vector<int16_t> samples;
  for (size_t cursor = 12; cursor + 8 <= file.size();) {
    const uint32_t length = read32(file.data() + cursor + 4);
    const size_t start = cursor + 8;
    if (length > file.size() - start) throw std::runtime_error("Malformed WAV chunk");
    if (!std::memcmp(file.data() + cursor, "fmt ", 4)) {
      if (length < 16 || read16(file.data() + start) != 1 || read16(file.data() + start + 2) != 1 ||
          read32(file.data() + start + 4) != kSampleRate || read16(file.data() + start + 14) != 16)
        throw std::runtime_error("Expected 24 kHz mono PCM16 WAV");
      formatSeen = true;
    } else if (!std::memcmp(file.data() + cursor, "data", 4)) {
      samples.resize(length / 2);
      std::memcpy(samples.data(), file.data() + start, samples.size() * 2);  // Android is little-endian.
    }
    cursor = start + length + (length & 1);
  }
  if (!formatSeen || samples.empty()) throw std::runtime_error("Missing WAV fmt or data chunk");
  return samples;
}

void writeWav(const std::string& path, const std::vector<int16_t>& samples) {
  std::ofstream output(path, std::ios::binary);
  const auto put32 = [&](uint32_t v) { output.write(reinterpret_cast<const char*>(&v), 4); };
  const auto put16 = [&](uint16_t v) { output.write(reinterpret_cast<const char*>(&v), 2); };
  const uint32_t dataBytes = static_cast<uint32_t>(samples.size() * 2);
  output.write("RIFF", 4); put32(36 + dataBytes); output.write("WAVEfmt ", 8);
  put32(16); put16(1); put16(1); put32(kSampleRate); put32(kSampleRate * 2); put16(2); put16(16);
  output.write("data", 4); put32(dataBytes);
  output.write(reinterpret_cast<const char*>(samples.data()), dataBytes);
  if (!output) throw std::runtime_error("Failed writing output WAV: " + path);
}

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

  void decode(const int32_t* codes, std::vector<int16_t>& pcm) {
    set(decoder_.input("codes"), codes, 8 * sizeof(int32_t), QNN_DATATYPE_INT_32);
    decoder_.execute();
    const auto* audio = reinterpret_cast<const float*>(decoder_.output("audio").data());
    for (size_t i = 0; i < kFrameSamples; ++i) pcm.push_back(toPcm16(audio[i]));
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

// Moshiko LM: frontend -> 16 Temporal shards (2 layers each) -> head -> 8 sequential DepFormer codebooks.
class MoshiLm {
 public:
  MoshiLm(Runtime& runtime, const std::string& directory) : runtime_(runtime) {
    std::vector<std::string> names = {"frontend"};
    for (int layer = 0; layer < 32; layer += 2) names.push_back(shardName(layer));
    names.push_back("head");
    for (int book = 0; book < 8; ++book) names.push_back("depformer_codebook_" + std::to_string(book));

    uint64_t spill = 0;
    for (const auto& name : names) spill = std::max(spill, spillFillBytes(directory + "/dlc/" + name + ".bin"));
    runtime_.groupSpillBytes = spill;
    for (const auto& name : names) graphs_[name] = std::make_unique<Graph>(runtime_, directory + "/dlc/" + name + ".bin", true);

    for (int layer = 0; layer < 32; layer += 2) {
      Graph& graph = *graphs_.at(shardName(layer));
      for (int j = 0; j < 4; ++j) {
        const std::string name = cacheName(layer + j / 2, j % 2);
        rings_[2 * layer + j] = std::make_unique<KvRing>(runtime_.sharedMemory, cacheCapacity(graph.input(name)));
        graph.bindInput(name, *rings_[2 * layer + j]);
      }
      shared_ += graph.sharedInputs();
    }
  }

  ~MoshiLm() {
    // Contexts first (they reference KV storage), and the group leader last.
    auto leader = std::move(graphs_.at("frontend"));
    graphs_.clear();
    leader.reset();
    runtime_.groupLeader = nullptr;
  }

  size_t sharedKvInputs() const { return shared_; }

  void reset() {
    delay_.reset();
    for (auto& ring : rings_) ring->reset();
    position_ = 0;
  }

  std::optional<std::array<int32_t, 9>> step(const std::array<int32_t, 8>& user) {
    const auto sequence = delay_.begin(user);
    Graph& frontend = *graphs_.at("frontend");
    set(frontend.input("sequence"), sequence.data(), sizeof(sequence), QNN_DATATYPE_INT_32);
    frontend.execute();
    const Buffer* hidden = &frontend.output("hidden");
    requireFinite(*hidden);

    const int32_t position = position_;
    for (int layer = 0; layer < 32; layer += 2) {
      Graph& graph = *graphs_.at(shardName(layer));
      set(graph.input("hidden"), hidden->bytes.data(), hidden->bytes.size(), QNN_DATATYPE_FLOAT_32);
      set(graph.input("position"), &position, sizeof(position), QNN_DATATYPE_INT_32);
      graph.execute();
      hidden = &graph.output(graph.hasOutput("output_hidden") ? "output_hidden" : "output_0");
      expect(*hidden, QNN_DATATYPE_FLOAT_32, 4096 * sizeof(float));
      requireFinite(*hidden);
      for (int j = 0; j < 4; ++j) {
        auto& output = temporalOutput(graph, layer, j);
        const size_t capacity = cacheCapacity(output);
        rings_[2 * layer + j]->write(position, output.data(), capacity, position % capacity);
      }
    }

    Graph& head = *graphs_.at("head");
    set(head.input("hidden"), hidden->bytes.data(), hidden->bytes.size(), QNN_DATATYPE_FLOAT_32);
    head.execute();
    const auto& temporal = head.output("temporal");
    requireFinite(temporal);
    auto& logits = head.output("text_logits");
    expect(logits, QNN_DATATYPE_FLOAT_32, 32000 * sizeof(float));
    requireFinite(logits);
    const auto* scores = reinterpret_cast<const float*>(logits.data());
    const int32_t text = static_cast<int32_t>(std::max_element(scores, scores + 32000) - scores);

    // The depth cache restarts every frame and is shared across the 8 codebooks.
    std::array<int32_t, 8> audio{};
    int32_t previous = text;
    Graph* last = nullptr;
    for (int book = 0; book < 8; ++book) {
      Graph& graph = *graphs_.at("depformer_codebook_" + std::to_string(book));
      set(graph.input("previous_token"), &previous, sizeof(previous), QNN_DATATYPE_INT_32);
      set(graph.input("temporal"), temporal.bytes.data(), temporal.bytes.size(), QNN_DATATYPE_FLOAT_32);
      for (int j = 0; j < 12; ++j) {
        auto& input = graph.input(cacheName(j / 2, j % 2));
        expect(input, QNN_DATATYPE_FLOAT_32, 16 * 8 * 64 * sizeof(float));
        if (last) std::memcpy(input.data(), last->output("output_" + input.name).data(), input.bytes.size());
        else std::fill(input.bytes.begin(), input.bytes.end(), 0);
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
  static std::string shardName(int layer) {
    return "temporal_layers_" + std::to_string(layer) + "_" + std::to_string(layer + 1);
  }
  static std::string cacheName(int layer, int value) {
    return "layer_" + std::to_string(layer) + (value ? "_value" : "_key");
  }
  static size_t cacheCapacity(const Buffer& buffer) {
    const auto* shape = tensorDimensions(buffer.tensor);
    if (tensorRank(buffer.tensor) != 4 || shape[0] != 1 || shape[1] != kTemporalHeads || !shape[2] ||
        shape[3] != 128 || tensorType(buffer.tensor) != QNN_DATATYPE_FLOAT_32)
      throw std::runtime_error("Unsupported Temporal cache layout: " + buffer.name);
    return shape[2];
  }
  // Only the delivered 0_1 shard numbers its outputs: hidden, then key/value per layer.
  static Buffer& temporalOutput(Graph& graph, int layer, int index) {
    const std::string name = "output_" + cacheName(layer + index / 2, index % 2);
    if (graph.hasOutput(name)) return graph.output(name);
    if (layer == 0) return graph.output("output_" + std::to_string(index + 1));
    throw std::runtime_error("Missing Temporal cache output: " + name);
  }

  Runtime& runtime_;
  std::array<std::unique_ptr<KvRing>, 64> rings_;
  std::map<std::string, std::unique_ptr<Graph>> graphs_;
  MoshiDelayRing delay_;
  int32_t position_ = 0;
  size_t shared_ = 0;
};

using FrameCallback = std::function<void(size_t, size_t, double, const std::vector<int16_t>&)>;

class Session {
 public:
  Session(const std::string& libraryDir, const std::string& encoder, const std::string& decoder,
          const std::string& lmDirectory)
      : runtime_(libraryDir) {
    if (!lmDirectory.empty()) lm_ = std::make_unique<MoshiLm>(runtime_, lmDirectory);
    codec_ = std::make_unique<Codec>(runtime_, encoder, decoder);
  }

  ~Session() { codec_.reset(); lm_.reset(); }

  size_t run(const std::string& inputWav, const std::string& outputWav, size_t frames,
             const FrameCallback& callback, bool highPerformance) {
    runtime_.clearTimings();
    const auto start = Clock::now();
    if (highPerformance) runtime_.beginPerformanceVote();
    try {
      const size_t emitted = lm_ ? runMoshi(inputWav, outputWav, frames, callback)
                                 : runCodec(inputWav, outputWav, frames, callback);
      runtime_.timing("pipeline", "wall_total", elapsedMs(start));
      runtime_.endPerformanceVote();
      runtime_.writeTimings(outputWav + ".performance.csv");
      writeConfiguration(outputWav + ".runtime.json", highPerformance);
      return emitted;
    } catch (...) {
      runtime_.endPerformanceVote();
      runtime_.timing("pipeline", "wall_failed", elapsedMs(start));
      try { runtime_.writeTimings(outputWav + ".performance.csv"); } catch (...) {}
      throw;
    }
  }

 private:
  static std::vector<int16_t> readFrames(const std::string& path, size_t& frames, size_t minimum) {
    auto samples = readWav(path);
    const size_t available = samples.size() / kFrameSamples;
    if (!frames) frames = available;
    if (frames < minimum || frames > available || frames > 3000)
      throw std::runtime_error("Requested frame count is outside 1..3000 complete 80 ms frames of the WAV");
    return samples;
  }

  size_t runCodec(const std::string& inputWav, const std::string& outputWav, size_t frames,
                  const FrameCallback& callback) {
    const auto samples = readFrames(inputWav, frames, 1);
    codec_->reset();
    std::vector<int16_t> result;
    std::ofstream codesFile(outputWav + ".codes.txt");
    for (size_t frame = 0; frame < frames; ++frame) {
      const auto start = Clock::now();
      const auto codes = codec_->encode(samples.data() + frame * kFrameSamples);
      std::vector<int16_t> chunk;
      codec_->decode(codes.data(), chunk);
      result.insert(result.end(), chunk.begin(), chunk.end());
      codesFile << frame;
      for (auto code : codes) codesFile << ',' << code;
      codesFile << '\n';
      if (callback) callback(frame + 1, frame + 1, elapsedMs(start), chunk);
    }
    if (!codesFile) throw std::runtime_error("Failed writing codes file");
    writeWav(outputWav, result);
    return frames;
  }

  // The first input frame only primes the delay ring, so N input frames yield N-1 output frames.
  size_t runMoshi(const std::string& inputWav, const std::string& outputWav, size_t frames,
                  const FrameCallback& callback) {
    const auto samples = readFrames(inputWav, frames, 2);
    codec_->reset();
    lm_->reset();
    std::vector<int16_t> result;
    std::ofstream codes(outputWav + ".codes.txt"), text(outputWav + ".text_tokens.txt");
    size_t emitted = 0;
    for (size_t frame = 0; frame < frames; ++frame) {
      const auto start = Clock::now();
      const auto generated = lm_->step(codec_->encode(samples.data() + frame * kFrameSamples));
      std::vector<int16_t> chunk;
      if (generated) {
        codec_->decode(generated->data() + 1, chunk);
        result.insert(result.end(), chunk.begin(), chunk.end());
        codes << emitted;
        for (size_t i = 1; i < 9; ++i) codes << ',' << (*generated)[i];
        codes << '\n';
        text << emitted << ',' << (*generated)[0] << '\n';
        ++emitted;
      }
      const double ms = elapsedMs(start);
      runtime_.timing("frame", "total", ms);
      if (callback) callback(frame + 1, emitted, ms, chunk);
    }
    if (!codes || !text) throw std::runtime_error("Failed writing Moshi token files");
    writeWav(outputWav, result);
    return emitted;
  }

  void writeConfiguration(const std::string& path, bool highPerformance) const {
    std::ofstream file(path);
    file << "{\"format\":\"moshi-runtime-configuration-v2\",\"burst_vote\":" << (highPerformance ? "true" : "false")
         << ",\"lm\":" << (lm_ ? "true" : "false")
         << ",\"shared_spill_bytes\":" << runtime_.groupSpillBytes
         << ",\"shared_memory_kv_inputs\":" << (lm_ ? lm_->sharedKvInputs() : 0) << "}\n";
    if (!file) throw std::runtime_error("Cannot write runtime configuration report");
  }

  Runtime runtime_;
  std::unique_ptr<MoshiLm> lm_;
  std::unique_ptr<Codec> codec_;
};

}  // namespace

// cacheDirectory and fast are accepted for JNI compatibility; context binaries need neither.
void* createConfiguredCodecSession(const std::string& libraryDir, const std::string& encoder,
                                   const std::string& decoder, const std::string& lmDirectory,
                                   const std::string& /*cacheDirectory*/, bool /*fast*/) {
  return new Session(libraryDir, encoder, decoder, lmDirectory);
}

size_t runMimiCodecSessionStreaming(void* session, const std::string& inputWav, const std::string& outputWav,
                                    size_t frames, const FrameCallback& callback, bool highPerformance) {
  if (!session) throw std::runtime_error("Models have not been loaded");
  return static_cast<Session*>(session)->run(inputWav, outputWav, frames, callback, highPerformance);
}

void destroyMimiCodecSession(void* session) { delete static_cast<Session*>(session); }

#ifndef MIMI_CODEC_LIBRARY
int main(int argc, char** argv) {
  try {
    if (argc < 7 || argc > 9 || (argc == 9 && std::string(argv[8]) != "--repeat")) {
      std::cerr << "Usage: mimi_qnn_runner LIB_DIR ENCODER.bin DECODER.bin INPUT.wav OUTPUT.wav FRAMES "
                   "[LM_BUNDLE_DIR [--repeat]]  (FRAMES 0 = whole WAV; MOSHI_HTP_PERFORMANCE=0 disables burst)\n";
      return 2;
    }
    const bool burst = environmentLevel("MOSHI_HTP_PERFORMANCE", 1) != 0;
    const auto prepare = Clock::now();
    Session session(argv[1], argv[2], argv[3], argc >= 8 ? argv[7] : "");
    std::cerr << "prepare_ms=" << elapsedMs(prepare) << '\n';
    for (int pass = 0; pass < (argc == 9 ? 2 : 1); ++pass) {
      const auto start = Clock::now();
      const std::string output = pass ? std::string(argv[5]) + ".repeat.wav" : argv[5];
      const size_t emitted = session.run(argv[4], output, std::stoul(argv[6]), {}, burst);
      std::cerr << "run=" << pass << " emitted_frames=" << emitted << " total_ms=" << elapsedMs(start) << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "mimi_qnn_runner error: " << error.what() << '\n';
    return 1;
  }
}
#endif
