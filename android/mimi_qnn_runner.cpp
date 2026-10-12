// Minimal on-device Mimi DLC loopback: PCM16 WAV -> QNN encoder -> QNN decoder -> WAV.
// Uses the QAIRT 2.45 public QNN interfaces. Each graph owns its streaming state.

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <map>
#include <stdexcept>
#include <sstream>
#include <set>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpSystemContext.h"
#include "moshi_lm_scheduler.h"
#include "moshi_cache_prefix.h"
#include "moshi_kv_ring.h"
#include "moshi_graph_io_pool.h"

namespace {

constexpr size_t kFrameSamples = 1920;
constexpr uint32_t kSampleRate = 24000;

double elapsedMs(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

uint64_t availableMemoryBytes() {
  std::ifstream file("/proc/meminfo");
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream row(line);
    std::string key; uint64_t kib = 0;
    if (row >> key >> kib && key == "MemAvailable:") return kib * 1024;
  }
  return 0; // No reliable budget: do not expand residency.
}

bool enabledEnvironment(const char* name) {
  const char* value = std::getenv(name);
  return value && std::string(value) == "1";
}

class ContextAllocationError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Context binaries are device/runtime-specific caches, not replacement model files.
// Keep the mapping alive until contextFree, including backends which retain it.
struct MappedBinary {
  void* data = MAP_FAILED;
  size_t size = 0;
  ~MappedBinary() { clear(); }
  void clear() { if (data != MAP_FAILED) munmap(data, size); data = MAP_FAILED; size = 0; }
  bool open(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat info{};
    if (fstat(fd, &info) || info.st_size <= 0) { close(fd); return false; }
    size = static_cast<size_t>(info.st_size);
    data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    return data != MAP_FAILED;
  }
};

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
    if (enabledEnvironment("MOSHI_QNN_PROFILE") && qnn.profileCreate) {
      const auto level = enabledEnvironment("MOSHI_QNN_PROFILE_DETAILED") ? QNN_PROFILE_LEVEL_DETAILED : QNN_PROFILE_LEVEL_BASIC;
      check(qnn.profileCreate(backend, level, &profile), "profileCreate");
    }
  }

  ~Runtime() {
    endPerformanceVote();
    if (profile && qnn.profileFree) qnn.profileFree(profile);
    if (device && qnn.deviceFree) qnn.deviceFree(device);
    if (backend && qnn.backendFree) qnn.backendFree(backend);
    // The HTP backend can retain process-wide cleanup hooks, so leave its SO loaded.
  }

  const QnnInterface_t& provider() const { return *qnnProvider_; }
  void beginPerformanceVote() {
    QnnDevice_Infrastructure_t infrastructure = nullptr;
    if (!qnn.deviceGetInfrastructure || qnn.deviceGetInfrastructure(&infrastructure) != QNN_SUCCESS ||
        !infrastructure || infrastructure->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) return;
    perf_ = infrastructure->perfInfra;
    if (!perf_.createPowerConfigId || !perf_.destroyPowerConfigId || !perf_.setPowerConfig ||
        perf_.createPowerConfigId(0, 0, &powerConfigId_) != QNN_SUCCESS) return;
    hasPowerConfig_ = true;
    QnnHtpPerfInfrastructure_PowerConfig_t config{};
    config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    auto& vote = config.dcvsV3Config;
    vote.contextId = powerConfigId_;
    vote.setDcvsEnable = 1; vote.dcvsEnable = 1; // Allow scaling; never disable thermal control.
    vote.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    vote.setSleepLatency = 1; vote.sleepLatency = 100;
    vote.setBusParams = 1; vote.setCoreParams = 1;
    vote.busVoltageCornerMin = vote.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS;
    vote.busVoltageCornerTarget = vote.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
    vote.busVoltageCornerMax = vote.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO;
    const QnnHtpPerfInfrastructure_PowerConfig_t* configs[] = {&config, nullptr};
    const auto status = perf_.setPowerConfig(powerConfigId_, configs);
    timing("HTP", status == QNN_SUCCESS ? "performance_vote_accepted" : "performance_vote_rejected", 0);
    if (status != QNN_SUCCESS) endPerformanceVote();
  }
  void endPerformanceVote() {
    if (hasPowerConfig_) {
      perf_.destroyPowerConfigId(powerConfigId_);
      hasPowerConfig_ = false;
    }
  }
  void timing(const std::string& graph, const std::string& phase, double milliseconds) {
    timings.push_back({graph, phase, milliseconds});
    std::cerr << "moshi_timing graph=" << graph << " phase=" << phase << " ms=" << milliseconds << '\n';
  }
  void writeTimings(const std::string& path) const {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Cannot write performance CSV: " + path);
    file << "graph,phase,milliseconds\n";
    for (const auto& row : timings) file << row.graph << ',' << row.phase << ',' << row.ms << '\n';
    if (!file) throw std::runtime_error("Cannot finish performance CSV: " + path);
    if (!profileRows.empty()) {
      std::ofstream events(path + ".qnn.csv");
      if (!events) throw std::runtime_error("Cannot write QNN profile CSV");
      events << "graph,event_type,event_unit,event_value,identifier\n";
      for (const auto& row : profileRows) {
        std::string label;
        for (char c : row.identifier) { if (c == '"') label += '"'; label += c; }
        events << row.graph << ',' << row.type << ',' << row.unit << ',' << row.value << ",\"" << label << "\"\n";
      }
    }
  }
  void collectProfile(const std::string& graph) {
    if (!profile || !qnn.profileGetEvents || !qnn.profileGetEventData) return;
    const QnnProfile_EventId_t* events = nullptr; uint32_t count = 0;
    if (qnn.profileGetEvents(profile, &events, &count) != QNN_SUCCESS) return;
    std::function<void(QnnProfile_EventId_t, int)> visit = [&](QnnProfile_EventId_t id, int depth) {
      QnnProfile_EventData_t data{};
      if (qnn.profileGetEventData(id, &data) == QNN_SUCCESS)
        profileRows.push_back({graph, data.type, data.unit, data.value, data.identifier ? data.identifier : ""});
      const QnnProfile_EventId_t* children = nullptr; uint32_t number = 0;
      if (depth < 4 && qnn.profileGetSubEvents && qnn.profileGetSubEvents(id, &children, &number) == QNN_SUCCESS)
        for (uint32_t i = 0; i < number; ++i) visit(children[i], depth + 1);
    };
    for (uint32_t i = 0; i < count; ++i) visit(events[i], 0);
  }
  struct Timing { std::string graph, phase; double ms; };
  std::vector<Timing> timings;
  struct ProfileRow { std::string graph; uint32_t type, unit; uint64_t value; std::string identifier; };
  std::vector<ProfileRow> profileRows;
  Qnn_ProfileHandle_t profile = nullptr;

  QNN_INTERFACE_VER_TYPE qnn{};
  QNN_SYSTEM_INTERFACE_VER_TYPE system{};
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;

 private:
  void* backendLibrary_ = nullptr;
  void* systemLibrary_ = nullptr;
  const QnnInterface_t* qnnProvider_ = nullptr;
  QnnHtpDevice_PerfInfrastructure_t perf_{};
  uint32_t powerConfigId_ = 0;
  bool hasPowerConfig_ = false;
};

class Graph {
 public:
  Graph(Runtime& runtime, const std::string& dlcPath, const std::string& cachePath = "",
        GraphIoPool* ioPool = nullptr) : runtime_(runtime), ioPool_(ioPool) {
    try {
    label_ = dlcPath.substr(dlcPath.find_last_of('/') + 1);
    const auto start = std::chrono::steady_clock::now();
    if (!cachePath.empty() && loadBinary(cachePath)) {
      runtime_.timing(label_, "restore_context", elapsedMs(start));
      return;
    }
    // SDK 1002 can hide an unreadable external-storage file. Fail clearly first.
    {
      std::ifstream readable(dlcPath, std::ios::binary);
      if (!readable || readable.peek() == std::char_traits<char>::eof())
        throw std::runtime_error("DLC is unreadable or empty: " + dlcPath +
                                 "; check ADB file permissions (644)");
    }
    check(runtime_.qnn.contextCreate(runtime_.backend, runtime_.device, nullptr, &context_),
          "contextCreate " + dlcPath);
    check(runtime_.system.systemDlcCreateFromFile(nullptr, dlcPath.c_str(), &dlc_),
          "systemDlcCreateFromFile " + dlcPath);
    runtime_.timing(label_, "open_dlc", elapsedMs(start));
    auto phaseStart = std::chrono::steady_clock::now();
    uint32_t count = 0;
    check(runtime_.system.systemDlcComposeGraphs(
              dlc_, nullptr, 0, runtime_.backend, context_, runtime_.provider(),
              QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1, &graphs_, &count),
          "systemDlcComposeGraphs " + dlcPath);
    runtime_.timing(label_, "compose", elapsedMs(phaseStart));
    if (count != 1 || !graphs_ || graphs_[0].version !=
                                  QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
      throw std::runtime_error("Expected one DLC graph with v1 graph metadata");
    }
    const auto& graphInfo = graphs_[0].graphInfoV1;
    name_ = graphInfo.graphName;
    check(runtime_.qnn.graphRetrieve(context_, name_.c_str(), &graph_),
          "graphRetrieve " + name_);
    phaseStart = std::chrono::steady_clock::now();
    check(runtime_.qnn.graphFinalize(graph_, nullptr, nullptr), "graphFinalize " + name_);
    runtime_.timing(label_, "finalize", elapsedMs(phaseStart));
    if (!cachePath.empty()) saveBinary(cachePath);
    phaseStart = std::chrono::steady_clock::now();
    prepare(graphInfo.graphInputs, graphInfo.numGraphInputs, inputs_, inputTensors_, "input");
    prepare(graphInfo.graphOutputs, graphInfo.numGraphOutputs, outputs_, outputTensors_, "output");
    runtime_.timing(label_, "allocate_io", elapsedMs(phaseStart));
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
    context_ = nullptr;
    if (ioPool_) {
      for (size_t i = 0; i < inputs_.size(); ++i) ioPool_->put("input", i, inputs_[i].bytes);
      for (size_t i = 0; i < outputs_.size(); ++i) ioPool_->put("output", i, outputs_[i].bytes);
    }
    if (graphs_) std::free(graphs_);
    graphs_ = nullptr;
    if (dlc_ && runtime_.system.systemDlcFree) runtime_.system.systemDlcFree(dlc_);
    dlc_ = nullptr;
    if (systemContext_ && runtime_.system.systemContextFree) runtime_.system.systemContextFree(systemContext_);
    systemContext_ = nullptr;
    binary_.clear();
  }

  bool loadBinary(const std::string& path) {
    if (!runtime_.qnn.contextCreateFromBinary || !runtime_.system.systemContextGetBinaryInfo || !binary_.open(path)) return false;
    try {
      check(runtime_.system.systemContextCreate(&systemContext_), "systemContextCreate");
      const QnnSystemContext_BinaryInfo_t* info = nullptr;
      Qnn_ContextBinarySize_t metadataSize = 0;
      check(runtime_.system.systemContextGetBinaryInfo(systemContext_, binary_.data, binary_.size, &info, &metadataSize), "context metadata");
      QnnSystemContext_GraphInfo_t* graphs = nullptr;
      uint32_t count = 0;
      if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) { graphs = info->contextBinaryInfoV1.graphs; count = info->contextBinaryInfoV1.numGraphs; }
      else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) { graphs = info->contextBinaryInfoV2.graphs; count = info->contextBinaryInfoV2.numGraphs; }
      else if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) { graphs = info->contextBinaryInfoV3.graphs; count = info->contextBinaryInfoV3.numGraphs; }
      if (count != 1 || !graphs) throw std::runtime_error("Expected one cached graph");
      QnnSystemContext_GraphInfoV1_t graphInfo{};
      if (graphs[0].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) graphInfo = graphs[0].graphInfoV1;
      else if (graphs[0].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        const auto& g = graphs[0].graphInfoV2;
        graphInfo.graphName = g.graphName; graphInfo.graphInputs = g.graphInputs; graphInfo.numGraphInputs = g.numGraphInputs;
        graphInfo.graphOutputs = g.graphOutputs; graphInfo.numGraphOutputs = g.numGraphOutputs;
      } else if (graphs[0].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        const auto& g = graphs[0].graphInfoV3;
        graphInfo.graphName = g.graphName; graphInfo.graphInputs = g.graphInputs; graphInfo.numGraphInputs = g.numGraphInputs;
        graphInfo.graphOutputs = g.graphOutputs; graphInfo.numGraphOutputs = g.numGraphOutputs;
      } else throw std::runtime_error("Unsupported cached graph metadata");
      check(runtime_.qnn.contextCreateFromBinary(runtime_.backend, runtime_.device, nullptr, binary_.data, binary_.size, &context_, nullptr), "contextCreateFromBinary");
      name_ = graphInfo.graphName;
      check(runtime_.qnn.graphRetrieve(context_, name_.c_str(), &graph_), "cached graphRetrieve");
      prepare(graphInfo.graphInputs, graphInfo.numGraphInputs, inputs_, inputTensors_, "input");
      prepare(graphInfo.graphOutputs, graphInfo.numGraphOutputs, outputs_, outputTensors_, "output");
      return true; // Already finalized: never finalize a deserialized context.
    } catch (const std::exception& error) {
      std::cerr << "Context cache rejected; rebuilding DLC: " << error.what() << '\n';
      release(); inputs_.clear(); outputs_.clear(); inputTensors_.clear(); outputTensors_.clear();
      return false;
    }
  }

  void saveBinary(const std::string& path) {
    if (!runtime_.qnn.contextGetBinarySize || !runtime_.qnn.contextGetBinary) return;
    const auto start = std::chrono::steady_clock::now();
    Qnn_ContextBinarySize_t size = 0, written = 0;
    if (runtime_.qnn.contextGetBinarySize(context_, &size) != QNN_SUCCESS || !size) return;
    struct statvfs space{};
    const std::string directory = path.substr(0, path.find_last_of('/'));
    if (statvfs(directory.c_str(), &space) || size + (512ULL << 20) >
        static_cast<uint64_t>(space.f_bavail) * space.f_frsize) {
      runtime_.timing(label_, "save_context_no_space", elapsedMs(start));
      return;
    }
    std::string temporary = path + ".tmp-XXXXXX";
    std::vector<char> filename(temporary.begin(), temporary.end()); filename.push_back('\0');
    int fd = mkstemp(filename.data());
    if (fd < 0) return; // Cache failures must not prevent model execution.
    bool saved = false;
    if (!ftruncate(fd, static_cast<off_t>(size))) {
      void* data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      if (data != MAP_FAILED) {
        saved = runtime_.qnn.contextGetBinary(context_, data, size, &written) == QNN_SUCCESS && written > 0 && written <= size;
        if (saved) saved = msync(data, size, MS_SYNC) == 0;
        munmap(data, size);
      }
    }
    if (saved) saved = ftruncate(fd, static_cast<off_t>(written)) == 0 && fsync(fd) == 0;
    close(fd);
    if (saved) saved = rename(filename.data(), path.c_str()) == 0;
    if (!saved) unlink(filename.data()); // Only our uniquely created temporary cache.
    runtime_.timing(label_, saved ? "save_context" : "save_context_failed", elapsedMs(start));
  }

 public:

  Buffer& input(const std::string& name) { return find(inputs_, name); }
  bool hasOutput(const std::string& name) const {
    return std::any_of(outputs_.begin(), outputs_.end(), [&](const Buffer& b) { return b.name == name; });
  }
  const std::vector<Buffer>& benchmarkOutputs() const { return outputs_; }
  Buffer& output(const std::string& name) {
    resumeIo();
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
    resumeIo();
    const auto start = std::chrono::steady_clock::now();
    check(runtime_.qnn.graphExecute(graph_, inputTensors_.data(), inputTensors_.size(),
                                    outputTensors_.data(), outputTensors_.size(), runtime_.profile,
                                    nullptr),
          "graphExecute " + name_);
    const auto end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(end - start).count();
    runtime_.timing(label_, "execute", ms);
    runtime_.collectProfile(label_);
    return ms;
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

  void suspendIo() {
    if (!ioPool_ || !ioActive_) return;
    for (size_t i = 0; i < inputs_.size(); ++i) ioPool_->put("input", i, inputs_[i].bytes);
    for (size_t i = 0; i < outputs_.size(); ++i) ioPool_->put("output", i, outputs_[i].bytes);
    ioActive_ = false;
  }

  void resumeIo() {
    if (ioActive_) return;
    const auto start = std::chrono::steady_clock::now();
    auto restore = [&](const std::string& role, std::vector<Buffer>& buffers, std::vector<Qnn_Tensor_t>& tensors) {
      for (size_t i = 0; i < buffers.size(); ++i) {
        auto& buffer = buffers[i];
        ioPool_->take(role, i, tensorBytes(buffer.tensor), buffer.bytes);
        setClientBuffer(buffer.tensor, buffer.bytes.data(), static_cast<uint32_t>(buffer.bytes.size()));
        tensors[i] = buffer.tensor;
      }
    };
    restore("input", inputs_, inputTensors_); restore("output", outputs_, outputTensors_);
    ioActive_ = true;
    runtime_.timing(label_, "rebind_io", elapsedMs(start));
  }

 private:
  Buffer& find(std::vector<Buffer>& buffers, const std::string& name) {
    resumeIo();
    for (auto& buffer : buffers) {
      if (buffer.name == name) return buffer;
    }
    throw std::runtime_error("Missing QNN graph tensor: " + name);
  }

  void prepare(Qnn_Tensor_t* metadata, uint32_t count,
               std::vector<Buffer>& buffers, std::vector<Qnn_Tensor_t>& tensors,
               const std::string& role) {
    if (count == 0) throw std::runtime_error("Graph has no tensors");
    buffers.resize(count);
    tensors.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
      auto& buffer = buffers[i];
      buffer.tensor = metadata[i];
      buffer.name = tensorName(metadata[i]);
      if (ioPool_) ioPool_->take(role, i, tensorBytes(metadata[i]), buffer.bytes);
      else buffer.bytes.resize(tensorBytes(metadata[i]));
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
  GraphIoPool* ioPool_;
  bool ioActive_ = true;
  Qnn_ContextHandle_t context_ = nullptr;
  QnnSystemDlc_Handle_t dlc_ = nullptr;
  QnnSystemContext_GraphInfo_t* graphs_ = nullptr;
  Qnn_GraphHandle_t graph_ = nullptr;
  QnnSystemContext_Handle_t systemContext_ = nullptr;
  MappedBinary binary_;
  std::string name_;
  std::string label_;
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
    // This is a conservative SOURCE-WEIGHT budget, not a claimed hard RSS cap.
    // Keep OS/headroom; never automatically try to fill all physical RAM.
    const uint64_t available = availableMemoryBytes();
    uint64_t budget = available >= (8ULL << 30) ? std::min<uint64_t>(3ULL << 30, available / 4) : 0;
    if (const char* experiment = std::getenv("MOSHI_TEMPORAL_BUDGET_MIB")) {
      const uint64_t requested = std::stoull(experiment);
      if (requested > 6144) throw std::runtime_error("Temporal experiment budget exceeds 6144 MiB");
      const uint64_t headroom = 6ULL << 30;
      budget = available > headroom ? std::min<uint64_t>(requested << 20, available - headroom) : 0;
    }
    if (const char* limit = std::getenv("MOSHI_TEMPORAL_RESIDENT_MIB"))
      budget = std::min<uint64_t>(budget, std::stoull(limit) << 20);
    std::vector<std::pair<uint64_t, int>> candidates;
    for (int layer = 0; layer < 32; layer += 2) {
      struct stat info{};
      if (!stat(path(shardName(layer)).c_str(), &info)) candidates.emplace_back(info.st_size, layer);
    }
    std::sort(candidates.begin(), candidates.end());
    for (const auto& candidate : candidates) {
      if (candidate.first <= budget) { residentTemporal_.insert(candidate.second); budget -= candidate.first; }
    }
    std::cerr << "Temporal resident plan: " << residentTemporal_.size() << " shards; shared I/O pool; available_memory=" << available << '\n';
  }

  void configureCache(const std::string& directory) {
    if (!cacheDirectory_.empty()) return;
    const char* overrideDirectory = std::getenv("MOSHI_CONTEXT_CACHE_DIR");
    cacheDirectory_ = overrideDirectory && *overrideDirectory ? overrideDirectory : directory;
    mkdir(cacheDirectory_.c_str(), 0700);
  }

  void reset() {
    ring_.reset();
    for (auto& cache : temporalCache_) std::vector<uint8_t>().swap(cache);
    for (auto& cache : temporalRing_) if (cache) cache->reset();
  }

  std::optional<std::array<int32_t, 9>> step(const std::array<int32_t, 8>& user,
                                           int32_t position) {
    auto sequence = ring_.begin(user);
    std::vector<uint8_t> hidden;
    {
      Graph& graph = resident("frontend");
      set(graph.input("sequence"), sequence.data(), sizeof(sequence), QNN_DATATYPE_INT_32);
      graph.execute();
      finite(graph.output("hidden"));
      hidden = graph.output("hidden").bytes;
    }
    for (int layer = 0; layer < 32; layer += 2) {
      std::unique_ptr<Graph> paged;
      Graph& graph = temporal(layer, paged);
      const auto hostStart = std::chrono::steady_clock::now();
      set(graph.input("hidden"), hidden.data(), hidden.size(), QNN_DATATYPE_FLOAT_32);
      set(graph.input("position"), &position, sizeof(position), QNN_DATATYPE_INT_32);
      for (int j = 0; j < 4; ++j) {
        const std::string name = cacheName(layer + j / 2, j % 2);
        auto& input = graph.input(name);
        const size_t capacity = cacheCapacity(input);
        auto& output = temporalOutput(graph, layer, j);
        const size_t outputCapacity = cacheCapacity(output);
        if (outputCapacity != 1 && outputCapacity != capacity)
          throw std::runtime_error("Temporal cache input/output capacity mismatch");
        auto& ring = temporalRing_[2 * layer + j];
        if (capacity <= 1000 || outputCapacity == 1) {
          if (!ring) ring = std::make_unique<MoshiKvRing>(32, capacity, 128 * sizeof(float));
          if (position < 0 || ring->capacity() != capacity || ring->nextPosition() != static_cast<uint64_t>(position))
            throw std::runtime_error("Temporal ring position or model capacity changed without reset");
          std::memcpy(input.bytes.data(), ring->bytes().data(), input.bytes.size());
          continue;
        }
        auto& cache = temporalCache_[2 * layer + j];
        // Preserve the delivered full-cache ABI and compact-prefix behavior.
        // Optimized/smaller models above use preallocated ring state instead.
        MoshiCachePrefix::restore(input.bytes, cache, 32, capacity, 128 * sizeof(float), position);
      }
      runtime_.timing(shardName(layer), "restore_kv", elapsedMs(hostStart));
      graph.execute();
      // This delivered 0_1 DLC alone uses numbered outputs. The verified source
      // order is hidden, layer_0_key, layer_0_value, layer_1_key, layer_1_value.
      if (!graph.hasOutput("output_hidden") && layer != 0)
        throw std::runtime_error("Unexpected numbered Temporal hidden output");
      auto& nextHidden = graph.output(graph.hasOutput("output_hidden") ? "output_hidden" : "output_0");
      expect(nextHidden, QNN_DATATYPE_FLOAT_32, 4096 * sizeof(float));
      finite(nextHidden);
      hidden = nextHidden.bytes;
      const auto packStart = std::chrono::steady_clock::now();
      for (int j = 0; j < 4; ++j) {
        auto& output = temporalOutput(graph, layer, j);
        const size_t outputCapacity = cacheCapacity(output);
        auto& ring = temporalRing_[2 * layer + j];
        if (ring) {
          std::array<uint8_t, 32 * 128 * sizeof(float)> delta{};
          for (size_t head = 0; head < 32; ++head)
            std::memcpy(delta.data() + head * 128 * sizeof(float),
                        output.bytes.data() + (head * outputCapacity + position % outputCapacity) * 128 * sizeof(float),
                        128 * sizeof(float));
          for (size_t offset = 0; offset < delta.size(); offset += sizeof(float)) {
            float value; std::memcpy(&value, delta.data() + offset, sizeof(value));
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite new K/V: " + output.name);
          }
          ring->append(position, delta.data(), delta.size());
          continue;
        }
        auto& cache = temporalCache_[2 * layer + j];
        MoshiCachePrefix::pack(output.bytes, cache, 32, outputCapacity, 128 * sizeof(float), position + 1);
      }
      runtime_.timing(shardName(layer), "pack_kv", elapsedMs(packStart));
      graph.suspendIo(); // Context may remain resident; scratch buffers do not.
    }
    std::vector<uint8_t> temporal;
    int32_t previous;
    {
      Graph& graph = resident("head");
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
      Graph& graph = resident("depformer_codebook_" + std::to_string(book));
      set(graph.input("previous_token"), &previous, sizeof(previous), QNN_DATATYPE_INT_32);
      set(graph.input("temporal"), temporal.data(), temporal.size(), QNN_DATATYPE_FLOAT_32);
      for (int j = 0; j < 12; ++j) {
        auto& input = graph.input(cacheName(j / 2, j % 2));
        expect(input, QNN_DATATYPE_FLOAT_32, 16 * 8 * 64 * sizeof(float));
        if (!depthCache[j].empty()) set(input, depthCache[j].data(), depthCache[j].size(), QNN_DATATYPE_FLOAT_32);
        else std::fill(input.bytes.begin(), input.bytes.end(), 0); // Resident graphs retain last frame's inputs.
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
  Graph& temporal(int layer, std::unique_ptr<Graph>& paged) {
    const std::string name = shardName(layer);
    auto found = temporalGraphs_.find(layer);
    if (found != temporalGraphs_.end()) {
      runtime_.timing(name, "resident_reuse", 0);
      return *found->second;
    }
    if (residentTemporal_.count(layer) && availableMemoryBytes() >= (4ULL << 30)) {
      auto graph = std::make_unique<Graph>(runtime_, path(name), cachePath(name), &temporalIoPool_);
      auto& result = *graph;
      temporalGraphs_[layer] = std::move(graph);
      return result;
    }
    paged = std::make_unique<Graph>(runtime_, path(name), cachePath(name), &temporalIoPool_);
    return *paged;
  }
  Graph& resident(const std::string& name) {
    auto& graph = residentGraphs_[name];
    if (!graph) graph = std::make_unique<Graph>(runtime_, path(name), cachePath(name));
    else runtime_.timing(name, "resident_reuse", 0);
    return *graph;
  }
  std::string cachePath(const std::string& name) const {
    struct stat info{}, directoryInfo{};
    if (stat(path(name).c_str(), &info) || stat(cacheDirectory_.c_str(), &directoryInfo) ||
        access(cacheDirectory_.c_str(), W_OK)) return "";
    const auto& api = runtime_.provider().apiVersion;
    // Bundles are SHA256-verified by the App; size+mtime invalidate replaced DLCs.
    // Cache remains on this phone. API/backend version changes select a new file.
    return cacheDirectory_ + "/" + name + "-" + std::to_string(info.st_size) + "-" +
        std::to_string(info.st_mtime) + "-" + std::to_string(api.coreApiVersion.major) + "." +
        std::to_string(api.coreApiVersion.minor) + "." + std::to_string(api.coreApiVersion.patch) + "-" +
        std::to_string(api.backendApiVersion.major) + "." + std::to_string(api.backendApiVersion.minor) + "." +
        std::to_string(api.backendApiVersion.patch) + ".bin";
  }
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
  static size_t cacheCapacity(Buffer& buffer) {
    const auto* shape = tensorDimensions(buffer.tensor);
    if (tensorRank(buffer.tensor) != 4 || shape[0] != 1 || shape[1] != 32 ||
        !shape[2] || shape[2] > 3000 || shape[3] != 128)
      throw std::runtime_error("Unsupported Temporal cache layout: " + buffer.name);
    expect(buffer, QNN_DATATYPE_FLOAT_32, 32ULL * shape[2] * 128 * sizeof(float));
    return shape[2];
  }
  static Buffer& temporalOutput(Graph& graph, int layer, int index) {
    const std::string name = "output_" + cacheName(layer + index / 2, index % 2);
    if (graph.hasOutput(name)) return graph.output(name);
    if (layer == 0) return graph.output("output_" + std::to_string(index + 1));
    throw std::runtime_error("Missing named Temporal cache output: " + name);
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
  std::string cacheDirectory_;
  GraphIoPool temporalIoPool_;
  std::set<int> residentTemporal_;
  std::map<int, std::unique_ptr<Graph>> temporalGraphs_;
  std::map<std::string, std::unique_ptr<Graph>> residentGraphs_;
  MoshiDelayRing ring_;
  std::array<std::vector<uint8_t>, 64> temporalCache_;
  std::array<std::unique_ptr<MoshiKvRing>, 64> temporalRing_;
};

using FrameCallback = std::function<void(size_t, size_t, double, const std::vector<int16_t>&)>;

size_t runMoshiFrames(Graph& encoder, Graph& decoder, PagedMoshiLm& lm,
                      const std::string& inputWav, const std::string& outputWav,
                      size_t requestedFrames, const FrameCallback& callback = {}) {
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
    std::vector<int16_t> audioChunk;
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
        const auto sample = static_cast<int16_t>(std::lrintf(std::clamp(value, -1.0f, 32767.0f / 32768.0f) * 32768));
        result.push_back(sample); audioChunk.push_back(sample);
      }
      codes << emitted;
      for (size_t i = 1; i < 9; ++i) codes << ',' << (*generated)[i];
      codes << '\n'; text << emitted << ',' << (*generated)[0] << '\n';
      ++emitted;
    }
    if (callback) callback(frame + 1, emitted, elapsedMs(start), audioChunk);
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
             size_t requestedFrames, const FrameCallback& callback = {}, bool highPerformance = false) {
    runtime_.timings.clear();
    runtime_.profileRows.clear();
    const auto start = std::chrono::steady_clock::now();
    if (lm_) lm_->configureCache(outputWav + ".contexts-v1");
    const char* boost = std::getenv("MOSHI_HTP_PERFORMANCE");
    if (highPerformance || (boost && std::string(boost) == "1")) runtime_.beginPerformanceVote();
    try {
      const auto frames = lm_ ? runMoshiFrames(encoder_, decoder_, *lm_, inputWav, outputWav, requestedFrames, callback)
                              : runCodecFrames(encoder_, decoder_, inputWav, outputWav, requestedFrames);
      runtime_.timing("pipeline", "wall_total", elapsedMs(start));
      runtime_.endPerformanceVote();
      runtime_.writeTimings(outputWav + ".performance.csv");
      return frames;
    } catch (...) {
      runtime_.endPerformanceVote();
      runtime_.timing("pipeline", "wall_failed", elapsedMs(start));
      try { runtime_.writeTimings(outputWav + ".performance.csv"); } catch (...) {}
      throw;
    }
  }

  void configureCache(const std::string& directory) {
    if (lm_) lm_->configureCache(directory);
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

size_t runMimiCodecSessionStreaming(void* session, const std::string& inputWav,
                                  const std::string& outputWav, size_t frames,
                                  const FrameCallback& callback, bool highPerformance) {
  if (!session) throw std::runtime_error("Models have not been loaded");
  return static_cast<PreparedMimiCodec*>(session)->run(inputWav, outputWav, frames, callback, highPerformance);
}

void destroyMimiCodecSession(void* session) {
  delete static_cast<PreparedMimiCodec*>(session);
}

void configureMimiCodecSessionCache(void* session, const std::string& directory) {
  if (!session) throw std::runtime_error("Models have not been loaded");
  static_cast<PreparedMimiCodec*>(session)->configureCache(directory);
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
    if (argc == 8 && std::string(argv[1]) == "--benchmark-temporal") {
      Runtime runtime(argv[2]);
      Graph graph(runtime, argv[3], argv[4]);
      std::ifstream file(argv[5], std::ios::binary);
      std::vector<uint8_t> hidden((std::istreambuf_iterator<char>(file)), {});
      if (!file.eof() && !file) throw std::runtime_error("Cannot read captured hidden tensor");
      auto& input = graph.input("hidden"); expect(input, QNN_DATATYPE_FLOAT_32, 4096 * sizeof(float));
      auto& position = graph.input("position"); expect(position, QNN_DATATYPE_INT_32, sizeof(int32_t));
      // Real captured frame-0 input: all incoming K/V caches are initially zero.
      const size_t repeats = std::stoul(argv[7]);
      if (!repeats || repeats > 20) throw std::runtime_error("Use 1..20 benchmark repetitions");
      const bool recurrent = enabledEnvironment("MOSHI_BENCH_RECURRENT");
      if (hidden.size() != input.bytes.size() * (recurrent ? repeats : 1))
        throw std::runtime_error("Captured hidden has incorrect size");
      std::vector<uint8_t> reference;
      if (enabledEnvironment("MOSHI_HTP_PERFORMANCE")) runtime.beginPerformanceVote();
      for (size_t i = 0; i < repeats; ++i) {
        const int32_t absolute = recurrent ? static_cast<int32_t>(i) : 0;
        std::memcpy(position.bytes.data(), &absolute, sizeof(absolute));
        std::memcpy(input.bytes.data(), hidden.data() + (recurrent ? i * input.bytes.size() : 0), input.bytes.size());
        graph.execute();
        auto& output = graph.output("output_hidden");
        expect(output, QNN_DATATYPE_FLOAT_32, 4096 * sizeof(float));
        for (size_t j = 0; j < output.bytes.size(); j += sizeof(float)) {
          float value; std::memcpy(&value, output.bytes.data() + j, sizeof(value));
          if (!std::isfinite(value)) throw std::runtime_error("Nonfinite Temporal benchmark output");
        }
        if (!recurrent) {
          if (reference.empty()) reference = output.bytes;
          else if (reference != output.bytes) throw std::runtime_error("Repeated captured-input output changed");
        }
        if ((i == 0 || recurrent) && enabledEnvironment("MOSHI_BENCH_DUMP")) {
          // Small parity artifacts, outside the timed execute: hidden plus the
          // written slot of each cache. Never dump megabytes of unwritten slots.
          const auto save = [&](const std::string& name, const std::vector<uint8_t>& bytes) {
            const std::string frame = recurrent ? ".frame_" + std::to_string(i) : "";
            std::ofstream stream(std::string(argv[6]) + frame + "." + name + ".raw", std::ios::binary);
            stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            if (!stream) throw std::runtime_error("Cannot save benchmark output");
          };
          save("hidden", output.bytes);
          for (const auto& buffer : graph.benchmarkOutputs()) {
            if (buffer.name.find("layer_") == std::string::npos) continue;
            const auto* shape = tensorDimensions(buffer.tensor);
            if (tensorRank(buffer.tensor) != 4 || shape[0] != 1 || shape[1] != 32 ||
                shape[3] != 128 || tensorType(buffer.tensor) != QNN_DATATYPE_FLOAT_32)
              throw std::runtime_error("Unexpected cache benchmark output ABI");
            std::vector<uint8_t> slot(32 * 128 * sizeof(float));
            for (size_t head = 0; head < 32; ++head)
              std::memcpy(slot.data() + head * 128 * sizeof(float),
                          buffer.bytes.data() + (head * shape[2] + absolute % shape[2]) * 128 * sizeof(float),
                          128 * sizeof(float));
            save(buffer.name, slot);
          }
        }
        if (recurrent) {
          // Inputs remain allocated across frames. Scatter only the newly
          // written per-head slice back into the persistent input cache.
          // Do not alias RAW QNN input/output pointers or copy full history.
          for (const auto& buffer : graph.benchmarkOutputs()) {
            if (buffer.name.rfind("output_layer_", 0) != 0) continue;
            auto& cache = graph.input(buffer.name.substr(7));
            const auto* inShape = tensorDimensions(cache.tensor);
            const auto* outShape = tensorDimensions(buffer.tensor);
            if (tensorRank(cache.tensor) != 4 || tensorRank(buffer.tensor) != 4 ||
                inShape[0] != 1 || inShape[1] != 32 || !inShape[2] || inShape[3] != 128 ||
                outShape[0] != 1 || outShape[1] != 32 || outShape[3] != 128 ||
                (outShape[2] != 1 && outShape[2] != inShape[2]) ||
                tensorType(cache.tensor) != QNN_DATATYPE_FLOAT_32 || tensorType(buffer.tensor) != QNN_DATATYPE_FLOAT_32)
              throw std::runtime_error("Unexpected recurrent cache ABI");
            for (size_t head = 0; head < 32; ++head)
              std::memcpy(cache.bytes.data() + (head * inShape[2] + absolute % inShape[2]) * 128 * sizeof(float),
                          buffer.bytes.data() + (head * outShape[2] + absolute % outShape[2]) * 128 * sizeof(float),
                          128 * sizeof(float));
          }
        }
      }
      runtime.endPerformanceVote(); runtime.writeTimings(argv[6]);
      std::cout << (recurrent ? "Captured-hidden recurrent Temporal probe: PASS\n"
                             : "Captured-input resident Temporal benchmark: PASS\n");
      return 0;
    }
    if (argc == 5 && std::string(argv[1]) == "--benchmark-frontend") {
      Runtime runtime(argv[2]);
      // Same deterministic streaming initial input, same DLC; change only lifecycle.
      MoshiDelayRing ring;
      const auto sequence = ring.begin(std::array<int32_t, 8>{});
      std::vector<uint8_t> reference;
      for (int pass = 0; pass < 2; ++pass) {
        Graph graph(runtime, argv[3], argv[4]);
        auto& input = graph.input("sequence");
        expect(input, QNN_DATATYPE_INT_32, sizeof(sequence));
        std::memcpy(input.bytes.data(), sequence.data(), sizeof(sequence));
        for (int repeat = 0; repeat < 2; ++repeat) {
          graph.execute();
          auto& output = graph.output("hidden");
          if (reference.empty()) reference = output.bytes;
          else if (reference != output.bytes) throw std::runtime_error("Context-cache/resident frontend output differs from first execution");
        }
      }
      runtime.writeTimings(std::string(argv[4]) + ".performance.csv");
      std::cout << "Frontend cache + resident repeat byte parity: PASS\n";
      return 0;
    }
    if (argc != 7 && argc != 8 && !(argc == 9 && std::string(argv[8]) == "--repeat")) {
      std::cerr << "Usage: mimi_qnn_runner LIB_DIR ENCODER.dlc DECODER.dlc "
                   "INPUT.wav OUTPUT.wav FRAMES [LM_BUNDLE_DIR [--repeat]] (0 = all complete frames)\n";
      return 2;
    }
    if (argc >= 8) {
      PreparedMimiCodec session(argv[1], argv[2], argv[3], argv[7]);
      session.run(argv[4], argv[5], std::stoul(argv[6]));
      if (argc == 9) session.run(argv[4], std::string(argv[5]) + ".repeat.wav", std::stoul(argv[6]));
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
