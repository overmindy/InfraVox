// On-device Moshi / Mimi runner over QNN HTP context binaries (QAIRT 2.45 APIs).
// Every model is a device-specific .bin context; all graphs stay resident for the session.

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
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "moshi_lm_scheduler.h"
#include "qnn_runtime.h"

namespace {

using namespace qnnrt;

constexpr size_t kFrameSamples = 1920;  // 80 ms at 24 kHz

int16_t toPcm16(float value) {
  if (!std::isfinite(value)) throw std::runtime_error("Nonfinite decoder audio");
  return static_cast<int16_t>(std::lrintf(std::clamp(value, -1.0f, 32767.0f / 32768.0f) * 32768.0f));
}

uint32_t read32(const uint8_t* d) { return d[0] | d[1] << 8 | d[2] << 16 | uint32_t(d[3]) << 24; }

// Mimi streaming state: swap each output into the next input instead of copying megabytes.
void advanceState(Graph& graph) {
  for (const std::string name : {"position", "kv_cache", "conv_state"}) graph.swapState(name + "_out", name);
}

void resetState(Graph& graph) {
  for (const auto* name : {"position", "kv_cache", "conv_state"}) {
    auto& state = graph.input(name);
    std::fill(state.bytes.begin(), state.bytes.end(), 0);
  }
}

// One Qualcomm-LLM-layout Temporal part (scripts/moshi_temporal.py): host attention mask and
// RoPE cos/sin, fixed-point KV rings (key [H,1,D,P], value [H,1,P,D]), new K/V outputs only.
class TemporalPart {
  struct Ring { Affine affine; std::unique_ptr<SharedBuffer> storage; };

 public:
  static constexpr size_t kHeads = 32, kHeadDim = 128, kDim = 4096;
  static constexpr float kMaskMin = -50.0f;
  // Frames 0-3 (frame 0 is the all-initial-token attention sink) keep their slots once the ring wraps.
  static constexpr int32_t kSinkFrames = 4;

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
    for (int32_t frame = 0; frame < std::min(position, kSinkFrames); ++frame) mask[frame] = 0.0f;
    for (int32_t back = 1; back <= std::min<int32_t>(position, static_cast<int32_t>(past_) - kSinkFrames); ++back)
      mask[slotOf(position - back)] = 0.0f;
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
    const size_t slot = slotOf(position);
    for (size_t r = 0; r < rings_.size(); ++r) {
      const bool key = r % 2 == 0;
      writeSlot(rings_[r], graph_.output(cacheName(layers_[r / 2], key, false)).data(), slot, key);
    }
  }

 private:
  // Keys keep their absolute RoPE positions, so evicting the oldest non-sink frame needs no rewrite.
  size_t slotOf(int32_t position) const {
    const int32_t past = static_cast<int32_t>(past_);
    return position < past ? position : kSinkFrames + (position - kSinkFrames) % (past - kSinkFrames);
  }

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

  void reset() { resetState(encoder_); resetState(decoder_); encoded_ = decoded_ = 0; }

  std::array<int32_t, 8> encode(const int16_t* pcm) {
    auto* audio = reinterpret_cast<float*>(encoder_.input("audio").data());
    for (size_t i = 0; i < kFrameSamples; ++i) audio[i] = pcm[i] / 32768.0f;
    encoder_.execute();
    std::array<int32_t, 8> codes{};
    std::memcpy(codes.data(), encoder_.output("codes").data(), sizeof(codes));
    checkPosition(encoder_, ++encoded_);
    advanceState(encoder_);
    return codes;
  }

  void decode(const int32_t* codes, int16_t* pcm) {
    set(decoder_.input("codes"), codes, 8 * sizeof(int32_t), QNN_DATATYPE_INT_32);
    decoder_.execute();
    const auto* audio = reinterpret_cast<const float*>(decoder_.output("audio").data());
    for (size_t i = 0; i < kFrameSamples; ++i) pcm[i] = toPcm16(audio[i]);
    checkPosition(decoder_, ++decoded_);
    advanceState(decoder_);
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
    for (const auto& name : names) spill = std::max(spill, runtime_.spillFillBytes(directory + "/dlc/" + name + ".bin"));
    for (const auto& path : parts) spill = std::max(spill, runtime_.spillFillBytes(path));
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

  // textTemperature 0 is greedy. Kyutai samples text at 0.7 with top-k 25; the quantized LM
  // tends to stay silent forever under greedy text after its first turn.
  void reset(float textTemperature, uint32_t seed) {
    delay_.reset();
    for (auto& part : temporal_) part->reset();
    position_ = 0;
    textTemperature_ = textTemperature;
    random_.seed(seed);
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
    const int32_t text = sampleText();

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
  // Only "speak or stay silent" is sampled (pad versus everything else, temperature + top-k as Kyutai);
  // the word itself is the best non-pad token, because the quantized head's tail holds junk like "<".
  int32_t sampleText() {
    constexpr int32_t kPad = 3;
    constexpr size_t kTopK = 25;
    const auto best = std::max_element(logits_.begin(), logits_.end());
    if (textTemperature_ <= 0.0f) return static_cast<int32_t>(best - logits_.begin());
    std::vector<int32_t> ids(logits_.size());
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int32_t>(i);
    std::partial_sort(ids.begin(), ids.begin() + kTopK, ids.end(),
                      [&](int32_t a, int32_t b) { return logits_[a] > logits_[b]; });
    double pad = 0, total = 0;
    int32_t word = -1;
    for (size_t i = 0; i < kTopK; ++i) {
      const double weight = std::exp((logits_[ids[i]] - *best) / textTemperature_);
      total += weight;
      if (ids[i] == kPad) pad = weight;
      else if (word < 0) word = ids[i];
    }
    return std::uniform_real_distribution<double>(0, total)(random_) < pad ? kPad : word;
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
  float textTemperature_ = 0.0f;
  std::mt19937 random_;
};

// Frame-by-frame duplex session: the app feeds one 80 ms microphone frame per step and plays
// whatever comes back. All calls for one session come from a single thread.
// Mimi stays on this thread: on a worker it only waits on the shared HTP execute lock, and a
// cool-phone A/B gave 100.4 ms per frame against 99.3 ms here.
class Session {
 public:
  Session(const std::string& libraryDir, const std::string& modelDirectory, bool codecOnly)
      : runtime_(libraryDir) {
    if (!codecOnly) lm_ = std::make_unique<MoshiLm>(runtime_, modelDirectory);
    codec_ = std::make_unique<Codec>(runtime_, modelDirectory + "/mimi_encoder.bin",
                                     modelDirectory + "/mimi_decoder.bin");
  }

  ~Session() { runtime_.endPerformanceVote(); codec_.reset(); lm_.reset(); }

  void begin(bool highPerformance, float textTemperature, uint32_t seed) {
    runtime_.endPerformanceVote();
    runtime_.clearTimings();
    codec_->reset();
    if (lm_) lm_->reset(textTemperature, seed);
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

void moshiBegin(void* session, bool highPerformance, float textTemperature, uint32_t seed) {
  static_cast<Session*>(session)->begin(highPerformance, textTemperature, seed);
}

bool moshiStep(void* session, const int16_t* input, int16_t* pcm, int32_t* tokens) {
  return static_cast<Session*>(session)->step(input, pcm, tokens);
}

void moshiEnd(void* session, const std::string& performanceCsv) { static_cast<Session*>(session)->end(performanceCsv); }

void moshiDestroy(void* session) { delete static_cast<Session*>(session); }
