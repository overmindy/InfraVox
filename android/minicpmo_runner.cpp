// On-device MiniCPM-o 4.5 full-duplex speech runner: QNN HTP context binaries for every network
// except the flow conformer, which runs on the CPU through ONNX Runtime (its relative positions
// depend on the cache length). The host logic mirrors scripts/minicpmo_sim.py, itself a port of
// MiniCPMODuplex.streaming_prefill / streaming_generate and stepaudio2 Token2wav.stream.

#include <array>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <thread>
#include <utility>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "onnxruntime_cxx_api.h"
#pragma GCC diagnostic pop
#include "qnn_runtime.h"

namespace {

using namespace qnnrt;

constexpr int32_t kVocab = 151748, kDim = 4096, kTtsDim = 768, kCodes = 6562;
constexpr int32_t kListen = 151705, kTtsBos = 151703, kChunkEos = 151718, kChunkTtsEos = 151721;
constexpr int32_t kTurnEos = 151717, kTtsPad = 151722, kUnit = 151683, kUnitEnd = 151684;
constexpr int32_t kTtsEos = 6561, kAudioBosText = 151687, kSilenceCode = 4218;
constexpr int kMaxSpeakTokens = 20, kArN = 16;
constexpr float kMaskMin = -50.0f;

float half(uint16_t bits) {
  __fp16 value;
  std::memcpy(&value, &bits, sizeof(value));
  return static_cast<float>(value);
}

// scripts/minicpmo_bundle.py: "MCPOTEN1" tensor container, 64-byte aligned data blocks.
class Assets {
 public:
  struct Tensor { uint8_t dtype; std::vector<uint32_t> dims; const uint8_t* data; size_t bytes; };

  explicit Assets(const std::string& path) : file_(path) {
    const auto* base = static_cast<const uint8_t*>(file_.data);
    if (file_.size < 12 || std::memcmp(base, "MCPOTEN1", 8)) throw std::runtime_error("Not a MiniCPM-o asset file: " + path);
    uint32_t count;
    std::memcpy(&count, base + 8, 4);
    size_t cursor = 12;
    const auto need = [&](size_t bytes) { if (cursor + bytes > file_.size) throw std::runtime_error("Truncated " + path); };
    for (uint32_t i = 0; i < count; ++i) {
      uint16_t length;
      need(2); std::memcpy(&length, base + cursor, 2); cursor += 2;
      need(length + 2u);
      std::string name(reinterpret_cast<const char*>(base + cursor), length);
      cursor += length;
      Tensor t;
      t.dtype = base[cursor];
      const uint8_t rank = base[cursor + 1];
      cursor += 2;
      need(4u * rank + 16);
      t.dims.resize(rank);
      std::memcpy(t.dims.data(), base + cursor, 4u * rank);
      cursor += 4u * rank;
      uint64_t offset, bytes;
      std::memcpy(&offset, base + cursor, 8);
      std::memcpy(&bytes, base + cursor + 8, 8);
      cursor += 16;
      if (offset + bytes > file_.size) throw std::runtime_error("Truncated tensor " + name + " in " + path);
      t.data = base + offset;
      t.bytes = bytes;
      tensors_[name] = t;
    }
  }

  const Tensor& get(const std::string& name, uint8_t dtype) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end() || it->second.dtype != dtype) throw std::runtime_error("Missing asset tensor " + name);
    return it->second;
  }
  const float* f32(const std::string& name) const { return reinterpret_cast<const float*>(get(name, 0).data); }
  const uint16_t* f16(const std::string& name) const { return reinterpret_cast<const uint16_t*>(get(name, 1).data); }
  const int32_t* i32(const std::string& name) const { return reinterpret_cast<const int32_t*>(get(name, 2).data); }
  const uint8_t* u8(const std::string& name) const { return get(name, 3).data; }

 private:
  MappedFile file_;
  std::map<std::string, Tensor> tensors_;
};

void rowF16(const uint16_t* table, size_t row, size_t width, float* out) {
  const uint16_t* source = table + row * width;
  for (size_t i = 0; i < width; ++i) out[i] = half(source[i]);
}

// 8-bit KV storage in rpcmem whose encoding is tied between a graph's past input and new output.
struct Ring {
  Affine affine;
  std::unique_ptr<SharedBuffer> storage;
  uint8_t zero = 0;  // quantized 0.0: empty slots must hold it, not byte 0
};

Ring makeRing(Runtime& runtime, Graph& graph, const std::string& input, const std::string& output) {
  Ring ring{affineOf(graph.input(input).tensor), nullptr, 0};
  if (affineWidth(ring.affine) != 1 || !(ring.affine == affineOf(graph.output(output).tensor)))
    throw std::runtime_error("KV must be 8-bit with tied input/output encodings: " + input + " in " + graph.label());
  // From the tensor, not the host bytes: binding set 0 already freed those for the DiT's later sets.
  ring.storage = std::make_unique<SharedBuffer>(runtime.sharedMemory, tensorBytes(graph.input(input).tensor));
  const float zero = 0.0f;
  quantizeTo(ring.affine, &zero, 1, &ring.zero);
  std::memset(ring.storage->data(), ring.zero, ring.storage->bytes());
  return ring;
}

std::string cacheName(int layer, bool key, bool input) {
  return std::string(key ? "past_key_" : "past_value_") + std::to_string(layer) + (input ? "_in" : "_out");
}

// Key out [H,1,D,n] column j -> ring [H,1,D,P] column slot; value out [H,1,n,D] row j -> ring row slot.
void writeKv(uint8_t* ring, const uint8_t* out, size_t heads, size_t dim, size_t past, size_t n, size_t j, size_t slot, bool key) {
  for (size_t h = 0; h < heads; ++h) {
    if (!key) { std::memcpy(ring + (h * past + slot) * dim, out + (h * n + j) * dim, dim); continue; }
    for (size_t d = 0; d < dim; ++d) ring[(h * dim + d) * past + slot] = out[(h * dim + d) * n + j];
  }
}

// An AR-1 / AR-16 graph pair in one weight-sharing context (scripts/minicpmo_llm.py, _tts.py layout):
// host additive mask, RoPE cos/sin, and one KV ring shared by both graphs. Position t is slot t % P.
class KvModel {
 public:
  KvModel(Runtime& runtime, const std::string& path, const std::string& prefix, const char* input, const char* output,
          double ropeTheta)
      : context_(std::make_shared<Context>(runtime, path, true)),
        ar1_(runtime, context_, prefix + "_ar1"), arN_(runtime, context_, prefix + "_ar16"),
        input_(input), output_(output), ropeTheta_(ropeTheta) {
    for (const auto& in : ar1_.inputs()) {
      int layer = -1;
      if (std::sscanf(in.name.c_str(), "past_key_%d_in", &layer) == 1) layers_.push_back(layer);
    }
    std::sort(layers_.begin(), layers_.end());
    if (layers_.empty()) throw std::runtime_error("No past_key inputs in " + path);
    const auto& key = ar1_.input(cacheName(layers_[0], true, true)).tensor;
    heads_ = tensorDimensions(key)[0];
    headDim_ = tensorDimensions(key)[2];
    past_ = tensorDimensions(key)[3];
    width_ = tensorDimensions(ar1_.input(input_).tensor)[2];
    for (int layer : layers_) {
      for (bool isKey : {true, false}) {
        rings_.push_back(makeRing(runtime, ar1_, cacheName(layer, isKey, true), cacheName(layer, isKey, false)));
        if (!(affineOf(arN_.input(cacheName(layer, isKey, true)).tensor) == rings_.back().affine))
          throw std::runtime_error("AR-1 and AR-16 KV encodings differ in " + path);
        ar1_.bindInput(cacheName(layer, isKey, true), *rings_.back().storage);
        arN_.bindInput(cacheName(layer, isKey, true), *rings_.back().storage);
      }
    }
  }

  size_t width() const { return width_; }
  size_t outputWidth() { return tensorDimensions(ar1_.output(output_).tensor)[2]; }

  void reset() {
    for (auto& ring : rings_) std::memset(ring.storage->data(), ring.zero, ring.storage->bytes());
  }

  // Runs `count` (1..16) rows at positions position.. and returns the output rows.
  std::vector<float> run(int64_t position, const float* rows, size_t count) {
    if (count < 1 || count > static_cast<size_t>(kArN)) throw std::runtime_error("KvModel chunk of " + std::to_string(count));
    Graph& graph = count == 1 ? ar1_ : arN_;
    const size_t n = count == 1 ? 1 : kArN;
    std::vector<float> x(n * width_, 0.0f);
    std::copy(rows, rows + count * width_, x.begin());
    setReal(graph.input(input_), x.data(), x.size());
    std::vector<float> mask(n * (past_ + n), kMaskMin);
    const size_t valid = static_cast<size_t>(std::min<int64_t>(position, past_));
    for (size_t i = 0; i < n; ++i) {
      std::fill(mask.begin() + i * (past_ + n), mask.begin() + i * (past_ + n) + valid, 0.0f);
      for (size_t j = 0; j <= std::min(i, count - 1); ++j) mask[i * (past_ + n) + past_ + j] = 0.0f;
    }
    setReal(graph.input("attention_mask"), mask.data(), mask.size());
    const size_t halfDim = headDim_ / 2;
    std::vector<float> cos(n * halfDim), sin(n * halfDim);
    for (size_t i = 0; i < n; ++i)
      for (size_t k = 0; k < halfDim; ++k) {
        const double angle = static_cast<double>(position + i) / std::pow(ropeTheta_, 2.0 * k / headDim_);
        cos[i * halfDim + k] = static_cast<float>(std::cos(angle));
        sin[i * halfDim + k] = static_cast<float>(std::sin(angle));
      }
    setReal(graph.input("position_cos"), cos.data(), cos.size());
    setReal(graph.input("position_sin"), sin.data(), sin.size());
    graph.execute();
    for (size_t r = 0; r < rings_.size(); ++r) {
      const bool isKey = r % 2 == 0;
      const uint8_t* out = graph.output(cacheName(layers_[r / 2], isKey, false)).data();
      for (size_t j = 0; j < count; ++j)
        writeKv(rings_[r].storage->data(), out, heads_, headDim_, past_, n, j, (position + j) % past_, isKey);
    }
    const size_t outWidth = tensorDimensions(graph.output(output_).tensor)[2];
    std::vector<float> all(n * outWidth);
    getReal(graph.output(output_), all.data(), all.size());
    all.resize(count * outWidth);
    return all;
  }

 private:
  std::shared_ptr<Context> context_;
  std::vector<Ring> rings_;  // Before the graphs: memory is deregistered before it is freed.
  Graph ar1_, arN_;
  std::string input_, output_;
  double ropeTheta_;
  std::vector<int> layers_;
  size_t heads_ = 0, headDim_ = 0, past_ = 0, width_ = 0;
};

// Whisper log-mel exactly as MiniCPM-o's StreamingMelProcessorExact: 1 s units (the first one
// 1030 ms), 2 frames of CNN context each side, features recomputed over the 30 s window with a
// -10 floor below 5 s of buffer and max - 8 above. Stable frames are cached; only the tail and,
// after a window slide, the first frame are recomputed.
class StreamingMel {
 public:
  static constexpr size_t kFft = 400, kHop = 160, kBins = 201, kMels = 80, kRate = 16000;

  explicit StreamingMel(const float* filters) : filters_(filters, filters + kBins * kMels) {
    window_.resize(kFft);
    for (size_t i = 0; i < kFft; ++i) window_[i] = 0.5 - 0.5 * std::cos(2 * M_PI * i / kFft);
    cos_.resize(kBins * kFft);
    sin_.resize(kBins * kFft);
    for (size_t k = 0; k < kBins; ++k)
      for (size_t i = 0; i < kFft; ++i) {
        cos_[k * kFft + i] = std::cos(2 * M_PI * k * i / kFft);
        sin_[k * kFft + i] = std::sin(2 * M_PI * k * i / kFft);
      }
  }

  void reset() { buffer_.clear(); frames_.clear(); emitted_ = base_ = 0; }

  // Appends one chunk and returns the emitted frames [80, F] (F = 102 for the first unit, else 104).
  std::vector<float> process(const float* samples, size_t count, size_t& frames) {
    buffer_.insert(buffer_.end(), samples, samples + count);
    if (buffer_.size() >= 30 * kRate) {
      size_t drop = std::min(10 * kRate, buffer_.size() - std::min(kRate, buffer_.size() - 10 * kRate));
      drop = drop / kHop * kHop;
      buffer_.erase(buffer_.begin(), buffer_.begin() + drop);
      base_ += drop / kHop;
      frames_.erase(frames_.begin(), frames_.begin() + std::min(frames_.size(), drop / kHop * kMels));
      // Frames 0 and 1 reach before the new start, so their reflect padding changed.
      for (size_t f = 0; f < std::min<size_t>(2, frames_.size() / kMels); ++f) logMel(f, frames_.data() + f * kMels);
    }
    const size_t total = buffer_.size() / kHop;  // torch.stft frames minus the dropped last one
    const size_t stable = buffer_.size() < kFft / 2 ? 0 : (buffer_.size() - kFft / 2) / kHop + 1;
    const size_t cached = frames_.size() / kMels;
    const size_t keep = std::min(stable, total);
    frames_.resize(keep * kMels);
    for (size_t f = cached; f < keep; ++f) logMel(f, frames_.data() + f * kMels);
    std::vector<float> tail((total - keep) * kMels);
    for (size_t f = keep; f < total; ++f) logMel(f, tail.data() + (f - keep) * kMels);
    float peak = -1e30f;
    for (float v : frames_) peak = std::max(peak, v);
    for (float v : tail) peak = std::max(peak, v);
    const float floor = buffer_.size() >= 5 * kRate ? peak - 8.0f : -10.0f;
    const size_t start = emitted_ >= base_ + 2 ? emitted_ - 2 - base_ : 0;
    const size_t end = std::min(emitted_ + 100 + 2 - base_, total);
    emitted_ += 100;
    frames = end - start;
    std::vector<float> out(kMels * frames);
    for (size_t f = start; f < end; ++f) {
      const float* row = f < keep ? frames_.data() + f * kMels : tail.data() + (f - keep) * kMels;
      for (size_t m = 0; m < kMels; ++m) out[m * frames + (f - start)] = (std::max(row[m], floor) + 4.0f) / 4.0f;
    }
    return out;
  }

 private:
  // log10 mel power of frame f (centred at f * hop, reflect padding at both ends).
  void logMel(size_t f, float* out) const {
    std::array<double, kFft> frame;
    const int64_t length = static_cast<int64_t>(buffer_.size());
    for (size_t i = 0; i < kFft; ++i) {
      int64_t at = static_cast<int64_t>(f * kHop + i) - static_cast<int64_t>(kFft / 2);
      if (at < 0) at = -at;
      if (at >= length) at = 2 * (length - 1) - at;
      frame[i] = buffer_[static_cast<size_t>(at)] * window_[i];
    }
    std::array<double, kBins> power;
    for (size_t k = 0; k < kBins; ++k) {
      double re = 0, im = 0;
      const double* c = cos_.data() + k * kFft;
      const double* s = sin_.data() + k * kFft;
      for (size_t i = 0; i < kFft; ++i) { re += frame[i] * c[i]; im -= frame[i] * s[i]; }
      power[k] = re * re + im * im;
    }
    for (size_t m = 0; m < kMels; ++m) {
      double energy = 0;
      for (size_t k = 0; k < kBins; ++k) energy += filters_[k * kMels + m] * power[k];
      out[m] = static_cast<float>(std::log10(std::max(energy, 1e-10)));
    }
  }

  std::vector<double> filters_, window_, cos_, sin_;
  std::vector<float> buffer_, frames_;
  size_t emitted_ = 0, base_ = 0;
};

// Streaming Whisper + projector + pooling (scripts/minicpmo_audio.py): KV window of 1450 frames,
// cleared once 1500 would be exceeded, as upstream.
class AudioEncoder {
 public:
  static constexpr size_t kFrames = 50, kMelIn = 104, kHeads = 16, kHeadDim = 64, kWidth = 1024;

  AudioEncoder(Runtime& runtime, const std::string& path, const uint16_t* positions)
      : graph_(runtime, path, true), positions_(positions) {
    for (int layer = 0; layer < 24; ++layer)
      for (bool key : {true, false}) {
        rings_.push_back(makeRing(runtime, graph_, cacheName(layer, key, true), cacheName(layer, key, false)));
        graph_.bindInput(cacheName(layer, key, true), *rings_.back().storage);
      }
    past_ = tensorDimensions(graph_.input("past_key_0_in").tensor)[3];
  }

  void reset() {
    length_ = 0;
    for (auto& ring : rings_) std::memset(ring.storage->data(), ring.zero, ring.storage->bytes());
  }

  // mel [80, frames] with frames 102 (first unit, left-padded here) or 104 -> 10 x 4096 embeddings.
  std::vector<float> encode(const std::vector<float>& mel, size_t frames, bool first) {
    if (length_ + kFrames >= 1500) length_ = 0;
    std::vector<float> padded(80 * kMelIn, 0.0f);
    for (size_t m = 0; m < 80; ++m)
      std::copy(mel.begin() + m * frames, mel.begin() + (m + 1) * frames, padded.begin() + m * kMelIn + (kMelIn - frames));
    setReal(graph_.input("mel"), padded.data(), padded.size());
    std::vector<float> frameMask(kMelIn - 2, 1.0f);  // unpadded conv1 frames 1..102
    if (first) frameMask[0] = 0.0f;
    setReal(graph_.input("frame_mask"), frameMask.data(), frameMask.size());
    std::vector<float> row(kWidth), positions(kWidth * kFrames);  // graph input is channel-first
    for (size_t f = 0; f < kFrames; ++f) {
      rowF16(positions_, length_ + f, kWidth, row.data());
      for (size_t c = 0; c < kWidth; ++c) positions[c * kFrames + f] = row[c];
    }
    setReal(graph_.input("positions"), positions.data(), positions.size());
    std::vector<float> mask(kFrames * (past_ + kFrames), kMaskMin);
    for (size_t i = 0; i < kFrames; ++i) {
      std::fill(mask.begin() + i * (past_ + kFrames), mask.begin() + i * (past_ + kFrames) + length_, 0.0f);
      std::fill(mask.begin() + i * (past_ + kFrames) + past_, mask.begin() + (i + 1) * (past_ + kFrames), 0.0f);
    }
    setReal(graph_.input("attention_mask"), mask.data(), mask.size());
    graph_.execute();
    for (size_t r = 0; r < rings_.size(); ++r) {
      const uint8_t* out = graph_.output(cacheName(static_cast<int>(r / 2), r % 2 == 0, false)).data();
      for (size_t j = 0; j < kFrames; ++j)
        writeKv(rings_[r].storage->data(), out, kHeads, kHeadDim, past_, kFrames, j, length_ + j, r % 2 == 0);
    }
    length_ += kFrames;
    std::vector<float> embeds(10 * kDim);
    getReal(graph_.output("audio_embeds"), embeds.data(), embeds.size());
    return embeds;
  }

 private:
  std::vector<Ring> rings_;
  Graph graph_;
  const uint16_t* positions_;
  size_t past_ = 0, length_ = 0;
};

// StreamDecoder.decode in sampling mode with the duplex defaults (modeling_minicpmo / utils.py).
class Sampler {
 public:
  void reset(uint64_t seed, float temperature) { random_.seed(seed); temperature_ = temperature; generated_.clear(); }

  int32_t operator()(std::vector<float> logits) {
    const bool greedy = temperature_ <= 0.0f;
    const int32_t first = greedy ? argmax(logits) : draw(logits, 1.0f);
    if (first == kChunkEos) return kChunkEos;
    logits[kTtsPad] = logits[kChunkEos] = -INFINITY;
    std::vector<int32_t> recent(generated_.end() - std::min<size_t>(generated_.size(), 512), generated_.end());
    std::sort(recent.begin(), recent.end());
    recent.erase(std::unique(recent.begin(), recent.end()), recent.end());
    for (int32_t token : recent) logits[token] /= 1.05f;
    int32_t token;
    if (greedy) {
      token = argmax(logits);
    } else {
      // temperature 0.7, top-k 100, top-p 0.8 (the first token over the threshold is kept)
      std::vector<int32_t> ids(logits.size());
      std::iota(ids.begin(), ids.end(), 0);
      std::partial_sort(ids.begin(), ids.begin() + 100, ids.end(), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
      ids.resize(100);
      std::vector<double> weights(ids.size());
      for (size_t i = 0; i < ids.size(); ++i) weights[i] = std::exp((logits[ids[i]] - logits[ids[0]]) / temperature_);
      const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
      double cumulative = 0;
      size_t keep = 0;
      while (keep < ids.size()) {
        const bool over = cumulative > 0.8 * total;
        if (over && keep > 0) break;
        cumulative += weights[keep++];
      }
      token = ids[pick(weights.data(), keep)];
    }
    generated_.push_back(token);
    return token;
  }

 private:
  static int32_t argmax(const std::vector<float>& v) {
    return static_cast<int32_t>(std::max_element(v.begin(), v.end()) - v.begin());
  }
  int32_t draw(const std::vector<float>& logits, float temperature) {
    const float top = *std::max_element(logits.begin(), logits.end());
    std::vector<double> weights(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) weights[i] = std::exp((logits[i] - top) / temperature);
    return static_cast<int32_t>(pick(weights.data(), weights.size()));
  }
  size_t pick(const double* weights, size_t count) {
    const double total = std::accumulate(weights, weights + count, 0.0);
    double u = std::uniform_real_distribution<double>(0, total)(random_);
    for (size_t i = 0; i < count; ++i) { if (u < weights[i]) return i; u -= weights[i]; }
    return count - 1;
  }
  std::mt19937_64 random_;
  float temperature_ = 0.7f;
  std::vector<int32_t> generated_;
};

// The 25-code TTS chunks of MiniCPMTTS.generate_chunk: temperature 0.8, repetition penalty 1.05
// over this chunk's last 16 codes, EOS 6561 blocked below min_tokens, the 26th code dropped.
class Tts {
 public:
  Tts(Runtime& runtime, const std::string& path, const uint16_t* embCode)
      : model_(runtime, path, "tts", "embeds", "logits", 1e4), embCode_(embCode) {}

  void reset() { model_.reset(); position_ = 0; }
  void seed(uint64_t seed) { random_.seed(seed); }

  std::vector<int32_t> generate(const std::vector<float>& condition, size_t rows, int minTokens) {
    std::vector<int32_t> codes;
    std::vector<float> logits = feed(condition.data(), rows);
    constexpr int kMax = 26;
    std::vector<float> embed(kTtsDim);
    for (int t = 0; t < kMax; ++t) {
      if (t > 0) {
        rowF16(embCode_, codes.back(), kTtsDim, embed.data());
        logits = feed(embed.data(), 1);
      }
      std::vector<double> scores(kCodes);
      for (int i = 0; i < kCodes; ++i) scores[i] = logits[i] / 0.8;
      if (t > 0) {
        std::map<int32_t, int> counts;
        for (size_t i = codes.size() - std::min<size_t>(codes.size(), 16); i < codes.size(); ++i) counts[codes[i]]++;
        for (const auto& [code, count] : counts) {
          const double alpha = std::pow(1.05, count);
          scores[code] = scores[code] < 0 ? scores[code] * alpha : scores[code] / alpha;
        }
      }
      if (t < minTokens) scores[kTtsEos] = -INFINITY;
      const double top = *std::max_element(scores.begin(), scores.end());
      double total = 0;
      for (auto& s : scores) { s = std::exp(s - top); total += s; }
      double u = std::uniform_real_distribution<double>(0, total)(random_);
      int32_t code = kCodes - 1;
      for (int i = 0; i < kCodes; ++i) { if (u < scores[i]) { code = i; break; } u -= scores[i]; }
      if (code == kTtsEos || t == kMax - 1) break;
      codes.push_back(code);
    }
    return codes;
  }

 private:
  std::vector<float> feed(const float* rows, size_t count) {
    std::vector<float> last;
    for (size_t start = 0; start < count; start += kArN) {
      const size_t n = std::min<size_t>(kArN, count - start);
      auto out = model_.run(position_, rows + start * kTtsDim, n);
      position_ += static_cast<int64_t>(n);
      last.assign(out.end() - kCodes, out.end());
    }
    return last;
  }

  KvModel model_;
  const uint16_t* embCode_;
  int64_t position_ = 0;
  std::mt19937_64 random_;
};

// CosyVoice2 DiT step (scripts/minicpmo_flow.py DiTStep). Each of the 10 Euler steps owns a
// binding set whose KV inputs live in that step's rings; slots hold estimator frames in any order.
class Dit {
 public:
  static constexpr size_t kSteps = 10, kBlocks = 16, kT = 50, kHeads = 16, kHeadDim = 64, kCnn = 2 * 1024 * 2;

  Dit(Runtime& runtime, const std::string& path) : graph_(runtime, path, true) {
    past_ = tensorDimensions(graph_.input("past_key_0_in").tensor)[3];
    for (size_t s = 0; s < kSteps; ++s) {
      const size_t set = s == 0 ? 0 : graph_.addBindingSet();
      for (size_t b = 0; b < kBlocks; ++b)
        for (bool key : {true, false}) {
          rings_.push_back(makeRing(runtime, graph_, cacheName(static_cast<int>(b), key, true), cacheName(static_cast<int>(b), key, false)));
          graph_.bindInput(set, cacheName(static_cast<int>(b), key, true), *rings_.back().storage);
        }
    }
    cnn_.assign(kSteps * kBlocks * kCnn, 0.0f);
  }

  size_t past() const { return past_; }
  Ring& ring(size_t step, size_t block, bool key) { return rings_[(step * kBlocks + block) * 2 + (key ? 0 : 1)]; }
  float* cnn(size_t step, size_t block) { return cnn_.data() + (step * kBlocks + block) * kCnn; }

  // The prompt state is quantized once; every new turn restores it with a copy.
  void saveInitial() {
    initial_.clear();
    for (auto& r : rings_) initial_.insert(initial_.end(), r.storage->data(), r.storage->data() + r.storage->bytes());
    initialCnn_ = cnn_;
  }
  bool restoreInitial() {
    if (initial_.empty()) return false;
    size_t at = 0;
    for (auto& r : rings_) { std::memcpy(r.storage->data(), initial_.data() + at, r.storage->bytes()); at += r.storage->bytes(); }
    cnn_ = initialCnn_;
    return true;
  }

  // Writes frame `source` of a float cache [2,8,frames,128] (K | V) into `slot` of step/block rings.
  void writeFrame(size_t step, size_t block, const float* cache, size_t frames, size_t source, size_t slot) {
    for (bool key : {true, false}) {
      Ring& r = ring(step, block, key);
      uint8_t* data = r.storage->data();
      for (size_t h = 0; h < kHeads; ++h)
        for (size_t d = 0; d < kHeadDim; ++d) {
          const float v = cache[(h * frames + source) * 128 + (key ? d : 64 + d)];
          uint8_t q;
          quantizeTo(r.affine, &v, 1, &q);
          if (key) data[(h * kHeadDim + d) * past_ + slot] = q;
          else data[(h * past_ + slot) * kHeadDim + d] = q;
        }
    }
  }

  // One Euler step's velocity [2,80,T]; the new frames' K/V stay in the graph outputs for copyNew.
  void run(size_t step, const float* x, const float* cond, const std::vector<float>& mask, const std::vector<float>& select,
           const float* modBlocks, const float* modFinal, float* velocity) {
    setReal(graph_.input("x"), x, 2 * 80 * kT);
    setReal(graph_.input("cond"), cond, 2 * 240 * kT);
    setReal(graph_.input("attention_mask"), mask.data(), mask.size());
    setReal(graph_.input("select"), select.data(), select.size());
    for (size_t b = 0; b < kBlocks; ++b) {
      setReal(graph_.input("mod_" + std::to_string(b)), modBlocks + b * 9 * 512, 9 * 512);
      setReal(graph_.input("cnn_" + std::to_string(b) + "_in"), cnn(step, b), kCnn);
    }
    setReal(graph_.input("mod_final"), modFinal, 2 * 512);
    graph_.execute(step);
    getReal(graph_.output("velocity"), velocity, 2 * 80 * kT);
    for (size_t b = 0; b < kBlocks; ++b) getReal(graph_.output("cnn_" + std::to_string(b) + "_out"), cnn(step, b), kCnn);
  }

  // Copies new frame j (graph output) of every block into `slot` of this step's rings.
  void copyNew(size_t step, size_t j, size_t slot) {
    for (size_t b = 0; b < kBlocks; ++b)
      for (bool key : {true, false})
        writeKv(ring(step, b, key).storage->data(), graph_.output(cacheName(static_cast<int>(b), key, false)).data(), kHeads,
                kHeadDim, past_, kT, j, slot, key);
  }

 private:
  Graph graph_;
  std::vector<Ring> rings_;
  std::vector<float> cnn_, initialCnn_;
  std::vector<uint8_t> initial_;
  size_t past_ = 0;
};

// The CPU tail of HiFT: SourceModuleHnNSF2 and the n_fft = 16 STFT pair (scripts/minicpmo_flow.py).
std::vector<float> nsfSource(const std::vector<float>& f0, std::mt19937_64& random, const float* weight, float bias,
                             bool noise = true) {
  constexpr size_t kUp = 480, kHarmonics = 9;
  const size_t frames = f0.size();
  std::vector<std::array<double, kHarmonics>> phase(frames);
  std::array<double, kHarmonics> sum{};
  for (size_t f = 0; f < frames; ++f)
    for (size_t h = 0; h < kHarmonics; ++h) {
      sum[h] += std::fmod(f0[f] * (h + 1) / 24000.0, 1.0);
      phase[f][h] = sum[h] * 2 * M_PI * kUp;
    }
  std::normal_distribution<double> gauss;
  std::vector<float> out(frames * kUp);
  for (size_t i = 0; i < out.size(); ++i) {
    const double pos = (i + 0.5) / kUp - 0.5;
    size_t lo = pos < 0 ? 0 : std::min(static_cast<size_t>(std::floor(pos)), frames - 1);
    const size_t hi = std::min(lo + 1, frames - 1);
    const double frac = pos < 0 ? 0.0 : std::clamp(pos - std::floor(pos), 0.0, 1.0);
    const bool voiced = f0[i / kUp] > 10.0f;
    double merged = bias;
    for (size_t h = 0; h < kHarmonics; ++h) {
      const double p = phase[lo][h] * (1 - frac) + phase[hi][h] * frac;
      const double sine = voiced ? std::sin(p) * 0.1 : 0.0;
      merged += (sine + (noise ? (voiced ? 0.003 : 0.1 / 3) * gauss(random) : 0.0)) * weight[h];
    }
    out[i] = static_cast<float>(std::tanh(merged));
  }
  return out;
}

constexpr size_t kNfft = 16, kHopT = 4;

std::array<double, kNfft> hann16() {
  std::array<double, kNfft> w;
  for (size_t i = 0; i < kNfft; ++i) w[i] = 0.5 - 0.5 * std::cos(2 * M_PI * i / kNfft);
  return w;
}

// cos / sin(2 pi k i / 16), indexed [k * 16 + i]: the transforms run ~7000 frames per vocoder call.
struct Twiddles16 {
  std::array<double, (kNfft / 2 + 1) * kNfft> cos, sin;
  Twiddles16() {
    for (size_t k = 0; k <= kNfft / 2; ++k)
      for (size_t i = 0; i < kNfft; ++i) {
        cos[k * kNfft + i] = std::cos(2 * M_PI * k * i / kNfft);
        sin[k * kNfft + i] = std::sin(2 * M_PI * k * i / kNfft);
      }
  }
};
const Twiddles16& twiddles16() { static const Twiddles16 table; return table; }

// torch.stft(center, reflect) -> [18, frames] (real 0..8 | imag 0..8) written at row stride `stride`.
void stft16(const std::vector<float>& signal, float* out, size_t stride) {
  const auto window = hann16();
  const auto& tw = twiddles16();
  const int64_t length = static_cast<int64_t>(signal.size());
  const size_t frames = signal.size() / kHopT + 1;
  for (size_t f = 0; f < frames; ++f) {
    std::array<double, kNfft> x;
    for (size_t i = 0; i < kNfft; ++i) {
      int64_t at = static_cast<int64_t>(f * kHopT + i) - static_cast<int64_t>(kNfft / 2);
      if (at < 0) at = -at;
      if (at >= length) at = 2 * (length - 1) - at;
      x[i] = signal[static_cast<size_t>(at)] * window[i];
    }
    for (size_t k = 0; k <= kNfft / 2; ++k) {
      double re = 0, im = 0;
      for (size_t i = 0; i < kNfft; ++i) {
        re += x[i] * tw.cos[k * kNfft + i];
        im -= x[i] * tw.sin[k * kNfft + i];
      }
      out[k * stride + f] = static_cast<float>(re);
      out[(9 + k) * stride + f] = static_cast<float>(im);
    }
  }
}

// exp / sin of conv_post [18, frames] (row stride `stride`) and torch.istft(center) -> 4 * (frames - 1) samples.
std::vector<float> istft16(const float* head, size_t stride, size_t frames) {
  const auto window = hann16();
  const auto& tw = twiddles16();
  const size_t length = kNfft + kHopT * (frames - 1);
  std::vector<double> out(length, 0.0), norm(length, 0.0);
  for (size_t f = 0; f < frames; ++f) {
    std::array<double, 9> re, im;
    for (size_t k = 0; k < 9; ++k) {
      const double magnitude = std::min(std::exp(static_cast<double>(head[k * stride + f])), 1e2);
      const double phase = std::sin(static_cast<double>(head[(9 + k) * stride + f]));
      re[k] = magnitude * std::cos(phase);
      im[k] = magnitude * std::sin(phase);
    }
    for (size_t i = 0; i < kNfft; ++i) {
      // irfft: DC and Nyquist once, the rest twice; imaginary parts of DC/Nyquist are ignored.
      double v = re[0] + re[8] * ((i % 2) ? -1.0 : 1.0);
      for (size_t k = 1; k < 8; ++k) v += 2 * (re[k] * tw.cos[k * kNfft + i] - im[k] * tw.sin[k * kNfft + i]);
      out[f * kHopT + i] += v / kNfft * window[i];
      norm[f * kHopT + i] += window[i] * window[i];
    }
  }
  std::vector<float> result(length - kNfft);
  for (size_t i = 0; i < result.size(); ++i) {
    const size_t at = i + kNfft / 2;
    result[i] = static_cast<float>(std::clamp(out[at] / std::max(norm[at], 1e-11), -0.99, 0.99));
  }
  return result;
}

// stepaudio2 Token2wav.stream with fixed graph shapes (scripts/minicpmo_flow.py Token2wavHost).
class Token2wav {
 public:
  static constexpr size_t kMel = 80, kMelCache = 8, kSourceCache = 8 * 480, kHiftFrames = 58;

  Token2wav(Runtime& runtime, const std::string& dir, const Assets& assets)
      : assets_(assets), dit_(runtime, dir + "/dit.bin"), f0_(runtime, dir + "/hift_f0.bin", true),
        decode_(runtime, dir + "/hift_decode.bin", true),
        env_(ORT_LOGGING_LEVEL_WARNING, "minicpmo"), session_(nullptr) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(4);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = Ort::Session(env_, (dir + "/flow_encoder.onnx").c_str(), options);
    prompt_ = static_cast<size_t>(assets_.i32("t2w_prompt_frames")[0]);
    if (prompt_ + 100 != dit_.past()) throw std::runtime_error("DiT graph past does not match the voice prompt");
  }

  void seed(uint64_t seed) { random_.seed(seed); }

  void reset() {
    const auto& att = assets_.get("t2w_conformer_att", 0);
    conformerAtt_.assign(assets_.f32("t2w_conformer_att"), assets_.f32("t2w_conformer_att") + att.bytes / 4);
    conformerFrames_ = att.dims[3];
    conformerCnn_.assign(assets_.f32("t2w_conformer_cnn"), assets_.f32("t2w_conformer_cnn") + 512 * 6);
    if (!dit_.restoreInitial()) {
      const uint16_t* est = assets_.f16("t2w_est_att");
      std::vector<float> cache(2 * 8 * prompt_ * 128);
      for (size_t s = 0; s < Dit::kSteps; ++s)
        for (size_t b = 0; b < Dit::kBlocks; ++b) {
          rowF16(est + (s * Dit::kBlocks + b) * cache.size(), 0, cache.size(), cache.data());
          for (size_t f = 0; f < prompt_; ++f) dit_.writeFrame(s, b, cache.data(), prompt_, f, f);
          std::copy(assets_.f32("t2w_est_cnn") + (s * Dit::kBlocks + b) * Dit::kCnn,
                    assets_.f32("t2w_est_cnn") + (s * Dit::kBlocks + b + 1) * Dit::kCnn, dit_.cnn(s, b));
        }
      dit_.saveInitial();
    }
    order_.resize(prompt_);
    std::iota(order_.begin(), order_.end(), 0);
    melCache_.clear();
    sourceCache_.clear();
    speechCache_.clear();
  }

  std::vector<float> stream(const std::vector<int32_t>& tokens, bool last) {
    const auto start = Clock::now();
    const auto mu = encode(tokens, last);
    const size_t n = mu.size() / kMel;
    timing_("flow_encoder_cpu", elapsedMs(start));
    const auto mel = estimate(mu, n);
    if (conformerFrames_ > prompt_ + 100) truncateConformer();
    std::vector<float> full((melCache_.size() / kMel + n) * kMel);
    const size_t cached = melCache_.size() / kMel, frames = cached + n;
    for (size_t m = 0; m < kMel; ++m) {
      std::copy(melCache_.begin() + m * cached, melCache_.begin() + (m + 1) * cached, full.begin() + m * frames);
      std::copy(mel.begin() + m * n, mel.begin() + (m + 1) * n, full.begin() + m * frames + cached);
    }
    std::vector<float> source;
    auto speech = vocode(full, frames, source);
    if (!speechCache_.empty()) {
      for (size_t i = 0; i < kSourceCache; ++i) {
        const double a = 0.54 - 0.46 * std::cos(2 * M_PI * i / (2 * kSourceCache - 1));
        const double b = 0.54 - 0.46 * std::cos(2 * M_PI * (i + kSourceCache) / (2 * kSourceCache - 1));
        speech[i] = static_cast<float>(speech[i] * a + speechCache_[i] * b);
      }
    }
    const bool first = speechCache_.empty();
    melCache_.assign(kMel * kMelCache, 0.0f);
    for (size_t m = 0; m < kMel; ++m)
      std::copy(full.begin() + m * frames + frames - kMelCache, full.begin() + (m + 1) * frames, melCache_.begin() + m * kMelCache);
    sourceCache_.assign(source.end() - kSourceCache, source.end());
    speechCache_.assign(speech.end() - kSourceCache, speech.end());
    if (!last) {
      speech.resize(speech.size() - kSourceCache);
      if (first) speech.insert(speech.begin(), kSourceCache, 0.0f);
    }
    return speech;
  }

  std::function<void(const char*, double)> timing_ = [](const char*, double) {};

 private:
  std::vector<float> encode(const std::vector<int32_t>& tokens, bool last) {
    std::vector<int64_t> ids(tokens.begin(), tokens.end());
    std::vector<float> pad(last ? 3 * 512 : 0, 0.0f);
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 2> idShape{1, static_cast<int64_t>(ids.size())};
    const std::array<int64_t, 3> padShape{1, last ? 3 : 0, 512};
    const std::array<int64_t, 3> cnnShape{1, 512, 6};
    const std::array<int64_t, 5> attShape{10, 1, 8, static_cast<int64_t>(conformerFrames_), 128};
    float padDummy = 0.0f;
    std::array<Ort::Value, 4> inputs{
        Ort::Value::CreateTensor<int64_t>(memory, ids.data(), ids.size(), idShape.data(), idShape.size()),
        Ort::Value::CreateTensor<float>(memory, pad.empty() ? &padDummy : pad.data(), pad.size(), padShape.data(), padShape.size()),
        Ort::Value::CreateTensor<float>(memory, conformerCnn_.data(), conformerCnn_.size(), cnnShape.data(), cnnShape.size()),
        Ort::Value::CreateTensor<float>(memory, conformerAtt_.data(), conformerAtt_.size(), attShape.data(), attShape.size())};
    const char* inNames[] = {"tokens", "pad", "cnn_cache", "att_cache"};
    const char* outNames[] = {"h", "cnn_cache_out", "att_cache_out"};
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, inNames, inputs.data(), inputs.size(), outNames, 3);
    const auto hShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const size_t n = static_cast<size_t>(hShape[1]);
    const float* h = outputs[0].GetTensorData<float>();
    std::vector<float> mu(kMel * n);
    for (size_t f = 0; f < n; ++f)
      for (size_t m = 0; m < kMel; ++m) mu[m * n + f] = h[f * kMel + m];
    const float* cnn = outputs[1].GetTensorData<float>();
    conformerCnn_.assign(cnn, cnn + 512 * 6);
    const auto attOut = outputs[2].GetTensorTypeAndShapeInfo();
    conformerFrames_ = static_cast<size_t>(attOut.GetShape()[3]);
    const float* att = outputs[2].GetTensorData<float>();
    conformerAtt_.assign(att, att + attOut.GetElementCount());
    return mu;
  }

  void truncateConformer() {
    const size_t frames = conformerFrames_, kept = prompt_ + 100;
    std::vector<float> out(10 * 8 * kept * 128);
    for (size_t lh = 0; lh < 10 * 8; ++lh)
      for (size_t f = 0; f < kept; ++f) {
        const size_t source = f < prompt_ ? f : frames - 100 + (f - prompt_);
        std::copy(conformerAtt_.begin() + (lh * frames + source) * 128, conformerAtt_.begin() + (lh * frames + source + 1) * 128,
                  out.begin() + (lh * kept + f) * 128);
      }
    conformerAtt_.swap(out);
    conformerFrames_ = kept;
  }

  // Ten CFG Euler steps over the cache (newest frames first, cut to prompt + its 100 oldest).
  std::vector<float> estimate(const std::vector<float>& mu, size_t n) {
    constexpr size_t T = Dit::kT;
    if (n > T) throw std::runtime_error("Token2wav chunk of " + std::to_string(n) + " mel frames");
    const size_t s = order_.size(), past = dit_.past();
    // Plan the cache after this chunk; slots of dropped frames are reused by surviving new ones.
    std::vector<int> combined(n, -1);
    combined.insert(combined.end(), order_.begin(), order_.end());
    std::vector<int> kept = combined;
    if (combined.size() > prompt_ + 100) {
      kept.assign(combined.begin(), combined.begin() + prompt_);
      kept.insert(kept.end(), combined.end() - 100, combined.end());
    }
    std::vector<bool> used(past, false);
    for (int slot : kept) if (slot >= 0) used[slot] = true;
    std::vector<int> freeSlots;
    for (size_t i = 0; i < past; ++i) if (!used[i]) freeSlots.push_back(static_cast<int>(i));
    std::vector<int> newSlot(n, -1);
    for (size_t i = 0, next = 0; i < std::min(n, kept.size()); ++i) {
      if (kept[i] != -1 || i >= n) continue;
      newSlot[i] = freeSlots.at(next++);
    }

    std::vector<float> mask(T * (past + T), kMaskMin), select((T + 2) * 2, 0.0f);
    for (size_t i = 0; i < T; ++i) {
      for (int slot : order_) mask[i * (past + T) + slot] = 0.0f;
      for (size_t j = 0; j < n; ++j) mask[i * (past + T) + past + j] = 0.0f;
    }
    select[n * 2] = select[(n + 1) * 2 + 1] = 1.0f;
    const float* noise = assets_.f32("t2w_noise");
    const size_t noiseFrames = assets_.get("t2w_noise", 0).dims[1];
    if (s + n > noiseFrames) throw std::runtime_error("Estimator noise table too short");
    std::vector<float> x(kMel * n);
    for (size_t m = 0; m < kMel; ++m)
      for (size_t f = 0; f < n; ++f) x[m * n + f] = noise[m * noiseFrames + s + f];
    std::vector<float> cond(2 * 240 * T, 0.0f), xin(2 * kMel * T, 0.0f), velocity(2 * kMel * T);
    const float* spk = assets_.f32("t2w_spk");
    for (size_t m = 0; m < kMel; ++m) {
      for (size_t f = 0; f < n; ++f) cond[m * T + f] = mu[m * n + f];
      for (size_t f = 0; f < n; ++f) cond[(kMel + m) * T + f] = spk[m];
    }
    const float* dts = assets_.f32("t2w_dts");
    for (size_t step = 0; step < Dit::kSteps; ++step) {
      for (size_t row = 0; row < 2; ++row)
        for (size_t m = 0; m < kMel; ++m)
          for (size_t f = 0; f < n; ++f) xin[(row * kMel + m) * T + f] = x[m * n + f];
      const auto start = Clock::now();
      dit_.run(step, xin.data(), cond.data(), mask, select, assets_.f32("t2w_mod_blocks") + step * 16 * 9 * 512,
               assets_.f32("t2w_mod_final") + step * 2 * 512, velocity.data());
      timing_("dit_step", elapsedMs(start));
      for (size_t m = 0; m < kMel; ++m)
        for (size_t f = 0; f < n; ++f)
          x[m * n + f] += dts[step] * (1.7f * velocity[m * T + f] - 0.7f * velocity[(kMel + m) * T + f]);
      for (size_t j = 0; j < n; ++j) if (newSlot[j] >= 0) dit_.copyNew(step, j, static_cast<size_t>(newSlot[j]));
    }
    order_.clear();
    for (size_t i = 0; i < kept.size(); ++i) order_.push_back(kept[i] >= 0 ? kept[i] : newSlot[i]);
    return x;
  }

  // HiFT: f0 graph, NSF source + STFT on the CPU, decode graph, iSTFT on the CPU.
  std::vector<float> vocode(const std::vector<float>& mel, size_t frames, std::vector<float>& source) {
    if (frames > kHiftFrames) throw std::runtime_error("HiFT chunk of " + std::to_string(frames) + " frames");
    std::vector<float> padded(kMel * kHiftFrames, 0.0f);
    for (size_t m = 0; m < kMel; ++m) std::copy(mel.begin() + m * frames, mel.begin() + (m + 1) * frames, padded.begin() + m * kHiftFrames);
    setReal(f0_.input("mel"), padded.data(), padded.size());
    f0_.execute();
    std::vector<float> f0(kHiftFrames);
    getReal(f0_.output("f0"), f0.data(), f0.size());
    f0.resize(frames);
    source = nsfSource(f0, random_, assets_.f32("t2w_source_weight"), assets_.f32("t2w_source_bias")[0]);
    std::copy(sourceCache_.begin(), sourceCache_.end(), source.begin());
    constexpr size_t kStft = kHiftFrames * 480 / kHopT + 1;
    std::vector<float> spectrum(18 * kStft, 0.0f);
    stft16(source, spectrum.data(), kStft);
    setReal(decode_.input("mel"), padded.data(), padded.size());
    setReal(decode_.input("s_stft"), spectrum.data(), spectrum.size());
    decode_.execute();
    std::vector<float> head(18 * kStft);
    getReal(decode_.output("head"), head.data(), head.size());
    return istft16(head.data(), kStft, frames * 480 / kHopT + 1);
  }

  const Assets& assets_;
  Dit dit_;
  Graph f0_, decode_;
  Ort::Env env_;
  Ort::Session session_;
  size_t prompt_ = 0, conformerFrames_ = 0;
  std::vector<float> conformerAtt_, conformerCnn_, melCache_, sourceCache_, speechCache_;
  std::vector<int> order_;
  std::mt19937_64 random_;
};

// One duplex conversation: 1 s units of <unit> + 10 audio embeddings, up to 20 decoded tokens,
// a TTS chunk per speaking unit, token2wav in 28-token windows.
class Session {
 public:
  Session(const std::string& libraryDir, const std::string& dir) : runtime_(libraryDir), assets_(dir + "/assets.bin") {
    uint64_t spill = 0;
    for (const char* name : {"llm_head", "audio_encoder", "tts", "dit", "hift_f0", "hift_decode", "llm_part_0", "llm_part_1",
                             "llm_part_2", "llm_part_3"})
      spill = std::max(spill, runtime_.spillFillBytes(dir + "/" + name + ".bin"));
    runtime_.groupSpillBytes = spill;
    head_ = std::make_unique<Graph>(runtime_, dir + "/llm_head.bin", true);  // group leader: created first, freed last
    encoder_ = std::make_unique<AudioEncoder>(runtime_, dir + "/audio_encoder.bin", assets_.f16("whisper_positions"));
    for (int p = 0; p < 4; ++p)
      parts_.push_back(std::make_unique<KvModel>(runtime_, dir + "/llm_part_" + std::to_string(p) + ".bin",
                                                 "llm_part_" + std::to_string(p), "hidden", "hidden_out", 1e6));
    tts_ = std::make_unique<Tts>(runtime_, dir + "/tts.bin", assets_.f16("tts_emb_code"));
    token2wav_ = std::make_unique<Token2wav>(runtime_, dir, assets_);
    token2wav_->timing_ = [this](const char* name, double ms) { runtime_.timing(name, "execute", ms); };
    mel_ = std::make_unique<StreamingMel>(assets_.f32("mel_filters"));
    worker_ = std::thread([this] { work(); });
  }

  ~Session() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    worker_.join();
    runtime_.endPerformanceVote();
    token2wav_.reset(); tts_.reset(); parts_.clear(); encoder_.reset();
    head_.reset();
    runtime_.groupLeader = nullptr;
  }

  void begin(bool highPerformance, float temperature, uint64_t seed) {
    idle();
    hasReady_ = false;
    runtime_.endPerformanceVote();
    runtime_.clearTimings();
    if (highPerformance) runtime_.beginPerformanceVote();
    started_ = Clock::now();
    for (auto& part : parts_) part->reset();
    encoder_->reset();
    mel_->reset();
    sampler_.reset(seed, temperature);
    tts_->seed(seed + 1);
    tts_->reset();
    token2wav_->seed(seed + 2);
    token2wav_->reset();
    position_ = 0;
    unit_ = 0;
    turnEnded_ = ttsTurnStart_ = true;
    codeBuffer_.assign(3, kSilenceCode);
    audio_.clear();
    pending_.clear();
    const auto& system = assets_.get("system_embeds", 1);
    std::vector<float> rows(system.dims[0] * kDim);
    rowF16(assets_.f16("system_embeds"), 0, rows.size(), rows.data());
    pending_ = rows;
    flush();
    runtime_.timing("system_prompt", "total", elapsedMs(started_));
  }

  // 100 ms of 16 kHz input; returns whether a unit completed (then pcm / text hold its output).
  bool step(const int16_t* input, size_t count, std::vector<int16_t>& pcm, std::vector<uint8_t>& text) {
    for (size_t i = 0; i < count; ++i) audio_.push_back(input[i] / 32768.0f);
    const size_t need = unit_ == 0 ? 16480 : 16000;
    if (unit_ == 0 && audio_.size() >= 16000 && audio_.size() < 16560) audio_.insert(audio_.begin(), 16560 - audio_.size(), 0.0f);
    if (audio_.size() < need || (unit_ == 0 && audio_.size() < 16560)) return false;
    const auto start = Clock::now();
    size_t frames = 0;
    const auto mel = mel_->process(audio_.data(), need, frames);
    audio_.erase(audio_.begin(), audio_.begin() + need);
    const auto embeds = encoder_->encode(mel, frames, unit_ == 0);
    ++unit_;
    std::vector<float> wave = exchange(generate(embeds, text));
    pcm.resize(wave.size());
    // vocode() already holds int16 levels / 32768, as upstream returns its PCM bytes.
    for (size_t i = 0; i < wave.size(); ++i) pcm[i] = static_cast<int16_t>(std::clamp<long>(std::lrint(wave[i] * 32768.0f), -32768, 32767));
    runtime_.timing("unit", "total", elapsedMs(start));
    return true;
  }

  void end(const std::string& performanceCsv) {
    idle();
    runtime_.timing("pipeline", "wall_total", elapsedMs(started_));
    runtime_.endPerformanceVote();
    if (!performanceCsv.empty()) runtime_.writeTimings(performanceCsv);
  }

 private:
  // One unit's speech work: the TTS condition, or silence for a listening unit.
  struct Speech {
    std::vector<float> condition;
    bool listen = true, endOfTurn = false;
  };

  // TTS + token2wav run on worker_ one unit behind, so their CPU work overlaps the next unit's LLM.
  void work() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      wake_.wait(lock, [this] { return stop_ || job_.has_value(); });
      if (stop_) return;
      Speech job = std::move(*job_);
      job_.reset();
      lock.unlock();
      std::vector<float> wave;
      std::exception_ptr error;
      try {
        wave = synthesize(job);
      } catch (...) {
        error = std::current_exception();
      }
      lock.lock();
      ready_ = std::move(wave);
      workerError_ = error;
      busy_ = false;
      wake_.notify_all();
    }
  }

  void idle() {
    std::unique_lock<std::mutex> lock(mutex_);
    wake_.wait(lock, [this] { return !busy_; });
    if (workerError_) std::rethrow_exception(std::exchange(workerError_, nullptr));
  }

  // Hands this unit's speech to the worker and returns the previous unit's audio (silence for unit 0).
  std::vector<float> exchange(Speech speech) {
    idle();
    std::vector<float> previous = hasReady_ ? std::move(ready_) : std::vector<float>(24000, 0.0f);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      job_ = std::move(speech);
      busy_ = hasReady_ = true;
    }
    wake_.notify_all();
    return previous;
  }

  void embed(int32_t token, std::vector<float>& rows) {
    const size_t at = rows.size();
    rows.resize(at + kDim);
    rowF16(assets_.f16("embed_tokens"), static_cast<size_t>(token), kDim, rows.data() + at);
  }

  // Feeds pending rows (+ `rows`) through the four parts; returns the last row's final hidden.
  std::vector<float> flush(const std::vector<float>& rows = {}) {
    pending_.insert(pending_.end(), rows.begin(), rows.end());
    std::vector<float> hidden;
    const size_t total = pending_.size() / kDim;
    for (size_t start = 0; start < total; start += kArN) {
      const size_t n = std::min<size_t>(kArN, total - start);
      std::vector<float> x(pending_.begin() + start * kDim, pending_.begin() + (start + n) * kDim);
      for (auto& part : parts_) x = part->run(position_, x.data(), n);
      position_ += static_cast<int64_t>(n);
      hidden.assign(x.end() - kDim, x.end());
    }
    pending_.clear();
    return hidden;
  }

  void logitsOf(const std::vector<float>& hidden, std::vector<float>& logits, std::vector<float>& semantic) {
    setReal(head_->input("hidden"), hidden.data(), kDim);
    head_->execute();
    logits.resize(kVocab);
    semantic.resize(kTtsDim);
    getReal(head_->output("logits"), logits.data(), kVocab);
    getReal(head_->output("semantic"), semantic.data(), kTtsDim);
  }

  void appendText(int32_t token, std::vector<uint8_t>& text) {
    const int32_t* offsets = assets_.i32("vocab_offsets");
    const uint8_t* bytes = assets_.u8("vocab_bytes");
    if (token >= 0 && token < kVocab) text.insert(text.end(), bytes + offsets[token], bytes + offsets[token + 1]);
  }

  Speech generate(const std::vector<float>& audio, std::vector<uint8_t>& text) {
    std::vector<float> rows;
    embed(kUnit, rows);
    rows.insert(rows.end(), audio.begin(), audio.end());
    std::vector<float> logits, semantic;
    logitsOf(flush(rows), logits, semantic);
    std::vector<float> condition;
    bool listen = false, endOfTurn = false;
    std::vector<float> emb(kTtsDim);
    for (int j = 0; j < kMaxSpeakTokens; ++j) {
      if (j == kMaxSpeakTokens - 1) { embed(kChunkEos, pending_); break; }
      int32_t token = sampler_(logits);
      if (token == kListen && !turnEnded_) token = kTtsBos;
      listen = token == kListen;
      if (token == kListen || token == kChunkEos || token == kChunkTtsEos) { embed(token, pending_); break; }
      turnEnded_ = false;
      std::vector<float> one;
      embed(token, one);
      logitsOf(flush(one), logits, semantic);
      endOfTurn = token == kTurnEos;
      if (endOfTurn) turnEnded_ = true;
      if (j != 0) {
        rowF16(assets_.f16("tts_emb_text"), static_cast<size_t>(token), kTtsDim, emb.data());
        for (int i = 0; i < kTtsDim; ++i) condition.push_back(emb[i] + semantic[i]);
        appendText(token, text);
      }
    }
    embed(kUnitEnd, pending_);
    Speech speech;
    speech.listen = listen;
    speech.endOfTurn = endOfTurn;
    if (listen) return speech;
    rowF16(assets_.f16("tts_emb_text"), kAudioBosText, kTtsDim, emb.data());
    condition.insert(condition.end(), emb.begin(), emb.end());
    speech.condition = std::move(condition);
    return speech;
  }

  // Worker thread only: owns tts_, token2wav_, codeBuffer_ and ttsTurnStart_.
  std::vector<float> synthesize(const Speech& speech) {
    if (speech.listen) return std::vector<float>(24000, 0.0f);
    const bool turnStart = ttsTurnStart_;
    const auto start = Clock::now();
    const auto codes = tts_->generate(speech.condition, speech.condition.size() / kTtsDim, speech.endOfTurn || turnStart ? 0 : 26);
    runtime_.timing("tts_chunk", "total", elapsedMs(start));
    ttsTurnStart_ = speech.endOfTurn;
    if (speech.endOfTurn) tts_->reset();
    auto wave = vocode(codes, speech.endOfTurn, turnStart);
    if (speech.endOfTurn) { token2wav_->reset(); codeBuffer_.assign(3, kSilenceCode); }
    runtime_.timing("speech", "total", elapsedMs(start));
    return wave;
  }

  std::vector<float> vocode(const std::vector<int32_t>& codes, bool last, bool forceFlush) {
    codeBuffer_.insert(codeBuffer_.end(), codes.begin(), codes.end());
    std::vector<float> wave;
    const auto emit = [&](size_t take, bool final) {
      std::vector<int32_t> window(codeBuffer_.begin(), codeBuffer_.begin() + take);
      const auto start = Clock::now();
      const auto piece = token2wav_->stream(window, final);
      runtime_.timing("token2wav_chunk", "total", elapsedMs(start));
      wave.insert(wave.end(), piece.begin(), piece.end());
    };
    if (forceFlush) {
      while (codeBuffer_.size() >= 3 + 5) {
        const size_t take = std::min<size_t>(28, codeBuffer_.size());
        emit(take, false);
        codeBuffer_.erase(codeBuffer_.begin(), codeBuffer_.begin() + std::min<size_t>(25, take - 3));
      }
    } else {
      while (codeBuffer_.size() >= 28) {
        emit(28, false);
        codeBuffer_.erase(codeBuffer_.begin(), codeBuffer_.begin() + 25);
      }
    }
    if (last && !codeBuffer_.empty()) {
      // The final window yields 2 mel frames per token: past 25 tokens it would overflow the DiT chunk (50)
      // and HiFT (58 frames), so the head goes out first as a lookahead window.
      if (codeBuffer_.size() > 25) {
        emit(codeBuffer_.size(), false);
        codeBuffer_.erase(codeBuffer_.begin(), codeBuffer_.end() - 3);
      }
      emit(codeBuffer_.size(), true);
      codeBuffer_.clear();
    }
    for (auto& v : wave) v = std::nearbyint(std::clamp(v, -1.0f, 1.0f) * 32767.0f) / 32768.0f;
    if (!last && wave.size() < 24000) wave.insert(wave.begin(), 24000 - wave.size(), 0.0f);
    return wave;
  }

  Runtime runtime_;
  Assets assets_;
  std::unique_ptr<Graph> head_;
  std::unique_ptr<AudioEncoder> encoder_;
  std::vector<std::unique_ptr<KvModel>> parts_;
  std::unique_ptr<Tts> tts_;
  std::unique_ptr<Token2wav> token2wav_;
  std::unique_ptr<StreamingMel> mel_;
  Sampler sampler_;
  std::vector<float> audio_, pending_;
  std::vector<int32_t> codeBuffer_;
  int64_t position_ = 0;
  size_t unit_ = 0;
  bool turnEnded_ = true, ttsTurnStart_ = true;
  Clock::time_point started_ = Clock::now();
  std::mutex mutex_;
  std::condition_variable wake_;
  std::optional<Speech> job_;
  std::vector<float> ready_;
  std::exception_ptr workerError_;
  bool busy_ = false, hasReady_ = false, stop_ = false;
  std::thread worker_;  // last: started after, and joined before, everything it uses
};

}  // namespace

void* minicpmoCreate(const std::string& libraryDir, const std::string& modelDirectory) {
  return new Session(libraryDir, modelDirectory);
}

void minicpmoBegin(void* session, bool highPerformance, float temperature, uint64_t seed) {
  static_cast<Session*>(session)->begin(highPerformance, temperature, seed);
}

bool minicpmoStep(void* session, const int16_t* input, size_t count, std::vector<int16_t>& pcm, std::vector<uint8_t>& text) {
  return static_cast<Session*>(session)->step(input, count, pcm, text);
}

void minicpmoEnd(void* session, const std::string& performanceCsv) { static_cast<Session*>(session)->end(performanceCsv); }

void minicpmoDestroy(void* session) { delete static_cast<Session*>(session); }
