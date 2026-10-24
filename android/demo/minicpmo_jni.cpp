#include <jni.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

void* minicpmoCreate(const std::string& libraryDir, const std::string& modelDirectory);
void minicpmoBegin(void* session, bool highPerformance, float temperature, uint64_t seed);
bool minicpmoStep(void* session, const int16_t* input, size_t count, std::vector<int16_t>& pcm, std::vector<uint8_t>& text);
void minicpmoEnd(void* session, const std::string& performanceCsv);
void minicpmoDestroy(void* session);

namespace {

// The runner session plus the text bytes of its last completed unit.
struct Handle {
  void* session;
  std::vector<uint8_t> text;
};

class JavaString {
 public:
  JavaString(JNIEnv* env, jstring value) : env_(env), value_(value) {
    chars_ = env_->GetStringUTFChars(value_, nullptr);
    if (!chars_) throw std::bad_alloc();
  }
  ~JavaString() { env_->ReleaseStringUTFChars(value_, chars_); }
  std::string str() const { return chars_; }

 private:
  JNIEnv* env_;
  jstring value_;
  const char* chars_;
};

void throwJava(JNIEnv* env, const char* message) {
  if (env->ExceptionCheck()) return;
  jclass exception = env->FindClass("java/lang/RuntimeException");
  if (exception) env->ThrowNew(exception, message);
}

Handle* handleOf(jlong value) {
  if (!value) throw std::runtime_error("Model is not loaded");
  return reinterpret_cast<Handle*>(value);
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_overmindy_duplex_engine_MiniCpmoNative_nativeCreate(JNIEnv* env, jclass, jstring libraryDir, jstring modelDir) {
  try {
    const std::string libraries = JavaString(env, libraryDir).str();
    setenv("ADSP_LIBRARY_PATH", libraries.c_str(), 1);
    return reinterpret_cast<jlong>(new Handle{minicpmoCreate(libraries, JavaString(env, modelDir).str()), {}});
  } catch (const std::exception& error) {
    throwJava(env, error.what());
    return 0;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_overmindy_duplex_engine_MiniCpmoNative_nativeBegin(
    JNIEnv* env, jclass, jlong handle, jboolean highPerformance, jfloat temperature, jint seed) {
  try {
    minicpmoBegin(handleOf(handle)->session, highPerformance == JNI_TRUE, temperature, static_cast<uint32_t>(seed));
  } catch (const std::exception& error) {
    throwJava(env, error.what());
  }
}

// Returns the 24 kHz output of a completed 1 s unit, or null while the unit is still filling.
extern "C" JNIEXPORT jshortArray JNICALL
Java_com_overmindy_duplex_engine_MiniCpmoNative_nativeStep(JNIEnv* env, jclass, jlong handle, jshortArray input) {
  try {
    Handle* h = handleOf(handle);
    const jsize count = env->GetArrayLength(input);
    std::vector<int16_t> in(static_cast<size_t>(count)), pcm;
    env->GetShortArrayRegion(input, 0, count, in.data());
    h->text.clear();
    if (!minicpmoStep(h->session, in.data(), in.size(), pcm, h->text)) return nullptr;
    jshortArray out = env->NewShortArray(static_cast<jsize>(pcm.size()));
    if (out) env->SetShortArrayRegion(out, 0, static_cast<jsize>(pcm.size()), pcm.data());
    return out;
  } catch (const std::exception& error) {
    throwJava(env, error.what());
    return nullptr;
  }
}

// UTF-8 bytes of the text decoded in the last completed unit (may split a character).
extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_overmindy_duplex_engine_MiniCpmoNative_nativeText(JNIEnv* env, jclass, jlong handle) {
  try {
    const auto& text = handleOf(handle)->text;
    jbyteArray out = env->NewByteArray(static_cast<jsize>(text.size()));
    if (out) env->SetByteArrayRegion(out, 0, static_cast<jsize>(text.size()), reinterpret_cast<const jbyte*>(text.data()));
    return out;
  } catch (const std::exception& error) {
    throwJava(env, error.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_overmindy_duplex_engine_MiniCpmoNative_nativeEnd(JNIEnv* env, jclass, jlong handle, jstring performanceCsv) {
  try {
    minicpmoEnd(handleOf(handle)->session, JavaString(env, performanceCsv).str());
  } catch (const std::exception& error) {
    throwJava(env, error.what());
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_overmindy_duplex_engine_MiniCpmoNative_nativeRelease(JNIEnv*, jclass, jlong handle) {
  if (!handle) return;
  Handle* h = reinterpret_cast<Handle*>(handle);
  minicpmoDestroy(h->session);
  delete h;
}
