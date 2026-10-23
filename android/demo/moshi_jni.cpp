#include <jni.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

void* moshiCreate(const std::string& libraryDir, const std::string& modelDirectory, bool codecOnly);
void moshiBegin(void* session, bool highPerformance);
bool moshiStep(void* session, const int16_t* input, int16_t* pcm, int32_t* tokens);
void moshiEnd(void* session, const std::string& performanceCsv);
void moshiDestroy(void* session);

namespace {

constexpr jsize kFrameSamples = 1920;
constexpr jsize kTokens = 9;

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

void* session(jlong handle) {
  if (!handle) throw std::runtime_error("Model is not loaded");
  return reinterpret_cast<void*>(handle);
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_overmindy_duplex_engine_MoshiNative_nativeCreate(
    JNIEnv* env, jclass, jstring libraryDir, jstring modelDir, jboolean codecOnly) {
  try {
    const std::string libraries = JavaString(env, libraryDir).str();
    setenv("ADSP_LIBRARY_PATH", libraries.c_str(), 1);
    return reinterpret_cast<jlong>(moshiCreate(libraries, JavaString(env, modelDir).str(), codecOnly == JNI_TRUE));
  } catch (const std::exception& error) {
    throwJava(env, error.what());
    return 0;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_overmindy_duplex_engine_MoshiNative_nativeBegin(JNIEnv* env, jclass, jlong handle, jboolean highPerformance) {
  try {
    moshiBegin(session(handle), highPerformance == JNI_TRUE);
  } catch (const std::exception& error) {
    throwJava(env, error.what());
  }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_overmindy_duplex_engine_MoshiNative_nativeStep(
    JNIEnv* env, jclass, jlong handle, jshortArray input, jshortArray output, jintArray tokens) {
  try {
    if (env->GetArrayLength(input) != kFrameSamples || env->GetArrayLength(output) != kFrameSamples ||
        env->GetArrayLength(tokens) != kTokens)
      throw std::runtime_error("Expected 1920-sample frames and 9 tokens");
    std::array<int16_t, kFrameSamples> in{}, pcm{};
    std::array<int32_t, kTokens> out{};
    env->GetShortArrayRegion(input, 0, kFrameSamples, in.data());
    if (!moshiStep(session(handle), in.data(), pcm.data(), out.data())) return JNI_FALSE;
    env->SetShortArrayRegion(output, 0, kFrameSamples, pcm.data());
    env->SetIntArrayRegion(tokens, 0, kTokens, out.data());
    return JNI_TRUE;
  } catch (const std::exception& error) {
    throwJava(env, error.what());
    return JNI_FALSE;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_overmindy_duplex_engine_MoshiNative_nativeEnd(JNIEnv* env, jclass, jlong handle, jstring performanceCsv) {
  try {
    moshiEnd(session(handle), JavaString(env, performanceCsv).str());
  } catch (const std::exception& error) {
    throwJava(env, error.what());
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_overmindy_duplex_engine_MoshiNative_nativeRelease(JNIEnv*, jclass, jlong handle) {
  if (handle) moshiDestroy(reinterpret_cast<void*>(handle));
}
