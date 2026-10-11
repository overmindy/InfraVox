#include <jni.h>

#include <cstdlib>
#include <cstdint>
#include <exception>
#include <string>
#include <stdexcept>
#include <functional>
#include <vector>

void* createMimiCodecSession(const std::string& libraryDir,
                             const std::string& encoderDlc,
                             const std::string& decoderDlc);
size_t runMimiCodecSession(void* session, const std::string& inputWav,
                           const std::string& outputWav, size_t requestedFrames);
size_t runMimiCodecSessionStreaming(void* session, const std::string& inputWav,
    const std::string& outputWav, size_t frames,
    const std::function<void(size_t, size_t, double, const std::vector<int16_t>&)>& callback, bool highPerformance);
void destroyMimiCodecSession(void* session);
void configureMimiCodecSessionCache(void* session, const std::string& directory);
void* createMoshiCodecSession(const std::string& libraryDir,
                              const std::string& encoderDlc,
                              const std::string& decoderDlc,
                              const std::string& lmDirectory);

namespace {

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

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_overmindy_mimicodecdemo_MainActivity_nativePrepare(
    JNIEnv* env, jclass, jstring libraryDir, jstring encoderDlc,
    jstring decoderDlc, jstring lmDirectory, jstring cacheDirectory) {
  try {
    const std::string libraries = JavaString(env, libraryDir).str();
    const std::string encoder = JavaString(env, encoderDlc).str();
    const std::string decoder = JavaString(env, decoderDlc).str();
    const std::string lm = JavaString(env, lmDirectory).str();
    setenv("ADSP_LIBRARY_PATH", libraries.c_str(), 1);
    void* session = lm.empty()
        ? createMimiCodecSession(libraries, encoder, decoder)
        : createMoshiCodecSession(libraries, encoder, decoder, lm);
    try { configureMimiCodecSessionCache(session, JavaString(env, cacheDirectory).str()); }
    catch (...) { destroyMimiCodecSession(session); throw; }
    return reinterpret_cast<jlong>(session);
  } catch (const std::exception& error) {
    throwJava(env, error.what());
    return 0;
  }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_overmindy_mimicodecdemo_MainActivity_nativeRunPrepared(
    JNIEnv* env, jclass, jlong handle, jstring inputWav, jstring outputWav, jint frames, jobject receiver, jboolean highPerformance) {
  try {
    const std::string input = JavaString(env, inputWav).str();
    const std::string output = JavaString(env, outputWav).str();
    if (frames < 0 || frames > 3000) throw std::runtime_error("Invalid frame count");
    jclass cls = env->GetObjectClass(receiver);
    jmethodID method = env->GetMethodID(cls, "onNativeFrame", "(IID[S)V");
    env->DeleteLocalRef(cls);
    if (!method || env->ExceptionCheck()) return 0;
    // Callback is synchronous on this Java worker/JNI thread. No saved JNIEnv or
    // global Activity reference, and no UI work on the native inference thread.
    return static_cast<jint>(runMimiCodecSessionStreaming(reinterpret_cast<void*>(handle), input, output,
        static_cast<size_t>(frames), [&](size_t consumed, size_t emitted, double ms, const std::vector<int16_t>& pcm) {
          jshortArray audio = env->NewShortArray(static_cast<jsize>(pcm.size()));
          if (!audio) throw std::runtime_error("Cannot allocate streaming audio callback");
          if (!pcm.empty()) env->SetShortArrayRegion(audio, 0, static_cast<jsize>(pcm.size()), pcm.data());
          if (!env->ExceptionCheck()) env->CallVoidMethod(receiver, method, static_cast<jint>(consumed), static_cast<jint>(emitted), ms, audio);
          env->DeleteLocalRef(audio);
          if (env->ExceptionCheck()) throw std::runtime_error("Streaming callback failed");
        }, highPerformance == JNI_TRUE));
  } catch (const std::exception& error) {
    throwJava(env, error.what());
    return 0;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_overmindy_mimicodecdemo_MainActivity_nativeRelease(
    JNIEnv*, jclass, jlong handle) {
  destroyMimiCodecSession(reinterpret_cast<void*>(handle));
}
