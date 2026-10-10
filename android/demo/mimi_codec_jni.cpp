#include <jni.h>

#include <cstdlib>
#include <cstdint>
#include <exception>
#include <string>
#include <stdexcept>

void* createMimiCodecSession(const std::string& libraryDir,
                             const std::string& encoderDlc,
                             const std::string& decoderDlc);
size_t runMimiCodecSession(void* session, const std::string& inputWav,
                           const std::string& outputWav, size_t requestedFrames);
void destroyMimiCodecSession(void* session);
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
  jclass exception = env->FindClass("java/lang/RuntimeException");
  if (exception) env->ThrowNew(exception, message);
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_overmindy_mimicodecdemo_MainActivity_nativePrepare(
    JNIEnv* env, jclass, jstring libraryDir, jstring encoderDlc,
    jstring decoderDlc, jstring lmDirectory) {
  try {
    const std::string libraries = JavaString(env, libraryDir).str();
    const std::string encoder = JavaString(env, encoderDlc).str();
    const std::string decoder = JavaString(env, decoderDlc).str();
    const std::string lm = JavaString(env, lmDirectory).str();
    setenv("ADSP_LIBRARY_PATH", libraries.c_str(), 1);
    return reinterpret_cast<jlong>(lm.empty()
        ? createMimiCodecSession(libraries, encoder, decoder)
        : createMoshiCodecSession(libraries, encoder, decoder, lm));
  } catch (const std::exception& error) {
    throwJava(env, error.what());
    return 0;
  }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_overmindy_mimicodecdemo_MainActivity_nativeRunPrepared(
    JNIEnv* env, jclass, jlong handle, jstring inputWav, jstring outputWav, jint frames) {
  try {
    const std::string input = JavaString(env, inputWav).str();
    const std::string output = JavaString(env, outputWav).str();
    if (frames < 0 || frames > 3000) throw std::runtime_error("Invalid frame count");
    return static_cast<jint>(runMimiCodecSession(reinterpret_cast<void*>(handle),
                                                input, output, static_cast<size_t>(frames)));
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
