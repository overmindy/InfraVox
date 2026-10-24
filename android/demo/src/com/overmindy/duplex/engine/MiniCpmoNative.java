package com.overmindy.duplex.engine;

/** JNI surface of android/minicpmo_runner.cpp via android/demo/minicpmo_jni.cpp. */
final class MiniCpmoNative {
    static {
        System.loadLibrary("onnxruntime");
        System.loadLibrary("minicpmo_qnn");
    }

    private MiniCpmoNative() { }

    static native long nativeCreate(String libraryDir, String modelDir);
    static native void nativeBegin(long handle, boolean highPerformance, float temperature, int seed);
    static native short[] nativeStep(long handle, short[] input);
    static native byte[] nativeText(long handle);
    static native void nativeEnd(long handle, String performanceCsv);
    static native void nativeRelease(long handle);
}
