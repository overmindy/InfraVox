package com.overmindy.duplex.engine;

/** JNI surface of android/mimi_qnn_runner.cpp via android/demo/moshi_jni.cpp. */
final class MoshiNative {
    static {
        System.loadLibrary("moshi_qnn");
    }

    private MoshiNative() { }

    static native long nativeCreate(String libraryDir, String modelDir, boolean codecOnly);
    static native void nativeBegin(long handle, boolean highPerformance);
    static native boolean nativeStep(long handle, short[] input, short[] output, int[] tokens);
    static native void nativeEnd(long handle, String performanceCsv);
    static native void nativeRelease(long handle);
}
