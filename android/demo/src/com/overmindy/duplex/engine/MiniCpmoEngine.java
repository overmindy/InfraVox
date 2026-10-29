package com.overmindy.duplex.engine;

import android.content.Context;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.nio.charset.StandardCharsets;

/**
 * OpenBMB MiniCPM-o 4.5 duplex speech on QNN HTP: 16 kHz input in 80 ms frames, which the runner
 * collects into 1 s units (Whisper -> Qwen3-8B listen/speak decoding -> TTS -> token2wav). A
 * completed unit returns its 24 kHz reply (1 s of silence while listening).
 */
public final class MiniCpmoEngine implements DuplexEngine {
    private static final int FRAME_SAMPLES = 1280;
    private static final int[] NO_CODES = new int[0];

    private long handle;
    private NpuGate npu;
    private final ByteArrayOutputStream textBytes = new ByteArrayOutputStream();
    private int emittedBytes;

    @Override public int sampleRate() { return 16_000; }
    @Override public int outputSampleRate() { return 24_000; }
    // Each 1 s unit's audio lands 150-950 ms after the unit, depending on how much it decoded.
    @Override public int playbackReserveMs() { return 1000; }
    @Override public int frameSamples() { return FRAME_SAMPLES; }
    @Override public String describe() { return "MiniCPM-o 4.5 双工"; }

    @Override
    public void load(Context context, File modelDir, Options options) throws Exception {
        release();
        npu = NpuGate.acquire(context, 1);
        try {
            handle = MiniCpmoNative.nativeCreate(context.getApplicationInfo().nativeLibraryDir, modelDir.getAbsolutePath());
        } catch (Exception e) {
            throw new Exception(e.getMessage() + "（NPU：" + npu.status + "）", e);
        }
        npu.notifyLoaded();
    }

    @Override
    public void begin(boolean highPerformance, float textTemperature) {
        textBytes.reset();
        emittedBytes = 0;
        MiniCpmoNative.nativeBegin(handle, highPerformance, textTemperature, (int) System.nanoTime());
    }

    @Override
    public Frame step(short[] input) {
        short[] pcm = MiniCpmoNative.nativeStep(handle, input);
        if (pcm == null) return null;
        byte[] text = MiniCpmoNative.nativeText(handle);
        textBytes.write(text, 0, text.length);
        return new Frame(pcm, completeText(), NO_CODES);
    }

    @Override
    public void end(File performanceCsv) {
        MiniCpmoNative.nativeEnd(handle, performanceCsv == null ? "" : performanceCsv.getAbsolutePath());
    }

    @Override
    public void release() {
        if (handle != 0) MiniCpmoNative.nativeRelease(handle);
        handle = 0;
        if (npu != null) npu.notifyUnloaded();
        npu = null;
    }

    // Byte-level BPE pieces can split a UTF-8 character across units; only whole characters are returned.
    private String completeText() {
        byte[] all = textBytes.toByteArray();
        int complete = all.length;
        for (int back = 1; back <= Math.min(3, all.length); back++) {
            int b = all[all.length - back] & 0xff;
            if ((b & 0xc0) == 0x80) continue;
            int needed = b >= 0xf0 ? 4 : b >= 0xe0 ? 3 : b >= 0xc0 ? 2 : 1;
            complete = needed > back ? all.length - back : all.length;
            break;
        }
        if (complete <= emittedBytes) return "";
        String text = new String(all, emittedBytes, complete - emittedBytes, StandardCharsets.UTF_8);
        emittedBytes = complete;
        return text;
    }
}
