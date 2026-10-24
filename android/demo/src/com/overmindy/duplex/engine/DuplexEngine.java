package com.overmindy.duplex.engine;

import android.content.Context;

import java.io.File;

/**
 * A full-duplex speech model driven one fixed-size audio frame at a time. Every call for one
 * engine must come from the same thread; load() is slow and never runs on the UI thread.
 */
public interface DuplexEngine {
    final class Options {
        public boolean codecOnly;
    }

    /** One step's output; text is the newly decoded part of the model's reply (may be empty). */
    final class Frame {
        public final short[] pcm;
        public final String text;
        public final int[] codes;

        public Frame(short[] pcm, String text, int[] codes) {
            this.pcm = pcm;
            this.text = text;
            this.codes = codes;
        }
    }

    /** Input (microphone) sample rate; frameSamples() is in this rate. */
    int sampleRate();
    /** Rate of the PCM in returned frames, whose length may vary per step. */
    default int outputSampleRate() { return sampleRate(); }
    int frameSamples();
    String describe();

    void load(Context context, File modelDir, Options options) throws Exception;

    /** Resets the conversation state; textTemperature 0 decodes greedily. */
    void begin(boolean highPerformance, float textTemperature);

    /** Consumes one input frame; returns null while the model has no output for it yet. */
    Frame step(short[] input);

    void end(File performanceCsv);

    void release();
}
