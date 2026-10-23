package com.overmindy.duplex.engine;

import android.content.Context;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;

/** Kyutai Moshi on QNN HTP: Mimi encoder -> Temporal + DepFormer -> Mimi decoder, 80 ms per step. */
public final class MoshiEngine implements DuplexEngine {
    private static final int FRAME_SAMPLES = 1920;

    private long handle;
    private boolean codecOnly;
    private String[] vocabulary;
    private final ByteArrayOutputStream textBytes = new ByteArrayOutputStream();
    private int emittedBytes;

    @Override public int sampleRate() { return 24_000; }
    @Override public int frameSamples() { return FRAME_SAMPLES; }
    @Override public String describe() { return codecOnly ? "Mimi 编解码回环" : "Mimi + Moshi LM"; }

    @Override
    public void load(Context context, File modelDir, Options options) throws Exception {
        release();
        codecOnly = options.codecOnly;
        vocabulary = codecOnly ? null : readVocabulary(new File(modelDir, "text_vocab.txt"));
        handle = MoshiNative.nativeCreate(context.getApplicationInfo().nativeLibraryDir,
                modelDir.getAbsolutePath(), codecOnly);
    }

    @Override
    public void begin(boolean highPerformance) {
        textBytes.reset();
        emittedBytes = 0;
        MoshiNative.nativeBegin(handle, highPerformance);
    }

    @Override
    public Frame step(short[] input) {
        short[] pcm = new short[FRAME_SAMPLES];
        int[] tokens = new int[9];
        if (!MoshiNative.nativeStep(handle, input, pcm, tokens)) return null;
        int[] codes = new int[8];
        System.arraycopy(tokens, 1, codes, 0, 8);
        return new Frame(pcm, appendText(tokens[0]), codes);
    }

    @Override
    public void end(File performanceCsv) {
        MoshiNative.nativeEnd(handle, performanceCsv == null ? "" : performanceCsv.getAbsolutePath());
    }

    @Override
    public void release() {
        if (handle != 0) MoshiNative.nativeRelease(handle);
        handle = 0;
    }

    // SentencePiece: ids 0-3 are <unk>/<s>/</s>/<pad>, "<0xHH>" pieces are raw bytes, "▁" is a space.
    // Only whole UTF-8 characters are returned, so a multi-byte character split across frames waits.
    private String appendText(int id) {
        if (vocabulary == null || id <= 3 || id >= vocabulary.length) return "";
        String piece = vocabulary[id];
        if (piece.length() == 6 && piece.startsWith("<0x") && piece.endsWith(">")) {
            textBytes.write(Integer.parseInt(piece.substring(3, 5), 16));
        } else {
            byte[] encoded = piece.replace('▁', ' ').getBytes(StandardCharsets.UTF_8);
            textBytes.write(encoded, 0, encoded.length);
        }
        byte[] all = textBytes.toByteArray();
        int complete = completeUtf8(all);
        if (complete <= emittedBytes) return "";
        String text = new String(all, emittedBytes, complete - emittedBytes, StandardCharsets.UTF_8);
        emittedBytes = complete;
        return text;
    }

    private static int completeUtf8(byte[] bytes) {
        for (int back = 1; back <= Math.min(3, bytes.length); back++) {
            int b = bytes[bytes.length - back] & 0xff;
            if ((b & 0xc0) == 0x80) continue;
            int needed = b >= 0xf0 ? 4 : b >= 0xe0 ? 3 : b >= 0xc0 ? 2 : 1;
            return needed > back ? bytes.length - back : bytes.length;
        }
        return bytes.length;
    }

    private static String[] readVocabulary(File file) throws IOException {
        ArrayList<String> pieces = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) pieces.add(unescape(line));
        }
        return pieces.toArray(new String[0]);
    }

    private static String unescape(String line) {
        StringBuilder out = new StringBuilder(line.length());
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '\\' && i + 1 < line.length()) {
                char next = line.charAt(++i);
                out.append(next == 'n' ? '\n' : next == 'r' ? '\r' : next);
            } else {
                out.append(c);
            }
        }
        return out.toString();
    }
}
