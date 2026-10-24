package com.overmindy.duplex.audio;

import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/** Mono PCM16 WAV reading and incremental writing. */
public final class Wav {
    private Wav() { }

    /** Reads a mono PCM16 WAV of any rate, linearly resampled to `sampleRate`. */
    public static short[] readMono16(InputStream input, int sampleRate) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        byte[] buffer = new byte[65536];
        int count;
        while ((count = input.read(buffer)) != -1) bytes.write(buffer, 0, count);
        ByteBuffer file = ByteBuffer.wrap(bytes.toByteArray()).order(ByteOrder.LITTLE_ENDIAN);
        if (file.limit() < 44 || !tag(file, 0, "RIFF") || !tag(file, 8, "WAVE")) throw new IOException("不是 RIFF/WAVE 文件");
        int rate = 0;
        short[] samples = null;
        for (int cursor = 12; cursor + 8 <= file.limit(); ) {
            long length = file.getInt(cursor + 4) & 0xffffffffL;
            int start = cursor + 8;
            if (length > file.limit() - start) throw new IOException("WAV 数据块损坏");
            if (tag(file, cursor, "fmt ")) {
                if (length < 16 || file.getShort(start) != 1 || file.getShort(start + 2) != 1 || file.getShort(start + 14) != 16)
                    throw new IOException("需要单声道 PCM16 WAV");
                rate = file.getInt(start + 4);
            } else if (tag(file, cursor, "data")) {
                samples = new short[(int) (length / 2)];
                for (int i = 0; i < samples.length; i++) samples[i] = file.getShort(start + 2 * i);
            }
            cursor = start + (int) length + (int) (length & 1);
        }
        if (rate <= 0 || samples == null || samples.length == 0) throw new IOException("WAV 缺少 fmt 或 data");
        return rate == sampleRate ? samples : resample(samples, rate, sampleRate);
    }

    private static short[] resample(short[] samples, int from, int to) {
        short[] out = new short[(int) ((long) samples.length * to / from)];
        for (int i = 0; i < out.length; i++) {
            double at = (double) i * from / to;
            int lo = Math.min((int) at, samples.length - 1);
            int hi = Math.min(lo + 1, samples.length - 1);
            double frac = at - lo;
            out[i] = (short) Math.round(samples[lo] * (1 - frac) + samples[hi] * frac);
        }
        return out;
    }

    private static boolean tag(ByteBuffer file, int offset, String name) {
        return new String(file.array(), offset, 4, StandardCharsets.US_ASCII).equals(name);
    }

    /** Streams samples to disk; the header is patched on close so a crash leaves at most a short file. */
    public static final class Writer implements Closeable {
        private final RandomAccessFile file;
        private final int sampleRate;
        private long samples;
        private byte[] scratch = new byte[0];

        public Writer(File path, int sampleRate) throws IOException {
            this.sampleRate = sampleRate;
            file = new RandomAccessFile(path, "rw");
            file.setLength(0);
            writeHeader();
        }

        public void write(short[] pcm) throws IOException {
            if (scratch.length != pcm.length * 2) scratch = new byte[pcm.length * 2];
            for (int i = 0; i < pcm.length; i++) {
                scratch[2 * i] = (byte) pcm[i];
                scratch[2 * i + 1] = (byte) (pcm[i] >>> 8);
            }
            file.write(scratch);
            samples += pcm.length;
        }

        public long samples() { return samples; }

        @Override
        public void close() throws IOException {
            try {
                file.seek(0);
                writeHeader();
            } finally {
                file.close();
            }
        }

        private void writeHeader() throws IOException {
            long bytes = Math.min(samples * 2, 0xffffffffL - 36);
            ByteBuffer header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN);
            header.put("RIFF".getBytes(StandardCharsets.US_ASCII)).putInt((int) (36 + bytes))
                    .put("WAVEfmt ".getBytes(StandardCharsets.US_ASCII)).putInt(16).putShort((short) 1).putShort((short) 1)
                    .putInt(sampleRate).putInt(sampleRate * 2).putShort((short) 2).putShort((short) 16)
                    .put("data".getBytes(StandardCharsets.US_ASCII)).putInt((int) bytes);
            file.write(header.array());
        }
    }
}
