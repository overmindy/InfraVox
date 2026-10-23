package com.overmindy.duplex.audio;

import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.media.audiofx.AcousticEchoCanceler;
import android.os.Build;

import com.overmindy.duplex.engine.DuplexEngine;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Streaming duplex conversation: a capture thread pushes 80 ms frames (microphone or a WAV paced in
 * real time) into a queue, and the engine thread steps the model per frame and plays every output
 * frame immediately. When the model is slower than real time the queue grows and latency with it.
 */
public final class ConversationSession {
    public interface Listener {
        void onStats(Stats stats);
        void onText(String delta);
        void onFinished(Throwable error, Stats stats);
    }

    public static final class Config {
        public short[] fileInput;  // null: microphone
        public boolean highPerformance = true;
        public float textTemperature = 0.7f;
        public boolean echoCancel = true;
        public File directory;
    }

    public static final class Stats {
        public int processed, emitted, backlog;
        public double lastMs, averageMs, latencySeconds, inputSeconds, inputLevel, outputLevel;
        public boolean draining;

        Stats copy() {
            Stats s = new Stats();
            s.processed = processed; s.emitted = emitted; s.backlog = backlog;
            s.lastMs = lastMs; s.averageMs = averageMs; s.latencySeconds = latencySeconds;
            s.inputSeconds = inputSeconds; s.inputLevel = inputLevel; s.outputLevel = outputLevel;
            s.draining = draining;
            return s;
        }

        public double realTimeFactor(double frameMs) { return averageMs / frameMs; }
    }

    private static final class Captured {
        final short[] pcm;
        final long nanos;
        Captured(short[] pcm, long nanos) { this.pcm = pcm; this.nanos = nanos; }
    }

    private final DuplexEngine engine;
    private final Config config;
    private final Listener listener;
    private final int frame, rate;
    private final LinkedBlockingQueue<Captured> queue = new LinkedBlockingQueue<>();
    private final Stats stats = new Stats();
    private volatile boolean capturing = true, captureDone, aborted;
    private volatile double inputLevel;
    private volatile Throwable captureError;

    public ConversationSession(DuplexEngine engine, Config config, Listener listener) {
        this.engine = engine;
        this.config = config;
        this.listener = listener;
        frame = engine.frameSamples();
        rate = engine.sampleRate();
    }

    public File inputWav() { return new File(config.directory, "input.wav"); }
    public File outputWav() { return new File(config.directory, "output.wav"); }

    public void start(ExecutorService engineThread) {
        config.directory.mkdirs();
        Thread capture = new Thread(this::capture, "duplex-capture");
        capture.setPriority(Thread.MAX_PRIORITY);
        capture.start();
        engineThread.execute(this::run);
    }

    /** Stops listening; frames already captured are still answered. */
    public void stopInput() { capturing = false; }

    public void abort() { aborted = true; capturing = false; }

    public boolean isDraining() { return !capturing && !aborted; }

    private void capture() {
        try {
            if (config.fileInput != null) captureFile(config.fileInput);
            else captureMicrophone();
        } catch (Throwable error) {
            captureError = error;
            aborted = true;
        } finally {
            capturing = false;
            captureDone = true;
        }
    }

    private void captureFile(short[] samples) throws InterruptedException {
        long start = System.nanoTime();
        for (int index = 0; capturing && (index + 1) * frame <= samples.length; index++) {
            long due = start + index * 1_000_000_000L * frame / rate;
            long wait = due - System.nanoTime();
            if (wait > 0) Thread.sleep(wait / 1_000_000L, (int) (wait % 1_000_000L));
            short[] pcm = new short[frame];
            System.arraycopy(samples, index * frame, pcm, 0, frame);
            inputLevel = level(pcm);
            queue.add(new Captured(pcm, System.nanoTime()));
        }
    }

    private void captureMicrophone() throws IOException {
        AudioRecord record = openRecorder(config.echoCancel);
        AcousticEchoCanceler canceler = null;
        try {
            if (config.echoCancel && AcousticEchoCanceler.isAvailable()) {
                canceler = AcousticEchoCanceler.create(record.getAudioSessionId());
                if (canceler != null) canceler.setEnabled(true);
            }
            record.startRecording();
            short[] pcm = new short[frame];
            while (capturing) {
                int filled = 0;
                while (filled < frame && capturing) {
                    int count = record.read(pcm, filled, frame - filled, AudioRecord.READ_BLOCKING);
                    if (count < 0) throw new IOException("录音读取失败：" + count);
                    filled += count;
                }
                if (filled < frame) break;
                inputLevel = level(pcm);
                queue.add(new Captured(pcm.clone(), System.nanoTime()));
            }
        } finally {
            try { record.stop(); } catch (IllegalStateException ignored) { }
            record.release();
            if (canceler != null) canceler.release();
        }
    }

    private AudioRecord openRecorder(boolean echoCancel) throws IOException {
        int minimum = AudioRecord.getMinBufferSize(rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        if (minimum <= 0) throw new IOException("本机不支持 " + rate / 1000 + " kHz 单声道录音");
        int[] sources = echoCancel
                ? new int[]{MediaRecorder.AudioSource.VOICE_COMMUNICATION, MediaRecorder.AudioSource.MIC}
                : new int[]{MediaRecorder.AudioSource.MIC};
        for (int source : sources) {
            AudioRecord record = new AudioRecord(source, rate, AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT, Math.max(minimum, frame * 8));
            if (record.getState() == AudioRecord.STATE_INITIALIZED) return record;
            record.release();
        }
        throw new IOException("麦克风初始化失败");
    }

    private void run() {
        Throwable failure = null;
        AudioTrack track = null;
        Wav.Writer input = null, output = null;
        Writer tokens = null;
        StringBuilder transcript = new StringBuilder();
        try {
            engine.begin(config.highPerformance, config.textTemperature);
            input = new Wav.Writer(inputWav(), rate);
            output = new Wav.Writer(outputWav(), rate);
            tokens = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(new File(config.directory, "tokens.csv")), StandardCharsets.UTF_8));
            tokens.write("frame,text,codes\n");
            track = openTrack();
            final double frameMs = 1000.0 * frame / rate;
            while (!aborted) {
                Captured captured = queue.poll(40, TimeUnit.MILLISECONDS);
                if (captured == null) {
                    if (captureDone) break;
                    publish(frameMs);
                    continue;
                }
                input.write(captured.pcm);
                long start = System.nanoTime();
                DuplexEngine.Frame result = engine.step(captured.pcm);
                double ms = (System.nanoTime() - start) / 1e6;
                stats.processed++;
                stats.lastMs = ms;
                stats.averageMs = stats.processed == 1 ? ms : stats.averageMs * 0.9 + ms * 0.1;
                stats.latencySeconds = (System.nanoTime() - captured.nanos) / 1e9 + frameMs / 1000;
                if (result != null) {
                    stats.emitted++;
                    writeFully(track, result.pcm);
                    output.write(result.pcm);
                    stats.outputLevel = level(result.pcm);
                    tokens.write(stats.emitted + "," + csv(result.text) + "," + join(result.codes) + "\n");
                    if (!result.text.isEmpty()) {
                        transcript.append(result.text);
                        listener.onText(result.text);
                    }
                }
                publish(frameMs);
            }
            if (captureError != null) throw captureError;
        } catch (Throwable error) {
            failure = error;
            abort();
        } finally {
            try {
                engine.end(new File(config.directory, "performance.csv"));
            } catch (Throwable error) {
                if (failure == null) failure = error;
            }
            for (java.io.Closeable closeable : new java.io.Closeable[]{input, output, tokens}) {
                try { if (closeable != null) closeable.close(); } catch (IOException ignored) { }
            }
            writeText(new File(config.directory, "transcript.txt"), transcript.toString().trim());
            if (track != null) finishTrack(track, output == null ? 0 : output.samples(), aborted);
            stats.backlog = queue.size();
            queue.clear();
            listener.onFinished(failure, stats.copy());
        }
    }

    private void publish(double frameMs) {
        stats.backlog = queue.size();
        stats.inputSeconds = stats.processed * frameMs / 1000;
        stats.inputLevel = inputLevel;
        stats.draining = isDraining();
        if (stats.emitted == 0) stats.outputLevel = 0;
        listener.onStats(stats.copy());
        stats.outputLevel *= 0.6;
    }

    private AudioTrack openTrack() {
        int minimum = AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT);
        AudioTrack track = new AudioTrack.Builder()
                .setAudioAttributes(new AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                .setAudioFormat(new AudioFormat.Builder().setSampleRate(rate)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT).build())
                .setBufferSizeInBytes(Math.max(minimum, rate * 2 * 2))
                .setTransferMode(AudioTrack.MODE_STREAM).build();
        // The default start threshold is the whole buffer; start playing after one frame instead.
        if (Build.VERSION.SDK_INT >= 31) track.setStartThresholdInFrames(frame);
        track.play();
        return track;
    }

    private static void writeFully(AudioTrack track, short[] pcm) throws IOException {
        int offset = 0;
        while (offset < pcm.length) {
            int count = track.write(pcm, offset, pcm.length - offset, AudioTrack.WRITE_BLOCKING);
            if (count <= 0) throw new IOException("音频播放写入失败：" + count);
            offset += count;
        }
    }

    private static void finishTrack(AudioTrack track, long samples, boolean immediately) {
        long deadline = System.currentTimeMillis() + 3000;
        while (!immediately && (track.getPlaybackHeadPosition() & 0xffffffffL) < samples
                && System.currentTimeMillis() < deadline) {
            try { Thread.sleep(20); } catch (InterruptedException error) { break; }
        }
        try { track.stop(); } catch (IllegalStateException ignored) { }
        track.release();
    }

    private static double level(short[] pcm) {
        double sum = 0;
        for (short sample : pcm) sum += (double) sample * sample;
        double rms = Math.sqrt(sum / Math.max(1, pcm.length)) / 32768.0;
        return Math.min(1.0, Math.sqrt(rms) * 1.6);
    }

    private static String join(int[] values) {
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < values.length; i++) out.append(i == 0 ? "" : " ").append(values[i]);
        return out.toString();
    }

    private static String csv(String text) {
        return "\"" + text.replace("\"", "\"\"") + "\"";
    }

    private static void writeText(File file, String text) {
        try (Writer writer = new OutputStreamWriter(new FileOutputStream(file), StandardCharsets.UTF_8)) {
            writer.write(text);
        } catch (IOException ignored) {
        }
    }
}
