package com.overmindy.mimicodecdemo;

import android.Manifest;
import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Typeface;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.text.InputType;
import android.view.Gravity;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Switch;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final int PICK_AUDIO = 1;
    private static final int REQUEST_MICROPHONE = 2;
    private static final int SAMPLE_RATE = 24_000;
    private static final int FRAME_SAMPLES = 1920;
    private static final int BACKGROUND = Color.rgb(10, 17, 29);
    private static final int CARD = Color.rgb(26, 39, 57);
    private static final int ACCENT = Color.rgb(68, 222, 194);
    private static final int MUTED = Color.rgb(176, 193, 210);
    // Context-only bundle: Mimi + frontend/head/8 DepFormer + 4 Temporal parts, listed in SHA256SUMS.
    private static final String BUNDLE = "lm-v2-w4w8vd";
    private static final String[] BUNDLE_FILES = {
        "mimi_encoder.bin", "mimi_decoder.bin", "dlc/frontend.bin", "dlc/head.bin",
        "dlc/depformer_codebook_0.bin", "dlc/depformer_codebook_1.bin", "dlc/depformer_codebook_2.bin",
        "dlc/depformer_codebook_3.bin", "dlc/depformer_codebook_4.bin", "dlc/depformer_codebook_5.bin",
        "dlc/depformer_codebook_6.bin", "dlc/depformer_codebook_7.bin",
        "dlc/temporal_part_0.bin", "dlc/temporal_part_1.bin", "dlc/temporal_part_2.bin", "dlc/temporal_part_3.bin",
    };

    private TextView status, modelStatus, audioStatus, tokens, replyText;
    private String[] textVocabulary; // SentencePiece pieces by id; loaded on the worker thread.
    private Button verifyButton, loadButton, recordButton, runButton, originalButton, outputButton, exportButton;
    private Switch moshiMode, streamPlayback, htpPerformance;
    private EditText frameLimit;
    private AudioTrack streamTrack; // Worker thread owns creation/write/release.
    private int streamedSamples, streamCallbacks, streamChunks, streamThresholdBefore, streamThresholdAfter;
    private MediaPlayer player;
    private boolean working, bundleVerified;
    private volatile boolean recording;
    private volatile long sessionHandle;
    private final ExecutorService worker = Executors.newSingleThreadExecutor();

    private static native long nativePrepare(String libraryDir, String encoder, String decoder, String lmDirectory);
    private static native int nativeRunPrepared(long handle, String inputWav, String outputWav, int frames,
                                                MainActivity receiver, boolean highPerformance);
    private static native void nativeRelease(long handle);

    static {
        System.loadLibrary("mimi_codec");
    }

    @Override
    public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().setStatusBarColor(BACKGROUND);
        getWindow().setNavigationBarColor(BACKGROUND);

        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setBackgroundColor(BACKGROUND);
        LinearLayout column = new LinearLayout(this);
        column.setOrientation(LinearLayout.VERTICAL);
        column.setPadding(dp(20), dp(26), dp(20), dp(26));
        scroll.addView(column);
        setContentView(scroll);

        label(column, "MOSHI / MIMI  •  QNN", 13, ACCENT, true);
        label(column, "听见编码的过程", 28, Color.WHITE, true);
        label(column, "WAV → Encoder → Moshi LM → Decoder → WAV（或 codec 对照）", 15, MUTED, false);

        LinearLayout steps = card(column);
        label(steps, "① 校验并加载模型", 19, Color.WHITE, true);
        label(steps, "模型包（ADB 放入，App 只读取并校验 SHA256SUMS）：\n" + bundleDirectory().getAbsolutePath(), 13, MUTED, false);
        verifyButton = button(steps, "校验模型包", () -> verifyBundle(true));
        moshiMode = new Switch(this);
        moshiMode.setText("启用 Moshi LM（关闭则只跑 Mimi codec 对照）");
        moshiMode.setTextColor(Color.WHITE);
        moshiMode.setChecked(true);
        steps.addView(moshiMode);
        moshiMode.setOnCheckedChangeListener((button, enabled) -> {
            releaseSession();
            modelStatus.setText("模式已更换，请重新加载");
            outputFile().delete(); codesFile().delete();
            refresh();
        });
        loadButton = button(steps, "加载模型", this::loadModels);
        modelStatus = label(steps, "正在检查模型包…", 14, MUTED, false);

        LinearLayout input = card(column);
        label(input, "② 准备音频", 19, Color.WHITE, true);
        label(input, "录音直接保存为 24 kHz 单声道 PCM16 WAV；点击停止后再推理。也可选已有 WAV。", 14, MUTED, false);
        recordButton = button(input, "● 开始录音", this::toggleRecording);
        button(input, "选择 24 kHz 单声道 WAV", this::pickAudio);
        button(input, "使用内置演示音频", this::useDemoAudio);
        audioStatus = label(input, inputFile().isFile() ? "已有输入音频，可直接推理" : "等待模型就绪", 14, MUTED, false);

        LinearLayout result = card(column);
        label(result, "③ 手机本地推理", 19, Color.WHITE, true);
        label(result, "帧数填 0 跑整段；每帧 80 ms。Temporal 为 4 段 W4/W8 量化，KV 窗口 1000 帧（约 80 秒）。",
              14, MUTED, false);
        frameLimit = new EditText(this);
        frameLimit.setInputType(InputType.TYPE_CLASS_NUMBER);
        frameLimit.setTextColor(Color.WHITE);
        frameLimit.setText("0");
        result.addView(frameLimit);
        streamPlayback = toggle(result, "每生成 80 ms 音频立即播放（仍可能有计算间隔）", false);
        htpPerformance = toggle(result, "推理期间请求 HTP burst 性能档（更耗电，系统温控仍有效）", true);
        runButton = button(result, "运行", this::runInference);
        status = label(result, "加载模型并准备音频后可推理", 14, MUTED, false);
        label(result, "回复文本", 15, Color.WHITE, true);
        replyText = label(result, "运行后显示 Moshi 生成的文本", 18, ACCENT, false);
        originalButton = button(result, "▶ 播放原音", () -> play(inputFile()));
        outputButton = button(result, "▶ 播放输出音频", () -> play(outputFile()));
        exportButton = button(result, "保存音频、tokens 与性能计时到下载目录", this::exportResults);

        LinearLayout tokenCard = card(column);
        label(tokenCard, "④ 每帧的 codec tokens", 19, Color.WHITE, true);
        label(tokenCard, "codec 模式显示 Encoder tokens；Moshi 模式显示延迟对齐后的生成音频 tokens。LM 采用 greedy，预热一帧，不做末尾补帧。",
              14, MUTED, false);
        tokens = label(tokenCard, "运行后显示编码结果", 13, ACCENT, false);
        tokens.setTypeface(Typeface.MONOSPACE);

        refresh();
        scroll.setFocusableInTouchMode(true);
        scroll.requestFocus();
        scroll.post(() -> scroll.scrollTo(0, 0));
        verifyBundle(false);
    }

    private File bundleDirectory() {
        File root = getExternalFilesDir("models");
        if (root == null) root = new File(getFilesDir(), "models");
        return new File(root, BUNDLE);
    }
    private File inputFile() { return new File(getFilesDir(), "input.wav"); }
    private File outputFile() { return new File(getFilesDir(), "reconstructed.wav"); }
    private File codesFile() { return new File(getFilesDir(), "reconstructed.wav.codes.txt"); }
    private File textTokensFile() { return new File(getFilesDir(), "reconstructed.wav.text_tokens.txt"); }

    private void releaseSession() {
        long previous = sessionHandle;
        sessionHandle = 0;
        if (previous != 0) worker.execute(() -> nativeRelease(previous));
    }

    // Hashes every bundle file against SHA256SUMS. A full pass is remembered by file size and
    // modification time, so unchanged bundles skip re-hashing 6 GB unless `force` is set.
    private void verifyBundle(boolean force) {
        if (working || recording) return;
        releaseSession();
        working = true; bundleVerified = false; refresh();
        worker.execute(() -> {
            try {
                File bundle = bundleDirectory();
                Map<String, String> expected = readChecksums(new File(bundle, "SHA256SUMS"));
                StringBuilder fingerprint = new StringBuilder();
                for (String name : BUNDLE_FILES) {
                    File file = new File(bundle, name);
                    if (!file.isFile() || file.length() == 0) throw new IOException("缺少模型文件：" + name);
                    if (!expected.containsKey(name)) throw new IOException("SHA256SUMS 未列出：" + name);
                    fingerprint.append(name).append(':').append(file.length()).append(':').append(file.lastModified()).append(';');
                }
                SharedPreferences preferences = getPreferences(MODE_PRIVATE);
                boolean cached = !force && fingerprint.toString().equals(preferences.getString("verified_bundle", ""));
                if (!cached) {
                    for (int i = 0; i < BUNDLE_FILES.length; i++) {
                        String name = BUNDLE_FILES[i];
                        final String progress = "校验 " + (i + 1) + "/" + BUNDLE_FILES.length + "：" + name;
                        runOnUiThread(() -> modelStatus.setText(progress));
                        if (!sha256(new File(bundle, name)).equals(expected.get(name)))
                            throw new IOException("SHA256 不匹配：" + name);
                    }
                    preferences.edit().putString("verified_bundle", fingerprint.toString()).apply();
                }
                runOnUiThread(() -> {
                    bundleVerified = true;
                    modelStatus.setText((cached ? "模型包未变化，沿用上次校验：\n" : "16 个模型文件 SHA256 校验通过：\n")
                        + bundle.getAbsolutePath() + "\n点击加载");
                });
            } catch (Exception error) {
                runOnUiThread(() -> modelStatus.setText("模型包不可用：" + error.getMessage()
                    + "\n请把 " + BUNDLE + " 放到 " + bundleDirectory().getParent()));
            } finally {
                runOnUiThread(() -> { working = false; refresh(); });
            }
        });
    }

    private static Map<String, String> readChecksums(File file) throws IOException {
        Map<String, String> sums = new LinkedHashMap<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] fields = line.trim().split("\\s+", 2);
                if (fields.length == 2) sums.put(fields[1].replaceFirst("^\\*", ""), fields[0].toLowerCase(Locale.US));
            }
        }
        return sums;
    }

    private static String sha256(File file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream input = new FileInputStream(file)) {
            byte[] buffer = new byte[1024 * 1024];
            int count;
            while ((count = input.read(buffer)) != -1) digest.update(buffer, 0, count);
        }
        StringBuilder hex = new StringBuilder();
        for (byte b : digest.digest()) hex.append(String.format(Locale.US, "%02x", b & 255));
        return hex.toString();
    }

    private void loadModels() {
        if (working || recording || !bundleVerified) return;
        working = true;
        final boolean useLm = moshiMode.isChecked();
        modelStatus.setText(useLm ? "正在加载 Mimi + Moshi LM（全部 context 常驻）…" : "正在加载 Mimi…");
        refresh();
        worker.execute(() -> {
            long start = System.nanoTime();
            try {
                long previous = sessionHandle;
                sessionHandle = 0;
                if (previous != 0) nativeRelease(previous);
                File bundle = bundleDirectory();
                sessionHandle = nativePrepare(getApplicationInfo().nativeLibraryDir,
                        new File(bundle, "mimi_encoder.bin").getAbsolutePath(),
                        new File(bundle, "mimi_decoder.bin").getAbsolutePath(),
                        useLm ? bundle.getAbsolutePath() : "");
                double seconds = (System.nanoTime() - start) / 1e9;
                runOnUiThread(() -> modelStatus.setText(String.format(Locale.US,
                        "%s已就绪；加载用时 %.1f 秒", useLm ? "Mimi + Moshi LM " : "Mimi ", seconds)));
            } catch (Exception error) {
                runOnUiThread(() -> modelStatus.setText("加载失败：" + error.getMessage()));
            } finally {
                runOnUiThread(() -> { working = false; refresh(); });
            }
        });
    }

    private void pickAudio() {
        if (working || recording) return;
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("audio/*");
        startActivityForResult(intent, PICK_AUDIO);
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request != PICK_AUDIO || result != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        copyAudio(() -> getContentResolver().openInputStream(uri), "音频已导入");
    }

    private interface Source { InputStream open() throws IOException; }

    private void useDemoAudio() {
        copyAudio(() -> getAssets().open("demo.wav"), "演示音频已导入");
    }

    private void copyAudio(Source source, String success) {
        if (working || recording) return;
        stopPlayer();
        outputFile().delete();
        codesFile().delete();
        tokens.setText("输入已更换；请重新运行");
        working = true;
        refresh();
        worker.execute(() -> {
            File target = inputFile();
            try (InputStream input = source.open(); OutputStream output = new FileOutputStream(target)) {
                if (input == null) throw new IOException("无法读取音频");
                byte[] buffer = new byte[1024 * 1024];
                int count;
                while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
                if (target.length() == 0) throw new IOException("文件为空");
                runOnUiThread(() -> { audioStatus.setText(success); status.setText(success); });
            } catch (Exception error) {
                target.delete();
                runOnUiThread(() -> status.setText("导入失败：" + error.getMessage()));
            } finally {
                runOnUiThread(() -> { working = false; refresh(); });
            }
        });
    }

    private void toggleRecording() {
        if (recording) {
            recording = false;
            recordButton.setText("正在结束录音…");
            audioStatus.setText("正在保存完整的 80 ms 音频帧…");
            return;
        }
        if (working || sessionHandle == 0) return;
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_MICROPHONE);
            return;
        }
        beginRecording();
    }

    @Override
    public void onRequestPermissionsResult(int request, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(request, permissions, results);
        if (request != REQUEST_MICROPHONE) return;
        if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) beginRecording();
        else audioStatus.setText("需要麦克风权限才能录音；也可选择现有 WAV");
    }

    private void beginRecording() {
        if (working || recording || sessionHandle == 0) return;
        stopPlayer();
        outputFile().delete();
        codesFile().delete();
        tokens.setText("录音后点击运行查看 tokens");
        recording = true;
        audioStatus.setText("正在录音；点击停止后生成 24 kHz PCM16 WAV");
        refresh();
        worker.execute(() -> {
            File temporary = new File(getFilesDir(), "recording.tmp.wav");
            AudioRecord microphone = null;
            try {
                int minimumBytes = AudioRecord.getMinBufferSize(SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
                if (minimumBytes <= 0) throw new IOException("本机不支持 24 kHz 单声道录音");
                microphone = new AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                        Math.max(minimumBytes, FRAME_SAMPLES * 4));
                if (microphone.getState() != AudioRecord.STATE_INITIALIZED) throw new IOException("麦克风初始化失败");
                short[] pcm = new short[FRAME_SAMPLES];
                byte[] packed = new byte[FRAME_SAMPLES * 2];
                long sampleCount = 0;
                try (RandomAccessFile wav = new RandomAccessFile(temporary, "rw")) {
                    wav.setLength(0);
                    wav.write(new byte[44]);
                    microphone.startRecording();
                    while (recording) {
                        int count = microphone.read(pcm, 0, pcm.length, AudioRecord.READ_BLOCKING);
                        if (count < 0) throw new IOException("录音读取失败：" + count);
                        for (int i = 0; i < count; i++) {
                            packed[2 * i] = (byte) pcm[i];
                            packed[2 * i + 1] = (byte) (pcm[i] >>> 8);
                        }
                        wav.write(packed, 0, count * 2);
                        sampleCount += count;
                        if (sampleCount > (0xffffffffL - 36) / 2) throw new IOException("录音超过 WAV 格式容量");
                    }
                    long completeSamples = sampleCount - sampleCount % FRAME_SAMPLES;
                    if (completeSamples == 0) throw new IOException("请至少录制 80 ms");
                    wav.setLength(44 + completeSamples * 2);
                    wav.seek(0);
                    writeWavHeader(wav, completeSamples);
                    sampleCount = completeSamples;
                }
                inputFile().delete();
                if (!temporary.renameTo(inputFile())) throw new IOException("录音文件保存失败");
                final long savedSamples = sampleCount;
                runOnUiThread(() -> audioStatus.setText(String.format(Locale.US, "录音完成：%.2f 秒 / %d 帧；点击运行",
                        savedSamples / (double) SAMPLE_RATE, savedSamples / FRAME_SAMPLES)));
            } catch (Exception error) {
                temporary.delete();
                runOnUiThread(() -> audioStatus.setText("录音失败：" + error.getMessage()));
            } finally {
                if (microphone != null) {
                    try { microphone.stop(); } catch (IllegalStateException ignored) { }
                    microphone.release();
                }
                recording = false;
                runOnUiThread(this::refresh);
            }
        });
    }

    private static void writeLittle16(RandomAccessFile wav, int value) throws IOException {
        wav.write(value & 255);
        wav.write((value >>> 8) & 255);
    }

    private static void writeLittle32(RandomAccessFile wav, long value) throws IOException {
        writeLittle16(wav, (int) value);
        writeLittle16(wav, (int) (value >>> 16));
    }

    private static void writeWavHeader(RandomAccessFile wav, long samples) throws IOException {
        long bytes = samples * 2;
        wav.write("RIFF".getBytes(StandardCharsets.US_ASCII));
        writeLittle32(wav, 36 + bytes);
        wav.write("WAVEfmt ".getBytes(StandardCharsets.US_ASCII));
        writeLittle32(wav, 16);
        writeLittle16(wav, 1);
        writeLittle16(wav, 1);
        writeLittle32(wav, SAMPLE_RATE);
        writeLittle32(wav, SAMPLE_RATE * 2);
        writeLittle16(wav, 2);
        writeLittle16(wav, 16);
        wav.write("data".getBytes(StandardCharsets.US_ASCII));
        writeLittle32(wav, bytes);
    }

    private void runInference() {
        if (working || recording || sessionHandle == 0 || !inputFile().isFile()) return;
        final int requestedFrames;
        try {
            requestedFrames = Integer.parseInt(frameLimit.getText().toString().trim());
            if (requestedFrames < 0 || (moshiMode.isChecked() && requestedFrames == 1) || requestedFrames > 3000)
                throw new NumberFormatException();
        } catch (NumberFormatException error) {
            status.setText("帧数请填 0（整段）或 2–3000");
            return;
        }
        stopPlayer();
        outputFile().delete();
        codesFile().delete();
        textTokensFile().delete();
        new File(outputFile().getAbsolutePath() + ".playback.csv").delete();
        tokens.setText("正在执行 QNN 图…");
        replyText.setText(moshiMode.isChecked() ? "生成中…" : "codec 对照模式没有文本输出");
        status.setText(moshiMode.isChecked() ? "正在逐帧执行 Encoder → LM → Decoder…" : "正在逐帧执行 Encoder → Decoder…");
        working = true;
        refresh();
        final boolean enableStreamPlayback = moshiMode.isChecked() && streamPlayback.isChecked();
        final boolean enableHtpPerformance = htpPerformance.isChecked();
        worker.execute(() -> {
            long start = System.nanoTime();
            try {
                stopStream();
                streamedSamples = 0;
                streamCallbacks = 0;
                streamChunks = 0;
                if (enableStreamPlayback) startStream();
                int frames = nativeRunPrepared(sessionHandle, inputFile().getAbsolutePath(),
                        outputFile().getAbsolutePath(), requestedFrames, this, enableHtpPerformance);
                final String playbackResult = finishStreamPlayback();
                String codeText = readCodes();
                final String reply = moshiMode.isChecked() && textTokensFile().isFile() ? decodeReply() : null;
                double seconds = (System.nanoTime() - start) / 1e9;
                runOnUiThread(() -> {
                    if (reply != null) replyText.setText(reply.isEmpty() ? "（本段没有生成文字，只有停顿）" : reply);
                    status.setText(String.format(Locale.US,
                            "推理完成：%d 帧 / %.2f 秒输出音频；本次运行 %.2f 秒，RTF %.2f（小于 1 才达到实时）。详细计时可导出。",
                            frames, frames * 0.08, seconds, seconds / Math.max(frames * 0.08, 1e-9)) + playbackResult);
                    tokens.setText(codeText);
                });
            } catch (Exception error) {
                stopStream();
                runOnUiThread(() -> {
                    status.setText("运行失败：" + error.getMessage());
                    tokens.setText("没有生成有效 tokens");
                });
            } finally {
                runOnUiThread(() -> { working = false; refresh(); });
            }
        });
    }

    private void startStream() {
        int minimum = AudioTrack.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT);
        streamTrack = new AudioTrack.Builder()
                .setAudioAttributes(new AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                .setAudioFormat(new AudioFormat.Builder().setSampleRate(SAMPLE_RATE)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT).build())
                .setBufferSizeInBytes(Math.max(minimum, FRAME_SAMPLES * 2 * 4))
                .setTransferMode(AudioTrack.MODE_STREAM).build();
        // The default start threshold is the whole buffer; start after one 80 ms frame instead.
        if (android.os.Build.VERSION.SDK_INT >= 31) {
            streamThresholdBefore = streamTrack.getStartThresholdInFrames();
            streamThresholdAfter = streamTrack.setStartThresholdInFrames(FRAME_SAMPLES);
        } else {
            streamThresholdBefore = streamTrack.getBufferSizeInFrames();
            streamTrack.setBufferSizeInFrames(FRAME_SAMPLES);
            streamThresholdAfter = streamTrack.getBufferSizeInFrames();
        }
        streamTrack.play();
    }

    // Called on the inference worker once per input frame, not the UI thread.
    private void onNativeFrame(int consumed, int emitted, double milliseconds, short[] pcm) {
        ++streamCallbacks;
        if (streamTrack != null && pcm.length > 0) {
            int offset = 0;
            while (offset < pcm.length) {
                int count = streamTrack.write(pcm, offset, pcm.length - offset, AudioTrack.WRITE_BLOCKING);
                if (count <= 0) throw new IllegalStateException("流式音频写入失败：" + count);
                offset += count;
            }
            streamedSamples += pcm.length;
            ++streamChunks;
        }
        runOnUiThread(() -> status.setText(String.format(Locale.US,
                "已处理输入 %d 帧，输出 %d 帧；刚完成一帧 %.1f ms", consumed, emitted, milliseconds)));
    }

    private void stopStream() {
        if (streamTrack != null) { streamTrack.release(); streamTrack = null; }
    }

    private String finishStreamPlayback() throws IOException, InterruptedException {
        if (streamTrack == null) return "";
        long deadline = android.os.SystemClock.uptimeMillis() + 2000;
        while (streamTrack.getPlaybackHeadPosition() < streamedSamples
                && android.os.SystemClock.uptimeMillis() < deadline) Thread.sleep(10);
        int played = streamTrack.getPlaybackHeadPosition();
        boolean timedOut = played < streamedSamples;
        String header = "callbacks,pcm_chunks,written_samples,played_samples,capacity_frames,threshold_before,threshold_after,underruns,drain_timeout\n";
        String row = String.format(Locale.US, "%d,%d,%d,%d,%d,%d,%d,%d,%s\n",
                streamCallbacks, streamChunks, streamedSamples, played, streamTrack.getBufferCapacityInFrames(),
                streamThresholdBefore, streamThresholdAfter, streamTrack.getUnderrunCount(), timedOut);
        try (FileOutputStream output = new FileOutputStream(outputFile().getAbsolutePath() + ".playback.csv")) {
            output.write((header + row).getBytes(StandardCharsets.UTF_8));
        }
        stopStream();
        return String.format(Locale.US, " 播放头消费 %d/%d 样本%s。", played, streamedSamples,
                timedOut ? "（等待超时，请检查音频输出）" : "");
    }

    private String readCodes() throws IOException {
        StringBuilder text = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(codesFile()), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(",");
                text.append(String.format(Locale.US, "%02d  ", Integer.parseInt(values[0])));
                for (int i = 1; i < values.length; i++) {
                    if (i > 1) text.append(' ');
                    text.append(values[i]);
                }
                text.append('\n');
            }
        }
        if (moshiMode.isChecked() && textTokensFile().isFile()) {
            text.append("\n文本 token ID（解码结果见“回复文本”）：\n");
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(textTokensFile()), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) text.append(line).append('\n');
            }
        }
        return text.toString();
    }

    // Decodes the runner's "frame,token" text stream with the SentencePiece vocabulary:
    // ids 0-3 are <unk>/<s>/</s>/<pad>, "<0xHH>" pieces are raw bytes, and "▁" marks a space.
    private String decodeReply() throws IOException {
        if (textVocabulary == null) {
            ArrayList<String> pieces = new ArrayList<>();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(getAssets().open("text_vocab.txt"), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) pieces.add(unescapePiece(line));
            }
            textVocabulary = pieces.toArray(new String[0]);
        }
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(textTokensFile()), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] fields = line.split(",");
                int id = Integer.parseInt(fields[fields.length - 1].trim());
                if (id <= 3 || id >= textVocabulary.length) continue;
                String piece = textVocabulary[id];
                if (piece.length() == 6 && piece.startsWith("<0x") && piece.endsWith(">")) {
                    bytes.write(Integer.parseInt(piece.substring(3, 5), 16));
                } else {
                    byte[] encoded = piece.replace('▁', ' ').getBytes(StandardCharsets.UTF_8);
                    bytes.write(encoded, 0, encoded.length);
                }
            }
        }
        return new String(bytes.toByteArray(), StandardCharsets.UTF_8).trim();
    }

    private static String unescapePiece(String line) {
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

    private void play(File file) {
        if (!file.isFile()) return;
        stopPlayer();
        try {
            player = new MediaPlayer();
            player.setDataSource(file.getAbsolutePath());
            player.setOnCompletionListener(done -> stopPlayer());
            player.prepare();
            player.start();
            status.setText("正在播放：" + file.getName());
        } catch (IOException error) {
            stopPlayer();
            status.setText("播放失败：" + error.getMessage());
        }
    }

    private void stopPlayer() {
        if (player != null) { player.release(); player = null; }
    }

    @Override
    protected void onDestroy() {
        recording = false;
        stopPlayer();
        worker.execute(() -> {
            stopStream();
            long handle = sessionHandle;
            sessionHandle = 0;
            if (handle != 0) nativeRelease(handle);
        });
        worker.shutdown();
        super.onDestroy();
    }

    private void exportResults() {
        if (!outputFile().isFile() || !codesFile().isFile()) return;
        working = true;
        refresh();
        status.setText("正在保存到 Downloads/MimiDemo…");
        worker.execute(() -> {
            try {
                String suffix = Long.toString(System.currentTimeMillis());
                String base = outputFile().getAbsolutePath();
                saveDownload(outputFile(), "moshi-" + suffix + ".wav", "audio/wav");
                saveDownload(codesFile(), "moshi-" + suffix + "-audio-tokens.csv", "text/csv");
                String[][] extras = {
                    {textTokensFile().getAbsolutePath(), "-text-token-ids.csv", "text/csv"},
                    {base + ".performance.csv", "-performance.csv", "text/csv"},
                    {base + ".performance.csv.qnn.csv", "-qnn-profile.csv", "text/csv"},
                    {base + ".runtime.json", "-runtime.json", "application/json"},
                    {base + ".playback.csv", "-playback.csv", "text/csv"},
                };
                for (String[] extra : extras) {
                    File file = new File(extra[0]);
                    if (file.isFile()) saveDownload(file, "moshi-" + suffix + extra[1], extra[2]);
                }
                runOnUiThread(() -> status.setText("已保存到 Downloads/MimiDemo（前缀 moshi-" + suffix + "）"));
            } catch (Exception error) {
                runOnUiThread(() -> status.setText("保存失败：" + error.getMessage()));
            } finally {
                runOnUiThread(() -> { working = false; refresh(); });
            }
        });
    }

    private void saveDownload(File source, String name, String mime) throws IOException {
        ContentValues values = new ContentValues();
        values.put(MediaStore.Downloads.DISPLAY_NAME, name);
        values.put(MediaStore.Downloads.MIME_TYPE, mime);
        values.put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/MimiDemo");
        Uri uri = getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);
        if (uri == null) throw new IOException("无法创建下载文件");
        try (InputStream input = new FileInputStream(source);
             OutputStream output = getContentResolver().openOutputStream(uri)) {
            if (output == null) throw new IOException("无法写入下载文件");
            byte[] buffer = new byte[65536];
            int count;
            while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
        }
    }

    private void refresh() {
        boolean idle = !working && !recording;
        if (idle) getWindow().clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        else getWindow().addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        streamPlayback.setEnabled(idle && moshiMode.isChecked());
        htpPerformance.setEnabled(idle);
        verifyButton.setEnabled(idle);
        moshiMode.setEnabled(idle);
        frameLimit.setEnabled(idle);
        loadButton.setText(moshiMode.isChecked() ? "加载 Mimi + Moshi LM" : "加载 Mimi");
        runButton.setText(moshiMode.isChecked() ? "运行 Encoder → Moshi LM → Decoder" : "运行 Encoder → Decoder 对照");
        loadButton.setEnabled(idle && bundleVerified);
        recordButton.setEnabled(!working && (recording || sessionHandle != 0));
        recordButton.setText(recording ? "■ 停止录音" : "● 开始录音");
        runButton.setEnabled(idle && sessionHandle != 0 && inputFile().isFile());
        originalButton.setEnabled(idle && inputFile().isFile());
        outputButton.setEnabled(idle && outputFile().isFile());
        exportButton.setEnabled(idle && outputFile().isFile() && codesFile().isFile());
    }

    private LinearLayout card(LinearLayout parent) {
        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(dp(16), dp(16), dp(16), dp(16));
        android.graphics.drawable.GradientDrawable background = new android.graphics.drawable.GradientDrawable();
        background.setColor(CARD);
        background.setCornerRadius(dp(18));
        panel.setBackground(background);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, -2);
        params.topMargin = dp(20);
        parent.addView(panel, params);
        return panel;
    }

    private TextView label(LinearLayout parent, String value, int size, int color, boolean bold) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextColor(color);
        view.setTextSize(size);
        view.setPadding(0, dp(4), 0, dp(8));
        if (bold) view.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        parent.addView(view);
        return view;
    }

    private Switch toggle(LinearLayout parent, String title, boolean checked) {
        Switch view = new Switch(this);
        view.setText(title);
        view.setTextColor(Color.WHITE);
        view.setChecked(checked);
        parent.addView(view);
        return view;
    }

    private Button button(LinearLayout parent, String title, Runnable action) {
        Button button = new Button(this);
        button.setText(title);
        button.setAllCaps(false);
        button.setTextColor(BACKGROUND);
        button.setBackgroundTintList(android.content.res.ColorStateList.valueOf(ACCENT));
        button.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, -2);
        params.topMargin = dp(8);
        parent.addView(button, params);
        button.setOnClickListener(view -> action.run());
        return button;
    }

    private int dp(int value) {
        return Math.round(getResources().getDisplayMetrics().density * value);
    }
}
