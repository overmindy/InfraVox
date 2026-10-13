package com.overmindy.mimicodecdemo;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Color;
import android.graphics.Typeface;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.AudioAttributes;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.provider.DocumentsContract;
import android.os.StatFs;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Switch;
import android.widget.EditText;
import android.text.InputType;

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
import java.util.Locale;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.security.MessageDigest;
import org.json.JSONObject;

public final class MainActivity extends Activity {
    private static final int PICK_AUDIO = 1;
    private static final int PICK_ENCODER = 2;
    private static final int PICK_DECODER = 3;
    private static final int REQUEST_MICROPHONE = 4;
    private static final int PICK_LM = 5;
    private static final int SAMPLE_RATE = 24_000;
    private static final int FRAME_SAMPLES = 1920;
    private static final int BACKGROUND = Color.rgb(10, 17, 29);
    private static final int CARD = Color.rgb(26, 39, 57);
    private static final int ACCENT = Color.rgb(68, 222, 194);
    private final File[] files = new File[4];
    private TextView status;
    private TextView modelStatus;
    private TextView audioStatus;
    private TextView tokens;
    private Button loadButton;
    private Button recordButton;
    private Button runButton;
    private Button originalButton;
    private Button outputButton;
    private Button exportButton;
    private Switch moshiMode;
    private Switch streamPlayback;
    private Switch htpPerformance;
    private Switch fastRuntime;
    private AudioTrack streamTrack; // Worker thread owns creation/write/release.
    private int streamedSamples;
    private int streamThresholdBefore, streamThresholdAfter, streamCallbacks, streamChunks;
    private EditText lmFrameLimit;
    private File selectedEncoder, selectedDecoder, selectedLm;
    private File contextCacheDirectory;
    private boolean localLmVerified;
    private Button scanModelsButton;
    private MediaPlayer player;
    private boolean working;
    private volatile boolean recording;
    private volatile long sessionHandle;
    private final ExecutorService worker = Executors.newSingleThreadExecutor();

    private static native long nativePrepare(String libraryDir, String encoderDlc,
                                             String decoderDlc, String lmDirectory, String cacheDirectory, boolean fast);
    private static native int nativeRunPrepared(long handle, String inputWav,
                                                 String outputWav, int frames, MainActivity receiver, boolean highPerformance);
    private static native void nativeRelease(long handle);

    static {
        System.loadLibrary("mimi_codec");
    }

    @Override
    public void onCreate(Bundle state) {
        super.onCreate(state);
        contextCacheDirectory = getExternalFilesDir("qnn-context-v1");
        if (contextCacheDirectory == null) contextCacheDirectory = new File(getFilesDir(), "qnn-context-v1");
        contextCacheDirectory.mkdirs();
        getWindow().setStatusBarColor(BACKGROUND);
        getWindow().setNavigationBarColor(BACKGROUND);
        files[PICK_AUDIO] = new File(getFilesDir(), "input.wav");
        files[PICK_ENCODER] = new File(getFilesDir(), "mimi_encoder.dlc");
        files[PICK_DECODER] = new File(getFilesDir(), "mimi_decoder.dlc");
        defaultModelsDirectory();
        selectedEncoder = rememberedModel("encoder");
        selectedDecoder = rememberedModel("decoder");
        selectedLm = rememberedModel("lm");

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
        label(column, "WAV → Encoder → Moshi LM → Decoder → WAV（或 codec 对照）",
              15, Color.rgb(176, 193, 210), false);

        LinearLayout steps = card(column);
        label(steps, "① 导入并加载模型", 19, Color.WHITE, true);
        label(steps, "两个 DLC 只需导入一次。点击加载，等 QNN 图就绪后再录音或推理。",
              14, Color.rgb(176, 193, 210), false);
        label(steps, "默认模型目录（ADB 直接放入，无需重复导入）：\n"
              + defaultModelsDirectory().getAbsolutePath()
              + "\nMimi：同一子目录放 mimi_encoder.dlc 与 mimi_decoder.dlc。"
              + "\nLM：子目录内放 dlc_inventory.json 与 dlc/。", 13,
              Color.rgb(176, 193, 210), false);
        scanModelsButton = button(steps, "扫描默认目录并选择模型", this::chooseDefaultModels);
        button(steps, "导入 Encoder DLC", () -> pick(PICK_ENCODER));
        button(steps, "导入 Decoder DLC", () -> pick(PICK_DECODER));
        button(steps, "导入完整 LM 模型目录（26 DLC + inventory）", this::pickLm);
        moshiMode = new Switch(this);
        moshiMode.setText("启用 Moshi LM（预算内常驻＋context 缓存，非实时）");
        moshiMode.setTextColor(Color.WHITE);
        steps.addView(moshiMode);
        moshiMode.setOnCheckedChangeListener((button, enabled) -> {
            long previous = sessionHandle;
            sessionHandle = 0;
            if (previous != 0) worker.execute(() -> nativeRelease(previous));
            modelStatus.setText("模式已更换，请重新加载；LM 使用常驻小图与本机 context 缓存");
            outputFile().delete(); codesFile().delete();
            refresh();
        });
        loadButton = button(steps, "加载两个 DLC 到 QNN", this::loadModels);
        modelStatus = label(steps, "等待导入两个 DLC", 14,
                            Color.rgb(176, 193, 210), false);

        LinearLayout input = card(column);
        label(input, "② 准备音频", 19, Color.WHITE, true);
        label(input, "录音直接保存为 24 kHz 单声道 PCM16 WAV；点击停止后再推理。也可选已有 WAV。",
              14, Color.rgb(176, 193, 210), false);
        recordButton = button(input, "● 开始录音", this::toggleRecording);
        button(input, "选择 24 kHz 单声道 WAV", () -> pick(PICK_AUDIO));
        button(input, "使用内置演示音频", this::useDemoAudio);
        audioStatus = label(input, "等待模型就绪", 14,
                            Color.rgb(176, 193, 210), false);

        LinearLayout result = card(column);
        label(result, "③ 手机本地编码与重建", 19, Color.WHITE, true);
        label(result, "先跑 2–4 帧检查。加速版需选择 lm-cache1000-v1，并使用本手机匹配的 context 缓存；首次运行仍要装载权重，重复运行复用。1000 帧窗口的长音频质量尚未验证。",
              14, Color.rgb(176, 193, 210), false);
        lmFrameLimit = new EditText(this);
        lmFrameLimit.setInputType(InputType.TYPE_CLASS_NUMBER);
        lmFrameLimit.setTextColor(Color.WHITE);
        lmFrameLimit.setText("2");
        result.addView(lmFrameLimit);
        streamPlayback = new Switch(this);
        streamPlayback.setText("每生成 80 ms 音频立即播放（仍可能有计算间隔）");
        streamPlayback.setTextColor(Color.WHITE);
        result.addView(streamPlayback);
        htpPerformance = new Switch(this);
        htpPerformance.setText("推理期间请求 HTP 性能档（更耗电，系统温控仍有效）");
        htpPerformance.setTextColor(Color.WHITE);
        htpPerformance.setChecked(true);
        result.addView(htpPerformance);
        fastRuntime = new Switch(this);
        fastRuntime.setText("已验证加速：cache1000＋共享工作区＋8 GiB 常驻预算（需完整本机缓存）");
        fastRuntime.setTextColor(Color.WHITE);
        fastRuntime.setChecked(true);
        result.addView(fastRuntime);
        fastRuntime.setOnCheckedChangeListener((button, enabled) -> {
            releaseSelectedSession();
            modelStatus.setText("运行配置已改变，请重新加载；未关闭系统温控。");
            refresh();
        });
        runButton = button(result, "运行 QNN Encoder → Decoder", this::runCodec);
        status = label(result, "加载模型并准备音频后可推理", 14,
                       Color.rgb(176, 193, 210), false);
        originalButton = button(result, "▶ 播放原音", () -> play(files[PICK_AUDIO]));
        outputButton = button(result, "▶ 播放重建音频", () -> play(outputFile()));
        exportButton = button(result, "保存音频、tokens 与性能计时到下载目录", this::exportResults);

        LinearLayout tokenCard = card(column);
        label(tokenCard, "④ 每帧的 codec tokens", 19, Color.WHITE, true);
        label(tokenCard, "codec 模式显示 Encoder tokens；Moshi 模式显示延迟对齐后的生成音频 tokens。LM 采用 greedy，预热一帧，不做末尾补帧。",
              14, Color.rgb(176, 193, 210), false);
        tokens = label(tokenCard, "运行后显示编码结果", 13, ACCENT, false);
        tokens.setTypeface(Typeface.MONOSPACE);
        if (encoderFile().isFile() && decoderFile().isFile()) {
            modelStatus.setText("已有 Mimi：\n" + encoderFile().getAbsolutePath()
                + "\n" + decoderFile().getAbsolutePath() + "\n点击加载即可");
        }
        if (files[PICK_AUDIO].isFile()) audioStatus.setText("已有输入音频，可直接推理");
        refresh();
        scroll.setFocusableInTouchMode(true);
        scroll.requestFocus();
        scroll.post(() -> scroll.scrollTo(0, 0));
        if (selectedLm != null) verifyLocalLm(selectedLm);
    }

    private File outputFile() { return new File(getFilesDir(), "reconstructed.wav"); }
    private File codesFile() { return new File(getFilesDir(), "reconstructed.wav.codes.txt"); }
    private File textTokensFile() { return new File(getFilesDir(), "reconstructed.wav.text_tokens.txt"); }
    private File lmDirectory() {
        return selectedLm != null ? selectedLm : new File(getFilesDir(), "moshi-lm");
    }
    private File encoderFile() { return selectedEncoder != null ? selectedEncoder : files[PICK_ENCODER]; }
    private File decoderFile() { return selectedDecoder != null ? selectedDecoder : files[PICK_DECODER]; }

    private File defaultModelsDirectory() {
        File root = getExternalFilesDir("models");
        if (root == null) root = new File(getFilesDir(), "models");
        root.mkdirs();
        return root;
    }

    private File rememberedModel(String key) {
        String value = getPreferences(MODE_PRIVATE).getString("model_" + key, "");
        if (value.isEmpty()) return null;
        File path = new File(value);
        try {
            String candidate = path.getCanonicalPath();
            String root = defaultModelsDirectory().getCanonicalPath();
            if ((candidate.startsWith(root + "/") || (key.equals("lm") && candidate.equals(root)))
                    && path.exists()) return path;
        } catch (IOException ignored) { }
        return null;
    }

    private void releaseSelectedSession() {
        long previous = sessionHandle; sessionHandle = 0;
        if (previous != 0) worker.execute(() -> nativeRelease(previous));
    }

    private void chooseDefaultModels() {
        if (working || recording) return;
        File root = defaultModelsDirectory();
        ArrayList<File> folders = new ArrayList<>(); folders.add(root);
        File[] children = root.listFiles(File::isDirectory);
        if (children != null) {
            Arrays.sort(children, Comparator.comparing(File::getName));
            folders.addAll(Arrays.asList(children));
        }
        ArrayList<String> labels = new ArrayList<>();
        ArrayList<Runnable> choices = new ArrayList<>();
        if (files[PICK_ENCODER].isFile() && files[PICK_DECODER].isFile()) {
            labels.add("Mimi：之前导入的私有模型");
            choices.add(() -> selectMimi(null, null));
        }
        File privateLm = new File(getFilesDir(), "moshi-lm");
        if (new File(privateLm, "READY").isFile()) {
            labels.add("LM：之前导入的私有模型");
            choices.add(() -> {
                releaseSelectedSession(); selectedLm = null; localLmVerified = false;
                getPreferences(MODE_PRIVATE).edit().remove("model_lm").apply();
                modelStatus.setText("已选择私有 LM，请重新加载"); refresh();
            });
        }
        for (File folder : folders) {
            File encoder = new File(folder, "mimi_encoder.dlc");
            File decoder = new File(folder, "mimi_decoder.dlc");
            if (encoder.isFile() && encoder.length() > 0 && decoder.isFile() && decoder.length() > 0) {
                labels.add("Mimi：" + folder.getName());
                choices.add(() -> selectMimi(encoder, decoder));
            }
            if (new File(folder, "dlc_inventory.json").isFile() && new File(folder, "dlc").isDirectory()) {
                labels.add("LM：" + folder.getName() + "（校验后直接使用）");
                choices.add(() -> verifyLocalLm(folder));
            }
        }
        if (labels.isEmpty()) {
            status.setText("没有发现模型。请放入：\n" + root.getAbsolutePath()); return;
        }
        new AlertDialog.Builder(this).setTitle("选择 Mimi 或 LM；可分别选择")
            .setItems(labels.toArray(new String[0]), (dialog, which) -> choices.get(which).run())
            .setNegativeButton("取消", null).show();
    }

    private void selectMimi(File encoder, File decoder) {
        releaseSelectedSession(); selectedEncoder = encoder; selectedDecoder = decoder;
        getPreferences(MODE_PRIVATE).edit()
            .putString("model_encoder", encoder == null ? "" : encoder.getAbsolutePath())
            .putString("model_decoder", decoder == null ? "" : decoder.getAbsolutePath()).apply();
        modelStatus.setText("已选择 Mimi：\n" + encoderFile().getAbsolutePath()
            + "\n" + decoderFile().getAbsolutePath() + "\n请点击加载");
        refresh();
    }

    private void verifyLocalLm(File folder) {
        if (working || recording) return;
        releaseSelectedSession(); working = true; localLmVerified = false; refresh();
        status.setText("正在校验默认目录中的 26 个 LM DLC（不复制文件）…");
        worker.execute(() -> {
            try {
                File root = defaultModelsDirectory().getCanonicalFile();
                File canonical = folder.getCanonicalFile();
                if (!canonical.equals(root) && !canonical.getPath().startsWith(root.getPath() + "/"))
                    throw new IOException("LM 目录必须位于默认模型目录内");
                File inventoryFile = new File(canonical, "dlc_inventory.json");
                if (inventoryFile.length() > 1024 * 1024) throw new IOException("模型清单过大");
                byte[] bytes;
                try (InputStream input = new FileInputStream(inventoryFile);
                     ByteArrayOutputStream output = new ByteArrayOutputStream()) {
                    byte[] buffer = new byte[8192]; int count;
                    while ((count = input.read(buffer)) != -1) {
                        output.write(buffer, 0, count);
                        if (output.size() > 1024 * 1024) throw new IOException("模型清单过大");
                    }
                    bytes = output.toByteArray();
                }
                JSONObject inventory = new JSONObject(new String(bytes, StandardCharsets.UTF_8));
                JSONObject graphs = inventory.getJSONObject("graphs");
                if (!inventory.getBoolean("all_graphs_verified") || graphs.length() != 26)
                    throw new IOException("需要完整的 26 图验证清单");
                ArrayList<String> names = new ArrayList<>(); names.add("frontend"); names.add("head");
                for (int i = 0; i < 32; i += 2) names.add("temporal_layers_" + i + "_" + (i+1));
                for (int i = 0; i < 8; i++) names.add("depformer_codebook_" + i);
                for (int i = 0; i < names.size(); i++) {
                    String name = names.get(i); JSONObject graph = graphs.getJSONObject(name);
                    String relative = "dlc/" + name + ".dlc";
                    if (!relative.equals(graph.getString("file"))) throw new IOException("模型路径不匹配：" + name);
                    File model = new File(canonical, relative);
                    if (!model.getCanonicalPath().startsWith(canonical.getPath() + "/")
                            || !model.isFile() || model.length() == 0) throw new IOException("模型缺失：" + name);
                    final int index = i + 1;
                    runOnUiThread(() -> status.setText("校验 " + index + "/26：" + name));
                    MessageDigest digest = MessageDigest.getInstance("SHA-256");
                    try (InputStream input = new FileInputStream(model)) {
                        byte[] buffer = new byte[1024 * 1024]; int count;
                        while ((count = input.read(buffer)) != -1) digest.update(buffer, 0, count);
                    }
                    StringBuilder hex = new StringBuilder();
                    for (byte b : digest.digest()) hex.append(String.format(Locale.US, "%02x", b & 255));
                    if (!hex.toString().equals(graph.getString("sha256"))) throw new IOException("SHA256 不匹配：" + name);
                }
                runOnUiThread(() -> {
                    selectedLm = canonical; localLmVerified = true;
                    getPreferences(MODE_PRIVATE).edit().putString("model_lm", canonical.getAbsolutePath()).apply();
                    modelStatus.setText("LM 26 文件校验通过，直接使用：\n" + canonical.getAbsolutePath());
                    status.setText("模型已选择；无需重复导入。请点击加载。");
                });
            } catch (Exception error) {
                runOnUiThread(() -> status.setText("LM 校验失败：" + error.getMessage()));
            } finally { runOnUiThread(() -> { working = false; refresh(); }); }
        });
    }

    private void pickLm() {
        if (working || recording) return;
        startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE), PICK_LM);
    }

    private void pick(int kind) {
        if (working || recording) return;
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        startActivityForResult(intent, kind);
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request == PICK_LM && result == RESULT_OK && data != null && data.getData() != null) {
            importLm(data.getData());
            return;
        }
        if (result != RESULT_OK || data == null || data.getData() == null ||
            request < PICK_AUDIO || request > PICK_DECODER) return;
        Uri uri = data.getData();
        if (request == PICK_ENCODER) {
            selectedEncoder = null;
            getPreferences(MODE_PRIVATE).edit().remove("model_encoder").apply();
        } else if (request == PICK_DECODER) {
            selectedDecoder = null;
            getPreferences(MODE_PRIVATE).edit().remove("model_decoder").apply();
        }
        doCopy(() -> getContentResolver().openInputStream(uri), files[request],
               request == PICK_AUDIO ? "音频已导入" : "DLC 已导入");
    }

    private interface Source { InputStream open() throws IOException; }

    private Uri child(Uri tree, String parentId, String name) throws IOException {
        Uri children = DocumentsContract.buildChildDocumentsUriUsingTree(tree, parentId);
        try (Cursor cursor = getContentResolver().query(children,
                new String[]{DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                             DocumentsContract.Document.COLUMN_DISPLAY_NAME}, null, null, null)) {
            if (cursor != null) while (cursor.moveToNext()) {
                if (name.equals(cursor.getString(1)))
                    return DocumentsContract.buildDocumentUriUsingTree(tree, cursor.getString(0));
            }
        }
        throw new IOException("模型目录缺少 " + name);
    }

    private byte[] smallDocument(Uri uri) throws IOException {
        try (InputStream input = getContentResolver().openInputStream(uri);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            if (input == null) throw new IOException("无法读取模型清单");
            byte[] buffer = new byte[8192]; int count;
            while ((count = input.read(buffer)) != -1) {
                output.write(buffer, 0, count);
                if (output.size() > 1024 * 1024) throw new IOException("模型清单过大");
            }
            return output.toByteArray();
        }
    }

    private void importLm(Uri tree) {
        if (working || recording) return;
        if (new StatFs(getFilesDir().getAbsolutePath()).getAvailableBytes() < 13L * 1024 * 1024 * 1024) {
            status.setText("导入 LM 至少需要 13 GiB 可用空间（约 12 GB 模型，不包含原目录）");
            return;
        }
        working = true;
        long previous = sessionHandle; sessionHandle = 0;
        refresh(); status.setText("正在读取 LM 清单并校验 26 个模型…");
        worker.execute(() -> {
            File incoming = new File(getFilesDir(), "lm-import-" + System.currentTimeMillis());
            try {
                if (previous != 0) nativeRelease(previous);
                String rootId = DocumentsContract.getTreeDocumentId(tree);
                byte[] inventoryBytes = smallDocument(child(tree, rootId, "dlc_inventory.json"));
                JSONObject inventory = new JSONObject(new String(inventoryBytes, StandardCharsets.UTF_8));
                JSONObject graphs = inventory.getJSONObject("graphs");
                if (!inventory.getBoolean("all_graphs_verified") || graphs.length() != 26)
                    throw new IOException("需要全部验证通过的 26 个 LM 模块");
                File dlcDir = new File(incoming, "dlc");
                if (!dlcDir.mkdirs()) throw new IOException("无法创建模型目录");
                Uri sourceDir = child(tree, rootId, "dlc");
                String parentId = DocumentsContract.getDocumentId(sourceDir);
                String[] names = new String[26];
                names[0] = "frontend"; names[1] = "head";
                for (int i = 0; i < 16; i++) names[2+i] = "temporal_layers_" + (2*i) + "_" + (2*i+1);
                for (int i = 0; i < 8; i++) names[18+i] = "depformer_codebook_" + i;
                for (int i = 0; i < names.length; i++) {
                    String name = names[i];
                    JSONObject graph = graphs.getJSONObject(name);
                    if (!graph.getString("file").equals("dlc/" + name + ".dlc"))
                        throw new IOException("模型路径不匹配：" + name);
                    final int index = i + 1;
                    runOnUiThread(() -> status.setText("导入并校验 " + index + "/26：" + name));
                    MessageDigest digest = MessageDigest.getInstance("SHA-256");
                    File target = new File(dlcDir, name + ".dlc");
                    try (InputStream input = getContentResolver().openInputStream(child(tree, parentId, name + ".dlc"));
                         OutputStream output = new FileOutputStream(target)) {
                        if (input == null) throw new IOException("无法读取 " + name);
                        byte[] buffer = new byte[1024 * 1024]; int count;
                        while ((count = input.read(buffer)) != -1) {
                            digest.update(buffer, 0, count); output.write(buffer, 0, count);
                        }
                    }
                    StringBuilder hex = new StringBuilder();
                    for (byte b : digest.digest()) hex.append(String.format(Locale.US, "%02x", b & 255));
                    if (!target.isFile() || target.length() == 0 || !hex.toString().equals(graph.getString("sha256")))
                        throw new IOException("SHA256 不匹配：" + name);
                }
                try (OutputStream out = new FileOutputStream(new File(incoming, "dlc_inventory.json"))) { out.write(inventoryBytes); }
                try (OutputStream out = new FileOutputStream(new File(incoming, "READY"))) {
                    out.write("26 DLC SHA256 verified\n".getBytes(StandardCharsets.UTF_8));
                }
                File old = new File(getFilesDir(), "lm-previous-" + System.currentTimeMillis());
                File privateTarget = new File(getFilesDir(), "moshi-lm");
                boolean hadOld = privateTarget.exists();
                if (hadOld && !privateTarget.renameTo(old)) throw new IOException("无法保留原模型目录");
                if (!incoming.renameTo(privateTarget)) {
                    if (hadOld && !old.renameTo(privateTarget)) throw new IOException("切换失败；原模型保留在 " + old);
                    throw new IOException("无法切换模型目录");
                }
                runOnUiThread(() -> {
                    selectedLm = null; localLmVerified = false;
                    getPreferences(MODE_PRIVATE).edit().remove("model_lm").apply();
                    moshiMode.setChecked(true);
                    modelStatus.setText("26 个 LM DLC 校验完成；原模型（如有）已保留。请加载模型后运行短音频");
                });
            } catch (Exception error) {
                runOnUiThread(() -> status.setText("LM 导入失败：" + error.getMessage()
                        + "；未完成文件保留在 " + incoming.getName()));
            } finally { runOnUiThread(() -> { working = false; refresh(); }); }
        });
    }

    private void useDemoAudio() {
        doCopy(() -> getAssets().open("demo.wav"), files[PICK_AUDIO],
               "演示音频已导入");
    }

    private void doCopy(Source source, File target, String success) {
        if (working || recording) return;
        stopPlayer();
        outputFile().delete();
        codesFile().delete();
        tokens.setText("输入或模型已更换；请重新运行编码");
        boolean modelChanged = target == files[PICK_ENCODER] ||
                               target == files[PICK_DECODER];
        long oldHandle = modelChanged ? sessionHandle : 0;
        if (modelChanged) {
            sessionHandle = 0;
            modelStatus.setText("模型文件已更换，请重新加载 DLC");
        }
        working = true;
        refresh();
        status.setText("正在导入 " + target.getName() + "…");
        worker.execute(() -> {
            if (oldHandle != 0) nativeRelease(oldHandle);
            try (InputStream input = source.open();
                 OutputStream output = new FileOutputStream(target)) {
                byte[] buffer = new byte[1024 * 1024];
                int count;
                while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
                if (target.length() == 0) throw new IOException("文件为空");
                runOnUiThread(() -> {
                    if (modelChanged) modelStatus.setText("DLC 已导入；点击加载两个 DLC 到 QNN");
                    else audioStatus.setText(success + "：" + target.getName());
                    status.setText(success + "：" + target.getName());
                });
            } catch (Exception error) {
                target.delete();
                runOnUiThread(() -> status.setText("导入失败：" + error.getMessage()));
            } finally {
                runOnUiThread(() -> { working = false; refresh(); });
            }
        });
    }

    private void loadModels() {
        if (working || recording || !modelFilesReady()) return;
        working = true;
        final boolean useLm = moshiMode.isChecked();
        final boolean requestFast = useLm && fastRuntime.isChecked();
        modelStatus.setText("正在加载 Mimi 图；LM 使用本机 context 缓存、预算内常驻和共享缓冲…");
        refresh();
        worker.execute(() -> {
            long start = System.nanoTime();
            try {
                long previous = sessionHandle;
                sessionHandle = 0;
                if (previous != 0) nativeRelease(previous);
                File cache = contextCacheDirectory;
                if (requestFast) {
                    // Only the separately hash-verified, short-chain-tested bundle
                    // opts into the fast profile. Original/SAF imports keep the control path.
                    if (selectedLm == null || !localLmVerified)
                        throw new IOException("请先从默认目录校验 lm-cache1000-v1；或关闭加速开关运行普通对照");
                    JSONObject manifest = new JSONObject(new String(
                        readSmallFile(new File(selectedLm, "dlc_inventory.json")), StandardCharsets.UTF_8));
                    if (!"cache1000-shared-v1".equals(manifest.optString("runtime_preset")))
                        throw new IOException("所选 LM 不是 cache1000 加速包；请选择新包或关闭加速开关");
                    cache = new File(contextCacheDirectory, "cache1000-v1");
                    if (!cache.isDirectory()) throw new IOException("缺少 cache1000-v1 本手机编译缓存；请部署缓存后再加载");
                }
                sessionHandle = nativePrepare(getApplicationInfo().nativeLibraryDir,
                        encoderFile().getAbsolutePath(),
                        decoderFile().getAbsolutePath(),
                        useLm ? lmDirectory().getAbsolutePath() : "", cache.getAbsolutePath(), requestFast);
                double seconds = (System.nanoTime() - start) / 1e9;
                runOnUiThread(() -> modelStatus.setText(String.format(Locale.US,
                        requestFast ? "加速配置已就绪；首次运行加载 LM，重复运行复用。准备用时 %.1f 秒"
                              : useLm ? "Mimi 就绪；普通 LM 缓存对照模式。准备用时 %.1f 秒"
                              : "两个 Mimi DLC 已就绪；本次加载用时 %.1f 秒", seconds)));
            } catch (Exception error) {
                runOnUiThread(() -> modelStatus.setText("加载失败：" + error.getMessage()));
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
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) !=
                PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO},
                               REQUEST_MICROPHONE);
            return;
        }
        beginRecording();
    }

    @Override
    public void onRequestPermissionsResult(int request, String[] permissions,
                                            int[] results) {
        super.onRequestPermissionsResult(request, permissions, results);
        if (request != REQUEST_MICROPHONE) return;
        if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) {
            beginRecording();
        } else {
            audioStatus.setText("需要麦克风权限才能录音；也可选择现有 WAV");
        }
    }

    private void beginRecording() {
        if (working || recording || sessionHandle == 0) return;
        stopPlayer();
        outputFile().delete();
        codesFile().delete();
        tokens.setText("录音后点击推理查看 codec tokens");
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
                microphone = new AudioRecord(MediaRecorder.AudioSource.MIC,
                        SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        Math.max(minimumBytes, FRAME_SAMPLES * 4));
                if (microphone.getState() != AudioRecord.STATE_INITIALIZED) {
                    throw new IOException("麦克风初始化失败");
                }
                short[] pcm = new short[FRAME_SAMPLES];
                byte[] packed = new byte[FRAME_SAMPLES * 2];
                long sampleCount = 0;
                try (RandomAccessFile wav = new RandomAccessFile(temporary, "rw")) {
                    wav.setLength(0);
                    wav.write(new byte[44]);
                    microphone.startRecording();
                    while (recording) {
                        int count = microphone.read(pcm, 0, pcm.length,
                                AudioRecord.READ_BLOCKING);
                        if (count < 0) throw new IOException("录音读取失败：" + count);
                        for (int i = 0; i < count; i++) {
                            packed[2 * i] = (byte) pcm[i];
                            packed[2 * i + 1] = (byte) (pcm[i] >>> 8);
                        }
                        wav.write(packed, 0, count * 2);
                        sampleCount += count;
                        if (sampleCount > (0xffffffffL - 36) / 2) {
                            throw new IOException("录音超过 WAV 格式容量");
                        }
                    }
                    long completeSamples = sampleCount - sampleCount % FRAME_SAMPLES;
                    if (completeSamples == 0) throw new IOException("请至少录制 80 ms");
                    wav.setLength(44 + completeSamples * 2);
                    wav.seek(0);
                    writeWavHeader(wav, completeSamples);
                    sampleCount = completeSamples;
                }
                files[PICK_AUDIO].delete();
                if (!temporary.renameTo(files[PICK_AUDIO])) {
                    throw new IOException("录音文件保存失败");
                }
                final long savedSamples = sampleCount;
                runOnUiThread(() -> audioStatus.setText(String.format(Locale.US,
                        "录音完成：%.2f 秒 / %d 帧；点击推理",
                        savedSamples / (double) SAMPLE_RATE,
                        savedSamples / FRAME_SAMPLES)));
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

    private static void writeWavHeader(RandomAccessFile wav, long samples)
            throws IOException {
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

    private void runCodec() {
        if (working || recording || sessionHandle == 0 || !files[PICK_AUDIO].isFile()) return;
        final int requestedFrames;
        try {
            requestedFrames = moshiMode.isChecked() ? Integer.parseInt(lmFrameLimit.getText().toString().trim()) : 0;
            if (requestedFrames < 0 || requestedFrames == 1 || requestedFrames > 3000)
                throw new NumberFormatException();
        } catch (NumberFormatException error) {
            status.setText("LM 帧数请填 2–3000，或 0 跑整段");
            return;
        }
        stopPlayer();
        outputFile().delete();
        codesFile().delete();
        textTokensFile().delete();
        new File(outputFile().getAbsolutePath() + ".playback.csv").delete();
        tokens.setText("正在执行 QNN 图…");
        status.setText(moshiMode.isChecked() ? "正在流式递进 Encoder → LM → Decoder；首次建立 context 缓存，之后复用…"
                                            : "正在逐帧执行 Encoder → Decoder…");
        working = true;
        refresh();
        final boolean enableStreamPlayback = moshiMode.isChecked() && streamPlayback.isChecked();
        final boolean enableHtpPerformance = htpPerformance.isChecked();
        worker.execute(() -> {
            long start = System.nanoTime();
            try {
                stopStream();
                streamedSamples = 0;
                streamCallbacks = 0; streamChunks = 0;
                if (enableStreamPlayback) {
                    int minimum = AudioTrack.getMinBufferSize(SAMPLE_RATE,
                            AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT);
                    streamTrack = new AudioTrack.Builder()
                            .setAudioAttributes(new AudioAttributes.Builder()
                                    .setUsage(AudioAttributes.USAGE_MEDIA)
                                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                            .setAudioFormat(new AudioFormat.Builder().setSampleRate(SAMPLE_RATE)
                                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT).build())
                            .setBufferSizeInBytes(Math.max(minimum, FRAME_SAMPLES * 2 * 4))
                            .setTransferMode(AudioTrack.MODE_STREAM).build();
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
                int frames = nativeRunPrepared(sessionHandle,
                                       files[PICK_AUDIO].getAbsolutePath(),
                                       outputFile().getAbsolutePath(), requestedFrames, this, enableHtpPerformance);
                final String playbackResult = finishStreamPlayback();
                String codeText = readCodes();
                double seconds = (System.nanoTime() - start) / 1e9;
                runOnUiThread(() -> {
                    status.setText(String.format(Locale.US,
                            "推理完成：%d 帧 / %.2f 秒输出音频；本次运行 %.2f 秒，输出 RTF %.2f（小于 1 才达到实时）。详细计时可导出。",
                            frames, frames * 0.08, seconds,
                            seconds / Math.max(frames * 0.08, 1e-9)) + playbackResult);
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

    // Called on the existing inference worker once per input frame, not UI thread.
    // A bounded AudioTrack buffer provides backpressure; no unbounded audio queue.
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
                "已处理输入 %d 帧，输出 %d 帧；刚完成一帧 %.1f ms。模型与状态继续复用。",
                consumed, emitted, milliseconds)));
    }

    private void stopStream() {
        if (streamTrack != null) { streamTrack.release(); streamTrack = null; }
    }

    private String finishStreamPlayback() throws IOException, InterruptedException {
        if (streamTrack == null) return "";
        // A bounded final queue; do not call consumption a proof of audible quality.
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
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(codesFile()), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(",");
                text.append(String.format(java.util.Locale.US, "%02d  ",
                                          Integer.parseInt(values[0])));
                for (int i = 1; i < values.length; i++) {
                    if (i > 1) text.append(' ');
                    text.append(values[i]);
                }
                text.append('\n');
            }
        }
        if (moshiMode.isChecked() && textTokensFile().isFile()) {
            text.append("\n文本 token ID（尚未接 SentencePiece 解码，不是文字转录）：\n");
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                    new FileInputStream(textTokensFile()), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) text.append(line).append('\n');
            }
        }
        return text.toString();
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
                saveDownload(outputFile(), "mimi-" + suffix + ".wav", "audio/wav");
                saveDownload(codesFile(), "mimi-" + suffix + "-tokens.csv", "text/csv");
                if (moshiMode.isChecked() && textTokensFile().isFile())
                    saveDownload(textTokensFile(), "moshi-" + suffix + "-text-token-ids.csv", "text/csv");
                File performance = new File(outputFile().getAbsolutePath() + ".performance.csv");
                if (performance.isFile())
                    saveDownload(performance, "moshi-" + suffix + "-performance.csv", "text/csv");
                File qnnProfile = new File(outputFile().getAbsolutePath() + ".performance.csv.qnn.csv");
                if (qnnProfile.isFile())
                    saveDownload(qnnProfile, "moshi-" + suffix + "-qnn-profile.csv", "text/csv");
                File configuration = new File(outputFile().getAbsolutePath() + ".runtime.json");
                if (configuration.isFile())
                    saveDownload(configuration, "moshi-" + suffix + "-runtime.json", "application/json");
                File playback = new File(outputFile().getAbsolutePath() + ".playback.csv");
                if (playback.isFile())
                    saveDownload(playback, "moshi-" + suffix + "-playback.csv", "text/csv");
                runOnUiThread(() -> status.setText("已保存到 Downloads/MimiDemo"));
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
        values.put(MediaStore.Downloads.RELATIVE_PATH,
                   Environment.DIRECTORY_DOWNLOADS + "/MimiDemo");
        Uri uri = getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI,
                                              values);
        if (uri == null) throw new IOException("无法创建下载文件");
        try (InputStream input = new FileInputStream(source);
             OutputStream output = getContentResolver().openOutputStream(uri)) {
            if (output == null) throw new IOException("无法写入下载文件");
            byte[] buffer = new byte[65536];
            int count;
            while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
        }
    }

    private boolean modelFilesReady() {
        return encoderFile().isFile() && decoderFile().isFile()
                && (!moshiMode.isChecked() || (selectedLm != null ? localLmVerified
                    : new File(lmDirectory(), "READY").isFile()));
    }

    private static byte[] readSmallFile(File file) throws IOException {
        try (InputStream input = new FileInputStream(file);
             ByteArrayOutputStream bytes = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192]; int count;
            while ((count = input.read(buffer)) != -1) {
                if (bytes.size() + count > 1024 * 1024) throw new IOException("清单超过 1 MiB");
                bytes.write(buffer, 0, count);
            }
            return bytes.toByteArray();
        }
    }

    private void refresh() {
        if (working || recording) getWindow().addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        else getWindow().clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        if (streamPlayback != null) streamPlayback.setEnabled(!working && !recording);
        if (htpPerformance != null) htpPerformance.setEnabled(!working && !recording);
        if (fastRuntime != null) fastRuntime.setEnabled(!working && !recording);
        scanModelsButton.setEnabled(!working && !recording);
        moshiMode.setEnabled(!working && !recording);
        lmFrameLimit.setEnabled(!working && !recording && moshiMode.isChecked());
        loadButton.setText(moshiMode.isChecked() ? "准备 Mimi + Moshi LM" : "加载两个 Mimi DLC 到 QNN");
        runButton.setText(moshiMode.isChecked() ? "运行 Encoder → Moshi LM → Decoder" : "运行 Encoder → Decoder 对照");
        loadButton.setEnabled(!working && !recording && modelFilesReady());
        recordButton.setEnabled(!working && (recording || sessionHandle != 0));
        recordButton.setText(recording ? "■ 停止录音" : "● 开始录音");
        runButton.setEnabled(!working && !recording && sessionHandle != 0 &&
                             files[PICK_AUDIO].isFile());
        originalButton.setEnabled(!working && !recording && files[PICK_AUDIO].isFile());
        outputButton.setEnabled(!working && !recording && outputFile().isFile());
        exportButton.setEnabled(!working && !recording && outputFile().isFile() &&
                                codesFile().isFile());
    }

    private LinearLayout card(LinearLayout parent) {
        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(dp(16), dp(16), dp(16), dp(16));
        android.graphics.drawable.GradientDrawable background =
                new android.graphics.drawable.GradientDrawable();
        background.setColor(CARD);
        background.setCornerRadius(dp(18));
        panel.setBackground(background);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, -2);
        params.topMargin = dp(20);
        parent.addView(panel, params);
        return panel;
    }

    private TextView label(LinearLayout parent, String value, int size,
                           int color, boolean bold) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextColor(color);
        view.setTextSize(size);
        view.setPadding(0, dp(4), 0, dp(8));
        if (bold) view.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
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
