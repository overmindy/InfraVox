package com.overmindy.duplex;

import android.Manifest;
import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Switch;
import android.widget.TextView;

import com.overmindy.duplex.audio.ConversationSession;
import com.overmindy.duplex.audio.Wav;
import com.overmindy.duplex.engine.DuplexEngine;
import com.overmindy.duplex.engine.Engines;
import com.overmindy.duplex.model.HubConfig;
import com.overmindy.duplex.model.HubDownloader;
import com.overmindy.duplex.model.ModelCatalog;
import com.overmindy.duplex.model.ModelStore;
import com.overmindy.duplex.ui.TalkButton;
import com.overmindy.duplex.ui.Ui;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final int PICK_AUDIO = 1;
    private static final int REQUEST_MICROPHONE = 2;

    private enum EngineState { NONE, VERIFYING, LOADING, READY, ERROR }
    private enum Source { MICROPHONE, DEMO, FILE }

    private Ui ui;
    private SharedPreferences preferences;
    private ModelStore store;
    private ModelCatalog catalog;
    private HubConfig hub;
    private final String soc = Build.VERSION.SDK_INT >= 31 ? Build.SOC_MODEL : "";

    // The engine thread owns the native model: load, every step of a session, and release.
    private final ExecutorService engineThread = Executors.newSingleThreadExecutor();
    private final ExecutorService ioThread = Executors.newSingleThreadExecutor();

    private DuplexEngine engine;
    private String loadedVariant;
    private EngineState engineState = EngineState.NONE;
    private String engineMessage = "";
    private ConversationSession session;
    private boolean hasRecording;
    private Source source = Source.MICROPHONE;
    private MediaPlayer player;

    private HubDownloader downloader;
    private String downloadingVariant;
    private boolean catalogRefreshing;
    private final Map<String, String> variantErrors = new HashMap<>();
    private final Map<String, VariantViews> variantViews = new HashMap<>();

    private final View[] pages = new View[3];
    private final TextView[] tabs = new TextView[3];
    private TextView headerModel;

    private TextView modelName, modelVariant, modelChip, modelDetail, talkHint, transcript, sessionStatus;
    private Button modelAction, playInput, playOutput, exportButton;
    private TalkButton talk;
    private final TextView[] metrics = new TextView[4];
    private final TextView[] sources = new TextView[3];

    private LinearLayout modelsList;
    private TextView hubSummary, hubStatus;
    private Button refreshButton;

    private EditText endpointField, repoField, tokenField;
    private final TextView[] endpointChoices = new TextView[3];
    private Switch highPerformance, echoCancel, codecOnly, textSampling;

    private static final class VariantViews {
        TextView status;
        ProgressBar progress;
        View pauseGap;
        Button download, pause, verify, remove, use;
    }

    @Override
    public void onCreate(Bundle state) {
        super.onCreate(state);
        ui = new Ui(this);
        preferences = getSharedPreferences("duplex", MODE_PRIVATE);
        hub = HubConfig.load(preferences);
        store = new ModelStore(this, preferences);
        store.migrateLegacy();
        try {
            catalog = store.loadCatalog();
        } catch (Exception error) {
            throw new IllegalStateException("内置模型列表损坏", error);
        }
        getWindow().setStatusBarColor(Ui.BACKGROUND);
        getWindow().setNavigationBarColor(Ui.SURFACE);
        setContentView(buildShell());
        renderModels();
        selectTab(store.status(selectedVariant()).state == ModelStore.State.READY ? 0 : 1);
        refreshAll();
    }

    // ---------------------------------------------------------------- layout

    private View buildShell() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Ui.BACKGROUND);

        LinearLayout header = ui.column(0);
        header.setPadding(ui.dp(20), ui.dp(18), ui.dp(20), ui.dp(6));
        ui.text(header, "Duplex", 26, Ui.TEXT, true);
        headerModel = ui.text(header, "", 13, Ui.MUTED, false);
        root.addView(header);

        FrameLayout content = new FrameLayout(this);
        pages[0] = scroll(buildChatPage());
        pages[1] = scroll(buildModelsPage());
        pages[2] = scroll(buildSettingsPage());
        for (View page : pages) content.addView(page, new FrameLayout.LayoutParams(-1, -1));
        root.addView(content, new LinearLayout.LayoutParams(-1, 0, 1));

        LinearLayout nav = ui.row();
        nav.setBackgroundColor(Ui.SURFACE);
        nav.setPadding(ui.dp(12), ui.dp(8), ui.dp(12), ui.dp(10));
        String[] titles = {"对话", "模型", "设置"};
        for (int i = 0; i < 3; i++) {
            final int index = i;
            TextView tab = ui.text(null, titles[i], 15, Ui.MUTED, true);
            tab.setGravity(Gravity.CENTER);
            tab.setPadding(0, ui.dp(10), 0, ui.dp(10));
            tab.setOnClickListener(v -> selectTab(index));
            nav.addView(tab, new LinearLayout.LayoutParams(0, -2, 1));
            tabs[i] = tab;
        }
        root.addView(nav);
        return root;
    }

    private ScrollView scroll(View child) {
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.addView(child);
        return scroll;
    }

    private void selectTab(int index) {
        for (int i = 0; i < 3; i++) {
            pages[i].setVisibility(i == index ? View.VISIBLE : View.GONE);
            tabs[i].setTextColor(i == index ? Ui.TEXT : Ui.MUTED);
            tabs[i].setBackground(i == index ? Ui.rounded(Ui.SURFACE_HIGH, ui.dp(12)) : null);
        }
    }

    private View buildChatPage() {
        LinearLayout page = ui.column(16);
        page.setPadding(ui.dp(16), 0, ui.dp(16), ui.dp(24));

        LinearLayout modelCard = ui.card(page);
        LinearLayout top = ui.row();
        LinearLayout names = ui.column(0);
        ui.eyebrow(names, "当前模型");
        modelName = ui.text(names, "", 18, Ui.TEXT, true);
        modelVariant = ui.text(names, "", 13, Ui.MUTED, false);
        top.addView(names, new LinearLayout.LayoutParams(0, -2, 1));
        modelChip = ui.chip(top, "", Ui.MUTED);
        modelCard.addView(top);
        modelDetail = ui.text(modelCard, "", 13, Ui.MUTED, false);
        modelDetail.setPadding(0, ui.dp(8), 0, 0);
        modelAction = ui.primary(modelCard, "加载模型", this::onModelAction);

        LinearLayout stage = ui.card(page);
        stage.setGravity(Gravity.CENTER_HORIZONTAL);
        talk = new TalkButton(this);
        talk.setOnClickListener(v -> onTalk());
        stage.addView(talk, new LinearLayout.LayoutParams(-1, -2));
        talkHint = ui.text(stage, "", 13, Ui.MUTED, false);
        talkHint.setGravity(Gravity.CENTER);

        LinearLayout sourceRow = ui.row();
        sourceRow.setBackground(Ui.rounded(Ui.SURFACE_HIGH, ui.dp(14)));
        sourceRow.setPadding(ui.dp(4), ui.dp(4), ui.dp(4), ui.dp(4));
        String[] labels = {"麦克风", "演示音频", "WAV 文件"};
        for (int i = 0; i < 3; i++) {
            final Source value = Source.values()[i];
            TextView option = ui.text(null, labels[i], 14, Ui.MUTED, true);
            option.setGravity(Gravity.CENTER);
            option.setPadding(0, ui.dp(9), 0, ui.dp(9));
            option.setOnClickListener(v -> chooseSource(value));
            sourceRow.addView(option, new LinearLayout.LayoutParams(0, -2, 1));
            sources[i] = option;
        }
        LinearLayout.LayoutParams sourceParams = new LinearLayout.LayoutParams(-1, -2);
        sourceParams.topMargin = ui.dp(14);
        stage.addView(sourceRow, sourceParams);

        LinearLayout grid = ui.card(page);
        ui.eyebrow(grid, "实时指标");
        String[] names2 = {"端到端延迟", "每帧耗时", "实时率 RTF", "积压帧"};
        for (int r = 0; r < 2; r++) {
            LinearLayout row = ui.row();
            for (int c = 0; c < 2; c++) {
                int i = r * 2 + c;
                LinearLayout tile = ui.column(12);
                tile.setBackground(Ui.rounded(Ui.SURFACE_HIGH, ui.dp(14)));
                metrics[i] = ui.text(tile, "—", 22, Ui.TEXT, true);
                metrics[i].setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
                ui.text(tile, names2[i], 12, Ui.MUTED, false);
                LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, -2, 1);
                params.setMargins(c == 0 ? 0 : ui.dp(5), ui.dp(5), c == 0 ? ui.dp(5) : 0, ui.dp(5));
                row.addView(tile, params);
            }
            grid.addView(row);
        }
        sessionStatus = ui.text(grid, "", 13, Ui.MUTED, false);
        sessionStatus.setPadding(0, ui.dp(6), 0, 0);

        LinearLayout reply = ui.card(page);
        ui.eyebrow(reply, "模型回复（文本流）");
        transcript = ui.text(reply, "开始对话后，这里会逐字显示模型说出的内容。", 17, Ui.MUTED, false);
        transcript.setTextIsSelectable(true);

        LinearLayout record = ui.card(page);
        ui.eyebrow(record, "本次对话录音");
        LinearLayout buttons = ui.buttonRow(record);
        playInput = ui.secondary(buttons, "▶ 我的输入", () -> play(new File(sessionDir(), "input.wav")));
        ui.gap(buttons);
        playOutput = ui.secondary(buttons, "▶ 模型输出", () -> play(new File(sessionDir(), "output.wav")));
        exportButton = ui.secondary(record, "导出音频、文本与性能数据到下载目录", this::exportSession);
        hasRecording = new File(sessionDir(), "output.wav").isFile();
        return page;
    }

    private View buildModelsPage() {
        LinearLayout page = ui.column(16);
        page.setPadding(ui.dp(16), 0, ui.dp(16), ui.dp(24));
        LinearLayout hubCard = ui.card(page);
        ui.eyebrow(hubCard, "模型仓库");
        hubSummary = ui.text(hubCard, "", 14, Ui.TEXT, false);
        hubStatus = ui.text(hubCard, "", 13, Ui.MUTED, false);
        hubStatus.setPadding(0, ui.dp(4), 0, 0);
        refreshButton = ui.secondary(hubCard, "从仓库刷新模型列表", this::refreshCatalog);
        modelsList = ui.column(0);
        page.addView(modelsList);
        return page;
    }

    private View buildSettingsPage() {
        LinearLayout page = ui.column(16);
        page.setPadding(ui.dp(16), 0, ui.dp(16), ui.dp(24));

        LinearLayout source = ui.card(page);
        ui.eyebrow(source, "下载源");
        LinearLayout choices = ui.row();
        String[] titles = {"HF 镜像", "Hugging Face", "自定义"};
        for (int i = 0; i < 3; i++) {
            final int index = i;
            TextView option = ui.chip(choices, titles[i], Ui.MUTED);
            option.setTextSize(14);
            option.setGravity(Gravity.CENTER);
            option.setPadding(0, ui.dp(9), 0, ui.dp(9));
            option.setOnClickListener(v -> chooseEndpoint(index));
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, -2, 1);
            params.setMargins(i == 0 ? 0 : ui.dp(4), 0, i == 2 ? 0 : ui.dp(4), 0);
            choices.removeView(option);
            choices.addView(option, params);
            endpointChoices[i] = option;
        }
        source.addView(choices);
        endpointField = ui.field(source, "https://your-mirror.example.com", hub.endpoint);
        endpointField.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        ui.text(source, "国内网络建议用 hf-mirror.com；所有模型都从仓库的 main 分支下载。", 12, Ui.MUTED, false)
                .setPadding(0, ui.dp(8), 0, 0);

        LinearLayout repo = ui.card(page);
        ui.eyebrow(repo, "模型仓库");
        repoField = ui.field(repo, "用户名/仓库名", hub.repo);
        tokenField = ui.field(repo, "访问令牌（仅私有仓库需要）", hub.token);
        tokenField.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);
        ui.primary(repo, "保存并刷新模型列表", this::saveHub);

        LinearLayout runtime = ui.card(page);
        ui.eyebrow(runtime, "推理");
        highPerformance = ui.toggle(runtime, "HTP 高性能档", "对话期间请求 burst 性能投票，更耗电", preferences.getBoolean("high_performance", true));
        echoCancel = ui.toggle(runtime, "回声消除", "用通话模式麦克风 + AEC；外放时减少模型听到自己", preferences.getBoolean("echo_cancel", true));
        textSampling = ui.toggle(runtime, "文本采样", "温度 0.7、top-k 25（Kyutai 默认）；关闭则贪心解码，模型容易一直沉默",
                preferences.getBoolean("text_sampling", true));
        textSampling.setOnCheckedChangeListener((b, on) -> preferences.edit().putBoolean("text_sampling", on).apply());
        codecOnly = ui.toggle(runtime, "仅编解码回环（调试）", "只跑 Mimi encoder → decoder，不跑语言模型", preferences.getBoolean("codec_only", false));
        highPerformance.setOnCheckedChangeListener((b, on) -> preferences.edit().putBoolean("high_performance", on).apply());
        echoCancel.setOnCheckedChangeListener((b, on) -> preferences.edit().putBoolean("echo_cancel", on).apply());
        codecOnly.setOnCheckedChangeListener((b, on) -> {
            preferences.edit().putBoolean("codec_only", on).apply();
            if (session == null) releaseEngine("模式已切换，请重新加载");
        });

        LinearLayout about = ui.card(page);
        ui.eyebrow(about, "关于");
        String version;
        try {
            version = getPackageManager().getPackageInfo(getPackageName(), 0).versionName;
        } catch (PackageManager.NameNotFoundException error) {
            version = "?";
        }
        ui.text(about, "版本 " + version + "\n设备 SoC：" + (soc.isEmpty() ? "未知" : soc)
                + "\n模型目录：" + store.root().getAbsolutePath(), 13, Ui.MUTED, false);
        return page;
    }

    // ---------------------------------------------------------------- models

    private ModelCatalog.Variant selectedVariant() {
        ModelCatalog.Variant chosen = catalog.variant(preferences.getString("selected_variant", ""));
        if (chosen != null) return chosen;
        ModelCatalog.Variant fallback = null;
        for (ModelCatalog.Model model : catalog.models)
            for (ModelCatalog.Variant variant : model.variants) {
                if (fallback == null) fallback = variant;
                if (variant.supports(soc) && Engines.supports(model.engine)) return variant;
            }
        return fallback;
    }

    private void renderModels() {
        modelsList.removeAllViews();
        variantViews.clear();
        for (ModelCatalog.Model model : catalog.models) {
            LinearLayout card = ui.card(modelsList);
            LinearLayout head = ui.row();
            LinearLayout names = ui.column(0);
            ui.text(names, model.name, 19, Ui.TEXT, true);
            String meta = model.publisher + (model.license.isEmpty() ? "" : " · " + model.license)
                    + (model.languages.isEmpty() ? "" : " · " + String.join("/", model.languages));
            ui.text(names, meta, 12, Ui.MUTED, false);
            head.addView(names, new LinearLayout.LayoutParams(0, -2, 1));
            ui.chip(head, model.engine, Ui.ACCENT);
            card.addView(head);
            if (!model.summary.isEmpty()) ui.text(card, model.summary, 14, Ui.TEXT, false).setPadding(0, ui.dp(8), 0, 0);
            for (ModelCatalog.Variant variant : model.variants) renderVariant(card, variant);
        }
    }

    private void renderVariant(LinearLayout card, ModelCatalog.Variant variant) {
        LinearLayout box = ui.column(14);
        box.setBackground(Ui.rounded(Ui.SURFACE_HIGH, ui.dp(16)));
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, -2);
        params.topMargin = ui.dp(12);
        card.addView(box, params);

        ui.text(box, variant.title, 15, Ui.TEXT, true);
        StringBuilder info = new StringBuilder(Ui.bytes(variant.totalBytes())).append(" · ").append(variant.files.size()).append(" 个文件");
        if (!variant.socs.isEmpty()) info.append(" · ").append(String.join("/", variant.socs));
        if (!variant.qairt.isEmpty()) info.append(" · QAIRT ").append(variant.qairt);
        ui.text(box, info.toString(), 12, Ui.MUTED, false);
        if (!variant.notes.isEmpty()) ui.text(box, variant.notes, 12, Ui.MUTED, false).setPadding(0, ui.dp(4), 0, 0);

        VariantViews views = new VariantViews();
        views.status = ui.text(box, "", 13, Ui.TEXT, false);
        views.status.setPadding(0, ui.dp(8), 0, 0);
        views.progress = ui.progress(box);
        LinearLayout row1 = ui.buttonRow(box);
        views.download = ui.primary(row1, "下载", () -> startDownload(variant));
        views.pauseGap = ui.gap(row1);
        views.pause = ui.secondary(row1, "暂停", this::pauseDownload);
        LinearLayout row2 = ui.buttonRow(box);
        views.use = ui.secondary(row2, "设为当前", () -> useVariant(variant));
        ui.gap(row2);
        views.verify = ui.secondary(row2, "校验", () -> verifyVariant(variant));
        ui.gap(row2);
        views.remove = ui.danger(row2, "删除", () -> deleteVariant(variant));
        variantViews.put(variant.id, views);
    }

    private void refreshModels() {
        hubSummary.setText(hub.label() + "\n" + hub.repo + " @ " + HubConfig.REVISION);
        Ui.enable(refreshButton, !catalogRefreshing);
        if (catalogRefreshing) hubStatus.setText("正在获取 catalog.json…");
        else if (hubStatus.getText().length() == 0)
            hubStatus.setText("剩余空间 " + Ui.bytes(store.freeBytes()) + " · 设备 " + (soc.isEmpty() ? "未知 SoC" : soc));
        ModelCatalog.Variant selected = selectedVariant();
        for (ModelCatalog.Model model : catalog.models) {
            for (ModelCatalog.Variant variant : model.variants) {
                VariantViews views = variantViews.get(variant.id);
                if (views == null) continue;
                ModelStore.Status status = store.status(variant);
                boolean downloading = variant.id.equals(downloadingVariant);
                boolean compatible = variant.supports(soc) && Engines.supports(model.engine);
                boolean inUse = variant.id.equals(loadedVariant) && (session != null || engineBusy());
                String error = variantErrors.get(variant.id);
                String text;
                int color = Ui.TEXT;
                if (downloading) text = views.status.getText().toString();
                else if (!Engines.supports(model.engine)) { text = "需要更新 App 才能运行此引擎"; color = Ui.WARN; }
                else if (error != null) { text = error; color = Ui.ERROR; }
                else if (status.state == ModelStore.State.READY) { text = store.verified(variant) ? "已下载并校验" : "已下载（加载前会校验）"; color = Ui.LIVE; }
                else if (status.state == ModelStore.State.PARTIAL)
                    text = "未完成：已有 " + Ui.bytes(status.presentBytes) + "，缺 " + status.missingFiles + " 个文件，可继续下载";
                else text = "未下载";
                if (!compatible && Engines.supports(model.engine)) { text += "\n此设备（" + (soc.isEmpty() ? "未知" : soc) + "）不兼容"; color = Ui.WARN; }
                if (variant == selected) text += " · 当前";
                views.status.setText(text);
                views.status.setTextColor(color);
                views.progress.setVisibility(downloading || status.state == ModelStore.State.PARTIAL ? View.VISIBLE : View.GONE);
                if (!downloading) views.progress.setProgress((int) (1000 * status.presentBytes / Math.max(1, variant.totalBytes())));
                boolean idle = downloadingVariant == null;
                views.download.setText(status.state == ModelStore.State.PARTIAL ? "继续下载" : status.state == ModelStore.State.READY ? "重新检查" : "下载");
                Ui.enable(views.download, idle && !inUse && compatible);
                views.pause.setVisibility(downloading ? View.VISIBLE : View.GONE);
                views.pauseGap.setVisibility(views.pause.getVisibility());
                Ui.enable(views.use, variant != selected && compatible && session == null);
                Ui.enable(views.verify, idle && !inUse && status.state == ModelStore.State.READY);
                Ui.enable(views.remove, idle && !inUse && status.state != ModelStore.State.MISSING);
            }
        }
    }

    private void refreshCatalog() {
        String problem = hub.validate();
        if (problem != null) { hubStatus.setText(problem); return; }
        catalogRefreshing = true;
        refreshAll();
        final HubConfig config = hub;
        ioThread.execute(() -> {
            String message;
            ModelCatalog fresh = null;
            try {
                String text = HubDownloader.fetchText(config, "catalog.json");
                fresh = ModelCatalog.parse(text);
                store.saveCatalog(text);
                message = "已更新：" + fresh.models.size() + " 个模型（" + timeNow() + "）";
            } catch (Exception error) {
                message = "刷新失败：" + error.getMessage() + "\n仍使用本地模型列表";
            }
            final ModelCatalog result = fresh;
            final String status = message;
            runOnUiThread(() -> {
                catalogRefreshing = false;
                if (result != null && downloadingVariant == null) {
                    catalog = result;
                    renderModels();
                }
                hubStatus.setText(status);
                refreshAll();
            });
        });
    }

    private void startDownload(ModelCatalog.Variant variant) {
        if (downloadingVariant != null) return;
        String problem = hub.validate();
        if (problem != null) { selectTab(2); toastStatus(problem); return; }
        long needed = variant.totalBytes() - store.status(variant).presentBytes;
        if (needed > store.freeBytes()) {
            variantErrors.put(variant.id, "空间不足：还需要 " + Ui.bytes(needed) + "，剩余 " + Ui.bytes(store.freeBytes()));
            refreshAll();
            return;
        }
        variantErrors.remove(variant.id);
        downloadingVariant = variant.id;
        downloader = new HubDownloader();
        final HubDownloader active = downloader;
        final HubConfig config = hub;
        VariantViews views = variantViews.get(variant.id);
        views.status.setText("正在连接 " + config.label() + "…");
        refreshAll();
        ioThread.execute(() -> {
            String error = null;
            try {
                active.download(config, store, variant, (done, total, file, speed) -> runOnUiThread(() -> {
                    VariantViews v = variantViews.get(variant.id);
                    if (v == null) return;
                    v.progress.setProgress((int) (1000 * done / Math.max(1, total)));
                    v.status.setText(String.format(Locale.US, "%s / %s%s\n%s", Ui.bytes(done), Ui.bytes(total),
                            speed > 0 ? String.format(Locale.US, " · %s/s", Ui.bytes((long) speed)) : "", file));
                }));
            } catch (InterruptedException | java.io.InterruptedIOException paused) {
                error = "已暂停，可继续下载";
            } catch (Exception failure) {
                error = "下载失败：" + failure.getMessage();
            }
            final String message = error;
            runOnUiThread(() -> {
                downloadingVariant = null;
                downloader = null;
                if (message != null) variantErrors.put(variant.id, message);
                refreshAll();
            });
        });
    }

    private void pauseDownload() {
        if (downloader != null) downloader.cancel();
    }

    private void verifyVariant(ModelCatalog.Variant variant) {
        if (downloadingVariant != null) return;
        downloadingVariant = variant.id;
        variantErrors.remove(variant.id);
        refreshAll();
        ioThread.execute(() -> {
            String error = null;
            try {
                store.verify(variant, (done, total, file) -> runOnUiThread(() -> {
                    VariantViews v = variantViews.get(variant.id);
                    v.progress.setProgress((int) (1000 * done / Math.max(1, total)));
                    v.status.setText("校验 " + (100 * done / Math.max(1, total)) + "% · " + file);
                }));
            } catch (Exception failure) {
                error = "校验失败：" + failure.getMessage();
            }
            final String message = error;
            runOnUiThread(() -> {
                downloadingVariant = null;
                if (message != null) variantErrors.put(variant.id, message);
                refreshAll();
            });
        });
    }

    private void deleteVariant(ModelCatalog.Variant variant) {
        if (session != null || downloadingVariant != null) return;
        new android.app.AlertDialog.Builder(this)
                .setTitle("删除 " + variant.model.name + "？")
                .setMessage("将删除本地 " + Ui.bytes(store.status(variant).presentBytes) + " 的模型文件，之后可以重新下载。")
                .setNegativeButton("取消", null)
                .setPositiveButton("删除", (dialog, which) -> {
                    if (variant.id.equals(loadedVariant)) releaseEngine("");
                    variantErrors.remove(variant.id);
                    ioThread.execute(() -> {
                        store.delete(variant);
                        runOnUiThread(this::refreshAll);
                    });
                })
                .show();
    }

    private void useVariant(ModelCatalog.Variant variant) {
        if (session != null) return;
        preferences.edit().putString("selected_variant", variant.id).apply();
        if (loadedVariant != null && !loadedVariant.equals(variant.id)) releaseEngine("");
        refreshAll();
        selectTab(0);
    }

    // ---------------------------------------------------------------- engine

    private boolean engineBusy() {
        return engineState == EngineState.VERIFYING || engineState == EngineState.LOADING;
    }

    private void onModelAction() {
        ModelCatalog.Variant variant = selectedVariant();
        if (variant == null || store.status(variant).state != ModelStore.State.READY) { selectTab(1); return; }
        if (engineBusy() || session != null) return;
        loadEngine(variant);
    }

    private void loadEngine(ModelCatalog.Variant variant) {
        final boolean verified = store.verified(variant);
        final DuplexEngine.Options options = new DuplexEngine.Options();
        options.codecOnly = codecOnly.isChecked();
        engineState = verified ? EngineState.LOADING : EngineState.VERIFYING;
        engineMessage = verified ? "正在把模型载入 HTP…" : "首次使用，正在校验模型文件…";
        refreshAll();
        engineThread.execute(() -> {
            try {
                if (!verified) {
                    store.verify(variant, (done, total, file) -> postEngine(EngineState.VERIFYING,
                            "校验 " + (100 * done / Math.max(1, total)) + "% · " + file));
                }
                postEngine(EngineState.LOADING, "正在把模型载入 HTP…");
                if (engine != null) engine.release();
                engine = null;
                long start = System.nanoTime();
                DuplexEngine created = Engines.create(variant.model.engine);
                created.load(this, store.dir(variant), options);
                double seconds = (System.nanoTime() - start) / 1e9;
                runOnUiThread(() -> {
                    engine = created;
                    loadedVariant = variant.id;
                    engineState = EngineState.READY;
                    engineMessage = String.format(Locale.US, "%s 已就绪 · 加载 %.1f 秒", created.describe(), seconds);
                    refreshAll();
                });
            } catch (Throwable error) {
                runOnUiThread(() -> {
                    engine = null;
                    loadedVariant = null;
                    engineState = EngineState.ERROR;
                    engineMessage = "加载失败：" + error.getMessage();
                    refreshAll();
                });
            }
        });
    }

    private void postEngine(EngineState state, String message) {
        runOnUiThread(() -> {
            engineState = state;
            engineMessage = message;
            refreshAll();
        });
    }

    private void releaseEngine(String message) {
        final DuplexEngine old = engine;
        engine = null;
        loadedVariant = null;
        engineState = EngineState.NONE;
        engineMessage = message;
        if (old != null) engineThread.execute(old::release);
        refreshAll();
    }

    // ---------------------------------------------------------------- conversation

    private void chooseSource(Source value) {
        if (session != null) return;
        if (value == Source.FILE) {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("audio/*");
            startActivityForResult(intent, PICK_AUDIO);
            return;
        }
        if (value == Source.DEMO && !hasDemo()) { toastStatus("此安装包没有内置演示音频"); return; }
        source = value;
        refreshAll();
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request != PICK_AUDIO || result != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        ioThread.execute(() -> {
            File target = pickedFile();
            String message;
            try (InputStream input = getContentResolver().openInputStream(uri);
                 OutputStream output = new FileOutputStream(target)) {
                if (input == null) throw new IOException("无法读取");
                byte[] buffer = new byte[1 << 16];
                int count;
                while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
                output.close();
                try (InputStream check = new FileInputStream(target)) {
                    short[] samples = Wav.readMono16(check, 24_000);
                    message = String.format(Locale.US, "已选择 WAV：%.1f 秒，将按实时速度送入模型", samples.length / 24_000.0);
                }
            } catch (Exception error) {
                target.delete();
                message = "无法使用此文件：" + error.getMessage();
            }
            final String status = message;
            runOnUiThread(() -> {
                if (pickedFile().isFile()) source = Source.FILE;
                toastStatus(status);
                refreshAll();
            });
        });
    }

    private void onTalk() {
        if (session != null) {
            if (session.isDraining()) session.abort();
            else session.stopInput();
            refreshAll();
            return;
        }
        if (engineState != EngineState.READY || engine == null) { onModelAction(); return; }
        if (source == Source.MICROPHONE
                && checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_MICROPHONE);
            return;
        }
        startSession();
    }

    @Override
    public void onRequestPermissionsResult(int request, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(request, permissions, results);
        if (request != REQUEST_MICROPHONE) return;
        if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) startSession();
        else toastStatus("需要麦克风权限；也可以改用演示音频或 WAV 文件");
    }

    private void startSession() {
        if (session != null || engine == null) return;
        stopPlayer();
        ConversationSession.Config config = new ConversationSession.Config();
        config.highPerformance = highPerformance.isChecked();
        config.textTemperature = textSampling.isChecked() ? 0.7f : 0f;
        config.echoCancel = echoCancel.isChecked();
        config.directory = sessionDir();
        try {
            if (source == Source.DEMO) {
                try (InputStream input = getAssets().open("demo.wav")) { config.fileInput = Wav.readMono16(input, engine.sampleRate()); }
            } else if (source == Source.FILE) {
                try (InputStream input = new FileInputStream(pickedFile())) { config.fileInput = Wav.readMono16(input, engine.sampleRate()); }
            }
        } catch (IOException error) {
            toastStatus("读取音频失败：" + error.getMessage());
            return;
        }
        final double frameMs = 1000.0 * engine.frameSamples() / engine.sampleRate();
        transcript.setText("");
        transcript.setTextColor(Ui.TEXT);
        hasRecording = false;
        session = new ConversationSession(engine, config, new ConversationSession.Listener() {
            @Override public void onStats(ConversationSession.Stats stats) { runOnUiThread(() -> showStats(stats, frameMs)); }

            @Override public void onText(String delta) { runOnUiThread(() -> transcript.append(delta)); }

            @Override
            public void onFinished(Throwable error, ConversationSession.Stats stats) {
                runOnUiThread(() -> {
                    session = null;
                    hasRecording = new File(sessionDir(), "output.wav").isFile();
                    showStats(stats, frameMs);
                    talk.levels(0, 0);
                    if (transcript.getText().length() == 0) {
                        transcript.setTextColor(Ui.MUTED);
                        transcript.setText(codecOnly.isChecked() ? "编解码回环模式没有文本输出" : "（这段对话里模型没有说出文字）");
                    }
                    sessionStatus.setText(error != null ? "对话中断：" + error.getMessage()
                            : String.format(Locale.US, "对话结束：输入 %.1f 秒，输出 %d 帧，平均每帧 %.0f ms%s", stats.inputSeconds,
                              stats.emitted, stats.averageMs, stats.backlog > 0 ? "，丢弃积压 " + stats.backlog + " 帧" : ""));
                    sessionStatus.setTextColor(error != null ? Ui.ERROR : Ui.MUTED);
                    refreshAll();
                });
            }
        });
        sessionStatus.setText("");
        session.start(engineThread);
        refreshAll();
    }

    private void showStats(ConversationSession.Stats stats, double frameMs) {
        metrics[0].setText(String.format(Locale.US, "%.1f s", stats.latencySeconds));
        metrics[1].setText(String.format(Locale.US, "%.0f ms", stats.averageMs));
        double rtf = stats.realTimeFactor(frameMs);
        metrics[2].setText(stats.processed == 0 ? "—" : String.format(Locale.US, "%.2f", rtf));
        metrics[2].setTextColor(stats.processed == 0 ? Ui.TEXT : rtf <= 1 ? Ui.LIVE : Ui.WARN);
        metrics[3].setText(String.valueOf(stats.backlog));
        if (session != null) {
            talk.levels(stats.inputLevel, stats.outputLevel);
            sessionStatus.setTextColor(Ui.MUTED);
            sessionStatus.setText(String.format(Locale.US, "已处理 %.1f 秒输入 · 输出 %d 帧%s", stats.inputSeconds, stats.emitted,
                    stats.draining ? " · 已停止收音，正在处理剩余 " + stats.backlog + " 帧（再点一次立即结束）" : ""));
            if (stats.draining) refreshTalk();
        }
    }

    // ---------------------------------------------------------------- refresh

    private void refreshAll() {
        refreshChat();
        refreshModels();
        refreshSettings();
        boolean keepAwake = session != null || downloadingVariant != null || engineBusy();
        if (keepAwake) getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        else getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    private void refreshChat() {
        ModelCatalog.Variant variant = selectedVariant();
        if (variant == null) {
            modelName.setText("没有可用模型");
            modelVariant.setText("请在“模型”页刷新模型列表");
            headerModel.setText("全双工语音 · 端侧推理");
            Ui.enable(modelAction, false);
            refreshTalk();
            return;
        }
        modelName.setText(variant.model.name);
        modelVariant.setText(variant.title);
        headerModel.setText("全双工语音 · " + variant.model.name);
        ModelStore.Status status = store.status(variant);
        boolean loaded = variant.id.equals(loadedVariant) && engineState == EngineState.READY;
        String chip;
        int color;
        if (engineBusy()) { chip = engineState == EngineState.VERIFYING ? "校验中" : "加载中"; color = Ui.WARN; }
        else if (loaded) { chip = "已就绪"; color = Ui.LIVE; }
        else if (engineState == EngineState.ERROR) { chip = "出错"; color = Ui.ERROR; }
        else if (status.state == ModelStore.State.READY) { chip = "未加载"; color = Ui.ACCENT; }
        else { chip = "未下载"; color = Ui.MUTED; }
        modelChip.setText(chip);
        Ui.setChipColor(modelChip, color);
        String detail = engineMessage;
        if (detail.isEmpty()) detail = status.state == ModelStore.State.READY
                ? "模型约 " + Ui.bytes(variant.totalBytes()) + "，加载后全部常驻 HTP。"
                : "需要先下载约 " + Ui.bytes(variant.totalBytes()) + " 的模型文件。";
        modelDetail.setText(detail);
        modelDetail.setTextColor(engineState == EngineState.ERROR ? Ui.ERROR : Ui.MUTED);
        modelAction.setText(status.state != ModelStore.State.READY ? "前往下载" : loaded ? "重新加载" : "加载模型");
        Ui.enable(modelAction, session == null && !engineBusy() && downloadingVariant == null);
        for (int i = 0; i < 3; i++) {
            boolean active = Source.values()[i] == source;
            sources[i].setTextColor(active ? Ui.TEXT : Ui.MUTED);
            sources[i].setBackground(active ? Ui.rounded(Ui.OUTLINE, ui.dp(11)) : null);
            Ui.enable(sources[i], session == null && (i != 1 || hasDemo()));
        }
        Ui.enable(playInput, session == null && hasRecording);
        Ui.enable(playOutput, session == null && hasRecording);
        Ui.enable(exportButton, session == null && hasRecording);
        refreshTalk();
    }

    private void refreshTalk() {
        if (session != null) {
            if (session.isDraining()) talk.set(TalkButton.Mode.BUSY, "处理剩余", "再点立即结束");
            else talk.set(TalkButton.Mode.LIVE, "结束", source == Source.MICROPHONE ? "正在收音" : "正在送入音频");
            talkHint.setText("模型边听边说；实时率大于 1 时延迟会逐渐累积。外放请开启回声消除，建议戴耳机。");
        } else if (engineState == EngineState.READY && engine != null) {
            talk.set(TalkButton.Mode.IDLE, "开始对话", source == Source.MICROPHONE ? "麦克风" : source == Source.DEMO ? "演示音频" : "WAV 文件");
            talkHint.setText("点击开始，像打电话一样直接说话；再点一次结束收音。");
        } else {
            talk.set(engineBusy() ? TalkButton.Mode.BUSY : TalkButton.Mode.DISABLED, engineBusy() ? "准备中" : "未加载", "");
            talkHint.setText("先在上方加载模型。");
        }
    }

    private void refreshSettings() {
        int index = hub.endpoint.equals(HubConfig.MIRROR) ? 0 : hub.endpoint.equals(HubConfig.HUGGING_FACE) ? 1 : 2;
        if (endpointField.getTag() != null) index = (Integer) endpointField.getTag();
        for (int i = 0; i < 3; i++) Ui.setChipColor(endpointChoices[i], i == index ? Ui.ACCENT : Ui.MUTED);
        endpointField.setVisibility(index == 2 ? View.VISIBLE : View.GONE);
        codecOnly.setEnabled(session == null && !engineBusy());
    }

    private void chooseEndpoint(int index) {
        endpointField.setTag(index);
        if (index == 0) endpointField.setText(HubConfig.MIRROR);
        if (index == 1) endpointField.setText(HubConfig.HUGGING_FACE);
        refreshSettings();
    }

    private void saveHub() {
        HubConfig next = new HubConfig(endpointField.getText().toString(), repoField.getText().toString(), tokenField.getText().toString());
        String problem = next.validate();
        if (problem != null) { toastStatus(problem); return; }
        if (downloadingVariant != null) { toastStatus("下载进行中，暂停后再修改仓库"); return; }
        hub = next;
        hub.save(preferences);
        endpointField.setTag(null);
        hubStatus.setText("");
        selectTab(1);
        refreshCatalog();
    }

    // ---------------------------------------------------------------- files

    private File sessionDir() { return new File(getFilesDir(), "sessions/latest"); }
    private File pickedFile() { return new File(getFilesDir(), "picked.wav"); }

    private boolean hasDemo() {
        try (InputStream ignored = getAssets().open("demo.wav")) {
            return true;
        } catch (IOException error) {
            return false;
        }
    }

    private void play(File file) {
        if (file == null || !file.isFile()) return;
        stopPlayer();
        try {
            player = new MediaPlayer();
            player.setDataSource(file.getAbsolutePath());
            player.setOnCompletionListener(done -> stopPlayer());
            player.prepare();
            player.start();
        } catch (IOException error) {
            stopPlayer();
            toastStatus("播放失败：" + error.getMessage());
        }
    }

    private void stopPlayer() {
        if (player != null) { player.release(); player = null; }
    }

    private void exportSession() {
        if (session != null || !hasRecording) return;
        Ui.enable(exportButton, false);
        ioThread.execute(() -> {
            String message;
            try {
                String prefix = "duplex-" + new java.text.SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new java.util.Date());
                String[][] files = {{"input.wav", "audio/wav"}, {"output.wav", "audio/wav"}, {"transcript.txt", "text/plain"},
                                    {"tokens.csv", "text/csv"}, {"performance.csv", "text/csv"}};
                for (String[] file : files) {
                    File source = new File(sessionDir(), file[0]);
                    if (source.isFile()) saveDownload(source, prefix + "-" + file[0], file[1]);
                }
                message = "已保存到 Download/DuplexVoice（" + prefix + "-*）";
            } catch (Exception error) {
                message = "导出失败：" + error.getMessage();
            }
            final String status = message;
            runOnUiThread(() -> { toastStatus(status); refreshAll(); });
        });
    }

    private void saveDownload(File source, String name, String mime) throws IOException {
        ContentValues values = new ContentValues();
        values.put(MediaStore.Downloads.DISPLAY_NAME, name);
        values.put(MediaStore.Downloads.MIME_TYPE, mime);
        values.put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/DuplexVoice");
        Uri uri = getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);
        if (uri == null) throw new IOException("无法创建下载文件");
        try (InputStream input = new FileInputStream(source); OutputStream output = getContentResolver().openOutputStream(uri)) {
            if (output == null) throw new IOException("无法写入下载文件");
            byte[] buffer = new byte[65536];
            int count;
            while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
        }
    }

    private void toastStatus(String message) {
        android.widget.Toast.makeText(this, message, android.widget.Toast.LENGTH_LONG).show();
    }

    private static String timeNow() {
        return new java.text.SimpleDateFormat("HH:mm", Locale.US).format(new java.util.Date());
    }

    @Override
    protected void onDestroy() {
        if (session != null) session.abort();
        if (downloader != null) downloader.cancel();
        stopPlayer();
        final DuplexEngine old = engine;
        engine = null;
        if (old != null) engineThread.execute(old::release);
        engineThread.shutdown();
        ioThread.shutdown();
        super.onDestroy();
    }
}
