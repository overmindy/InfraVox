package com.overmindy.duplex;

import android.Manifest;
import android.animation.ValueAnimator;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.ContentValues;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.text.InputType;
import android.text.TextUtils;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.PopupMenu;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import com.overmindy.duplex.audio.ConversationSession;
import com.overmindy.duplex.audio.Wav;
import com.overmindy.duplex.engine.DuplexEngine;
import com.overmindy.duplex.engine.Engines;
import com.overmindy.duplex.history.ConversationStore;
import com.overmindy.duplex.history.Turn;
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
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final int PICK_AUDIO = 1;
    private static final int REQUEST_MICROPHONE = 2;
    private static final double FRAME_SECONDS = 0.08;

    private enum EngineState { NONE, VERIFYING, LOADING, READY, ERROR }
    private enum Source { MICROPHONE, DEMO, FILE }
    private enum Screen { CHAT, MODELS, SETTINGS }

    private Ui ui;
    private SharedPreferences preferences;
    private ModelStore store;
    private ConversationStore conversations;
    private ModelCatalog catalog;
    private HubConfig hub;
    private final String soc = Build.VERSION.SDK_INT >= 31 ? Build.SOC_MODEL : "";
    private final Handler main = new Handler(Looper.getMainLooper());

    // The engine thread owns the native model: load, every step of a call, and release.
    private final ExecutorService engineThread = Executors.newSingleThreadExecutor();
    private final ExecutorService ioThread = Executors.newSingleThreadExecutor();

    private DuplexEngine engine;
    private String loadedVariant;
    private EngineState engineState = EngineState.NONE;
    private String engineMessage = "";
    private boolean startAfterLoad;

    private ConversationSession session;
    private ConversationStore.Meta current;  // null: a new, empty conversation
    private List<Turn> turns = new ArrayList<>();
    private ConversationSession.Stats liveStats;
    private Source source = Source.MICROPHONE;
    private MediaPlayer player;
    private final Runnable stopPlayback = this::stopPlayer;

    private HubDownloader downloader;
    private String downloadingVariant;
    private boolean catalogRefreshing;
    private final Map<String, String> variantErrors = new HashMap<>();
    private final Map<String, VariantViews> variantViews = new HashMap<>();

    private Screen screen = Screen.CHAT;
    private View chatPage, modelsPage, settingsPage, bottomBar, scrim;
    private LinearLayout drawer, drawerList, chatList;
    private ScrollView chatScroll;
    private TextView navButton, actionButton, title, subtitle, statusLine, sourceChip, callStats;
    private TalkButton talk;
    private boolean drawerOpen;

    private LinearLayout modelsList;
    private TextView hubSummary, hubStatus;
    private Button refreshButton;

    private EditText endpointField, repoField;
    private final TextView[] endpointChoices = new TextView[3];
    private LinearLayout advanced;
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
        conversations = new ConversationStore(this);
        try {
            catalog = store.loadCatalog();
        } catch (Exception error) {
            throw new IllegalStateException("内置模型列表损坏", error);
        }
        getWindow().setStatusBarColor(Ui.BACKGROUND);
        getWindow().setNavigationBarColor(Ui.BACKGROUND);
        setContentView(buildShell());
        renderModels();
        showScreen(Screen.CHAT);
        refreshAll();
    }

    // ---------------------------------------------------------------- shell, drawer, navigation

    private View buildShell() {
        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Ui.BACKGROUND);

        LinearLayout column = new LinearLayout(this);
        column.setOrientation(LinearLayout.VERTICAL);
        root.addView(column, new FrameLayout.LayoutParams(-1, -1));

        LinearLayout bar = ui.row();
        bar.setPadding(ui.dp(8), ui.dp(10), ui.dp(8), ui.dp(6));
        navButton = iconButton(bar, "☰", this::onNav);
        LinearLayout titles = ui.column(0);
        titles.setPadding(ui.dp(6), 0, ui.dp(6), 0);
        title = ui.text(titles, "", 18, Ui.TEXT, true);
        title.setSingleLine(true);
        title.setEllipsize(TextUtils.TruncateAt.END);
        subtitle = ui.text(titles, "", 12.5f, Ui.MUTED, false);
        subtitle.setSingleLine(true);
        subtitle.setEllipsize(TextUtils.TruncateAt.END);
        bar.addView(titles, new LinearLayout.LayoutParams(0, -2, 1));
        actionButton = iconButton(bar, "＋", this::onAction);
        column.addView(bar);

        statusLine = ui.text(column, "", 12.5f, Ui.LIVE, false);
        statusLine.setGravity(Gravity.CENTER);
        statusLine.setPadding(ui.dp(16), ui.dp(6), ui.dp(16), ui.dp(6));
        statusLine.setBackgroundColor(Color.argb(28, 64, 220, 160));

        FrameLayout content = new FrameLayout(this);
        chatPage = buildChatPage();
        modelsPage = scroll(buildModelsPage());
        settingsPage = scroll(buildSettingsPage());
        for (View page : new View[]{chatPage, modelsPage, settingsPage}) content.addView(page, new FrameLayout.LayoutParams(-1, -1));
        column.addView(content, new LinearLayout.LayoutParams(-1, 0, 1));
        bottomBar = buildBottomBar();
        column.addView(bottomBar);

        scrim = new View(this);
        scrim.setBackgroundColor(Color.argb(150, 0, 0, 0));
        scrim.setAlpha(0f);
        scrim.setVisibility(View.GONE);
        scrim.setOnClickListener(v -> setDrawer(false));
        root.addView(scrim, new FrameLayout.LayoutParams(-1, -1));
        drawer = buildDrawer();
        root.addView(drawer, new FrameLayout.LayoutParams(ui.dp(300), -1, Gravity.START));
        drawer.setTranslationX(-ui.dp(300));
        return root;
    }

    private TextView iconButton(LinearLayout parent, String glyph, Runnable action) {
        TextView view = ui.text(parent, glyph, 22, Ui.TEXT, false);
        view.setGravity(Gravity.CENTER);
        view.setBackground(ui.pressable(Ui.rounded(Color.TRANSPARENT, ui.dp(22)), ui.dp(22)));
        view.setOnClickListener(v -> action.run());
        view.setLayoutParams(new LinearLayout.LayoutParams(ui.dp(44), ui.dp(44)));
        return view;
    }

    private ScrollView scroll(View child) {
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.addView(child);
        return scroll;
    }

    private LinearLayout buildDrawer() {
        LinearLayout panel = ui.column(0);
        panel.setBackgroundColor(Ui.SURFACE);
        panel.setClickable(true);
        panel.setPadding(ui.dp(12), ui.dp(18), ui.dp(12), ui.dp(12));
        ui.text(panel, "Duplex", 22, Ui.TEXT, true).setPadding(ui.dp(8), 0, 0, ui.dp(10));
        ui.primary(panel, "＋  新对话", () -> { openConversation(null); setDrawer(false); });
        TextView label = ui.eyebrow(panel, "历史对话");
        label.setPadding(ui.dp(8), ui.dp(18), 0, ui.dp(6));
        drawerList = ui.column(0);
        ScrollView list = new ScrollView(this);
        list.addView(drawerList);
        panel.addView(list, new LinearLayout.LayoutParams(-1, 0, 1));
        View divider = new View(this);
        divider.setBackgroundColor(Ui.OUTLINE);
        panel.addView(divider, new LinearLayout.LayoutParams(-1, 1));
        drawerRow(panel, "模型管理", null, () -> { showScreen(Screen.MODELS); setDrawer(false); });
        drawerRow(panel, "设置", null, () -> { showScreen(Screen.SETTINGS); setDrawer(false); });
        return panel;
    }

    private TextView drawerRow(LinearLayout parent, String text, String detail, Runnable action) {
        TextView row = ui.text(parent, detail == null ? text : text + "\n" + detail, 15, Ui.TEXT, false);
        row.setPadding(ui.dp(12), ui.dp(12), ui.dp(12), ui.dp(12));
        row.setBackground(ui.pressable(Ui.rounded(Color.TRANSPARENT, ui.dp(12)), ui.dp(12)));
        row.setOnClickListener(v -> action.run());
        return row;
    }

    private void renderDrawer() {
        drawerList.removeAllViews();
        List<ConversationStore.Meta> metas = conversations.list();
        if (metas.isEmpty()) {
            ui.text(drawerList, "还没有对话。", 14, Ui.MUTED, false).setPadding(ui.dp(12), ui.dp(8), 0, 0);
            return;
        }
        SimpleDateFormat format = new SimpleDateFormat("M月d日 HH:mm", Locale.CHINA);
        for (ConversationStore.Meta meta : metas) {
            String name = meta.title.isEmpty() ? "语音对话" : meta.title;
            String detail = format.format(new Date(meta.createdMillis)) + String.format(Locale.US, " · %.0f 秒", meta.seconds);
            TextView row = drawerRow(drawerList, name, detail, () -> { openConversation(meta); setDrawer(false); });
            row.setSingleLine(false);
            row.setMaxLines(2);
            row.setEllipsize(TextUtils.TruncateAt.END);
            if (current != null && current.id.equals(meta.id)) row.setBackground(Ui.rounded(Ui.SURFACE_HIGH, ui.dp(12)));
            row.setOnLongClickListener(v -> { confirmDelete(meta); return true; });
        }
    }

    private void setDrawer(boolean open) {
        if (open && session != null) { toast("通话中不能切换对话"); return; }
        drawerOpen = open;
        if (open) { renderDrawer(); scrim.setVisibility(View.VISIBLE); }
        float from = drawer.getTranslationX(), to = open ? 0 : -drawer.getWidth();
        ValueAnimator animator = ValueAnimator.ofFloat(from, to).setDuration(220);
        animator.addUpdateListener(a -> {
            float x = (float) a.getAnimatedValue();
            drawer.setTranslationX(x);
            scrim.setAlpha(1 + x / Math.max(1, drawer.getWidth()));
            if (!open && x <= to) scrim.setVisibility(View.GONE);
        });
        animator.start();
    }

    private void showScreen(Screen next) {
        screen = next;
        chatPage.setVisibility(next == Screen.CHAT ? View.VISIBLE : View.GONE);
        modelsPage.setVisibility(next == Screen.MODELS ? View.VISIBLE : View.GONE);
        settingsPage.setVisibility(next == Screen.SETTINGS ? View.VISIBLE : View.GONE);
        bottomBar.setVisibility(next == Screen.CHAT ? View.VISIBLE : View.GONE);
        navButton.setText(next == Screen.CHAT ? "☰" : "←");
        refreshAll();
    }

    private void onNav() {
        if (screen != Screen.CHAT) showScreen(Screen.CHAT);
        else setDrawer(!drawerOpen);
    }

    private void onAction() {
        if (screen != Screen.CHAT || session != null) return;
        if (current == null) return;
        PopupMenu menu = new PopupMenu(this, actionButton);
        menu.getMenu().add(0, 1, 0, "新对话");
        menu.getMenu().add(0, 2, 1, "导出音频与文本");
        menu.getMenu().add(0, 3, 2, "删除此对话");
        menu.setOnMenuItemClickListener(item -> {
            if (item.getItemId() == 1) openConversation(null);
            if (item.getItemId() == 2) exportConversation(current);
            if (item.getItemId() == 3) confirmDelete(current);
            return true;
        });
        menu.show();
    }

    @Override
    public void onBackPressed() {
        if (drawerOpen) setDrawer(false);
        else if (screen != Screen.CHAT) showScreen(Screen.CHAT);
        else super.onBackPressed();
    }

    // ---------------------------------------------------------------- chat

    private View buildChatPage() {
        chatScroll = new ScrollView(this);
        chatScroll.setFillViewport(true);
        chatList = ui.column(0);
        chatList.setPadding(ui.dp(14), ui.dp(8), ui.dp(14), ui.dp(16));
        chatScroll.addView(chatList);
        return chatScroll;
    }

    private View buildBottomBar() {
        LinearLayout bar = ui.row();
        bar.setPadding(ui.dp(14), ui.dp(8), ui.dp(14), ui.dp(12));
        bar.setBackgroundColor(Ui.BACKGROUND);
        sourceChip = ui.chip(null, "", Ui.TEXT);
        sourceChip.setTextSize(13);
        sourceChip.setPadding(ui.dp(12), ui.dp(8), ui.dp(12), ui.dp(8));
        sourceChip.setOnClickListener(v -> chooseSourceMenu());
        LinearLayout left = ui.row();
        left.addView(sourceChip);
        bar.addView(left, new LinearLayout.LayoutParams(0, -2, 1));
        talk = new TalkButton(this, 96);
        talk.setOnClickListener(v -> onTalk());
        bar.addView(talk, new LinearLayout.LayoutParams(ui.dp(110), -2));
        callStats = ui.text(null, "", 12, Ui.MUTED, false);
        callStats.setGravity(Gravity.END | Gravity.CENTER_VERTICAL);
        bar.addView(callStats, new LinearLayout.LayoutParams(0, -2, 1));
        return bar;
    }

    private void openConversation(ConversationStore.Meta meta) {
        if (session != null) return;
        stopPlayer();
        current = meta;
        turns = meta == null ? new ArrayList<>() : conversations.turns(meta.id);
        liveStats = null;
        renderChat();
        refreshAll();
    }

    private void renderChat() {
        chatList.removeAllViews();
        if (turns.isEmpty() && session == null) {
            renderEmptyState();
            return;
        }
        boolean live = session != null;
        for (Turn turn : turns) renderBubble(turn, live);
        if (live && turns.isEmpty()) {
            TextView waiting = ui.text(chatList, "正在聆听…直接说话，模型会边听边说。", 14, Ui.MUTED, false);
            waiting.setGravity(Gravity.CENTER);
            waiting.setPadding(0, ui.dp(40), 0, 0);
        }
        if (!live && current != null) {
            String summary = String.format(Locale.US, "通话 %.1f 秒 · 平均每帧 %.0f ms（RTF %.2f）· %s",
                    current.seconds, current.averageMs, current.averageMs / (FRAME_SECONDS * 1000), current.modelName);
            TextView footer = ui.text(chatList, summary, 12, Ui.MUTED, false);
            footer.setGravity(Gravity.CENTER);
            footer.setPadding(0, ui.dp(18), 0, ui.dp(4));
            TextView hint = ui.text(chatList, "模型不会记住上一通对话；点下方按钮开始一通新的对话。", 12, Ui.MUTED, false);
            hint.setGravity(Gravity.CENTER);
        }
        chatScroll.post(() -> chatScroll.fullScroll(View.FOCUS_DOWN));
    }

    private void renderEmptyState() {
        LinearLayout box = ui.column(8);
        box.setGravity(Gravity.CENTER_HORIZONTAL);
        chatList.addView(box, new LinearLayout.LayoutParams(-1, -2));
        ui.spacer(box, 48);
        ModelCatalog.Variant variant = selectedVariant();
        TextView heading = ui.text(box, variant == null ? "Duplex" : variant.model.name, 26, Ui.TEXT, true);
        heading.setGravity(Gravity.CENTER);
        TextView lead = ui.text(box, "全双工语音对话，在手机上本地运行。\n点下方按钮后直接说话，模型会边听边说，你也可以随时打断它。",
                15, Ui.MUTED, false);
        lead.setGravity(Gravity.CENTER);
        lead.setPadding(0, ui.dp(8), 0, ui.dp(20));
        if (variant == null) return;

        LinearLayout card = ui.card(box);
        LinearLayout top = ui.row();
        LinearLayout names = ui.column(0);
        ui.text(names, variant.title, 14, Ui.TEXT, true);
        ui.text(names, Ui.bytes(variant.totalBytes()) + (variant.model.languages.isEmpty() ? "" : " · " + String.join("/", variant.model.languages)),
                12, Ui.MUTED, false);
        top.addView(names, new LinearLayout.LayoutParams(0, -2, 1));
        String[] chip = engineChip(variant);
        ui.chip(top, chip[0], Integer.parseInt(chip[1]));
        card.addView(top);
        if (!engineMessage.isEmpty()) {
            TextView message = ui.text(card, engineMessage, 13, engineState == EngineState.ERROR ? Ui.ERROR : Ui.MUTED, false);
            message.setPadding(0, ui.dp(8), 0, 0);
        }
        ModelStore.State state = store.status(variant).state;
        if (state != ModelStore.State.READY) {
            ui.primary(card, "下载模型（" + Ui.bytes(variant.totalBytes()) + "）", () -> showScreen(Screen.MODELS));
        } else if (engineState != EngineState.READY && !engineBusy()) {
            ui.secondary(card, "预先加载模型", this::onLoadModel);
        }
    }

    private void renderBubble(Turn turn, boolean live) {
        boolean user = turn.role == Turn.Role.USER;
        double seconds = (turn.endFrame - turn.startFrame + 1) * FRAME_SECONDS;
        LinearLayout bubble = ui.column(0);
        bubble.setPadding(ui.dp(14), ui.dp(10), ui.dp(14), ui.dp(10));
        bubble.setBackground(user ? Ui.rounded(Color.rgb(37, 49, 92), ui.dp(18)) : Ui.rounded(Ui.SURFACE_HIGH, ui.dp(18)));
        if (user) {
            ui.text(bubble, String.format(Locale.US, "🎤  你的语音 · %.1f 秒", seconds), 15, Ui.TEXT, false);
        } else {
            ui.text(bubble, turn.text.trim().isEmpty() ? "…" : turn.text.trim(), 16.5f, Ui.TEXT, false);
        }
        if (!live && current != null) {
            File dir = conversations.dir(current.id);
            // Output frame k answers input frame k + 1 (the first input frame only primes the model).
            double start = user ? turn.startFrame * FRAME_SECONDS : Math.max(0, turn.startFrame - 1) * FRAME_SECONDS;
            double length = seconds + (user ? 0.1 : 0.5);
            TextView play = ui.text(bubble, "▶ 播放", 13, Ui.ACCENT, true);
            play.setPadding(0, ui.dp(6), 0, 0);
            play.setOnClickListener(v -> playSegment(new File(dir, user ? "input.wav" : "output.wav"), start, length));
        }
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-2, -2);
        params.gravity = user ? Gravity.END : Gravity.START;
        params.topMargin = ui.dp(10);
        if (user) params.leftMargin = ui.dp(64);
        else params.rightMargin = ui.dp(40);
        chatList.addView(bubble, params);
    }

    private String[] engineChip(ModelCatalog.Variant variant) {
        boolean loaded = variant.id.equals(loadedVariant) && engineState == EngineState.READY;
        if (engineBusy()) return new String[]{engineState == EngineState.VERIFYING ? "校验中" : "加载中", String.valueOf(Ui.WARN)};
        if (loaded) return new String[]{"已就绪", String.valueOf(Ui.LIVE)};
        if (engineState == EngineState.ERROR) return new String[]{"出错", String.valueOf(Ui.ERROR)};
        if (store.status(variant).state == ModelStore.State.READY) return new String[]{"未加载", String.valueOf(Ui.ACCENT)};
        return new String[]{"未下载", String.valueOf(Ui.MUTED)};
    }

    private void chooseSourceMenu() {
        if (session != null) return;
        PopupMenu menu = new PopupMenu(this, sourceChip);
        menu.getMenu().add(0, 0, 0, "🎤 麦克风");
        if (hasDemo()) menu.getMenu().add(0, 1, 1, "♪ 演示音频");
        menu.getMenu().add(0, 2, 2, "📄 选择 WAV 文件（24 kHz 单声道）");
        menu.setOnMenuItemClickListener(item -> {
            if (item.getItemId() == 2) {
                Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                intent.setType("audio/*");
                startActivityForResult(intent, PICK_AUDIO);
            } else {
                source = item.getItemId() == 0 ? Source.MICROPHONE : Source.DEMO;
                refreshAll();
            }
            return true;
        });
        menu.show();
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
                    message = String.format(Locale.US, "已选择 WAV：%.1f 秒，会按实时速度送入模型", samples.length / 24_000.0);
                }
            } catch (Exception error) {
                target.delete();
                message = "无法使用此文件：" + error.getMessage();
            }
            final String status = message;
            runOnUiThread(() -> {
                if (pickedFile().isFile()) source = Source.FILE;
                toast(status);
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
        ModelCatalog.Variant variant = selectedVariant();
        if (variant == null || store.status(variant).state != ModelStore.State.READY) { showScreen(Screen.MODELS); return; }
        if (source == Source.MICROPHONE
                && checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_MICROPHONE);
            return;
        }
        if (engineState == EngineState.READY && engine != null && variant.id.equals(loadedVariant)) startSession();
        else if (!engineBusy()) { startAfterLoad = true; loadEngine(variant); }
    }

    @Override
    public void onRequestPermissionsResult(int request, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(request, permissions, results);
        if (request != REQUEST_MICROPHONE) return;
        if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) onTalk();
        else toast("需要麦克风权限；也可以在左下角改用演示音频或 WAV 文件");
    }

    private void startSession() {
        if (session != null || engine == null) return;
        stopPlayer();
        ModelCatalog.Variant variant = selectedVariant();
        ConversationStore.Meta meta = conversations.create(variant.model.name, variant.id);
        ConversationSession.Config config = new ConversationSession.Config();
        config.highPerformance = highPerformance.isChecked();
        config.echoCancel = echoCancel.isChecked();
        config.textTemperature = textSampling.isChecked() ? 0.7f : 0f;
        config.directory = conversations.dir(meta.id);
        try {
            if (source == Source.DEMO) {
                try (InputStream input = getAssets().open("demo.wav")) { config.fileInput = Wav.readMono16(input, engine.sampleRate()); }
            } else if (source == Source.FILE) {
                try (InputStream input = new FileInputStream(pickedFile())) { config.fileInput = Wav.readMono16(input, engine.sampleRate()); }
            }
        } catch (IOException error) {
            conversations.delete(meta.id);
            toast("读取音频失败：" + error.getMessage());
            return;
        }
        current = meta;
        turns = new ArrayList<>();
        liveStats = null;
        session = new ConversationSession(engine, config, new ConversationSession.Listener() {
            @Override public void onStats(ConversationSession.Stats stats) { runOnUiThread(() -> showStats(stats)); }

            @Override public void onTurns(List<Turn> snapshot) { runOnUiThread(() -> { turns = snapshot; renderChat(); }); }

            @Override
            public void onFinished(Throwable error, ConversationSession.Stats stats, List<Turn> snapshot) {
                runOnUiThread(() -> finishSession(meta, error, stats, snapshot));
            }
        });
        session.start(engineThread);
        renderChat();
        refreshAll();
    }

    private void finishSession(ConversationStore.Meta meta, Throwable error, ConversationSession.Stats stats, List<Turn> snapshot) {
        session = null;
        turns = snapshot;
        liveStats = null;
        talk.levels(0, 0);
        if (stats.processed == 0) {
            conversations.delete(meta.id);
            current = null;
        } else {
            meta.seconds = stats.inputSeconds;
            meta.averageMs = stats.averageMs;
            meta.title = ConversationStore.titleFrom(snapshot);
            if (meta.title.isEmpty())
                meta.title = "语音对话 " + new SimpleDateFormat("HH:mm", Locale.US).format(new Date(meta.createdMillis));
            try {
                conversations.save(meta, snapshot);
            } catch (Exception saveError) {
                toast("保存对话失败：" + saveError.getMessage());
            }
        }
        if (error != null) toast("对话中断：" + error.getMessage());
        renderChat();
        refreshAll();
    }

    private void showStats(ConversationSession.Stats stats) {
        liveStats = stats;
        talk.levels(stats.inputLevel, stats.outputLevel);
        refreshStatus();
        if (stats.draining) refreshTalk();
    }

    // ---------------------------------------------------------------- engine

    private boolean engineBusy() {
        return engineState == EngineState.VERIFYING || engineState == EngineState.LOADING;
    }

    private void onLoadModel() {
        ModelCatalog.Variant variant = selectedVariant();
        if (variant == null || store.status(variant).state != ModelStore.State.READY) { showScreen(Screen.MODELS); return; }
        if (!engineBusy() && session == null) loadEngine(variant);
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
                            "校验模型文件 " + (100 * done / Math.max(1, total)) + "%"));
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
                    if (startAfterLoad) { startAfterLoad = false; startSession(); }
                });
            } catch (Throwable error) {
                runOnUiThread(() -> {
                    engine = null;
                    loadedVariant = null;
                    startAfterLoad = false;
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

    // ---------------------------------------------------------------- refresh

    private void refreshAll() {
        refreshHeader();
        refreshStatus();
        refreshTalk();
        if (screen == Screen.CHAT && session == null && turns.isEmpty()) renderChat();
        refreshModels();
        refreshSettings();
        boolean keepAwake = session != null || downloadingVariant != null || engineBusy();
        if (keepAwake) getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        else getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    private void refreshHeader() {
        ModelCatalog.Variant variant = selectedVariant();
        actionButton.setVisibility(screen == Screen.CHAT && current != null && session == null ? View.VISIBLE : View.INVISIBLE);
        actionButton.setText("⋮");
        if (screen == Screen.MODELS) { title.setText("模型管理"); subtitle.setText(hub.label()); return; }
        if (screen == Screen.SETTINGS) { title.setText("设置"); subtitle.setText(""); return; }
        title.setText(current == null || current.title.isEmpty() ? (session != null ? "通话中" : "新对话") : current.title);
        if (variant == null) { subtitle.setText("没有可用模型"); return; }
        String state = engineChip(variant)[0];
        if (engineBusy()) state = engineMessage;
        subtitle.setText(variant.model.name + " · " + state);
    }

    private void refreshStatus() {
        if (session == null || screen != Screen.CHAT) { statusLine.setVisibility(View.GONE); callStats.setText(""); return; }
        statusLine.setVisibility(View.VISIBLE);
        ConversationSession.Stats s = liveStats;
        if (s == null) { statusLine.setText("● 正在接通…"); return; }
        double rtf = s.averageMs / (FRAME_SECONDS * 1000);
        String line = String.format(Locale.US, "● 通话中 %.0f 秒 · 延迟 %.1f 秒 · 每帧 %.0f ms · RTF %.2f · 积压 %d",
                s.inputSeconds, s.latencySeconds, s.averageMs, rtf, s.backlog);
        if (s.draining) line = (s.backlog > 0 ? "已停止收音，正在处理剩余 " + s.backlog + " 帧" : "已停止收音，等模型说完")
                + " · 再点一次立即结束";
        statusLine.setText(line);
        statusLine.setTextColor(s.draining ? Ui.WARN : rtf <= 1 ? Ui.LIVE : Ui.WARN);
        callStats.setText(String.format(Locale.US, "延迟\n%.1f s", s.latencySeconds));
    }

    private void refreshTalk() {
        sourceChip.setText(source == Source.MICROPHONE ? "🎤 麦克风" : source == Source.DEMO ? "♪ 演示音频" : "📄 WAV");
        Ui.enable(sourceChip, session == null);
        if (session != null) {
            if (session.isDraining()) talk.set(TalkButton.Mode.BUSY, "处理中", "");
            else talk.set(TalkButton.Mode.LIVE, "结束", "");
        } else if (engineBusy()) {
            talk.set(TalkButton.Mode.BUSY, "准备", "");
        } else {
            ModelCatalog.Variant variant = selectedVariant();
            boolean ready = variant != null && store.status(variant).state == ModelStore.State.READY;
            talk.set(ready ? TalkButton.Mode.IDLE : TalkButton.Mode.DISABLED, "通话", "");
        }
    }

    // ---------------------------------------------------------------- models page

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

    private View buildModelsPage() {
        LinearLayout page = ui.column(16);
        page.setPadding(ui.dp(16), 0, ui.dp(16), ui.dp(24));
        LinearLayout hubCard = ui.card(page);
        ui.eyebrow(hubCard, "模型来源");
        hubSummary = ui.text(hubCard, "", 14, Ui.TEXT, false);
        hubStatus = ui.text(hubCard, "", 13, Ui.MUTED, false);
        hubStatus.setPadding(0, ui.dp(4), 0, 0);
        refreshButton = ui.secondary(hubCard, "刷新模型列表", this::refreshCatalog);
        modelsList = ui.column(0);
        page.addView(modelsList);
        return page;
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
        hubSummary.setText(hub.label() + (hub.isDefaultRepo() ? " · 官方模型仓库" : "\n" + hub.repo));
        Ui.enable(refreshButton, !catalogRefreshing);
        if (catalogRefreshing) hubStatus.setText("正在获取模型列表…");
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
                message = "已更新：" + fresh.models.size() + " 个模型（" + new SimpleDateFormat("HH:mm", Locale.US).format(new Date()) + "）";
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
        if (problem != null) { toast(problem); return; }
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
        variantViews.get(variant.id).status.setText("正在连接 " + config.label() + "…");
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
        new AlertDialog.Builder(this)
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
        openConversation(null);
        showScreen(Screen.CHAT);
    }

    // ---------------------------------------------------------------- settings page

    private View buildSettingsPage() {
        LinearLayout page = ui.column(16);
        page.setPadding(ui.dp(16), 0, ui.dp(16), ui.dp(24));

        LinearLayout source = ui.card(page);
        ui.eyebrow(source, "下载源");
        LinearLayout choices = ui.row();
        String[] titles = {"HF 镜像", "Hugging Face", "自定义"};
        for (int i = 0; i < 3; i++) {
            final int index = i;
            TextView option = ui.chip(null, titles[i], Ui.MUTED);
            option.setTextSize(14);
            option.setGravity(Gravity.CENTER);
            option.setPadding(0, ui.dp(9), 0, ui.dp(9));
            option.setOnClickListener(v -> chooseEndpoint(index));
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, -2, 1);
            params.setMargins(i == 0 ? 0 : ui.dp(4), 0, i == 2 ? 0 : ui.dp(4), 0);
            choices.addView(option, params);
            endpointChoices[i] = option;
        }
        source.addView(choices);
        endpointField = ui.field(source, "https://your-mirror.example.com", hub.endpoint);
        endpointField.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        ui.text(source, "模型是公开的，下载不需要账号或令牌。国内网络建议用 hf-mirror.com。", 12, Ui.MUTED, false)
                .setPadding(0, ui.dp(8), 0, 0);

        LinearLayout runtime = ui.card(page);
        ui.eyebrow(runtime, "对话");
        highPerformance = ui.toggle(runtime, "HTP 高性能档", "通话期间请求 burst 性能投票，更耗电", preferences.getBoolean("high_performance", true));
        echoCancel = ui.toggle(runtime, "回声消除", "外放时减少模型听到自己的声音；戴耳机效果最好", preferences.getBoolean("echo_cancel", true));
        textSampling = ui.toggle(runtime, "文本采样", "温度 0.7、top-k 25（Kyutai 默认）；关闭则贪心解码，模型容易一直沉默",
                preferences.getBoolean("text_sampling", true));
        highPerformance.setOnCheckedChangeListener((b, on) -> preferences.edit().putBoolean("high_performance", on).apply());
        echoCancel.setOnCheckedChangeListener((b, on) -> preferences.edit().putBoolean("echo_cancel", on).apply());
        textSampling.setOnCheckedChangeListener((b, on) -> preferences.edit().putBoolean("text_sampling", on).apply());

        LinearLayout developer = ui.card(page);
        TextView toggle = ui.eyebrow(developer, "高级 ▸");
        toggle.setOnClickListener(v -> {
            boolean show = advanced.getVisibility() != View.VISIBLE;
            advanced.setVisibility(show ? View.VISIBLE : View.GONE);
            toggle.setText(show ? "高级 ▾" : "高级 ▸");
        });
        advanced = ui.column(0);
        advanced.setVisibility(View.GONE);
        developer.addView(advanced);
        ui.text(advanced, "模型仓库：App 默认从官方仓库 " + HubConfig.DEFAULT_REPO
                + " 下载，一般不需要改。只有自己发布了模型（scripts/hf_publish.py）时才填自己的仓库。", 12.5f, Ui.MUTED, false);
        repoField = ui.field(advanced, HubConfig.DEFAULT_REPO, hub.isDefaultRepo() ? "" : hub.repo);
        codecOnly = ui.toggle(advanced, "仅编解码回环（调试）", "只跑 Mimi encoder → decoder，不跑语言模型", preferences.getBoolean("codec_only", false));
        codecOnly.setOnCheckedChangeListener((b, on) -> {
            preferences.edit().putBoolean("codec_only", on).apply();
            if (session == null) releaseEngine("模式已切换，下次通话会重新加载");
        });

        ui.primary(page, "保存下载设置", this::saveHub);

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
        if (index == 2 && (hub.endpoint.equals(HubConfig.MIRROR) || hub.endpoint.equals(HubConfig.HUGGING_FACE))) endpointField.setText("");
        refreshSettings();
    }

    private void saveHub() {
        HubConfig next = new HubConfig(endpointField.getText().toString(), repoField.getText().toString());
        String problem = next.validate();
        if (problem != null) { toast(problem); return; }
        if (downloadingVariant != null) { toast("下载进行中，暂停后再修改"); return; }
        hub = next;
        hub.save(preferences);
        endpointField.setTag(null);
        hubStatus.setText("");
        showScreen(Screen.MODELS);
        refreshCatalog();
    }

    // ---------------------------------------------------------------- files

    private File pickedFile() { return new File(getFilesDir(), "picked.wav"); }

    private boolean hasDemo() {
        try (InputStream ignored = getAssets().open("demo.wav")) {
            return true;
        } catch (IOException error) {
            return false;
        }
    }

    private void playSegment(File file, double startSeconds, double lengthSeconds) {
        if (!file.isFile()) return;
        stopPlayer();
        try {
            player = new MediaPlayer();
            player.setDataSource(file.getAbsolutePath());
            player.setOnCompletionListener(done -> stopPlayer());
            player.prepare();
            player.seekTo((int) (startSeconds * 1000));
            player.start();
            main.postDelayed(stopPlayback, (long) (lengthSeconds * 1000));
        } catch (IOException error) {
            stopPlayer();
            toast("播放失败：" + error.getMessage());
        }
    }

    private void stopPlayer() {
        main.removeCallbacks(stopPlayback);
        if (player != null) { player.release(); player = null; }
    }

    private void confirmDelete(ConversationStore.Meta meta) {
        if (session != null) return;
        new AlertDialog.Builder(this)
                .setTitle("删除这段对话？")
                .setMessage(meta.title.isEmpty() ? "语音对话" : meta.title)
                .setNegativeButton("取消", null)
                .setPositiveButton("删除", (dialog, which) -> {
                    stopPlayer();
                    conversations.delete(meta.id);
                    if (current != null && current.id.equals(meta.id)) openConversation(null);
                    if (drawerOpen) renderDrawer();
                })
                .show();
    }

    private void exportConversation(ConversationStore.Meta meta) {
        ioThread.execute(() -> {
            String message;
            try {
                String prefix = "duplex-" + new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date(meta.createdMillis));
                String[][] files = {{"input.wav", "audio/wav"}, {"output.wav", "audio/wav"}, {"transcript.txt", "text/plain"},
                                    {"tokens.csv", "text/csv"}, {"performance.csv", "text/csv"}};
                for (String[] file : files) {
                    File source = new File(conversations.dir(meta.id), file[0]);
                    if (source.isFile()) saveDownload(source, prefix + "-" + file[0], file[1]);
                }
                message = "已保存到 Download/DuplexVoice（" + prefix + "-*）";
            } catch (Exception error) {
                message = "导出失败：" + error.getMessage();
            }
            final String status = message;
            runOnUiThread(() -> toast(status));
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

    private void toast(String message) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show();
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
