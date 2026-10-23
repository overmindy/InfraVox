package com.overmindy.duplex.history;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Saved calls under files/conversations/<id>/: meta.json, turns.json and the session files written by
 * ConversationSession (input.wav, output.wav, transcript.txt, tokens.csv, performance.csv).
 */
public final class ConversationStore {
    public static final class Meta {
        public final String id;
        public String title, modelName, variantId;
        public long createdMillis;
        public double seconds, averageMs;

        Meta(String id) { this.id = id; }
    }

    private final File root;

    public ConversationStore(Context context) {
        root = new File(context.getFilesDir(), "conversations");
        root.mkdirs();
    }

    public File dir(String id) { return new File(root, id); }

    public Meta create(String modelName, String variantId) {
        Meta meta = new Meta(Long.toString(System.currentTimeMillis()));
        meta.createdMillis = System.currentTimeMillis();
        meta.modelName = modelName;
        meta.variantId = variantId;
        meta.title = "";
        dir(meta.id).mkdirs();
        return meta;
    }

    /** Newest first; conversations without a saved meta.json (interrupted before saving) are skipped. */
    public List<Meta> list() {
        List<Meta> out = new ArrayList<>();
        File[] dirs = root.listFiles(File::isDirectory);
        if (dirs != null) {
            for (File dir : dirs) {
                try {
                    out.add(load(dir.getName()));
                } catch (Exception ignored) {
                }
            }
        }
        out.sort((a, b) -> Long.compare(b.createdMillis, a.createdMillis));
        return out;
    }

    public Meta load(String id) throws IOException, JSONException {
        JSONObject json = new JSONObject(read(new File(dir(id), "meta.json")));
        Meta meta = new Meta(id);
        meta.title = json.optString("title", "");
        meta.modelName = json.optString("model", "");
        meta.variantId = json.optString("variant", "");
        meta.createdMillis = json.optLong("created", 0);
        meta.seconds = json.optDouble("seconds", 0);
        meta.averageMs = json.optDouble("average_ms", 0);
        return meta;
    }

    public List<Turn> turns(String id) {
        List<Turn> out = new ArrayList<>();
        try {
            JSONArray array = new JSONArray(read(new File(dir(id), "turns.json")));
            for (int i = 0; i < array.length(); i++) out.add(Turn.fromJson(array.getJSONObject(i)));
        } catch (Exception ignored) {
        }
        return out;
    }

    public void save(Meta meta, List<Turn> turns) throws IOException, JSONException {
        JSONArray array = new JSONArray();
        for (Turn turn : turns) array.put(turn.toJson());
        write(new File(dir(meta.id), "turns.json"), array.toString());
        write(new File(dir(meta.id), "meta.json"), new JSONObject().put("title", meta.title).put("model", meta.modelName)
                .put("variant", meta.variantId).put("created", meta.createdMillis).put("seconds", meta.seconds)
                .put("average_ms", meta.averageMs).toString());
    }

    public void delete(String id) { deleteTree(dir(id)); }

    /** A title from the model's first words, like a chat app's auto-named thread. */
    public static String titleFrom(List<Turn> turns) {
        for (Turn turn : turns) {
            if (turn.role != Turn.Role.MODEL || turn.text.trim().isEmpty()) continue;
            String text = turn.text.trim();
            return text.length() > 28 ? text.substring(0, 28) + "…" : text;
        }
        return "";
    }

    private static String read(File file) throws IOException {
        try (InputStream input = new FileInputStream(file)) {
            byte[] bytes = new byte[(int) file.length()];
            int offset = 0, count;
            while (offset < bytes.length && (count = input.read(bytes, offset, bytes.length - offset)) > 0) offset += count;
            return new String(bytes, 0, offset, StandardCharsets.UTF_8);
        }
    }

    private static void write(File file, String text) throws IOException {
        File temporary = new File(file.getPath() + ".tmp");
        try (OutputStream output = new FileOutputStream(temporary)) {
            output.write(text.getBytes(StandardCharsets.UTF_8));
        }
        if (!temporary.renameTo(file)) throw new IOException("无法保存 " + file.getName());
    }

    private static void deleteTree(File file) {
        File[] children = file.listFiles();
        if (children != null) for (File child : children) deleteTree(child);
        file.delete();
    }
}
