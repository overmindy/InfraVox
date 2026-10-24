package com.overmindy.duplex.model;

import android.content.Context;
import android.content.SharedPreferences;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Locale;

/**
 * Installed variants live in files/models/<variant id>/ with the repo's relative layout.
 * SHA256SUMS is written last (by the downloader or an ADB deploy), so its presence marks a complete copy;
 * the catalog's sizes and hashes are the reference. A full hash pass is remembered by size + mtime.
 */
public final class ModelStore {
    public interface Progress { void update(long done, long total, String file); }

    public enum State { MISSING, PARTIAL, READY }

    public static final class Status {
        public final State state;
        public final int missingFiles;
        public final long presentBytes;

        Status(State state, int missingFiles, long presentBytes) {
            this.state = state;
            this.missingFiles = missingFiles;
            this.presentBytes = presentBytes;
        }
    }

    // Bundles pushed by older builds of scripts/moshi_deploy.sh.
    private static final String[][] LEGACY_DIRS = {{"lm-v4-lpbq4", "moshiko-lm-v4-lpbq4-sm8850"}};

    private final Context context;
    private final SharedPreferences preferences;
    private final File root;

    public ModelStore(Context context, SharedPreferences preferences) {
        this.context = context;
        this.preferences = preferences;
        File external = context.getExternalFilesDir("models");
        root = external != null ? external : new File(context.getFilesDir(), "models");
        root.mkdirs();
    }

    public File root() { return root; }
    public File dir(ModelCatalog.Variant variant) { return new File(root, variant.id); }
    public long freeBytes() { return root.getUsableSpace(); }

    public void migrateLegacy() {
        for (String[] pair : LEGACY_DIRS) {
            File old = new File(root, pair[0]), target = new File(root, pair[1]);
            if (old.isDirectory() && !target.exists()) old.renameTo(target);
        }
    }

    public Status status(ModelCatalog.Variant variant) {
        File dir = dir(variant);
        int missing = 0;
        long present = 0;
        for (ModelCatalog.FileEntry entry : variant.files) {
            File file = new File(dir, entry.path);
            if (file.isFile() && file.length() == entry.size) present += entry.size;
            else missing++;
        }
        if (missing == 0 && new File(dir, "SHA256SUMS").isFile()) return new Status(State.READY, 0, present);
        boolean any = present > 0 || new File(dir, "SHA256SUMS").exists();
        return new Status(any ? State.PARTIAL : State.MISSING, missing, present);
    }

    public boolean verified(ModelCatalog.Variant variant) {
        return status(variant).state == State.READY
                && fingerprint(variant).equals(preferences.getString(key(variant), ""));
    }

    /** Hashes every file against the catalog; throws on the first mismatch. */
    public void verify(ModelCatalog.Variant variant, Progress progress) throws Exception {
        File dir = dir(variant);
        long done = 0, total = variant.totalBytes();
        for (ModelCatalog.FileEntry entry : variant.files) {
            File file = new File(dir, entry.path);
            if (!file.isFile() || file.length() != entry.size) throw new IOException("缺少或大小不符：" + entry.path);
            final long base = done;
            String hash = sha256(file, read -> progress.update(base + read, total, entry.path));
            if (!hash.equals(entry.sha256)) throw new IOException("SHA256 不匹配：" + entry.path);
            done += entry.size;
        }
        writeChecksums(variant);
        markVerified(variant);
    }

    void markVerified(ModelCatalog.Variant variant) {
        preferences.edit().putString(key(variant), fingerprint(variant)).apply();
    }

    void writeChecksums(ModelCatalog.Variant variant) throws IOException {
        StringBuilder sums = new StringBuilder();
        for (ModelCatalog.FileEntry entry : variant.files) sums.append(entry.sha256).append("  ").append(entry.path).append('\n');
        File target = new File(dir(variant), "SHA256SUMS");
        File temporary = new File(target.getPath() + ".tmp");
        try (OutputStream output = new FileOutputStream(temporary)) {
            output.write(sums.toString().getBytes(StandardCharsets.UTF_8));
        }
        if (!temporary.renameTo(target)) throw new IOException("无法写入 SHA256SUMS");
    }

    void clearComplete(ModelCatalog.Variant variant) {
        new File(dir(variant), "SHA256SUMS").delete();
        preferences.edit().remove(key(variant)).apply();
    }

    public void delete(ModelCatalog.Variant variant) {
        preferences.edit().remove(key(variant)).apply();
        deleteTree(dir(variant));
    }

    public ModelCatalog loadCatalog() throws Exception {
        ModelCatalog bundled;
        try (InputStream input = context.getAssets().open("catalog.json")) {
            bundled = ModelCatalog.parse(readText(input));
        }
        File cached = new File(context.getFilesDir(), "catalog.json");
        if (cached.isFile()) {
            try (InputStream input = new FileInputStream(cached)) {
                ModelCatalog fetched = ModelCatalog.parse(readText(input));
                if (fetched.revision >= bundled.revision) return fetched;
            } catch (Exception ignored) {
                cached.delete();
            }
        }
        return bundled;
    }

    public void saveCatalog(String text) throws IOException {
        File target = new File(context.getFilesDir(), "catalog.json");
        File temporary = new File(target.getPath() + ".tmp");
        try (OutputStream output = new FileOutputStream(temporary)) {
            output.write(text.getBytes(StandardCharsets.UTF_8));
        }
        if (!temporary.renameTo(target)) throw new IOException("无法保存模型列表");
    }

    interface ReadProgress { void update(long read) throws IOException; }

    static String sha256(File file, ReadProgress progress) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        long read = 0, reported = 0;
        try (InputStream input = new FileInputStream(file)) {
            byte[] buffer = new byte[1 << 20];
            int count;
            while ((count = input.read(buffer)) != -1) {
                digest.update(buffer, 0, count);
                read += count;
                if (read - reported >= (32 << 20)) { progress.update(read); reported = read; }
            }
        }
        progress.update(read);
        return hex(digest.digest());
    }

    static String hex(byte[] bytes) {
        StringBuilder out = new StringBuilder();
        for (byte b : bytes) out.append(String.format(Locale.US, "%02x", b & 255));
        return out.toString();
    }

    static String readText(InputStream input) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        byte[] buffer = new byte[65536];
        int count;
        while ((count = input.read(buffer)) != -1) {
            bytes.write(buffer, 0, count);
            if (bytes.size() > (8 << 20)) throw new IOException("模型列表过大");
        }
        return new String(bytes.toByteArray(), StandardCharsets.UTF_8);
    }

    private String fingerprint(ModelCatalog.Variant variant) {
        StringBuilder out = new StringBuilder();
        File dir = dir(variant);
        for (ModelCatalog.FileEntry entry : variant.files) {
            File file = new File(dir, entry.path);
            out.append(entry.path).append(':').append(entry.sha256).append(':').append(file.length())
                    .append(':').append(file.lastModified()).append(';');
        }
        return out.toString();
    }

    private static String key(ModelCatalog.Variant variant) { return "verified_" + variant.id; }

    private static void deleteTree(File file) {
        File[] children = file.listFiles();
        if (children != null) for (File child : children) deleteTree(child);
        file.delete();
    }
}
