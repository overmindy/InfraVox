package com.overmindy.duplex.model;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.security.MessageDigest;

/**
 * Resumable downloads from a Hugging Face compatible endpoint. Each file streams into "<file>.part"
 * with HTTP Range resume, is hashed on the fly, and is renamed only when its SHA-256 matches the catalog.
 */
public final class HubDownloader {
    public interface Progress { void update(long done, long total, String file, double bytesPerSecond); }

    private volatile boolean cancelled;

    public void cancel() { cancelled = true; }

    public static String fetchText(HubConfig hub, String path) throws IOException {
        HttpURLConnection connection = open(hub.resolve(path), 0);
        try {
            int code = connection.getResponseCode();
            if (code != 200) throw httpError(code, path);
            try (InputStream input = connection.getInputStream()) {
                return ModelStore.readText(input);
            }
        } finally {
            connection.disconnect();
        }
    }

    public void download(HubConfig hub, ModelStore store, ModelCatalog.Variant variant, Progress progress) throws Exception {
        cancelled = false;
        store.clearComplete(variant);
        File dir = store.dir(variant);
        long total = variant.totalBytes(), done = 0;
        Meter meter = new Meter(progress, total);
        for (ModelCatalog.FileEntry entry : variant.files) {
            File target = new File(dir, entry.path);
            final long base = done;
            if (target.isFile() && target.length() == entry.size) {
                String hash = ModelStore.sha256(target, read -> meter.report(base + read, "校验 " + entry.path, false));
                if (hash.equals(entry.sha256)) { done += entry.size; continue; }
                target.delete();
            }
            downloadFile(hub, variant.path + "/" + entry.path, entry, target, base, meter);
            done += entry.size;
        }
        store.writeChecksums(variant);
        store.markVerified(variant);
        meter.report(total, "完成", false);
    }

    private void downloadFile(HubConfig hub, String repoPath, ModelCatalog.FileEntry entry, File target, long base,
                              Meter meter) throws Exception {
        File partial = new File(target.getPath() + ".part");
        File parent = target.getParentFile();
        if (parent != null && !parent.isDirectory() && !parent.mkdirs()) throw new IOException("无法创建目录：" + parent);
        if (partial.length() > entry.size) partial.delete();

        for (int attempt = 0; ; attempt++) {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            long offset = hashExisting(partial, digest);
            try {
                if (offset < entry.size) fetchInto(hub, repoPath, entry, partial, offset, digest, base, meter);
                break;
            } catch (IOException error) {
                if (cancelled || attempt >= 4) throw error;
                Thread.sleep(2000L * (attempt + 1));
            }
        }
        if (cancelled) throw new InterruptedException("已暂停");
        if (partial.length() != entry.size) throw new IOException("下载不完整：" + entry.path);
        String hash = ModelStore.sha256(partial, read -> { });
        if (!hash.equals(entry.sha256)) {
            partial.delete();
            throw new IOException("SHA256 不匹配，已删除重下：" + entry.path);
        }
        if (!partial.renameTo(target)) throw new IOException("无法保存：" + entry.path);
    }

    private void fetchInto(HubConfig hub, String repoPath, ModelCatalog.FileEntry entry, File partial, long offset,
                           MessageDigest digest, long base, Meter meter) throws IOException, InterruptedException {
        HttpURLConnection connection = open(hub.resolve(repoPath), offset);
        try {
            int code = connection.getResponseCode();
            boolean append = code == 206;
            if (code == 416 && offset == entry.size) return;
            if (code != 200 && code != 206) throw httpError(code, repoPath);
            if (!append && offset > 0) {
                digest.reset();
                offset = 0;
            }
            long written = offset;
            try (InputStream input = connection.getInputStream();
                 OutputStream output = new FileOutputStream(partial, append)) {
                byte[] buffer = new byte[256 * 1024];
                int count;
                while ((count = input.read(buffer)) != -1) {
                    if (cancelled) throw new InterruptedException("已暂停");
                    output.write(buffer, 0, count);
                    written += count;
                    if (written > entry.size) throw new IOException("服务器返回的数据比预期大：" + entry.path);
                    meter.report(base + written, entry.path, true);
                }
            }
        } finally {
            connection.disconnect();
        }
    }

    private static long hashExisting(File partial, MessageDigest digest) throws IOException {
        if (!partial.isFile()) return 0;
        long read = 0;
        try (InputStream input = new FileInputStream(partial)) {
            byte[] buffer = new byte[1 << 20];
            int count;
            while ((count = input.read(buffer)) != -1) { digest.update(buffer, 0, count); read += count; }
        }
        return read;
    }

    // Redirects are followed by hand so the Range header survives the hop to the CDN.
    private static HttpURLConnection open(String url, long offset) throws IOException {
        URL current = new URL(url);
        for (int hop = 0; hop < 10; hop++) {
            HttpURLConnection connection = (HttpURLConnection) current.openConnection();
            connection.setInstanceFollowRedirects(false);
            connection.setConnectTimeout(20_000);
            connection.setReadTimeout(60_000);
            connection.setRequestProperty("User-Agent", "InfraVox-Android/1.0");
            connection.setRequestProperty("Accept-Encoding", "identity");
            if (offset > 0) connection.setRequestProperty("Range", "bytes=" + offset + "-");
            int code = connection.getResponseCode();
            if (code < 300 || code >= 400 || code == 304) return connection;
            String location = connection.getHeaderField("Location");
            connection.disconnect();
            if (location == null) throw new IOException("重定向缺少 Location（HTTP " + code + "）");
            current = new URL(current, location);
        }
        throw new IOException("重定向次数过多：" + url);
    }

    private static IOException httpError(int code, String path) {
        if (code == 401 || code == 403) return new IOException("无权访问（HTTP " + code + "）：仓库不存在或不是公开仓库");
        if (code == 404) return new IOException("仓库里找不到 " + path + "（HTTP 404），请检查仓库名");
        return new IOException("下载 " + path + " 失败：HTTP " + code);
    }

    private final class Meter {
        private final Progress progress;
        private final long total;
        private long lastReport, windowStart, windowBytes;
        private double speed;

        Meter(Progress progress, long total) {
            this.progress = progress;
            this.total = total;
            windowStart = System.nanoTime();
        }

        void report(long done, String file, boolean network) throws InterruptedIOException {
            if (cancelled) throw new InterruptedIOException("已暂停");
            long now = System.nanoTime();
            if (network) {
                if (windowBytes == 0) { windowBytes = done; windowStart = now; }
                double seconds = (now - windowStart) / 1e9;
                if (seconds >= 1.0) {
                    speed = (done - windowBytes) / seconds;
                    windowBytes = done;
                    windowStart = now;
                }
            }
            if (now - lastReport < 200_000_000L && done < total) return;
            lastReport = now;
            progress.update(done, total, file, network ? speed : 0);
        }
    }
}
