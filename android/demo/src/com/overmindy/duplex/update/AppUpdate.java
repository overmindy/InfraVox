package com.overmindy.duplex.update;

import android.app.Activity;
import android.app.PendingIntent;
import android.content.Intent;
import android.content.pm.PackageInstaller;
import android.os.Build;

import com.overmindy.duplex.model.HubConfig;
import com.overmindy.duplex.model.HubDownloader;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * App updates. The newest release is described by app/latest.json in the official model repo, read through
 * the configured Hugging Face endpoint first (hf-mirror.com works in mainland China), then the other HF
 * endpoint, then the GitHub Releases API. The APK downloads from the same sources in that order.
 */
public final class AppUpdate {
    public static final String GITHUB_REPO = "overmindy/InfraVox";
    public static final String RELEASES_PAGE = "https://github.com/" + GITHUB_REPO + "/releases/latest";
    public static final String MANIFEST = "app/latest.json";
    public static final String ACTION_INSTALL_STATUS = "com.overmindy.duplex.INSTALL_STATUS";

    public static final class Release {
        public final int versionCode;  // -1 when only the GitHub release answered
        public final String versionName, notes, sha256, source;
        public final long size;
        public final List<String> urls;

        Release(int versionCode, String versionName, String notes, long size, String sha256, String source, List<String> urls) {
            this.versionCode = versionCode;
            this.versionName = versionName;
            this.notes = notes;
            this.size = size;
            this.sha256 = sha256;
            this.source = source;
            this.urls = Collections.unmodifiableList(urls);
        }

        public boolean newerThan(int code, String name) {
            return versionCode >= 0 ? versionCode > code : compareVersions(versionName, name) > 0;
        }

        public String fileName() { return "infravox-" + versionName + ".apk"; }
    }

    private AppUpdate() { }

    public static Release check(String preferredEndpoint) throws IOException {
        List<String> endpoints = endpoints(preferredEndpoint);
        IOException last = null;
        for (String endpoint : endpoints) {
            try {
                HubConfig hub = new HubConfig(endpoint, HubConfig.DEFAULT_REPO);
                return fromManifest(new JSONObject(HubDownloader.fetchText(hub, MANIFEST)), hub.label(), endpoints);
            } catch (IOException error) {
                last = error;
            } catch (JSONException error) {
                last = new IOException("更新信息格式错误：" + error.getMessage());
            }
        }
        try {
            return fromGitHub(new JSONObject(HubDownloader.fetchUrl(
                    "https://api.github.com/repos/" + GITHUB_REPO + "/releases/latest", "GitHub Releases")));
        } catch (IOException | JSONException error) {
            throw new IOException("无法获取更新信息：" + (last != null ? last.getMessage() : error.getMessage()));
        }
    }

    /** Downloads the APK, falling through the release's sources; a partial file resumes on the next source. */
    public static File download(Release release, File dir, HubDownloader downloader, HubDownloader.Progress progress)
            throws Exception {
        File target = new File(dir, release.fileName());
        File[] stale = dir.listFiles();
        if (stale != null)
            for (File file : stale) if (!file.getName().startsWith(release.fileName())) file.delete();
        if (target.isFile() && target.length() == release.size) return target;
        Exception last = null;
        for (String url : release.urls) {
            try {
                downloader.download(url, release.fileName(), release.size, release.sha256, target, progress);
                return target;
            } catch (InterruptedException | java.io.InterruptedIOException paused) {
                throw paused;
            } catch (Exception error) {
                last = error;
            }
        }
        throw last != null ? last : new IOException("没有可用的下载地址");
    }

    /** Streams the APK into a PackageInstaller session; the result arrives as ACTION_INSTALL_STATUS. */
    public static void install(Activity activity, File apk) throws IOException {
        PackageInstaller installer = activity.getPackageManager().getPackageInstaller();
        PackageInstaller.SessionParams params = new PackageInstaller.SessionParams(PackageInstaller.SessionParams.MODE_FULL_INSTALL);
        params.setAppPackageName(activity.getPackageName());
        int id = installer.createSession(params);
        try (PackageInstaller.Session session = installer.openSession(id)) {
            try (InputStream input = new FileInputStream(apk); OutputStream output = session.openWrite("base.apk", 0, apk.length())) {
                byte[] buffer = new byte[1 << 20];
                int count;
                while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
                session.fsync(output);
            }
            Intent intent = new Intent(ACTION_INSTALL_STATUS).setPackage(activity.getPackageName());
            int flags = PendingIntent.FLAG_UPDATE_CURRENT | (Build.VERSION.SDK_INT >= 31 ? PendingIntent.FLAG_MUTABLE : 0);
            session.commit(PendingIntent.getBroadcast(activity, id, intent, flags).getIntentSender());
        } catch (IOException | RuntimeException error) {
            installer.abandonSession(id);
            throw error;
        }
    }

    private static List<String> endpoints(String preferred) {
        List<String> endpoints = new ArrayList<>();
        String trimmed = preferred.trim().replaceAll("/+$", "");
        if (!trimmed.isEmpty()) endpoints.add(trimmed);
        for (String known : new String[]{HubConfig.MIRROR, HubConfig.HUGGING_FACE})
            if (!endpoints.contains(known)) endpoints.add(known);
        return endpoints;
    }

    private static Release fromManifest(JSONObject json, String source, List<String> endpoints) throws JSONException {
        JSONObject apk = json.getJSONObject("apk");
        String path = apk.getString("path");
        List<String> urls = new ArrayList<>();
        for (String endpoint : endpoints) urls.add(new HubConfig(endpoint, HubConfig.DEFAULT_REPO).resolve(path));
        String github = json.optString("github", "");
        if (!github.isEmpty()) urls.add(github);
        return new Release(json.getInt("versionCode"), json.getString("versionName"), json.optString("notes", ""),
                apk.getLong("size"), apk.optString("sha256", ""), source, urls);
    }

    private static Release fromGitHub(JSONObject json) throws JSONException, IOException {
        JSONArray assets = json.getJSONArray("assets");
        for (int i = 0; i < assets.length(); i++) {
            JSONObject asset = assets.getJSONObject(i);
            if (!asset.getString("name").endsWith(".apk")) continue;
            String digest = asset.optString("digest", "");
            return new Release(-1, json.getString("tag_name").replaceFirst("^v", ""), json.optString("body", ""),
                    asset.getLong("size"), digest.startsWith("sha256:") ? digest.substring(7) : "", "GitHub",
                    Collections.singletonList(asset.getString("browser_download_url")));
        }
        throw new IOException("GitHub 最新发布里没有 APK");
    }

    static int compareVersions(String a, String b) {
        String[] x = a.split("[.-]"), y = b.split("[.-]");
        for (int i = 0; i < Math.max(x.length, y.length); i++) {
            int left = i < x.length ? number(x[i]) : 0, right = i < y.length ? number(y[i]) : 0;
            if (left != right) return Integer.compare(left, right);
        }
        return 0;
    }

    private static int number(String part) {
        try {
            return Integer.parseInt(part.replaceAll("\\D.*$", ""));
        } catch (NumberFormatException error) {
            return 0;
        }
    }
}
