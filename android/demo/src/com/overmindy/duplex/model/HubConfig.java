package com.overmindy.duplex.model;

import android.content.SharedPreferences;

import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;

/**
 * Where models come from: a Hugging Face compatible endpoint (official or mirror) and the public model repo,
 * always on its main branch. Downloads are anonymous; the repo is only configurable for development.
 */
public final class HubConfig {
    public static final String HUGGING_FACE = "https://huggingface.co";
    public static final String MIRROR = "https://hf-mirror.com";
    public static final String DEFAULT_REPO = "ELAINAy/duplex-voice-models";
    public static final String REVISION = "main";

    public final String endpoint, repo;

    public HubConfig(String endpoint, String repo) {
        String trimmed = endpoint.trim();
        while (trimmed.endsWith("/")) trimmed = trimmed.substring(0, trimmed.length() - 1);
        this.endpoint = trimmed;
        this.repo = repo.trim().isEmpty() ? DEFAULT_REPO : repo.trim();
    }

    public static HubConfig load(SharedPreferences preferences) {
        return new HubConfig(preferences.getString("hub_endpoint", MIRROR), preferences.getString("hub_repo", DEFAULT_REPO));
    }

    public void save(SharedPreferences preferences) {
        preferences.edit().putString("hub_endpoint", endpoint).putString("hub_repo", repo).remove("hub_token").apply();
    }

    public boolean isDefaultRepo() { return repo.equals(DEFAULT_REPO); }

    public String label() {
        if (endpoint.equals(HUGGING_FACE)) return "Hugging Face";
        if (endpoint.equals(MIRROR)) return "HF 镜像 hf-mirror.com";
        return endpoint.replaceFirst("^https?://", "");
    }

    public String resolve(String path) {
        StringBuilder url = new StringBuilder(endpoint).append('/').append(repo).append("/resolve/").append(REVISION);
        for (String segment : path.split("/")) url.append('/').append(encode(segment));
        return url.toString();
    }

    public String validate() {
        if (!endpoint.startsWith("https://") && !endpoint.startsWith("http://")) return "下载源需以 https:// 开头";
        if (!repo.matches("[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")) return "仓库格式应为 用户名/仓库名";
        return null;
    }

    private static String encode(String segment) {
        try {
            return URLEncoder.encode(segment, "UTF-8").replace("+", "%20");
        } catch (UnsupportedEncodingException error) {
            throw new IllegalStateException(error);
        }
    }
}
