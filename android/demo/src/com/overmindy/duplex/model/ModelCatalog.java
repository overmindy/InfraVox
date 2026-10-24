package com.overmindy.duplex.model;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** catalog.json at the root of the model repository (source: android/models/catalog.json). */
public final class ModelCatalog {
    public static final int SCHEMA = 1;

    public static final class FileEntry {
        public final String path;
        public final long size;
        public final String sha256;

        FileEntry(JSONObject json) throws JSONException {
            path = json.getString("path");
            size = json.getLong("size");
            sha256 = json.getString("sha256").toLowerCase(java.util.Locale.US);
            if (path.startsWith("/") || path.contains("..")) throw new JSONException("Unsafe file path: " + path);
        }
    }

    public static final class Variant {
        public final Model model;
        public final String id, title, path, htp, qairt, notes;
        public final List<String> socs;
        public final List<FileEntry> files;

        Variant(Model model, JSONObject json) throws JSONException {
            this.model = model;
            id = json.getString("id");
            title = json.optString("title", id);
            path = json.getString("path");
            htp = json.optString("htp", "");
            qairt = json.optString("qairt", "");
            notes = json.optString("notes", "");
            socs = strings(json.optJSONArray("socs"));
            List<FileEntry> entries = new ArrayList<>();
            JSONArray array = json.getJSONArray("files");
            for (int i = 0; i < array.length(); i++) entries.add(new FileEntry(array.getJSONObject(i)));
            files = Collections.unmodifiableList(entries);
            if (id.contains("/") || id.contains("..")) throw new JSONException("Unsafe variant id: " + id);
        }

        public long totalBytes() {
            long total = 0;
            for (FileEntry file : files) total += file.size;
            return total;
        }

        public boolean supports(String soc) {
            return socs.isEmpty() || socs.contains(soc);
        }
    }

    public static final class Model {
        public final String id, engine, name, publisher, summary, license, upstream;
        public final List<String> languages;
        public final List<Variant> variants;
        public final boolean recommended;  // default pick for a fresh install, and tagged in the list

        Model(JSONObject json) throws JSONException {
            id = json.getString("id");
            engine = json.getString("engine");
            name = json.optString("name", id);
            publisher = json.optString("publisher", "");
            summary = json.optString("summary", "");
            license = json.optString("license", "");
            upstream = json.optString("upstream", "");
            languages = strings(json.optJSONArray("languages"));
            recommended = json.optBoolean("recommended", false);
            List<Variant> list = new ArrayList<>();
            JSONArray array = json.getJSONArray("variants");
            for (int i = 0; i < array.length(); i++) list.add(new Variant(this, array.getJSONObject(i)));
            variants = Collections.unmodifiableList(list);
        }
    }

    public final List<Model> models;
    /** Bumped on every catalog change, so a newer catalog bundled in the APK beats an older cached one. */
    public final int revision;

    private ModelCatalog(List<Model> models, int revision) {
        this.models = models;
        this.revision = revision;
    }

    public static ModelCatalog parse(String text) throws JSONException {
        JSONObject json = new JSONObject(text);
        if (json.optInt("schema", 0) > SCHEMA) throw new JSONException("模型列表格式较新，请升级 App");
        List<Model> list = new ArrayList<>();
        JSONArray array = json.getJSONArray("models");
        for (int i = 0; i < array.length(); i++) list.add(new Model(array.getJSONObject(i)));
        return new ModelCatalog(Collections.unmodifiableList(list), json.optInt("revision", 0));
    }

    public Variant variant(String id) {
        for (Model model : models)
            for (Variant variant : model.variants)
                if (variant.id.equals(id)) return variant;
        return null;
    }

    private static List<String> strings(JSONArray array) throws JSONException {
        List<String> out = new ArrayList<>();
        if (array != null) for (int i = 0; i < array.length(); i++) out.add(array.getString(i));
        return Collections.unmodifiableList(out);
    }
}
