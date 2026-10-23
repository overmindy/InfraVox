package com.overmindy.duplex.history;

import org.json.JSONException;
import org.json.JSONObject;

/** One chat bubble: a stretch of the user's speech or of the model's reply, in input-frame indices. */
public final class Turn {
    public enum Role { USER, MODEL }

    public final Role role;
    public final int startFrame;
    public int endFrame;
    public String text;

    public Turn(Role role, int startFrame, int endFrame, String text) {
        this.role = role;
        this.startFrame = startFrame;
        this.endFrame = endFrame;
        this.text = text;
    }

    public Turn copy() { return new Turn(role, startFrame, endFrame, text); }

    JSONObject toJson() throws JSONException {
        return new JSONObject().put("role", role.name()).put("start", startFrame).put("end", endFrame).put("text", text);
    }

    static Turn fromJson(JSONObject json) throws JSONException {
        return new Turn(Role.valueOf(json.getString("role")), json.getInt("start"), json.getInt("end"), json.optString("text", ""));
    }
}
