package com.overmindy.duplex.history;

import java.util.ArrayList;
import java.util.List;

/**
 * Splits a duplex call into chat turns. The user speaks while the microphone frame is above an energy
 * threshold (with a short hangover); the model talks while text keeps arriving, and a pause ends its turn.
 * Frame indices are input frames, so both sides share one timeline.
 */
public final class TurnTracker {
    private static final double SPEECH_DB = -45;
    private static final int USER_HANGOVER = 8;    // 0.64 s of quiet ends the user's turn
    private static final int USER_MINIMUM = 4;     // shorter blips are noise
    private static final int MODEL_PAUSE = 15;     // 1.2 s without text ends the model's turn

    private final List<Turn> turns = new ArrayList<>();
    private Turn user, model;
    private int userQuiet, lastText;

    /** Returns true when the visible turns changed. */
    public boolean onFrame(int frame, double inputDb, String text) {
        boolean changed = false;
        if (inputDb > SPEECH_DB) {
            if (user == null) {
                user = new Turn(Turn.Role.USER, frame, frame, "");
                turns.add(user);
            }
            user.endFrame = frame;
            userQuiet = 0;
            changed = true;
        } else if (user != null && ++userQuiet > USER_HANGOVER) {
            if (user.endFrame - user.startFrame + 1 < USER_MINIMUM) turns.remove(user);
            user = null;
            changed = true;
        }
        if (!text.isEmpty()) {
            if (model == null || frame - lastText > MODEL_PAUSE) {
                model = new Turn(Turn.Role.MODEL, frame, frame, "");
                turns.add(model);
                text = text.replaceFirst("^\\s+", "");
            }
            model.text += text;
            model.endFrame = frame;
            lastText = frame;
            changed = true;
        }
        return changed;
    }

    public List<Turn> snapshot() {
        List<Turn> out = new ArrayList<>();
        for (Turn turn : turns)
            if (turn != user || turn.endFrame - turn.startFrame + 1 >= USER_MINIMUM) out.add(turn.copy());
        out.sort((a, b) -> Integer.compare(a.startFrame, b.startFrame));
        return out;
    }
}
