package com.overmindy.duplex.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.view.View;

/** Round start/stop control; its two rings follow the microphone (inner) and model output (outer) levels. */
public final class TalkButton extends View {
    public enum Mode { IDLE, LIVE, BUSY, DISABLED }

    private final Paint fill = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint ring = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint label = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint caption = new Paint(Paint.ANTI_ALIAS_FLAG);
    private Mode mode = Mode.DISABLED;
    private String title = "开始对话", subtitle = "";
    private float input, output, shownInput, shownOutput, phase;

    public TalkButton(Context context) {
        super(context);
        ring.setStyle(Paint.Style.STROKE);
        label.setTextAlign(Paint.Align.CENTER);
        label.setColor(Color.WHITE);
        label.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
        caption.setTextAlign(Paint.Align.CENTER);
        caption.setColor(Color.argb(200, 255, 255, 255));
    }

    public void set(Mode mode, String title, String subtitle) {
        this.mode = mode;
        this.title = title;
        this.subtitle = subtitle;
        setEnabled(mode != Mode.DISABLED);
        invalidate();
    }

    public void levels(double input, double output) {
        this.input = (float) input;
        this.output = (float) output;
        invalidate();
    }

    @Override
    protected void onMeasure(int widthSpec, int heightSpec) {
        int size = Math.round(getResources().getDisplayMetrics().density * 240);
        setMeasuredDimension(resolveSize(size, widthSpec), size);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        float density = getResources().getDisplayMetrics().density;
        float cx = getWidth() / 2f, cy = getHeight() / 2f;
        float core = 78 * density;
        boolean live = mode == Mode.LIVE;
        shownInput += ((live ? input : 0) - shownInput) * 0.35f;
        shownOutput += ((live ? output : 0) - shownOutput) * 0.25f;
        phase = (phase + 0.03f) % 1f;

        int base = mode == Mode.LIVE ? Ui.ERROR : mode == Mode.BUSY ? Ui.WARN : mode == Mode.IDLE ? Ui.ACCENT : Ui.OUTLINE;
        if (live) {
            ring.setStrokeWidth(3 * density);
            ring.setColor(withAlpha(Ui.ACCENT, (int) (60 + 160 * shownOutput)));
            canvas.drawCircle(cx, cy, core + (18 + 32 * shownOutput) * density, ring);
            ring.setStrokeWidth(6 * density);
            ring.setColor(withAlpha(Ui.LIVE, (int) (90 + 160 * shownInput)));
            canvas.drawCircle(cx, cy, core + (6 + 14 * shownInput) * density, ring);
        } else if (mode == Mode.BUSY) {
            ring.setStrokeWidth(2 * density);
            ring.setColor(withAlpha(Ui.WARN, (int) (160 * (1 - phase))));
            canvas.drawCircle(cx, cy, core + 40 * density * phase, ring);
        } else if (mode == Mode.IDLE) {
            ring.setStrokeWidth(2 * density);
            ring.setColor(withAlpha(Ui.ACCENT, 70));
            canvas.drawCircle(cx, cy, core + 12 * density, ring);
        }
        fill.setColor(base);
        canvas.drawCircle(cx, cy, core, fill);

        label.setTextSize(20 * density);
        label.setColor(mode == Mode.DISABLED ? Ui.MUTED : mode == Mode.IDLE ? Color.rgb(8, 12, 24) : Color.WHITE);
        caption.setColor(withAlpha(label.getColor(), 190));
        caption.setTextSize(12.5f * density);
        boolean two = subtitle != null && !subtitle.isEmpty();
        canvas.drawText(title, cx, cy + (two ? -2 : 7) * density, label);
        if (two) canvas.drawText(subtitle, cx, cy + 20 * density, caption);
        if (live || mode == Mode.BUSY || Math.abs(shownInput) + Math.abs(shownOutput) > 0.01f) postInvalidateOnAnimation();
    }

    private static int withAlpha(int color, int alpha) {
        return Color.argb(Math.max(0, Math.min(255, alpha)), Color.red(color), Color.green(color), Color.blue(color));
    }
}
