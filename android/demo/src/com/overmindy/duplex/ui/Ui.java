package com.overmindy.duplex.ui;

import android.content.Context;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.RippleDrawable;
import android.text.TextUtils;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.Switch;
import android.widget.TextView;

/** Framework-only styling (the app is built without Gradle or AndroidX). */
public final class Ui {
    public static final int BACKGROUND = Color.rgb(9, 12, 20);
    public static final int SURFACE = Color.rgb(19, 24, 36);
    public static final int SURFACE_HIGH = Color.rgb(29, 36, 52);
    public static final int OUTLINE = Color.rgb(44, 53, 74);
    public static final int TEXT = Color.rgb(236, 240, 247);
    public static final int MUTED = Color.rgb(140, 152, 175);
    public static final int ACCENT = Color.rgb(110, 142, 255);
    public static final int LIVE = Color.rgb(64, 220, 160);
    public static final int WARN = Color.rgb(255, 184, 84);
    public static final int ERROR = Color.rgb(255, 110, 110);

    private final Context context;

    public Ui(Context context) { this.context = context; }

    public int dp(float value) { return Math.round(context.getResources().getDisplayMetrics().density * value); }

    public static GradientDrawable rounded(int color, float radiusPx) {
        GradientDrawable shape = new GradientDrawable();
        shape.setColor(color);
        shape.setCornerRadius(radiusPx);
        return shape;
    }

    public static GradientDrawable outlined(int fill, int stroke, int strokePx, float radiusPx) {
        GradientDrawable shape = rounded(fill, radiusPx);
        shape.setStroke(strokePx, stroke);
        return shape;
    }

    public Drawable pressable(Drawable content, float radiusPx) {
        return new RippleDrawable(ColorStateList.valueOf(Color.argb(60, 255, 255, 255)), content, rounded(Color.WHITE, radiusPx));
    }

    public LinearLayout column(int paddingDp) {
        LinearLayout layout = new LinearLayout(context);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(dp(paddingDp), dp(paddingDp), dp(paddingDp), dp(paddingDp));
        return layout;
    }

    public LinearLayout row() {
        LinearLayout layout = new LinearLayout(context);
        layout.setOrientation(LinearLayout.HORIZONTAL);
        layout.setGravity(Gravity.CENTER_VERTICAL);
        layout.setBaselineAligned(false);
        return layout;
    }

    public LinearLayout card(ViewGroup parent) {
        LinearLayout card = column(16);
        card.setBackground(outlined(SURFACE, OUTLINE, 1, dp(20)));
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, -2);
        params.topMargin = dp(12);
        parent.addView(card, params);
        return card;
    }

    public TextView text(ViewGroup parent, String value, float sp, int color, boolean bold) {
        TextView view = new TextView(context);
        view.setText(value);
        view.setTextColor(color);
        view.setTextSize(sp);
        view.setLineSpacing(0, 1.15f);
        if (bold) view.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
        if (parent != null) parent.addView(view);
        return view;
    }

    public TextView eyebrow(ViewGroup parent, String value) {
        TextView view = text(parent, value, 12, MUTED, true);
        view.setLetterSpacing(0.08f);
        view.setAllCaps(true);
        view.setPadding(0, 0, 0, dp(6));
        return view;
    }

    public TextView chip(ViewGroup parent, String value, int color) {
        TextView view = text(parent, value, 12, color, true);
        view.setPadding(dp(10), dp(4), dp(10), dp(4));
        view.setSingleLine(true);
        view.setEllipsize(TextUtils.TruncateAt.END);
        setChipColor(view, color);
        return view;
    }

    public static void setChipColor(TextView chip, int color) {
        chip.setTextColor(color);
        chip.setBackground(rounded(Color.argb(38, Color.red(color), Color.green(color), Color.blue(color)), 999));
    }

    public Button primary(ViewGroup parent, String title, Runnable action) {
        return button(parent, title, action, ACCENT, Color.rgb(8, 12, 24));
    }

    public Button secondary(ViewGroup parent, String title, Runnable action) {
        return button(parent, title, action, SURFACE_HIGH, TEXT);
    }

    public Button danger(ViewGroup parent, String title, Runnable action) {
        return button(parent, title, action, Color.rgb(58, 28, 34), ERROR);
    }

    private Button button(ViewGroup parent, String title, Runnable action, int fill, int textColor) {
        Button button = new Button(context);
        button.setText(title);
        button.setAllCaps(false);
        button.setTextSize(15);
        button.setTextColor(textColor);
        button.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
        button.setStateListAnimator(null);
        button.setMinHeight(dp(46));
        button.setMinimumHeight(dp(46));
        button.setPadding(dp(14), 0, dp(14), 0);
        button.setBackground(pressable(rounded(fill, dp(14)), dp(14)));
        button.setOnClickListener(view -> action.run());
        if (parent != null) {
            boolean inRow = parent instanceof LinearLayout && ((LinearLayout) parent).getOrientation() == LinearLayout.HORIZONTAL;
            LinearLayout.LayoutParams params = inRow ? new LinearLayout.LayoutParams(0, dp(46), 1) : new LinearLayout.LayoutParams(-1, -2);
            if (!inRow) params.topMargin = dp(8);
            parent.addView(button, params);
        }
        return button;
    }

    /** A row of equal-width buttons, spaced like a stacked button. */
    public LinearLayout buttonRow(ViewGroup parent) {
        LinearLayout row = row();
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, -2);
        params.topMargin = dp(8);
        parent.addView(row, params);
        return row;
    }

    public View gap(LinearLayout row) {
        View gap = new View(context);
        row.addView(gap, new LinearLayout.LayoutParams(dp(8), 1));
        return gap;
    }

    public static void enable(View view, boolean enabled) {
        view.setEnabled(enabled);
        view.setAlpha(enabled ? 1f : 0.38f);
    }

    public Switch toggle(ViewGroup parent, String title, String detail, boolean checked) {
        Switch view = new Switch(context);
        view.setText(detail == null ? title : title + "\n" + detail);
        view.setTextColor(TEXT);
        view.setTextSize(15);
        view.setChecked(checked);
        view.setPadding(0, dp(10), 0, dp(10));
        view.setThumbTintList(new ColorStateList(new int[][]{{android.R.attr.state_checked}, {}}, new int[]{ACCENT, MUTED}));
        view.setTrackTintList(new ColorStateList(new int[][]{{android.R.attr.state_checked}, {}},
                new int[]{Color.argb(120, Color.red(ACCENT), Color.green(ACCENT), Color.blue(ACCENT)), OUTLINE}));
        parent.addView(view, new LinearLayout.LayoutParams(-1, -2));
        return view;
    }

    public EditText field(ViewGroup parent, String hint, String value) {
        EditText field = new EditText(context);
        field.setHint(hint);
        field.setText(value);
        field.setSingleLine(true);
        field.setTextColor(TEXT);
        field.setHintTextColor(MUTED);
        field.setTextSize(15);
        field.setPadding(dp(14), dp(12), dp(14), dp(12));
        field.setBackground(outlined(SURFACE_HIGH, OUTLINE, 1, dp(12)));
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, -2);
        params.topMargin = dp(8);
        parent.addView(field, params);
        return field;
    }

    public ProgressBar progress(ViewGroup parent) {
        ProgressBar bar = new ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal);
        bar.setMax(1000);
        bar.setProgressTintList(ColorStateList.valueOf(ACCENT));
        bar.setProgressBackgroundTintList(ColorStateList.valueOf(OUTLINE));
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, dp(6));
        params.topMargin = dp(10);
        parent.addView(bar, params);
        return bar;
    }

    public View spacer(ViewGroup parent, int heightDp) {
        View view = new View(context);
        parent.addView(view, new LinearLayout.LayoutParams(-1, dp(heightDp)));
        return view;
    }

    public static String bytes(long value) {
        if (value >= (1L << 30)) return String.format(java.util.Locale.US, "%.2f GB", value / (double) (1L << 30));
        if (value >= (1L << 20)) return String.format(java.util.Locale.US, "%.1f MB", value / (double) (1L << 20));
        return String.format(java.util.Locale.US, "%.0f KB", value / 1024.0);
    }
}
