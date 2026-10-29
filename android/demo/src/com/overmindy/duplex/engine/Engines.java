package com.overmindy.duplex.engine;

/** Maps a catalog model's "engine" field to its runtime. Register new duplex models here. */
public final class Engines {
    private Engines() { }

    public static boolean supports(String engine) {
        return "moshi".equals(engine) || "minicpmo".equals(engine);
    }

    /** Engines that can see photos (when the bundle has a vision graph; see DuplexEngine.supportsImages). */
    public static boolean takesImages(String engine) {
        return "minicpmo".equals(engine);
    }

    public static DuplexEngine create(String engine) {
        if ("moshi".equals(engine)) return new MoshiEngine();
        if ("minicpmo".equals(engine)) return new MiniCpmoEngine();
        throw new IllegalArgumentException("此版本 App 不支持引擎：" + engine);
    }
}
