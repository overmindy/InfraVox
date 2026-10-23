package com.overmindy.duplex.engine;

/** Maps a catalog model's "engine" field to its runtime. Register new duplex models here. */
public final class Engines {
    private Engines() { }

    public static boolean supports(String engine) {
        return "moshi".equals(engine);
    }

    public static DuplexEngine create(String engine) {
        if ("moshi".equals(engine)) return new MoshiEngine();
        throw new IllegalArgumentException("此版本 App 不支持引擎：" + engine);
    }
}
