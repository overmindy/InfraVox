package com.overmindy.duplex.engine;

import android.content.Context;
import android.util.Log;

import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Android 17 NPU Manager handshake (android.npumanager.NpuManager, API 37), called by reflection so the app still
 * runs unchanged on older systems. On QSPA phones (SM8975 + Android 17) the NPU scheduler rejects FastRPC work
 * from apps that never got a grant this way: "remote work start failed: 3", then QNN deviceCreate fails with 14001.
 * Same handshake as OXR_Mobile's NpuGate.
 */
final class NpuGate {
    private static final String TAG = "InfraVox";
    private static final int SIZE_LESS_THAN_1GB = 0;
    private static final int PRIORITY_NORMAL = 0;
    private static final long TIMEOUT_MS = 10_000;

    private final Object request;
    private final Object listener;
    final String status;

    private NpuGate(Object request, Object listener, String status) {
        this.request = request;
        this.listener = listener;
        this.status = status;
    }

    /** Requests NPU access and waits for the grant; status says what happened ("no_npu_manager" before Android 17). */
    static NpuGate acquire(Context context, int requestId) {
        final Class<?> managerClass;
        try {
            managerClass = Class.forName("android.npumanager.NpuManager");
        } catch (ClassNotFoundException e) {
            return new NpuGate(null, null, "no_npu_manager");
        }
        try {
            Object manager = context.getSystemService(managerClass);
            if (manager == null) return new NpuGate(null, null, "npu_service_unavailable");
            Class<?> builderClass = Class.forName("android.npumanager.ModelLoadRequest$Builder");
            Object builder = builderClass.getConstructor(int.class).newInstance(requestId);
            builderClass.getMethod("setSize", int.class).invoke(builder, SIZE_LESS_THAN_1GB);
            builderClass.getMethod("setPriority", int.class).invoke(builder, PRIORITY_NORMAL);
            Object request = builderClass.getMethod("build").invoke(builder);

            Class<?> callbackClass = Class.forName("android.npumanager.ModelLoadRequestCallback");
            final Object[] granted = new Object[2];  // {status Integer, listener}
            final CountDownLatch latch = new CountDownLatch(1);
            Object callback = Proxy.newProxyInstance(callbackClass.getClassLoader(), new Class<?>[] {callbackClass},
                    (proxy, method, args) -> {
                        switch (method.getName()) {
                            case "onCanLoadModel":
                                granted[0] = args[1];
                                granted[1] = args[2];
                                latch.countDown();
                                break;
                            case "onRequestUnloadModel":
                                Log.w(TAG, "NPU manager asked to unload the model");
                                break;
                            case "onModelLoadRequestComplete":
                                Log.i(TAG, "NPU model load request complete, status " + args[1]);
                                break;
                            case "hashCode":
                                return System.identityHashCode(proxy);
                            case "equals":
                                return proxy == args[0];
                            case "toString":
                                return "InfraVoxNpuCallback";
                        }
                        return null;
                    });
            Method requestMethod = managerClass.getMethod("requestCanLoadModel", request.getClass(), callbackClass,
                    Executor.class);
            requestMethod.invoke(manager, request, callback, Executors.newSingleThreadExecutor());
            if (!latch.await(TIMEOUT_MS, TimeUnit.MILLISECONDS)) return new NpuGate(request, null, "npu_grant_timeout");
            // 0 = CAN_LOAD_NOW, 1 = WAIT_FOR_UNLOAD, 2 = NOT_PRIORITIZED
            NpuGate gate = new NpuGate(request, granted[1], "npu_grant_status_" + granted[0]);
            Log.i(TAG, "NPU manager: " + gate.status);
            return gate;
        } catch (Throwable t) {
            Log.e(TAG, "NPU manager handshake failed", t);
            return new NpuGate(null, null, "npu_handshake_error_" + t.getClass().getSimpleName());
        }
    }

    void notifyLoaded() { call("notifyModelLoaded"); }

    void notifyUnloaded() { call("notifyModelUnloaded"); }

    private void call(String name) {
        if (listener == null || request == null) return;
        try {
            listener.getClass().getMethod(name, request.getClass()).invoke(listener, request);
        } catch (Throwable t) {
            Log.e(TAG, "NPU manager " + name + " failed", t);
        }
    }
}
