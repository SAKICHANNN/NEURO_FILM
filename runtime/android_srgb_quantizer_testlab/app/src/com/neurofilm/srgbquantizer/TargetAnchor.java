package com.neurofilm.srgbquantizer;

/**
 * Minimal target-APK code anchor required by Firebase Test Lab validation.
 *
 * The quantizer and its assertions remain entirely in the instrumentation APK.
 */
public final class TargetAnchor {
    private TargetAnchor() {}

    public static int protocolVersion() {
        return 1;
    }
}
