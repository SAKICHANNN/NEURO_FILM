package com.neurofilm.srgbquantizer;

import android.app.Activity;
import android.app.Instrumentation;
import android.os.Bundle;

public final class QuantizerInstrumentation extends Instrumentation {
    private static final String THRESHOLD_ID =
            "fae645ef1aad04fcd1233631a32f820cf"
            + "7696e3ca65d31939acf60d7f123674c";
    private static final String VECTOR_ID =
            "3d4205e51de80392ea7a4e5eccaf05ab"
            + "6d322a48475603764c28e47d61aa7628";

    static {
        System.loadLibrary("nf_srgb_quantizer_testlab");
    }

    private static native String nativeRun();

    private static void require(String result, String token) {
        if (!result.contains(token)) {
            throw new IllegalStateException("missing runtime token: " + token);
        }
    }

    @Override
    public void onCreate(Bundle arguments) {
        super.onCreate(arguments);
        start();
    }

    @Override
    public void onStart() {
        Bundle result = new Bundle();
        try {
            String first = nativeRun();
            String second = nativeRun();
            require(first, "\"schema\":\"neuro-film.android-srgb-quantizer-runtime.v1\"");
            require(first, "\"status\":\"PASS\"");
            require(first, "\"sample_count\":4096");
            require(first, "\"threshold_identity\":\"" + THRESHOLD_ID + "\"");
            require(first, "\"vector_sha256\":\"" + VECTOR_ID + "\"");
            require(first, "\"q8_exact\":true");
            require(first, "\"q16_exact\":true");
            require(first, "\"inner_replay_exact\":true");
            require(first, "\"failure_atomic\":true");
            require(first, "\"icc_exact\":true");
            require(first, "\"eotf_q8_roundtrip_exact\":true");
            require(first, "\"eotf_q16_roundtrip_exact\":true");
            require(first, "\"eotf_failure_atomic\":true");
            require(first, "\"product_chain_vector_count\":10");
            require(first, "\"product_chain_canonical_bytes\":14254");
            require(first, "\"product_chain_hashes_exact\":true");
            require(first, "\"product_chain_sha_failure_atomic\":true");
            require(first, "\"staging_truth_table_exact\":true");
            if (!first.equals(second)) {
                throw new IllegalStateException("outer replay differs");
            }
            result.putString("nf_runtime_result", first);
            result.putInt("outer_replays", 2);
            result.putString("outer_replay_exact", "true");
            finish(Activity.RESULT_OK, result);
        } catch (Throwable error) {
            result.putString("nf_runtime_error", error.toString());
            finish(Activity.RESULT_CANCELED, result);
        }
    }
}
