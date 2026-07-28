from __future__ import annotations

import json

import pytest

from scripts.run_android_srgb_quantizer_emulator_v1 import (
    EMULATOR_TARGETS,
    EmulatorRuntimeError,
    execute,
    _instrumentation_result,
    _validate_host_architecture,
    _validate_instrumentation,
)


def _output(payload: dict) -> str:
    return (
        "INSTRUMENTATION_RESULT: nf_runtime_result="
        + json.dumps(payload, separators=(",", ":"))
        + "\nINSTRUMENTATION_RESULT: outer_replays=2"
        + "\nINSTRUMENTATION_RESULT: outer_replay_exact=true"
        + "\nINSTRUMENTATION_CODE: -1\n"
    )


def test_instrumentation_parser_accepts_complete_runtime() -> None:
    payload = {
        "schema": "neuro-film.android-srgb-quantizer-runtime.v1",
        "status": "PASS",
        "sample_count": 4096,
        "threshold_identity": (
            "fae645ef1aad04fcd1233631a32f820c"
            "f7696e3ca65d31939acf60d7f123674c"
        ),
        "vector_sha256": (
            "3d4205e51de80392ea7a4e5eccaf05ab"
            "6d322a48475603764c28e47d61aa7628"
        ),
        "q8_exact": True,
        "q16_exact": True,
        "inner_replay_exact": True,
        "failure_atomic": True,
        "icc_exact": True,
        "eotf_q8_roundtrip_exact": True,
        "eotf_q16_roundtrip_exact": True,
        "eotf_failure_atomic": True,
        "product_chain_vector_count": 10,
        "product_chain_canonical_bytes": 14254,
        "product_chain_hashes_exact": True,
        "product_chain_sha_failure_atomic": True,
        "staging_truth_table_exact": True,
    }
    output = _output(payload)
    assert _instrumentation_result(output) == payload
    assert _validate_instrumentation(output) == payload


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.replace("INSTRUMENTATION_CODE: -1", ""),
        lambda value: value.replace("outer_replay_exact=true", "false"),
        lambda value: value.replace('"q16_exact":true', '"q16_exact":false'),
    ],
)
def test_instrumentation_parser_fails_closed(mutation) -> None:
    complete = _output(
        {
            "schema": "neuro-film.android-srgb-quantizer-runtime.v1",
            "status": "PASS",
            "sample_count": 4096,
            "threshold_identity": (
                "fae645ef1aad04fcd1233631a32f820c"
                "f7696e3ca65d31939acf60d7f123674c"
            ),
            "vector_sha256": (
                "3d4205e51de80392ea7a4e5eccaf05ab"
                "6d322a48475603764c28e47d61aa7628"
            ),
            "q8_exact": True,
            "q16_exact": True,
            "inner_replay_exact": True,
            "failure_atomic": True,
            "icc_exact": True,
            "eotf_q8_roundtrip_exact": True,
            "eotf_q16_roundtrip_exact": True,
            "eotf_failure_atomic": True,
            "product_chain_vector_count": 10,
            "product_chain_canonical_bytes": 14254,
            "product_chain_hashes_exact": True,
            "product_chain_sha_failure_atomic": True,
            "staging_truth_table_exact": True,
        }
    )
    with pytest.raises(EmulatorRuntimeError):
        _validate_instrumentation(mutation(complete))


def test_emulator_targets_separate_abi_resources() -> None:
    assert set(EMULATOR_TARGETS) == {"x86_64", "arm64-v8a"}
    x86 = EMULATOR_TARGETS["x86_64"]
    arm = EMULATOR_TARGETS["arm64-v8a"]
    assert x86["port"] != arm["port"]
    assert x86["avd_name"] != arm["avd_name"]
    assert x86["report"] != arm["report"]
    assert x86["accel"] == "on"
    assert arm["accel"] == "off"
    assert str(arm["system_image"]).endswith("default;arm64-v8a")
    _validate_host_architecture("x86_64", machine="AMD64")
    _validate_host_architecture("arm64-v8a", machine="aarch64")
    with pytest.raises(
        EmulatorRuntimeError,
        match="requires a matching host architecture",
    ):
        _validate_host_architecture("arm64-v8a", machine="AMD64")
    with pytest.raises(EmulatorRuntimeError, match="unsupported emulator ABI"):
        execute("armeabi-v7a")
