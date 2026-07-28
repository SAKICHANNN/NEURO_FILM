from __future__ import annotations

import json

import pytest

from scripts.run_android_srgb_quantizer_emulator_v1 import (
    EmulatorRuntimeError,
    _instrumentation_result,
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
        }
    )
    with pytest.raises(EmulatorRuntimeError):
        _validate_instrumentation(mutation(complete))
