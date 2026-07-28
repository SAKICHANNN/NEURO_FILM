from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

import numpy as np
import pytest

from scripts.build_android_srgb_quantizer_testlab_v1 import (
    DEFAULT_NDK,
    DEFAULT_SDK,
    HARNESS,
    VECTOR_COUNT,
    VECTOR_PREFIX,
    _vectors,
    build,
    encode_vector_header,
)


def test_frozen_android_vector_matches_independent_oracle() -> None:
    linear, q8, q16, identities = _vectors()
    assert linear.shape == (VECTOR_COUNT,)
    assert q8.dtype == np.uint8
    assert q16.dtype == np.uint16
    payload = (
        VECTOR_PREFIX
        + linear.view(np.uint32).astype(">u4", copy=False).tobytes()
        + q8.tobytes()
        + q16.astype(">u2", copy=False).tobytes()
    )
    assert hashlib.sha256(payload).hexdigest() == (
        "3d4205e51de80392ea7a4e5eccaf05ab"
        "6d322a48475603764c28e47d61aa7628"
    )
    assert identities == {
        "threshold_identity": (
            "fae645ef1aad04fcd1233631a32f820c"
            "f7696e3ca65d31939acf60d7f123674c"
        ),
        "vector_sha256": hashlib.sha256(payload).hexdigest(),
        "input_sha256": hashlib.sha256(linear.tobytes()).hexdigest(),
        "q8_sha256": hashlib.sha256(q8.tobytes()).hexdigest(),
        "q16_sha256": hashlib.sha256(q16.tobytes()).hexdigest(),
    }
    header, header_identities = encode_vector_header()
    assert header_identities == identities
    assert f"#define NF_SRGB_QUANTIZER_VECTOR_COUNT {VECTOR_COUNT}u" in header
    java = (
        HARNESS
        / "test/src/com/neurofilm/srgbquantizer/QuantizerInstrumentation.java"
    ).read_text(encoding="utf-8")
    assert identities["threshold_identity"][:32] in java
    assert identities["vector_sha256"][:32] in java


def test_android_package_builds_and_binds_exact_apks(tmp_path: Path) -> None:
    if not DEFAULT_SDK.is_dir() or not DEFAULT_NDK.is_dir():
        pytest.skip("Android SDK/NDK are unavailable")
    first = build(DEFAULT_SDK, DEFAULT_NDK, tmp_path / "first")
    second = build(DEFAULT_SDK, DEFAULT_NDK, tmp_path / "second")
    assert first["status"] == "PASS"
    assert first["schema"] == (
        "neuro-film.android-srgb-quantizer-testlab-package.v1"
    )
    assert first["package_identity"] == second["package_identity"]
    assert (
        "runtime/android_srgb_quantizer_testlab/app/src/com/neurofilm/"
        "srgbquantizer/TargetAnchor.java"
    ) in first["sources"]
    assert first["native"] == second["native"]
    assert first["vector"] == second["vector"]
    assert first["target"]["device_form"] == "PHYSICAL"
    assert first["target"]["test_model"] == "shiba"
    assert first["target"]["test_version"] == "34"
    assert first["manifest_validation"] == {
        "app_package": "com.neurofilm.srgbquantizer.target",
        "test_package": "com.neurofilm.srgbquantizer.test",
        "runner": (
            "com.neurofilm.srgbquantizer.QuantizerInstrumentation"
        ),
        "runner_target": "com.neurofilm.srgbquantizer.target",
        "aapt2_badging_and_xmltree_checked": True,
    }
    assert first["runtime_protocol"]["outer_replays"] == 2
    test_apk = Path(first["artifacts"]["test"]["path"])
    app_apk = Path(first["artifacts"]["app"]["path"])
    assert hashlib.sha256(app_apk.read_bytes()).hexdigest() == (
        first["artifacts"]["app"]["sha256"]
    )
    with zipfile.ZipFile(app_apk) as archive:
        app_names = set(archive.namelist())
    assert "classes.dex" in app_names
    assert hashlib.sha256(test_apk.read_bytes()).hexdigest() == (
        first["artifacts"]["test"]["sha256"]
    )
    with zipfile.ZipFile(test_apk) as archive:
        names = set(archive.namelist())
    assert "classes.dex" in names
    assert (
        "lib/arm64-v8a/libnf_srgb_quantizer_testlab.so" in names
    )
    assert (
        "lib/arm64-v8a/"
        "libneuro_film_srgb_oetf_quantize_arm64-v8a.so"
    ) in names
