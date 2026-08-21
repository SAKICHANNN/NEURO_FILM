from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.dng_camera_to_pcs_native_conformance import (
    build_probes,
    python_oracle,
)


def test_p96_probe_identity_is_fixed_and_bounded() -> None:
    first = build_probes()
    second = build_probes()
    assert first.shape == (257, 3)
    assert first.dtype == np.float64
    assert first.tobytes() == second.tobytes()
    assert np.all(np.isfinite(first))
    assert np.min(first) == 0.0
    assert np.max(first) == 4.0


def test_p96_python_oracle_preserves_identity() -> None:
    probes = build_probes()
    result = python_oracle(np.eye(3, dtype=np.float64), probes)
    np.testing.assert_array_equal(result, probes)


def test_p96_config_binds_parent_and_exact_triplet_count() -> None:
    config = json.loads(
        Path("configs/p96_dng_camera_to_pcs_portable_abi_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["bindings"]["p94_report"]["sha256"] == (
        "f5a4aeb31070efcc7bddd9ff485dccadaa6b04e95d0251483c4bc1c24a1ae216"
    )
    assert config["gates"]["required_profiles"] == 5
    assert config["gates"]["required_transformed_triplets"] == 5 * 257
