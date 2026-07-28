from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.film_physics.native_profile import (
    NATIVE_PRINT_ABI,
    build_native_print_oracle,
    compile_native_print_profile_payload,
    native_print_payload_sha256,
    validate_native_print_profile_payload,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


def _artifact() -> dict:
    config = json.loads(
        (
            ROOT / "configs/u6_p8b_artifact_only_cpu_consumer_v1.json"
        ).read_text()
    )
    return compile_standalone_profile_artifact(root=ROOT, config=config)


def test_native_profile_is_exact_hash_bound_subset() -> None:
    artifact = _artifact()
    first = compile_native_print_profile_payload(artifact)
    second = compile_native_print_profile_payload(artifact)
    assert first == second
    assert first["abi"] == NATIVE_PRINT_ABI
    assert len(native_print_payload_sha256(first)) == 64
    binding = next(
        row
        for row in artifact["film_profile_bundle"]["components"]
        if row["component_id"] == "physical-chain-4000dpi"
    )
    assert first["source_component"]["sha256"] == binding["sha256"]
    assert (
        first["operator"]
        == artifact["component_payloads"]["physical-chain-4000dpi"][
            "print_operator"
        ]
    )


def test_native_profile_rejects_provenance_and_capacity_drift() -> None:
    artifact = _artifact()
    payload = compile_native_print_profile_payload(artifact)
    forged = copy.deepcopy(payload)
    forged["source_component"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        validate_native_print_profile_payload(forged, artifact=artifact)
    oversized = copy.deepcopy(payload)
    spline = oversized["operator"]["sensitometry"]["curves"][0][
        "spline"
    ]
    spline["x_knots"] = list(np.arange(17, dtype=float))
    spline["y_knots"] = list(np.arange(17, dtype=float))
    spline["derivatives"] = [1.0] * 17
    with pytest.raises(ValueError, match="capacity"):
        validate_native_print_profile_payload(oversized)


def test_native_oracle_is_stable_and_bounded() -> None:
    payload = compile_native_print_profile_payload(_artifact())
    first = build_native_print_oracle(payload)
    second = build_native_print_oracle(payload)
    assert first == second
    output = np.asarray(first["expected_scan_linear_f64"])
    assert output.shape == (12, 3)
    assert np.all(np.isfinite(output))
    assert np.all((output >= 0.0) & (output <= 1.0))
