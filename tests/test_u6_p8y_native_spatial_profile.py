from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.film_physics.native_spatial_profile import (
    build_native_gaussian_oracle,
    compile_native_gaussian_profile_payload,
    native_gaussian_payload_sha256,
    validate_native_gaussian_profile_payload,
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


def test_native_gaussian_profile_is_exact_bound_subset() -> None:
    artifact = _artifact()
    payload = compile_native_gaussian_profile_payload(artifact)
    assert payload == compile_native_gaussian_profile_payload(artifact)
    assert len(native_gaussian_payload_sha256(payload)) == 64
    assert [row["stage"] for row in payload["stages"]] == [
        "forward_scatter",
        "development_adjacency",
        "dye_diffusion",
        "scanner_mtf",
    ]
    assert [row["maximum_radius"] for row in payload["stages"]] == [
        2,
        1,
        1,
        1,
    ]


def test_native_gaussian_profile_rejects_radius_and_provenance_drift() -> None:
    artifact = _artifact()
    payload = compile_native_gaussian_profile_payload(artifact)
    forged = copy.deepcopy(payload)
    forged["source_component"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        validate_native_gaussian_profile_payload(
            forged, artifact=artifact
        )
    oversized = copy.deepcopy(payload)
    oversized["stages"][0]["radius_rgb"][0] = 65
    oversized["stages"][0]["maximum_radius"] = 65
    with pytest.raises(ValueError, match="stage"):
        validate_native_gaussian_profile_payload(oversized)


def test_native_gaussian_oracle_is_stable_and_finite() -> None:
    payload = compile_native_gaussian_profile_payload(_artifact())
    first = build_native_gaussian_oracle(payload)
    second = build_native_gaussian_oracle(payload)
    assert first == second
    for output in first["expected_by_stage_f64"].values():
        values = np.asarray(output)
        assert values.shape == (11, 13, 3)
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)
