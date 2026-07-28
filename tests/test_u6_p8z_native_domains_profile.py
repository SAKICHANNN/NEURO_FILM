from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.film_physics.native_profile import (
    NATIVE_DOMAINS_ABI,
    build_native_domains_oracle,
    compile_native_domains_profile_payload,
    validate_native_domains_profile_payload,
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


def test_native_domains_payload_preserves_operator_and_changes_abi() -> None:
    artifact = _artifact()
    payload = compile_native_domains_profile_payload(artifact)
    assert payload == compile_native_domains_profile_payload(artifact)
    assert payload["abi"] == NATIVE_DOMAINS_ABI
    assert (
        payload["operator"]
        == artifact["component_payloads"]["physical-chain-4000dpi"][
            "print_operator"
        ]
    )


def test_native_domains_payload_rejects_identity_and_provenance_drift() -> None:
    artifact = _artifact()
    payload = compile_native_domains_profile_payload(artifact)
    wrong_abi = copy.deepcopy(payload)
    wrong_abi["abi"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        validate_native_domains_profile_payload(wrong_abi)
    forged = copy.deepcopy(payload)
    forged["source_component"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        validate_native_domains_profile_payload(
            forged, artifact=artifact
        )


def test_native_domains_oracle_exposes_both_physical_boundaries() -> None:
    payload = compile_native_domains_profile_payload(_artifact())
    first = build_native_domains_oracle(payload)
    assert first == build_native_domains_oracle(payload)
    density = np.asarray(first["expected_developed_density_f64"])
    scan = np.asarray(first["expected_scan_linear_f64"])
    assert density.shape == scan.shape == (12, 3)
    assert np.all(np.isfinite(density))
    assert np.all(np.isfinite(scan))
    assert np.all((scan >= 0.0) & (scan <= 1.0))
