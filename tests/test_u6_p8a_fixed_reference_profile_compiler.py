from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.film_physics.profile_compiler import (
    compile_profile_artifact,
    reconstruct_runtime,
    validate_contract,
    validate_profile_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p8a_fixed_reference_profile_compiler_v1.json"
)
DECISION = (
    ROOT
    / "configs"
    / "u6_p8a_fixed_reference_profile_compiler_decision_v1.json"
)


def _config() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_p8a_contract_binds_fixed_reference_generic_claim() -> None:
    runtime, _ = validate_contract(ROOT, _config())
    assert set(_config()["replay"]["real_ids"]) <= set(
        runtime.eligible_ids
    )


def test_compiler_is_canonical_and_roundtrips() -> None:
    first = compile_profile_artifact(root=ROOT, config=_config())
    second = compile_profile_artifact(root=ROOT, config=_config())
    assert first == second
    bundle = validate_profile_artifact(first)
    assert bundle.claim_level == "generic-physical-inspired"
    assert bundle.stock_id == "unknown"
    assert bundle.process_id == "unknown"
    assert bundle.scanner_profile_id == "unknown"
    assert len(bundle.components) == 4


def test_component_tamper_fails_closed() -> None:
    artifact = compile_profile_artifact(root=ROOT, config=_config())
    tampered = copy.deepcopy(artifact)
    tampered["component_payloads"]["physical-chain-4000dpi"][
        "reference_sampling_dpi"
    ] = 3999
    with pytest.raises(ValueError, match="hash drift"):
        validate_profile_artifact(tampered)


def test_reconstruction_keeps_fixed_4000dpi_scale() -> None:
    runtime, _ = validate_contract(ROOT, _config())
    artifact = compile_profile_artifact(root=ROOT, config=_config())
    rebuilt, _ = reconstruct_runtime(artifact, runtime)
    assert rebuilt.profile.pixel_pitch_um == pytest.approx(6.35)


def test_p8a_decision_freezes_exact_formal_result() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["software_commit"] == "ec8e5ee"
    assert decision["two_run_byte_exact"]
    assert decision["artifact_exact"]
    assert decision["bundle_roundtrip_exact"]
    assert decision["replay_exact"]
    assert decision["maximum_absolute_replay_error"] == 0.0
    assert not decision[
        "independent_low_resolution_equivalence_claim_allowed"
    ]
    assert decision["next_leaf"].startswith("U6.P8B")
