from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
    reconstruct_standalone_runtime,
    render_standalone_profile,
    validate_contract,
    validate_standalone_profile_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p8b_artifact_only_cpu_consumer_v1.json"
)
DECISION = (
    ROOT
    / "configs"
    / "u6_p8b_artifact_only_cpu_consumer_decision_v1.json"
)


def _config() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_p8b_contract_requires_artifact_only_reconstruction() -> None:
    _, runtime, _ = validate_contract(ROOT, _config())
    assert runtime.display_look_payload[
        "source_context_scope"
    ] == "one-full-frame"


def test_standalone_artifact_is_canonical_and_reconstructs() -> None:
    first = compile_standalone_profile_artifact(
        root=ROOT, config=_config()
    )
    second = compile_standalone_profile_artifact(
        root=ROOT, config=_config()
    )
    assert first == second
    bundle = validate_standalone_profile_artifact(first)
    runtime, gauge = reconstruct_standalone_runtime(first)
    assert bundle.claim_level == "generic-physical-inspired"
    assert runtime.profile.pixel_pitch_um == pytest.approx(6.35)
    assert len(gauge.inverse_neutral_splines) == 3


def test_standalone_display_payload_tamper_fails_closed() -> None:
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=_config()
    )
    tampered = copy.deepcopy(artifact)
    tampered["component_payloads"][
        "ao6-source-context-display-look"
    ]["residual"]["tone_strength"] = 0.16
    with pytest.raises(ValueError, match="hash drift"):
        validate_standalone_profile_artifact(tampered)


def test_standalone_render_needs_only_artifact_and_pixels() -> None:
    _, reference_runtime, reference_gauge = validate_contract(
        ROOT, _config()
    )
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=_config()
    )
    source = np.random.default_rng(2026072906).random((33, 35, 3))
    from src.eval.physical_neutral_gauged_invariance import (
        render_challenger,
    )

    expected = render_challenger(
        source,
        reference_runtime,
        reference_gauge,
        sampling_dpi=4000,
    )
    actual = render_standalone_profile(artifact, source)
    assert np.array_equal(expected, actual)


def test_p8b_decision_freezes_artifact_only_exact_replay() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["software_commit"] == "b838491"
    assert decision["two_run_byte_exact"]
    assert decision["artifact_only_reconstruction"]
    assert decision["replay_exact"]
    assert decision["maximum_absolute_replay_error"] == 0.0
    assert not decision["native_runtime_opened"]
    assert decision["next_leaf"].startswith("U6.P8C")
