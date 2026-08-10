from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.gate_weave_ou import load_contract, run_audit
from src.film_physics.gate_weave import (
    GateWeaveDomainError,
    GateWeaveProfile,
    advance_gate_weave,
    generate_gate_weave,
    initial_gate_weave_state,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p9a_gate_weave_ou_v1.json"


def _profile() -> GateWeaveProfile:
    payload = load_contract(CONTRACT)["profile"]
    return GateWeaveProfile(
        **{key: value for key, value in payload.items() if key != "schema"}
    )


def test_gate_weave_partition_matches_full_trajectory_exactly() -> None:
    profile = _profile()
    full = generate_gate_weave(profile, 257)
    state = initial_gate_weave_state(profile)
    parts = []
    for count in (13, 71, 172):
        segment, state = advance_gate_weave(
            profile, total_frame_count=257, state=state, transition_count=count
        )
        parts.append(segment.x_pixels)
    assert np.array_equal(np.concatenate(parts), full.x_pixels[1:])
    assert state.frame_index == 256


def test_gate_weave_is_bounded_and_rejects_invalid_requests() -> None:
    profile = _profile()
    result = generate_gate_weave(profile, 128)
    assert np.max(np.abs(result.x_pixels)) <= profile.padding_x_pixels
    assert np.max(np.abs(result.y_pixels)) <= profile.padding_y_pixels
    with pytest.raises(GateWeaveDomainError):
        generate_gate_weave(profile, 0)
    with pytest.raises(GateWeaveDomainError):
        advance_gate_weave(
            profile,
            total_frame_count=4,
            state=initial_gate_weave_state(profile),
            transition_count=4,
        )


def test_formal_gate_weave_audit_passes_frozen_gates() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    assert report["automatic_pass"] is True
    assert report["measurements"]["partition_byte_exact"] is True
    assert report["measurements"]["random_walk_late_to_early_variance_ratio"] > 1.75
