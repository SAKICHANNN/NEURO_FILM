from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.temporal_exposure_flicker import load_contract, run_audit
from src.film_physics.temporal_exposure import (
    TemporalExposureDomainError,
    TemporalExposureProfile,
    advance_temporal_exposure,
    apply_temporal_exposure,
    generate_temporal_exposure,
    initial_temporal_exposure_state,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p9e_temporal_exposure_flicker_v1.json"


def _profile() -> TemporalExposureProfile:
    payload = load_contract(CONTRACT)["profile"]
    return TemporalExposureProfile(
        **{key: value for key, value in payload.items() if key != "schema"}
    )


def test_temporal_exposure_partition_matches_full_trajectory_exactly() -> None:
    profile = _profile()
    full = generate_temporal_exposure(profile, 257)
    state = initial_temporal_exposure_state()
    parts = []
    for count in (13, 71, 172):
        segment, state = advance_temporal_exposure(
            profile, total_frame_count=257, state=state, transition_count=count
        )
        parts.append(segment.offset_stops)
    assert np.array_equal(np.concatenate(parts), full.offset_stops[1:])
    assert state.frame_index == 256


def test_temporal_exposure_is_typed_bounded_and_neutral_exact() -> None:
    profile = _profile()
    result = generate_temporal_exposure(profile, 128)
    assert np.max(np.abs(result.offset_stops)) <= profile.maximum_absolute_stops
    exposure = np.linspace(0.0, 4.0, 128, dtype=np.float32)
    assert np.array_equal(apply_temporal_exposure(exposure, 0.0), exposure)
    adjusted = apply_temporal_exposure(exposure, 1.0)
    assert np.allclose(adjusted, exposure * 2.0, rtol=0.0, atol=0.0)
    with pytest.raises(TemporalExposureDomainError):
        apply_temporal_exposure(np.array([-1.0], dtype=np.float32), 0.0)
    with pytest.raises(TemporalExposureDomainError):
        generate_temporal_exposure(profile, 0)


def test_formal_temporal_exposure_audit_passes_frozen_gates() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    assert report["automatic_pass"] is True
    assert report["measurements"]["partition_byte_exact"] is True
    assert report["measurements"]["candidate_to_iid_p95_jump_ratio"] < 0.65
