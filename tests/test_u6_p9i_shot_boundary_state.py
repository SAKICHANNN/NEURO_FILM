from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.shot_boundary_state import load_contract, run_audit
from src.film_physics.temporal_sequence import (
    TemporalSequenceIdentity,
    TemporalSequenceIdentityError,
    derive_temporal_stream_seed,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p9i_shot_boundary_state_v1.json"


def test_formal_shot_boundary_state_passes() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    assert report["automatic_pass"] is True
    assert report["measurements"]["cross_shot_grain_duplicate_frame_count"] == 0
    assert report["measurements"]["naive_local_frame_grain_duplicate_fraction"] == 1.0


def test_sequence_identity_is_replayable_and_role_separated() -> None:
    sequence = TemporalSequenceIdentity("shot-a")
    exposure = derive_temporal_stream_seed(
        base_seed=7, sequence=sequence, stream_role="exposure_flicker"
    )
    assert exposure == derive_temporal_stream_seed(
        base_seed=7, sequence=sequence, stream_role="exposure_flicker"
    )
    assert exposure != derive_temporal_stream_seed(
        base_seed=7, sequence=sequence, stream_role="density_grain"
    )
    assert exposure != derive_temporal_stream_seed(
        base_seed=7,
        sequence=TemporalSequenceIdentity("shot-b"),
        stream_role="exposure_flicker",
    )


@pytest.mark.parametrize(
    "identity", ["", "UPPER", " space", "ends-", "../escape", "a" * 129]
)
def test_sequence_identity_rejects_ambiguous_values(identity: str) -> None:
    with pytest.raises(TemporalSequenceIdentityError):
        TemporalSequenceIdentity(identity)
