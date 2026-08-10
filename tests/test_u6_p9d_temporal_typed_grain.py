from __future__ import annotations

from pathlib import Path

from src.eval.temporal_typed_grain import load_contract, run_audit
from src.film_physics.temporal_grain import temporal_grain_realization_seeds

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p9d_temporal_typed_grain_v1.json"
PROFILE_SHA = "e542b99e052a4d3d91ca8005473e42ad9fa94546329195d052dae5b245faa361"


def test_temporal_realization_seeds_are_frame_and_layer_addressed() -> None:
    first = temporal_grain_realization_seeds(
        profile_sha256=PROFILE_SHA, seed=17, frame=0, layer_count=3
    )
    repeat = temporal_grain_realization_seeds(
        profile_sha256=PROFILE_SHA, seed=17, frame=0, layer_count=3
    )
    second = temporal_grain_realization_seeds(
        profile_sha256=PROFILE_SHA, seed=17, frame=1, layer_count=3
    )
    assert first == repeat
    assert len(set(first + second)) == 6


def test_formal_temporal_typed_grain_audit_passes() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    assert report["automatic_pass"] is True
    assert report["measurements"]["row_partition_byte_exact"] is True
