from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.temporal_grain_innovation import load_contract, run_audit
from src.film_physics.temporal_grain import (
    temporal_grain_innovation_frame,
    temporal_grain_innovation_region,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p9c_temporal_grain_innovation_v1.json"
PROFILE_SHA = "e542b99e052a4d3d91ca8005473e42ad9fa94546329195d052dae5b245faa361"


def test_temporal_grain_frame_and_tiles_are_exact() -> None:
    frame = temporal_grain_innovation_frame(
        profile_sha256=PROFILE_SHA,
        seed=17,
        frame=3,
        full_shape=(23, 19),
        layer_count=3,
    )
    tile = temporal_grain_innovation_region(
        profile_sha256=PROFILE_SHA,
        seed=17,
        frame=3,
        layer=2,
        full_shape=(23, 19),
        origin_yx=(7, 0),
        shape=(9, 19),
    )
    assert np.array_equal(tile, frame[7:16, :, 2])


def test_temporal_grain_changes_with_frame_and_layer() -> None:
    first = temporal_grain_innovation_frame(
        profile_sha256=PROFILE_SHA,
        seed=17,
        frame=3,
        full_shape=(23, 19),
        layer_count=3,
    )
    second = temporal_grain_innovation_frame(
        profile_sha256=PROFILE_SHA,
        seed=17,
        frame=4,
        full_shape=(23, 19),
        layer_count=3,
    )
    assert not np.array_equal(first, second)
    assert not np.array_equal(first[..., 0], first[..., 1])


def test_formal_temporal_grain_audit_passes() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    assert report["automatic_pass"] is True
    assert report["measurements"]["all_frame_hashes_unique"] is True
