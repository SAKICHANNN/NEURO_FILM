from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_scanner_profile import _profile
from src.eval.physical_scanner_streaming import (
    evaluate_scanner_streaming,
    load_contract,
)
from src.film_physics import (
    apply_scanner_profile,
    apply_scanner_profile_row_tiled,
    compile_scanner_context,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p6b_scanner_streaming_v1.json"
P6A = ROOT / "configs" / "u6_p6a_scanner_profile_boundary_v1.json"


def test_scanner_context_closes_tile_local_global_mean() -> None:
    p6a = json.loads(P6A.read_text(encoding="utf-8"))
    profile = _profile(p6a["profiles"]["scanner_a"])
    rng = np.random.default_rng(7)
    values = rng.uniform(1e-4, 1.0, size=(301, 97, 3))
    context = compile_scanner_context(values, profile)
    full = apply_scanner_profile(
        values, profile, pixel_pitch_um=1.0, context=context
    )
    tiled = apply_scanner_profile_row_tiled(
        values,
        profile,
        pixel_pitch_um=1.0,
        context=context,
        tile_rows=31,
    )
    assert np.array_equal(full, tiled)


def test_scanner_context_rejects_unbound_input_shape() -> None:
    p6a = json.loads(P6A.read_text(encoding="utf-8"))
    profile = _profile(p6a["profiles"]["scanner_a"])
    values = np.full((20, 30, 3), 0.5, dtype=np.float64)
    context = compile_scanner_context(values, profile)
    with pytest.raises(ValueError):
        apply_scanner_profile_row_tiled(
            values[:19],
            profile,
            pixel_pitch_um=1.0,
            context=context,
            tile_rows=5,
        )


def test_frozen_scanner_streaming_contract_passes() -> None:
    report = evaluate_scanner_streaming(
        load_contract(CONTRACT),
        json.loads(P6A.read_text(encoding="utf-8")),
    )
    assert report["automatic_pass"] is True
    assert report["stable_evidence_id"] == (
        "4a29efa8b839c3b119c449ba56211550206a53684970f5c0457a7035e64535cf"
    )
    assert all(report["decisions"].values())
