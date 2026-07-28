from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_scanner_profile import _profile
from src.eval.physical_scanner_standard import (
    deterministic_gradient,
    evaluate_standard_parity,
)
from src.film_physics import (
    apply_scanner_profile_standard_row_tiled,
    compile_scanner_standard_context,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p6c_scanner_standard_benchmark_v1.json"
P6A = ROOT / "configs" / "u6_p6a_scanner_profile_boundary_v1.json"


def test_standard_scanner_is_float32_and_partition_exact() -> None:
    p6a = json.loads(P6A.read_text(encoding="utf-8"))
    profile = _profile(p6a["profiles"]["scanner_b"])
    values = deterministic_gradient((301, 97))
    context = compile_scanner_standard_context(values, profile)
    first = apply_scanner_profile_standard_row_tiled(
        values,
        profile,
        pixel_pitch_um=1.0,
        context=context,
        tile_rows=31,
    )
    second = apply_scanner_profile_standard_row_tiled(
        values,
        profile,
        pixel_pitch_um=1.0,
        context=context,
        tile_rows=127,
    )
    assert first.dtype == np.float32
    assert np.array_equal(first, second)


def test_standard_scanner_rejects_float64_input() -> None:
    p6a = json.loads(P6A.read_text(encoding="utf-8"))
    profile = _profile(p6a["profiles"]["scanner_b"])
    values = deterministic_gradient((20, 30))
    context = compile_scanner_standard_context(values, profile)
    with pytest.raises(TypeError):
        apply_scanner_profile_standard_row_tiled(
            values.astype(np.float64),
            profile,
            pixel_pitch_um=1.0,
            context=context,
            tile_rows=10,
        )


def test_frozen_standard_parity_fixture_passes() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    p6a = json.loads(P6A.read_text(encoding="utf-8"))
    report = evaluate_standard_parity(contract, p6a)
    assert report["partitions_exact"] is True
    assert (
        report["maximum_abs_vs_float64"]
        <= contract["automatic_gates"]["parity_max_abs"]
    )
