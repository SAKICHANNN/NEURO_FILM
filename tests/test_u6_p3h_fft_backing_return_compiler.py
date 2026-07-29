from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_fft_backing_return import (
    P3D_SCHEMA,
    P3H_SCHEMA,
    evaluate_fft_backing_return,
    load_json,
)
from src.film_physics import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    apply_fft_backing_return,
    apply_fft_backing_return_row_tiled,
    backing_return_profile_from_contract,
    compile_backing_return_profile,
)


ROOT = Path(__file__).resolve().parents[1]
P3D = ROOT / "configs" / "u6_p3d_backing_return_reference_v1.json"
P3H = ROOT / "configs" / "u6_p3h_fft_backing_return_compiler_v1.json"


def _input(values: np.ndarray, pitch: float = 8.0) -> PhysicalDomainArray:
    return PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
        PhysicalScale(pitch),
    )


def test_fft_candidate_requires_exact_float32_domain_and_scale() -> None:
    reference = backing_return_profile_from_contract(load_json(P3D, P3D_SCHEMA))
    compiled = compile_backing_return_profile(reference)
    with pytest.raises(TypeError, match="float32"):
        apply_fft_backing_return(
            _input(np.ones((5, 7, 3), dtype=np.float64)), compiled
        )
    wrong_domain = PhysicalDomainArray(
        np.ones((5, 7, 3), dtype=np.float32),
        PhysicalDomain.SCAN_LINEAR,
        PhysicalUnit.RELATIVE_SCAN_SIGNAL,
        ("red", "green", "blue"),
        PhysicalScale(8.0),
    )
    with pytest.raises(ValueError, match="domain mismatch"):
        apply_fft_backing_return(wrong_domain, compiled)
    with pytest.raises(ValueError, match="scale"):
        apply_fft_backing_return(
            _input(np.ones((5, 7, 3), dtype=np.float32), 10.0), compiled
        )


def test_fft_row_tiling_is_tolerance_close_and_rejects_invalid_rows() -> None:
    reference = backing_return_profile_from_contract(load_json(P3D, P3D_SCHEMA))
    compiled = compile_backing_return_profile(reference)
    values = np.random.default_rng(43).random((257, 263, 3), dtype=np.float32)
    source = _input(values)
    full = apply_fft_backing_return(source, compiled).values
    tiled = apply_fft_backing_return_row_tiled(
        source, compiled, tile_rows=71
    ).values
    assert float(np.max(np.abs(full - tiled))) <= 2e-6
    with pytest.raises(ValueError, match="positive integer"):
        apply_fft_backing_return_row_tiled(source, compiled, tile_rows=0)


def test_frozen_fft_report_passes_and_repeats() -> None:
    first = evaluate_fft_backing_return(
        load_json(P3D, P3D_SCHEMA), load_json(P3H, P3H_SCHEMA)
    )
    second = evaluate_fft_backing_return(
        load_json(P3D, P3D_SCHEMA), load_json(P3H, P3H_SCHEMA)
    )
    assert first == second
    assert first["automatic_pass"] is True
    assert all(first["decisions"].values())
