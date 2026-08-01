from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_base_return_geometry import evaluate_geometry, load_contract
from src.film_physics.base_return_geometry import (
    BaseReturnGeometryProfile,
    apply_positive_base_return_geometry,
    base_return_geometry_kernel,
    equal_second_moment_gaussian,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p3q1_analytical_base_return_geometry_v1.json"


def _profile() -> BaseReturnGeometryProfile:
    return BaseReturnGeometryProfile(120.0, 1.5, 1200.0, 1200.0, 8.0)


def test_kernel_is_annular_normalized_and_distinct() -> None:
    profile = _profile()
    kernel = base_return_geometry_kernel(profile)
    gaussian = equal_second_moment_gaussian(kernel, profile.pixel_pitch_um)
    centre = kernel.shape[0] // 2
    assert kernel[centre, centre] == 0.0
    assert float(np.min(kernel)) == 0.0
    assert float(np.sum(kernel)) == pytest.approx(1.0, abs=1e-15)
    assert float(np.sum(np.abs(kernel - gaussian))) > 0.5


def test_positive_spread_preserves_constant_and_is_repeat_exact() -> None:
    kernel = base_return_geometry_kernel(_profile())
    constant = np.full((33, 35, 3), 0.18, dtype=np.float64)
    first = apply_positive_base_return_geometry(constant, kernel, (0.065, 0.018, 0.006))
    second = apply_positive_base_return_geometry(constant, kernel, (0.065, 0.018, 0.006))
    assert np.array_equal(first.output, second.output)
    assert np.max(np.abs(first.residual)) <= 1e-14


def test_contract_rejects_posthoc_topology_gate_change(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["automatic_gates"]["minimum_l1_distance_from_equal_moment_gaussian"] = 0.1
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        load_contract(path)


def test_formal_geometry_is_repeatable_and_passes() -> None:
    config = load_contract(CONTRACT)
    first, first_arrays = evaluate_geometry(config, ROOT)
    second, second_arrays = evaluate_geometry(config, ROOT)
    assert first == second
    assert first["automatic_pass"]
    assert all(first["checks"].values())
    for key in first_arrays:
        assert np.array_equal(first_arrays[key], second_arrays[key])
