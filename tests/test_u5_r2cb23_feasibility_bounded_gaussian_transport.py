from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.feasibility_bounded_gaussian_transport import (
    _maximum_feasible_step,
    feasibility_bounded_gaussian_transport_target,
    load_contract,
)
from src.eval.global_chroma_gaussian_transport import (
    global_chroma_gaussian_transport_target,
)
from src.eval.safe_base_ao6_chroma_direction import apply_safe_base_direction_target

ROOT = Path(__file__).resolve().parents[1]


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2300)
    source = rng.uniform(0.02, 0.98, size=(83, 97, 3)).astype(np.float32)
    base = np.asarray(0.7 * source + 0.15, dtype=np.float32)
    ao6 = np.asarray(
        np.clip(
            base.astype(np.float64)
            @ np.asarray([[1.2, -0.1, 0.0], [0.0, 0.9, 0.1], [-0.1, 0.0, 1.2]]),
            0.0,
            1.0,
        ),
        dtype=np.float32,
    )
    return source, base, ao6


def test_cb23_contract_and_global_dose() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cb23_feasibility_bounded_gaussian_transport_v1.json"
    )
    assert contract["experiment_id"] == "U5.R2CB23"
    source, base, ao6 = _fixture()
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    gaussian = global_chroma_gaussian_transport_target(
        base,
        ao6,
        weights=weights,
        minimum_transport_eigenvalue=0.01,
        maximum_transport_eigenvalue=10.0,
    )
    bounded = feasibility_bounded_gaussian_transport_target(
        source,
        base,
        ao6,
        weights=weights,
        boundary_epsilon=1.0 / 65535.0,
        minimum_transport_eigenvalue=0.01,
        maximum_transport_eigenvalue=10.0,
    )
    full_norm = np.linalg.norm(gaussian.astype(np.float64) - base, axis=-1)
    bounded_norm = np.linalg.norm(bounded.astype(np.float64) - base, axis=-1)
    ratios = bounded_norm[full_norm > 1e-7] / full_norm[full_norm > 1e-7]
    assert float(np.max(ratios) - np.min(ratios)) <= 2e-6
    assert 0.0 < float(np.median(ratios)) <= 1.0


def test_cb23_meets_per_image_subhalf_budget_by_construction() -> None:
    source, base, ao6 = _fixture()
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    epsilon = 1.0 / 65535.0
    bounded = feasibility_bounded_gaussian_transport_target(
        source,
        base,
        ao6,
        weights=weights,
        boundary_epsilon=epsilon,
        minimum_transport_eigenvalue=0.01,
        maximum_transport_eigenvalue=10.0,
    )
    _, scale, _ = apply_safe_base_direction_target(
        source,
        base,
        bounded,
        weights=weights,
        boundary_epsilon=epsilon,
    )
    assert float(np.mean(scale < 0.5)) <= 0.1


def test_cb23_feasible_step_is_bounded() -> None:
    source, base, ao6 = _fixture()
    step = _maximum_feasible_step(
        source,
        base,
        ao6,
        boundary_epsilon=1.0 / 65535.0,
    )
    assert np.isfinite(step).all()
    assert float(np.min(step)) >= 0.0
    assert float(np.max(step)) <= 1.0
