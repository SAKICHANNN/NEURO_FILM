from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.global_chroma_gaussian_transport import (
    GlobalChromaGaussianTransportError,
    global_chroma_gaussian_transport_target,
    load_contract,
)
from src.eval.global_chroma_procrustes import _plane_basis

ROOT = Path(__file__).resolve().parents[1]


def _chroma_coordinates(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    image64 = image.astype(np.float64)
    luma = np.sum(image64 * weights, axis=-1)
    return ((image64 - luma[..., None]) @ _plane_basis(weights)).reshape(-1, 2)


def test_cb22_recovers_gaussian_mean_and_covariance() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cb22_global_chroma_gaussian_transport_v1.json"
    )
    assert contract["experiment_id"] == "U5.R2CB22"
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    basis = _plane_basis(weights)
    rng = np.random.default_rng(2200)
    base_xy = rng.normal(size=(83, 97, 2)) @ np.asarray([[0.025, 0.007], [0.0, 0.018]])
    target_xy = rng.normal(size=(83, 97, 2)) @ np.asarray(
        [[0.042, -0.006], [0.0, 0.011]]
    )
    base_xy += np.asarray([0.006, -0.003])
    target_xy += np.asarray([-0.004, 0.009])
    luma = rng.uniform(0.45, 0.55, size=(83, 97))
    base = np.asarray(luma[..., None] + base_xy @ basis.T, dtype=np.float32)
    ao6 = np.asarray(luma[..., None] + target_xy @ basis.T, dtype=np.float32)
    target = global_chroma_gaussian_transport_target(
        base,
        ao6,
        weights=weights,
        minimum_transport_eigenvalue=0.01,
        maximum_transport_eigenvalue=10.0,
    )
    actual_xy = _chroma_coordinates(target, weights)
    expected_xy = _chroma_coordinates(ao6, weights)
    np.testing.assert_allclose(
        np.mean(actual_xy, axis=0), np.mean(expected_xy, axis=0), atol=1e-8
    )
    np.testing.assert_allclose(
        np.cov(actual_xy, rowvar=False, bias=True),
        np.cov(expected_xy, rowvar=False, bias=True),
        atol=2e-9,
    )


def test_cb22_preserves_base_luminance() -> None:
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    rng = np.random.default_rng(2201)
    base = rng.uniform(0.2, 0.8, size=(31, 29, 3)).astype(np.float32)
    ao6 = rng.uniform(0.2, 0.8, size=(31, 29, 3)).astype(np.float32)
    target = global_chroma_gaussian_transport_target(
        base,
        ao6,
        weights=weights,
        minimum_transport_eigenvalue=0.01,
        maximum_transport_eigenvalue=10.0,
    )
    np.testing.assert_allclose(
        np.sum(target.astype(np.float64) * weights, axis=-1),
        np.sum(base.astype(np.float64) * weights, axis=-1),
        atol=4e-8,
    )


def test_cb22_rejects_degenerate_chroma_covariance() -> None:
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    gray = np.full((9, 11, 3), 0.5, dtype=np.float32)
    with pytest.raises(GlobalChromaGaussianTransportError, match="covariance envelope"):
        global_chroma_gaussian_transport_target(gray, gray, weights=weights)
