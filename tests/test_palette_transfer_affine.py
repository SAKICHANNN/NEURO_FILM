from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2x0_palette_transfer_affine_audit import run_audit
from src.roll2film.palette_transfer_affine import (
    apply_palette_transfer_direct,
    collapse_palette_transfer_affine,
    generalized_affine_barycentric_weights,
)


def _tetrahedron() -> np.ndarray:
    return np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float64,
    )


def _config() -> dict:
    root = Path(__file__).resolve().parents[1]
    return json.loads(
        (
            root
            / "configs"
            / "u5_r2x0_palette_transfer_affine_collapse_v1.json"
        ).read_text(encoding="utf-8")
    )


def test_barycentric_weights_reconstruct_and_sum_to_one() -> None:
    source = _tetrahedron()
    weights = generalized_affine_barycentric_weights(source, source)
    assert np.allclose(weights @ source, source, atol=1e-14, rtol=0.0)
    assert np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-14, rtol=0.0)


def test_published_direct_path_is_exactly_one_affine_operator() -> None:
    rng = np.random.default_rng(29500)
    source = rng.uniform(0.05, 0.95, size=(7, 3))
    target = rng.uniform(0.05, 0.95, size=(7, 3))
    identity = np.eye(7)
    transport = 0.4 * identity + 0.6 * np.roll(identity, 1, axis=1)
    colours = rng.uniform(0.0, 1.0, size=(1000, 3))
    direct = apply_palette_transfer_direct(
        source, target, transport, colours
    )
    collapsed = collapse_palette_transfer_affine(
        source, target, transport
    ).apply(colours)
    assert np.allclose(direct, collapsed, atol=1e-14, rtol=0.0)


def test_valid_transport_can_reverse_orientation() -> None:
    palette = _tetrahedron()
    transport = np.eye(4)[[0, 3, 2, 1]]
    operator = collapse_palette_transfer_affine(
        palette, palette, transport
    )
    assert np.array_equal(np.sum(transport, axis=0), np.ones(4))
    assert np.array_equal(np.sum(transport, axis=1), np.ones(4))
    assert np.isclose(np.linalg.det(operator.matrix), -1.0)


def test_negative_barycentric_extrapolation_can_exit_rgb_cube() -> None:
    target = _tetrahedron()
    source = 0.4 + 0.2 * target
    operator = collapse_palette_transfer_affine(
        source, target, np.eye(4)
    )
    assert np.allclose(operator.matrix, 5.0 * np.eye(3), atol=1e-12)
    assert np.allclose(operator.bias, -2.0, atol=1e-12)
    corners = np.stack(
        np.meshgrid(*([np.array([0.0, 1.0])] * 3), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    output = operator.apply(corners)
    assert np.isclose(np.min(output), -2.0)
    assert np.isclose(np.max(output), 3.0)


def test_palette_and_transport_contract_fail_closed() -> None:
    target = _tetrahedron()
    with pytest.raises(ValueError):
        collapse_palette_transfer_affine(
            np.zeros((4, 3)), target, np.eye(4)
        )
    invalid = np.eye(4)
    invalid[0, 0] = 0.5
    with pytest.raises(ValueError):
        collapse_palette_transfer_affine(target, target, invalid)
    negative = np.eye(4)
    negative[0, 0] = -1.0
    negative[0, 1] = 2.0
    with pytest.raises(ValueError):
        collapse_palette_transfer_affine(target, target, negative)


def test_frozen_runner_reproduces_theorem_and_both_witnesses() -> None:
    config = _config()
    first = run_audit(config, config_sha256="a" * 64, software_commit="commit")
    second = run_audit(
        copy.deepcopy(config),
        config_sha256="a" * 64,
        software_commit="commit",
    )
    assert first == second
    assert first["decision_branch_before_repeat"] == "analytic_affine_only_close"
    assert all(first["gate_results"].values())
    assert all(first["expected_witnesses_match"].values())
    assert first["images_accessed"] == 0
