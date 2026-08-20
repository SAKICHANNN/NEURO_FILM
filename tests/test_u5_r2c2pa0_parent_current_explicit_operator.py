from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.eval.c2pa_parent_current_explicit_operator import (
    C2PAPairedOperatorError,
    apply_diagonal_logit_affine,
    canonical_sha256,
    evaluate,
    fit_and_score_registered,
    fit_diagonal_logit_affine,
    register_parent_current,
    verify_contract,
)
from src.eval.c2pa_parent_current_explicit_operator import (
    _aggregate as aggregate,
)
from src.eval.c2pa_parent_current_explicit_operator import (
    _resize_maximum_side as resize_maximum_side,
)
from src.roll2film.triangular_logit_transport import TriangularLogitTransport

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2c2pa0_parent_current_explicit_operator_d0_v1.json"


def _contract() -> dict:
    return json.loads(CONTRACT.read_text("utf-8"))


def _textured_rgb(height: int = 256, width: int = 320) -> np.ndarray:
    yy, xx = np.indices((height, width), dtype=np.float64)
    checker = ((xx // 13 + yy // 17) % 2) * 0.18
    return np.stack(
        (
            0.15 + 0.6 * xx / width + checker,
            0.12 + 0.65 * yy / height + 0.5 * checker,
            0.2 + 0.25 * np.sin(xx / 9.0) + 0.2 * np.cos(yy / 11.0),
        ),
        axis=-1,
    ).clip(0.02, 0.98)


def test_canonical_sha256_is_order_independent() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256(
        {"a": 1, "b": 2}
    )


def test_registration_resize_is_finite_and_cube_bounded() -> None:
    source = np.asarray([[[0.0, 0.5, 1.0], [1.0, 0.0, 0.5]]], dtype=np.float64)
    resized = resize_maximum_side(source, 32)
    assert np.isfinite(resized).all()
    assert float(resized.min()) >= 0.0
    assert float(resized.max()) <= 1.0


def test_verify_contract_binds_p210_without_pixel_decode() -> None:
    contract, sources = verify_contract(CONTRACT, ROOT)
    assert contract["parent_source_audit"]["expected_strict_rows"] == 48
    assert len(sources["rows"]) == 48


def test_registration_recovers_candidate_independent_homography() -> None:
    parent = _textured_rgb()
    transform = np.asarray(
        [[1.0, 0.0, 3.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    current = cv2.warpPerspective(
        parent,
        transform,
        (parent.shape[1], parent.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    result = register_parent_current(
        parent, current, _contract()["decode_and_registration"]
    )
    assert result["eligible"] is True
    assert result["inliers"] >= 12
    assert result["overlap_fraction"] >= 0.5


def test_registered_operator_recovers_cross_channel_effect() -> None:
    parent = _textured_rgb(256, 256)
    parameters = np.asarray(
        [
            0.18,
            0.12,
            -0.08,
            0.22,
            0.05,
            -0.16,
            0.12,
            -0.18,
            0.16,
            0.14,
            -0.03,
            0.10,
            -0.12,
            0.08,
        ]
    )
    current = TriangularLogitTransport(parameters).apply(parent)
    result = fit_and_score_registered(
        {
            "parent": parent,
            "current_aligned": current,
            "valid_mask": np.ones(parent.shape[:2], dtype=bool),
        },
        _contract(),
    )
    assert result["errors"]["candidate_rmse"] < result["errors"]["identity_rmse"]
    assert result["errors"]["candidate_rmse"] < result["errors"]["diagonal_rmse"]
    assert result["out_of_cube_fraction"] == 0.0
    assert result["new_boundary_fraction"] == 0.0


def test_diagonal_control_is_bounded() -> None:
    rng = np.random.default_rng(20260821)
    source = rng.uniform(0.05, 0.95, size=(500, 3))
    target = np.clip(source ** np.asarray([0.9, 1.1, 1.05]), 0.0, 1.0)
    operator = _contract()["operator"]
    parameters = fit_diagonal_logit_affine(
        source,
        target,
        np.asarray(operator["lower_bounds"]),
        np.asarray(operator["upper_bounds"]),
        operator["identity_shrinkage"],
        operator["maximum_fit_evaluations"],
    )
    output = apply_diagonal_logit_affine(source, parameters)
    assert np.isfinite(output).all()
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0


def test_aggregate_fails_closed_without_minimum_rows() -> None:
    rows = [{"status": "registration-rejected"} for _ in range(48)]
    result = aggregate(rows, _contract())
    assert result["automatic_pass"] is False
    assert result["failed_gates"] == ["minimum registration-eligible rows"]


def test_invalid_order_fails_before_source_scan(tmp_path: Path) -> None:
    with pytest.raises(C2PAPairedOperatorError, match="order must be canonical"):
        evaluate(CONTRACT, ROOT, tmp_path / "report.json", "invalid")
