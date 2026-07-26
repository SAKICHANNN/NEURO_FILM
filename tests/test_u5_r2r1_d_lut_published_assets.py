from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.eval.d_lut_published_assets import (
    PublishedDLUTError,
    analyze_asset,
    canonical_lut_manifest,
    parse_cube,
    tetrahedral_jacobians,
    trilinear_jacobians,
)


def _write_cube(path: Path, values: np.ndarray) -> None:
    size = values.shape[0]
    rows = np.transpose(values, (2, 1, 0, 3)).reshape(-1, 3)
    body = "\n".join(" ".join(f"{item:.9f}" for item in row) for row in rows)
    path.write_text(
        f"LUT_3D_SIZE {size}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n{body}\n",
        encoding="utf-8",
    )


def _identity(size: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, size)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)


def _gates() -> dict[str, float]:
    return {
        "minimum_output_node": 0.0,
        "maximum_output_node": 1.0,
        "minimum_jacobian_determinant_exclusive": 0.0,
        "maximum_jacobian_spectral_norm": 8.0,
    }


def test_cube_parser_preserves_standard_red_fastest_axis_order(
    tmp_path: Path,
) -> None:
    values = _identity(3)
    values[2, 0, 1] = [0.7, 0.2, 0.9]
    path = tmp_path / "LUT_0.CUBE"
    _write_cube(path, values)
    parsed = parse_cube(path)
    assert np.array_equal(parsed.values, values)
    assert np.array_equal(parsed.domain_min, np.zeros(3))
    assert np.array_equal(parsed.domain_max, np.ones(3))


def test_affine_jacobians_match_for_both_renderers(tmp_path: Path) -> None:
    grid = _identity(4)
    matrix = np.array(
        [[0.8, 0.1, 0.05], [0.05, 0.85, 0.05], [0.02, 0.08, 0.82]]
    )
    offset = np.array([0.02, 0.01, 0.03])
    values = grid @ matrix.T + offset
    path = tmp_path / "LUT_0.CUBE"
    _write_cube(path, values)
    asset = parse_cube(path)
    expected = np.broadcast_to(matrix, (len(tetrahedral_jacobians(asset)), 3, 3))
    assert np.max(np.abs(tetrahedral_jacobians(asset) - expected)) < 2e-8
    trilinear = trilinear_jacobians(asset, [0.0, 0.5, 1.0])
    assert np.max(np.abs(trilinear - matrix)) < 2e-8


def test_fold_and_out_of_range_fail_structural_gate(tmp_path: Path) -> None:
    values = _identity(3)
    values[1, :, :, 0] = -0.1
    path = tmp_path / "LUT_1.CUBE"
    _write_cube(path, values)
    report = analyze_asset(
        parse_cube(path), _gates(), subcell_axis=[0.0, 0.5, 1.0]
    )
    assert not report["range_pass"]
    assert not report["orientation_pass"]
    assert not report["structural_pass"]
    assert report["tetrahedral"]["nonpositive_determinant_count"] > 0
    assert report["trilinear"]["nonpositive_determinant_count"] > 0


def test_manifest_is_numeric_step_order_and_deterministic(tmp_path: Path) -> None:
    for step in (10, 2, 0):
        path = tmp_path / f"LUT_{step}.CUBE"
        _write_cube(path, _identity(2))
    rows, digest = canonical_lut_manifest(tmp_path.glob("LUT_*.CUBE"))
    assert [row["path"] for row in rows] == [
        "LUT_0.CUBE",
        "LUT_2.CUBE",
        "LUT_10.CUBE",
    ]
    payload = "".join(
        f"{row['path']}\t{row['bytes']}\t{row['sha256']}\n" for row in rows
    ).encode()
    assert digest == hashlib.sha256(payload).hexdigest()


def test_invalid_cube_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "LUT_0.CUBE"
    path.write_text("LUT_3D_SIZE 2\n0 0 0\n", encoding="utf-8")
    with pytest.raises(PublishedDLUTError):
        parse_cube(path)
