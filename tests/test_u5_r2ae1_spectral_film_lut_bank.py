from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.velvia_datasheet_witness import encoded_srgb_to_linear
from src.eval.spectral_film_lut_bank import (
    array_sha256,
    canonical_sha256,
    evaluate_manifests,
    local_jacobian_metrics,
    median_delta_e76,
    sha256_file,
    strength_path_metrics,
    symmetric_basic_residual,
    synthetic_cube,
)


def test_synthetic_cube_has_exact_endpoints_and_shape() -> None:
    cube = synthetic_cube(5)
    assert cube.shape == (5, 5, 5, 3)
    np.testing.assert_array_equal(cube[0, 0, 0], np.zeros(3))
    np.testing.assert_array_equal(cube[-1, -1, -1], np.ones(3))


def test_identity_jacobian_is_positive_and_unit_norm() -> None:
    metrics = local_jacobian_metrics(synthetic_cube(7))
    assert abs(metrics["minimum_jacobian_determinant"] - 1.0) < 1e-12
    assert metrics["negative_jacobian_fraction"] == 0.0
    assert abs(metrics["maximum_jacobian_spectral_norm"] - 1.0) < 1e-12


def test_folded_red_axis_is_detected() -> None:
    cube = synthetic_cube(7)
    cube[..., 0] = 1.0 - cube[..., 0]
    metrics = local_jacobian_metrics(cube)
    assert metrics["minimum_jacobian_determinant"] < 0.0
    assert metrics["negative_jacobian_fraction"] == 1.0


def test_strength_path_is_one_direction_not_a_second_mode() -> None:
    identity = synthetic_cube(5)
    full = np.clip(identity**1.4 + np.array([0.03, 0.0, 0.01]), 0.0, 1.0)
    metrics = strength_path_metrics(identity, full, 0.75)
    assert abs(metrics["fitted_strength"] - 0.75) < 1e-15
    assert metrics["residual_rgb_rmse"] < 1e-15
    assert metrics["explained_energy_fraction"] == 1.0


def test_array_hash_binds_dtype_and_shape() -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    assert array_sha256(values) == array_sha256(values.copy())
    assert array_sha256(values) != array_sha256(values.astype(np.float64))
    assert array_sha256(values) != array_sha256(values.reshape(4, 2, 3))


def test_symmetric_basic_residual_absorbs_simple_exposure() -> None:
    cube = synthetic_cube(7)
    linear = encoded_srgb_to_linear(cube) * 0.8
    darker = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * linear ** (1.0 / 2.4) - 0.055,
    )
    residual = symmetric_basic_residual(cube, darker)
    assert residual["conservative_minimum_delta_e76_median"] < 1e-3


def test_delta_e_detects_nonidentity() -> None:
    cube = synthetic_cube(5)
    shifted = np.clip(cube + np.array([0.05, 0.0, 0.0]), 0.0, 1.0)
    assert median_delta_e76(cube, shifted) > 1.0


def test_full_evaluator_returns_a_closed_or_retained_decision(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/u5_r2ae1_spectral_film_lut_structural_bank_v1.json").read_text(
            encoding="utf-8"
        )
    )
    cube = synthetic_cube(config["synthetic_population"]["cube_size"]).astype("<f4")
    neutral_axis = np.linspace(
        0.0,
        1.0,
        config["synthetic_population"]["neutral_samples"],
        dtype=np.float32,
    )
    neutral = np.repeat(neutral_axis[:, None], 3, axis=1)

    def manifest(run_id: str) -> dict:
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        records = []
        for index, chain in enumerate(config["chains"]):
            output = np.clip(cube ** (1.0 + 0.02 * index), 0.0, 1.0).astype("<f4")
            output_path = run_dir / f"{chain['id']}.npy"
            neutral_path = run_dir / f"{chain['id']}_neutral.npy"
            np.save(output_path, output, allow_pickle=False)
            np.save(neutral_path, neutral, allow_pickle=False)
            records.append(
                {
                    "chain_id": chain["id"],
                    "output_path": str(output_path),
                    "neutral_path": str(neutral_path),
                    "output_file_sha256": sha256_file(output_path),
                    "neutral_file_sha256": sha256_file(neutral_path),
                    "output_array_sha256": array_sha256(output),
                    "neutral_array_sha256": array_sha256(neutral),
                }
            )
        duplicate_path = run_dir / "duplicate.npy"
        np.save(duplicate_path, cube, allow_pickle=False)
        return {
            "schema_version": "u5-r2ae1-external-spectral-bank-manifest-v1",
            "run_id": run_id,
            "external_revision": config["external_source"]["revision"],
            "python_version": config["runtime"]["python_version"],
            "package_versions": config["runtime"]["package_versions"],
            "config_sha256": canonical_sha256(config),
            "records": records,
            "duplicate_control": {
                "output_path": str(duplicate_path),
                "output_file_sha256": sha256_file(duplicate_path),
                "output_array_sha256": array_sha256(cube),
            },
        }

    report = evaluate_manifests(
        manifest("a"), manifest("b"), config, root=root
    )
    assert report["decision"] in {
        "close_range_fold_or_monotonicity_failure",
        "close_basic_only_or_insufficient_family_diversity",
        "retain_external_structural_comparison_bank_only",
    }
    assert report["visual_review_allowed"] is False
