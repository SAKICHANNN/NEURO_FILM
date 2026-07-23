from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.roll2film.canonicalizer_sensitivity import (
    build_canonical_observations,
    build_neutral_bank,
    canonical_rows_for_split,
    neutral_bank_manifest,
    neutral_bank_manifest_sha256,
    normalized_rgb_quantiles,
    retrieval_diagnostics,
    signature_matrix,
)
from src.roll2film.synthetic_recovery import generate_operator_manifest


ROOT = Path(__file__).resolve().parents[1]
PARENT_CONFIG = ROOT / "configs/u5_r2d1_synthetic_operator_recovery_v1.json"
CONFIG = ROOT / "configs/u5_r2d2_canonicalizer_retrieval_sensitivity_v1.json"


def _tiny_parent() -> dict:
    config = json.loads(PARENT_CONFIG.read_text(encoding="utf-8"))
    config["operator"].update(
        {
            "operators_per_family": 8,
            "fit_per_family": 4,
            "validation_per_family": 2,
            "confirmation_per_family": 1,
            "stress_per_family": 1,
        }
    )
    config["palettes"]["pixels_per_cloud"] = 64
    return config


def _tiny_config() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["neutral_bank"]["clouds_per_palette"] = 2
    config["neutral_bank"]["pixels_per_cloud"] = 64
    config["policies"]["pca_components"] = 3
    config["evaluation"]["uniform_probe_grid_size"] = 5
    config["evaluation"]["bootstrap_replicates"] = 10
    return config


def test_normalized_quantiles_are_affine_invariant_without_clipping() -> None:
    rng = np.random.default_rng(11)
    rgb = rng.uniform(0.2, 0.6, size=(1024, 3))
    transformed = rgb * np.array([0.7, 1.1, 0.9]) + np.array([0.1, 0.05, 0.08])
    first = normalized_rgb_quantiles(rgb, quantile_count=33, minimum_iqr=1e-6)
    second = normalized_rgb_quantiles(
        transformed, quantile_count=33, minimum_iqr=1e-6
    )
    assert np.max(np.abs(first - second)) < 1e-12


def test_neutral_bank_is_deterministic_independent_and_hashed() -> None:
    config = _tiny_config()
    first = build_neutral_bank(config)
    second = build_neutral_bank(config)
    manifest = neutral_bank_manifest(first)

    assert len(first) == 18
    assert len({item.item_id for item in first}) == 18
    assert neutral_bank_manifest_sha256(first) == neutral_bank_manifest_sha256(second)
    assert manifest == neutral_bank_manifest(second)
    assert all(item.rgb.shape == (64, 3) for item in first)


def test_canonical_signatures_and_retrieval_controls() -> None:
    parent = _tiny_parent()
    config = _tiny_config()
    manifest = generate_operator_manifest(parent)
    bank = build_neutral_bank(config)
    observations = build_canonical_observations(parent, manifest, config, bank)
    confirmation = canonical_rows_for_split(observations, "confirmation")

    assert len(observations) == 2 * len(manifest)
    assert signature_matrix(confirmation, "query_as_neutral").shape == (
        len(confirmation),
        2304,
    )
    exact = signature_matrix(confirmation, "exact_raw_reference_oracle")
    query = signature_matrix(confirmation, "query_as_neutral")
    assert not np.array_equal(exact, query)
    diagnostics = retrieval_diagnostics(confirmation, bank)
    assert diagnostics["palette_oracle_nearest"]["same_palette_fraction"] == 1.0
    assert (
        diagnostics["nearest_raw_lab_leave_true_palette_out"][
            "same_palette_fraction"
        ]
        == 0.0
    )


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((5, 2)),
        np.array([[np.nan, 0.0, 0.0]]),
        np.array([[1.1, 0.0, 0.0]]),
    ],
)
def test_invalid_quantile_inputs_fail_closed(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError):
        normalized_rgb_quantiles(invalid, quantile_count=33, minimum_iqr=1e-6)
