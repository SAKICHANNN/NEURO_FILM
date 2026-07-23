from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.roll2film.synthetic_recovery import (
    SyntheticBoundedOperator,
    audit_synthetic_operator,
    generate_operator_manifest,
    manifest_sha256,
    operator_from_manifest,
    palette_cloud,
    palette_pairs,
    parameters_to_matrix,
    project_operator_parameters,
    uniform_probe_grid,
)
from src.roll2film.synthetic_benchmark import (
    PCAPredictor,
    build_observations,
    evaluate_predictions,
    feature_matrix,
    rows_for_split,
    shuffled_targets,
    target_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2d1_synthetic_operator_recovery_v1.json"


def _identity_parameters() -> np.ndarray:
    return np.array([0, 0, 0, 0, 0, 0, 0.25, 0.5, 0.75], dtype=np.float64)


def test_parameter_projection_is_feasible_and_deterministic() -> None:
    raw = np.array([-1.0, 4.0, 0.5, 0.8, 0.7, -2.0, 0.9, 0.1, 2.0])
    first = project_operator_parameters(raw)
    second = project_operator_parameters(raw)
    operator = SyntheticBoundedOperator(first, "combined", "projected")

    assert np.array_equal(first, second)
    assert audit_synthetic_operator(operator)["valid"]
    assert first[0:2].sum() <= 0.3 + 1e-12
    assert first[2:4].sum() <= 0.3 + 1e-12
    assert first[4:6].sum() <= 0.3 + 1e-12
    assert np.min(np.diff(np.r_[0.0, first[6:9], 1.0])) >= 0.08 - 1e-12


def test_identity_operator_and_scalar_reference() -> None:
    identity = SyntheticBoundedOperator(_identity_parameters(), "combined", "identity")
    probes = uniform_probe_grid(9)
    assert np.allclose(identity.apply(probes), probes, atol=2e-16)

    parameters = np.array(
        [0.1, 0.05, 0.02, 0.08, 0.12, 0.04, 0.18, 0.55, 0.82]
    )
    operator = SyntheticBoundedOperator(parameters, "combined", "reference")
    point = np.array([[0.2, 0.6, 0.9]])
    mixed = parameters_to_matrix(parameters) @ point[0]
    expected = np.array(
        [
            np.interp(channel, np.linspace(0.0, 1.0, 5), np.r_[0.0, parameters[6:9], 1.0])
            for channel in mixed
        ]
    )
    assert np.allclose(operator.apply(point)[0], expected, atol=1e-15)


def test_manifest_counts_splits_families_and_replay() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    first = generate_operator_manifest(config)
    second = generate_operator_manifest(config)
    expected_total = 3 * 128

    assert len(first) == expected_total
    assert manifest_sha256(first) == manifest_sha256(second)
    assert len({row["operator_id"] for row in first}) == expected_total
    for family in config["operator"]["families"]:
        rows = [row for row in first if row["family"] == family]
        assert len(rows) == 128
        assert sum(row["split"] == "fit" for row in rows) == 48
        assert sum(row["split"] == "validation" for row in rows) == 16
        assert sum(row["split"] == "confirmation" for row in rows) == 32
        assert sum(row["split"] == "stress" for row in rows) == 32
        assert all(audit_synthetic_operator(operator_from_manifest(row))["valid"] for row in rows)


def test_family_components_are_exact() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = generate_operator_manifest(config)
    identity = _identity_parameters()
    for row in rows:
        parameters = np.asarray(row["parameters"])
        if row["family"] == "matrix_only":
            assert np.array_equal(parameters[6:9], identity[6:9])
        elif row["family"] == "tone_only":
            assert np.array_equal(parameters[:6], identity[:6])
        else:
            assert np.any(parameters[:6] != identity[:6])
            assert np.any(parameters[6:9] != identity[6:9])


def test_palette_clouds_and_pairs_are_deterministic_and_distinct() -> None:
    names = ["balanced", "warm", "cool", "foliage_like"]
    for name in names:
        first = palette_cloud(name, 128, 7)
        second = palette_cloud(name, 128, 7)
        assert np.array_equal(first, second)
        assert first.shape == (128, 3)
        assert np.min(first) >= 0.0
        assert np.max(first) <= 1.0
    pairs = palette_pairs(names, operator_index=2, observations=7)
    assert all(query != reference for query, reference in pairs)


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros(8),
        np.full(9, np.nan),
    ],
)
def test_invalid_parameters_fail_closed(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError):
        project_operator_parameters(invalid)


def test_invalid_rgb_and_palette_fail_closed() -> None:
    operator = SyntheticBoundedOperator(_identity_parameters(), "combined", "identity")
    with pytest.raises(ValueError):
        operator.apply(np.array([[1.1, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        palette_cloud("unknown", 10, 1)
    with pytest.raises(ValueError):
        palette_pairs(["balanced"], operator_index=0)


def _tiny_config() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
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
    config["predictors"]["pca_components"] = 3
    config["evaluation"]["uniform_probe_grid_size"] = 5
    config["evaluation"]["operator_group_bootstrap_replicates"] = 10
    return config


def test_observation_features_and_split_groups() -> None:
    config = _tiny_config()
    manifest = generate_operator_manifest(config)
    observations = build_observations(config, manifest)

    assert len(observations) == 2 * len(manifest)
    split_ids = {
        split: {row.operator_id for row in rows_for_split(observations, split)}
        for split in ("fit", "validation", "confirmation", "stress")
    }
    for left, right in (
        ("fit", "validation"),
        ("fit", "confirmation"),
        ("fit", "stress"),
        ("validation", "confirmation"),
        ("validation", "stress"),
        ("confirmation", "stress"),
    ):
        assert not split_ids[left] & split_ids[right]
    row = observations[0]
    assert row.features["style_target_only"].shape == (2304,)
    assert row.features["interaction_source_target"].shape == (4608,)


def test_pca_predictor_and_identity_evaluation() -> None:
    config = _tiny_config()
    observations = build_observations(config, generate_operator_manifest(config))
    fit_rows = rows_for_split(observations, "fit")
    validation_rows = rows_for_split(observations, "validation")
    x_fit = feature_matrix(fit_rows, "canonical_reference_delta_oracle")
    predictor = PCAPredictor(
        kind="ridge",
        hyperparameter=0.01,
        components=3,
        seed=7,
    ).fit(x_fit, target_matrix(fit_rows))
    raw = predictor.predict(
        feature_matrix(validation_rows, "canonical_reference_delta_oracle")
    )
    result = evaluate_predictions(validation_rows, raw, config)

    assert raw.shape == (len(validation_rows), 9)
    assert result["summary"]["overall"]["operators"] == 6
    assert result["summary"]["overall"]["valid_fraction"] == 1.0


def test_shuffled_targets_preserve_operator_group_coherence() -> None:
    config = _tiny_config()
    observations = build_observations(config, generate_operator_manifest(config))
    fit_rows = rows_for_split(observations, "fit")
    shuffled = shuffled_targets(fit_rows, seed=41)
    values_by_id: dict[str, list[np.ndarray]] = {}
    for row, target in zip(fit_rows, shuffled):
        values_by_id.setdefault(row.operator_id, []).append(target)
    assert all(
        all(np.array_equal(values[0], value) for value in values[1:])
        for values in values_by_id.values()
    )
    assert not np.array_equal(shuffled, target_matrix(fit_rows))
