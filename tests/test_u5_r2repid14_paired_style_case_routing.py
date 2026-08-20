from __future__ import annotations

import numpy as np
import pytest

from src.eval.repid_paired_style_retrieval import (
    predict_case_routes,
    predict_latent_oracle,
    train_case_bank,
)


def _spec() -> tuple[dict[str, float | int], dict[str, float | int]]:
    return (
        {"principal_components": 2, "student_ridge_alpha": 1.0, "operator_ridge_alpha": 1.0},
        {"primary_top_k": 3, "softmax_temperature": 0.1, "cyclic_wrong_label_shift": 1},
    )


def test_case_routing_is_deterministic_and_bounded_to_observed_parameters() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(8, 5))
    deltas = rng.normal(size=(8, 4))
    parameters = rng.normal(size=(8, 3))
    latent, retrieval = _spec()
    model = train_case_bank(features, deltas, parameters, latent_spec=latent, retrieval_spec=retrieval)
    first = predict_case_routes(model, features[:2], retrieval)
    second = predict_case_routes(model, features[:2], retrieval)
    assert np.array_equal(first["candidate_parameters"], second["candidate_parameters"])
    assert first["neighbor_indices"] == second["neighbor_indices"]
    assert np.allclose(np.sum(first["neighbor_weights"], axis=1), 1.0)
    assert np.all(first["candidate_parameters"] <= np.max(parameters, axis=0) + 1e-12)
    assert np.all(first["candidate_parameters"] >= np.min(parameters, axis=0) - 1e-12)


def test_medoid_is_one_observed_case_and_top1_is_observed() -> None:
    rng = np.random.default_rng(8)
    features = rng.normal(size=(7, 5))
    deltas = rng.normal(size=(7, 4))
    parameters = rng.normal(size=(7, 3))
    latent, retrieval = _spec()
    model = train_case_bank(features, deltas, parameters, latent_spec=latent, retrieval_spec=retrieval)
    output = predict_case_routes(model, features[:3], retrieval)
    assert any(np.array_equal(output["medoid_parameters"][0], row) for row in parameters)
    assert all(any(np.array_equal(row, case) for case in parameters) for row in output["top1_parameters"])


def test_latent_oracle_returns_exact_shape_and_fit_neighbors() -> None:
    rng = np.random.default_rng(9)
    features = rng.normal(size=(9, 5))
    deltas = rng.normal(size=(9, 4))
    parameters = rng.normal(size=(9, 3))
    latent, retrieval = _spec()
    model = train_case_bank(features, deltas, parameters, latent_spec=latent, retrieval_spec=retrieval)
    output = predict_latent_oracle(model, deltas[:2], retrieval)
    assert output["parameters"].shape == (2, 3)
    assert all(len(row) == 3 for row in output["neighbor_indices"])


def test_invalid_or_zero_norm_features_fail_closed() -> None:
    latent, retrieval = _spec()
    with pytest.raises(ValueError):
        train_case_bank(
            np.ones((8, 5)),
            np.ones((8, 4)),
            np.ones((8, 3)),
            latent_spec=latent,
            retrieval_spec=retrieval,
        )
