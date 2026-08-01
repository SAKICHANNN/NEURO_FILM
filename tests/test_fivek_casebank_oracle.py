from __future__ import annotations

import hashlib
import numpy as np
import pytest

from src.eval.fivek_casebank_oracle import (
    FiveKCasebankOracleError,
    evaluate_offdiagonal_oracle,
    load_split_population,
)
from src.roll2film.triangular_logit_transport import TriangularLogitTransport


def _config() -> dict:
    return {
        "operator": {
            "family": "monotone_triangular_logit_transport",
            "parameter_count": 14,
            "lower_bounds": [-0.8] * 14,
            "upper_bounds": [0.8] * 14,
            "fit_samples_per_image": 128,
            "pooled_samples_per_image": 64,
            "identity_shrinkage": 0.0001,
            "maximum_fit_evaluations": 80,
            "safe_dose_grid_size": 5,
            "safe_dose_finite_difference": 0.00001,
            "minimum_jacobian_determinant": 0.01,
            "maximum_jacobian_condition": 50.0,
            "safe_dose_bisection_iterations": 16,
        },
        "evaluation": {
            "samples_per_confirmation_image": 128,
            "strength_doses": [0.0, 0.25, 0.5, 0.75, 1.0],
            "random_case_seed": 29,
            "bootstrap_seed": 31,
            "bootstrap_repetitions": 200,
            "gates": {
                "minimum_mean_improvement_over_global": 0.05,
                "minimum_win_fraction_over_global": 0.5,
                "maximum_p95_ratio_to_global": 1.0,
                "maximum_worst_ratio_to_global": 1.0,
                "minimum_mean_improvement_over_strength_oracle": 0.02,
                "minimum_win_fraction_over_strength_oracle": 0.5,
                "minimum_mean_improvement_over_random_case": 0.05,
                "minimum_win_fraction_over_random_case": 0.5,
                "minimum_bootstrap_lower_improvement": 0.0,
                "minimum_distinct_selected_cases": 2,
                "maximum_selected_case_share": 0.75,
            },
        },
        "router_training_allowed": False,
        "final_rgb_learning_allowed": False,
    }


def _populations() -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(20260801)
    first = np.zeros(14)
    first[[1, 5, 10]] = [0.32, -0.28, 0.22]
    second = np.zeros(14)
    second[[1, 5, 10]] = [-0.3, 0.25, -0.2]
    development = []
    confirmation = []
    for index in range(8):
        source = rng.uniform(0.05, 0.95, size=(16, 16, 3))
        parameters = first if index % 2 == 0 else second
        row = {
            "pair_id": f"dev-{index}",
            "group": f"dev-camera-{index % 4}",
            "source": source,
            "target": TriangularLogitTransport(parameters).apply(source),
        }
        development.append(row)
    for index in range(8):
        source = rng.uniform(0.05, 0.95, size=(16, 16, 3))
        parameters = first if index % 2 == 0 else second
        confirmation.append(
            {
                "pair_id": f"confirm-{index}",
                "group": f"confirm-camera-{index % 4}",
                "source": source,
                "target": TriangularLogitTransport(parameters).apply(source),
            }
        )
    return development, confirmation


def test_offdiagonal_oracle_recovers_multiple_explicit_directions() -> None:
    development, confirmation = _populations()
    report = evaluate_offdiagonal_oracle(development, confirmation, _config())
    assert report["automatic_pass"] is True
    assert report["metrics"]["mean_improvement_over_global"] > 0.5
    assert report["metrics"]["mean_improvement_over_strength_oracle"] > 0.5
    assert report["metrics"]["mean_improvement_over_random_case"] > 0.5
    assert report["metrics"]["distinct_selected_cases"] >= 2
    assert len(report["pooled_operator"]["parameters"]) == 14
    assert all(len(case["parameters"]) == 14 for case in report["case_bank"])
    assert report["router_training_allowed"] is False


def test_oracle_is_repeat_exact_and_input_order_independent() -> None:
    development, confirmation = _populations()
    first = evaluate_offdiagonal_oracle(development, confirmation, _config())
    second = evaluate_offdiagonal_oracle(
        list(reversed(development)), list(reversed(confirmation)), _config()
    )
    assert first == second


def test_oracle_rejects_self_case_overlap() -> None:
    development, confirmation = _populations()
    confirmation[0]["pair_id"] = development[0]["pair_id"]
    with pytest.raises(FiveKCasebankOracleError, match="off-diagonal"):
        evaluate_offdiagonal_oracle(development, confirmation, _config())


def test_split_loader_binds_paths_hashes_and_target_variant(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source.tif"
    filtered = tmp_path / "filtered.tif"
    expert = tmp_path / "expert.tif"
    for path, payload in (
        (source, b"source"),
        (filtered, b"filtered"),
        (expert, b"expert"),
    ):
        path.write_bytes(payload)

    def fake_load(path, maximum_side):
        assert maximum_side == 256
        value = {source: 0.2, filtered: 0.4, expert: 0.8}[path]
        return np.full((4, 5, 3), value, dtype=np.float32)

    monkeypatch.setattr(
        "src.eval.fivek_casebank_oracle.load_srgb16", fake_load
    )
    manifest = {
        "rows": [
            {
                "pair_id": "casebank-1",
                "camera_model": "camera-a",
                "split": "confirmation",
                "source_path": str(source),
                "source_sha256": hashlib.sha256(b"source").hexdigest(),
                "target_path": str(filtered),
                "target_sha256": hashlib.sha256(b"filtered").hexdigest(),
                "aligned_expert_path": str(expert),
                "aligned_expert_sha256": hashlib.sha256(b"expert").hexdigest(),
            }
        ]
    }
    loaded = load_split_population(
        manifest,
        split="confirmation",
        target_variant="aligned_expert",
        maximum_side=256,
    )
    assert loaded[0]["group"] == "camera-a"
    assert np.all(loaded[0]["target"] == np.float32(0.8))

    manifest["rows"][0]["aligned_expert_sha256"] = "0" * 64
    with pytest.raises(FiveKCasebankOracleError, match="asset drift"):
        load_split_population(
            manifest,
            split="confirmation",
            target_variant="aligned_expert",
            maximum_side=256,
        )
