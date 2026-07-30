from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.evaluate_cfsm_empirical_prior import confirmation_decision
from scripts.evaluate_cfsm_prior_selection import (
    evaluate_split,
    selected_manifest_rows,
)
from src.color_match.research import EmpiricalNeutralPrior
from src.roll2film.synthetic_recovery import (
    generate_operator_manifest,
    manifest_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs" / "reference_match_cfsm_empirical_prior_v1.json"
)


def _config_and_parent() -> tuple[dict, dict, list[dict]]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parent = json.loads(
        (ROOT / config["parent_config"]).read_text(encoding="utf-8")
    )
    manifest = generate_operator_manifest(parent)
    assert manifest_sha256(manifest) == config["parent_manifest_sha256"]
    return config, parent, manifest


def _prior() -> EmpiricalNeutralPrior:
    return EmpiricalNeutralPrior(
        prior_id="a" * 64,
        dataset_id="fixture",
        mean=np.array([0.16, 0.15, 0.14], dtype=np.float64),
        covariance=np.array(
            [
                [0.040, 0.034, 0.030],
                [0.034, 0.038, 0.034],
                [0.030, 0.034, 0.041],
            ],
            dtype=np.float64,
        ),
        source_image_count=128,
        source_pixel_count=200_000_000,
        artifact={},
    )


def _result(
    median: float,
    rate: float,
    worst: float,
    *,
    boundary: float = 0.0,
) -> dict:
    return {
        "summary": {
            "median_captured_style_fraction": median,
            "improvement_rate": rate,
            "worst_captured_style_fraction": worst,
            "maximum_new_boundary_fraction": boundary,
            "constraint_pass_fraction": 1.0,
            "identity_fallback_count": 0,
        }
    }


def test_empirical_experiment_freezes_unseen_confirmation() -> None:
    config, _, manifest = _config_and_parent()
    development = selected_manifest_rows(config, manifest, "development")
    confirmation = selected_manifest_rows(config, manifest, "confirmation")
    assert len(development) == 24
    assert len(confirmation) == 24
    assert {row["split"] for row in development} == {"fit"}
    assert {row["split"] for row in confirmation} == {"confirmation"}
    assert not (
        {row["operator_id"] for row in development}
        & {row["operator_id"] for row in confirmation}
    )
    assert config["execution"]["parameter_tuning_allowed"] is False
    assert config["execution"]["product_integration_allowed"] is False


def test_one_operator_executes_empirical_and_uniform_controls() -> None:
    config, parent, manifest = _config_and_parent()
    row = selected_manifest_rows(config, manifest, "development")[0]
    results = evaluate_split(
        config,
        parent,
        [row],
        split="development",
        prior_kinds=tuple(config["prior_kinds"]),
        empirical_prior=_prior(),
    )
    assert set(results) == {
        "fixed-uniform-cube",
        "empirical-neutral-photo-v1",
    }
    for result in results.values():
        assert result["summary"]["observations"] == 2
        assert result["summary"]["constraint_pass_fraction"] == 1.0
        assert result["summary"]["identity_fallback_count"] == 0


def test_confirmation_gate_never_opens_product_or_commercial_use() -> None:
    config, _, _ = _config_and_parent()
    passed = confirmation_decision(
        config,
        {
            "fixed-uniform-cube": _result(0.10, 0.80, -0.10),
            "empirical-neutral-photo-v1": _result(0.14, 0.80, -0.11),
        },
    )
    assert passed["status"].endswith("passed")
    assert passed["research_challenger_retained"] is True
    assert passed["product_integration_open"] is False
    assert passed["commercial_use_open"] is False

    failed = confirmation_decision(
        config,
        {
            "fixed-uniform-cube": _result(0.10, 0.80, -0.10),
            "empirical-neutral-photo-v1": _result(0.12, 0.80, -0.10),
        },
    )
    assert failed["status"] == "empirical-prior-route-closed"
    assert failed["research_challenger_retained"] is False
