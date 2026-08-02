from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.lognormal_disc_boolean_nps_compatibility import (
    LognormalDiscBooleanNPSCompatibilityError,
    load_contract,
)
from src.film_physics.variable_disc_boolean_nps import (
    bounded_lognormal_radius_quadrature,
    variable_disc_boolean_covariance,
    variable_disc_boolean_radial_nps,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bq_lognormal_disc_boolean_nps_compatibility_v1.json"
DECISION = (
    ROOT / "configs/u6_p4bq_lognormal_disc_boolean_nps_compatibility_decision_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_p4bq_contract_is_frozen_before_variable_radius_fit() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["schema"] == (
        "neuro_film.u6_p4bq_lognormal_disc_boolean_nps_compatibility_contract.v1"
    )
    model = payload["model"]
    assert model["median_radius_micrometres_bounds"] == [0.25, 12.0]
    assert model["log_radius_sigma_bounds"] == [0.05, 0.55]
    assert model["radius_cdf_interval"] == [0.00001, 0.99999]
    assert model["radius_distribution_quadrature_nodes"] == 32
    assert model["radial_hankel_quadrature_samples"] == 4097
    assert (
        payload["fit"][
            "post_score_distribution_bound_quadrature_frequency_split_model_or_gate_tuning_allowed"
        ]
        is False
    )
    assert model["source_code_reuse_allowed"] is False


def test_p4bq_contract_binds_measured_bundle_and_closed_fixed_disc_parent() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parents = payload["parents"]
    for stem in ("p4bl_bundle", "p4bp_contract", "p4bp_decision", "p4bp_report"):
        assert _sha256(ROOT / parents[f"{stem}_path"]) == parents[f"{stem}_sha256"]


def test_bounded_lognormal_quadrature_and_covariance_are_physical() -> None:
    radii, weights = bounded_lognormal_radius_quadrature(0.0034, 0.3)
    assert np.all(np.diff(radii) > 0.0)
    assert np.sum(weights) == pytest.approx(1.0, abs=1e-15)
    coverage = 0.41
    covariance = variable_disc_boolean_covariance(
        np.asarray([0.0, 2.0 * float(np.max(radii)), 3.0 * float(np.max(radii))]),
        0.0034,
        0.3,
        coverage,
    )
    assert covariance[0] == pytest.approx(coverage * (1.0 - coverage), rel=1e-14)
    assert covariance[1] == 0.0
    assert covariance[2] == 0.0


def test_variable_disc_radial_nps_is_positive_and_stable() -> None:
    frequency = np.arange(5.0, 501.0, 5.0)
    coarse = variable_disc_boolean_radial_nps(
        frequency, 0.0034, 0.3, 0.41, radial_samples=2049
    )
    frozen = variable_disc_boolean_radial_nps(
        frequency, 0.0034, 0.3, 0.41, radial_samples=4097
    )
    assert np.all(frozen > 0.0)
    assert np.max(np.abs(coarse / frozen - 1.0)) < 2e-5


def test_p4bq_contract_rejects_distribution_rescue(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["model"]["log_radius_sigma_bounds"] = [0.0, 1.0]
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        LognormalDiscBooleanNPSCompatibilityError, match="contract drift"
    ):
        load_contract(path)


def test_p4bq_decision_binds_exact_family_closure() -> None:
    payload = json.loads(DECISION.read_text(encoding="utf-8"))
    assert payload["decision"] == (
        "close_independent_disc_boolean_measured_nps_equation_family"
    )
    assert (
        payload["report_sha256"]
        == payload["repeat_report_sha256"]
        == ("efc4f154659922bf0a40972b170bc52dc274145cbf74dd4324b59c3ed4617ee9")
    )
    assert payload["stable_evidence_id"] == (
        "968cb5a14ebbf19877ab8a07322c6b3dab3389719b56349f908da7f1194b4ffc"
    )
    assert payload["confirmation_median_relative_improvement_over_fixed_disc"] < 0.0
