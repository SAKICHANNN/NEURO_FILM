from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.fixed_disc_boolean_nps_compatibility import (
    FixedDiscBooleanNPSCompatibilityError,
    load_contract,
)
from src.film_physics.fixed_disc_boolean_nps import (
    fixed_disc_boolean_covariance,
    fixed_disc_boolean_radial_nps,
    fixed_disc_overlap_area,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bp_fixed_disc_boolean_nps_compatibility_v1.json"
DECISION = (
    ROOT / "configs/u6_p4bp_fixed_disc_boolean_nps_compatibility_decision_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_p4bp_contract_is_frozen_before_measured_spectrum_fit() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["schema"] == (
        "neuro_film.u6_p4bp_fixed_disc_boolean_nps_compatibility_contract.v1"
    )
    fit = payload["fit"]
    assert fit["build_bands_lines_per_mm"] == [[5.0, 100.0], [255.0, 350.0]]
    assert fit["confirmation_bands_lines_per_mm"] == [
        [105.0, 250.0],
        [355.0, 500.0],
    ]
    assert fit["candidate_radius_micrometres_bounds"] == [0.25, 20.0]
    assert fit["candidate_coverage_probability_bounds"] == [0.02, 0.98]
    assert fit["post_score_radius_coverage_frequency_split_model_or_gate_tuning_allowed"] is False
    assert payload["primary_model_source"]["source_code_reuse_allowed"] is False


def test_p4bp_contract_binds_primary_paper_and_measured_parent() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source = payload["primary_model_source"]
    assert _sha256(ROOT / source["article_low_resolution_pdf_path"]) == source[
        "article_low_resolution_pdf_sha256"
    ]
    assert _sha256(ROOT / source["official_source_archive_path"]) == source[
        "official_source_archive_sha256"
    ]
    parents = payload["parents"]
    for stem in ("p4bl_contract", "p4bl_decision", "p4bl_bundle", "p4bl_report"):
        assert _sha256(ROOT / parents[f"{stem}_path"]) == parents[f"{stem}_sha256"]


def test_fixed_disc_overlap_and_boolean_covariance_endpoints() -> None:
    radius = 0.004
    coverage = 0.37
    distance = np.asarray([0.0, radius, 2.0 * radius, 3.0 * radius])
    overlap = fixed_disc_overlap_area(distance, radius)
    covariance = fixed_disc_boolean_covariance(distance, radius, coverage)
    assert overlap[0] == pytest.approx(np.pi * radius * radius, rel=1e-15)
    assert overlap[2] == 0.0
    assert overlap[3] == 0.0
    assert covariance[0] == pytest.approx(coverage * (1.0 - coverage), rel=1e-15)
    assert covariance[2] == 0.0
    assert covariance[3] == 0.0


def test_fixed_disc_boolean_nps_is_positive_and_quadrature_stable() -> None:
    frequency = np.arange(5.0, 501.0, 5.0)
    coarse = fixed_disc_boolean_radial_nps(
        frequency, 0.0035, 0.42, quadrature_samples=2049
    )
    frozen = fixed_disc_boolean_radial_nps(
        frequency, 0.0035, 0.42, quadrature_samples=4097
    )
    assert np.all(frozen > 0.0)
    assert np.max(np.abs(coarse / frozen - 1.0)) < 2e-7


def test_p4bp_contract_rejects_postscore_radius_rescue(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["fit"]["candidate_radius_micrometres_bounds"] = [0.1, 30.0]
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FixedDiscBooleanNPSCompatibilityError, match="contract drift"):
        load_contract(path)


def test_p4bp_decision_binds_two_exact_formal_negative_reports() -> None:
    payload = json.loads(DECISION.read_text(encoding="utf-8"))
    assert payload["decision"] == "close_fixed_disc_boolean_measured_nps_equation_family"
    assert payload["automatic_pass"] is False
    assert payload["report_sha256"] == payload["repeat_report_sha256"] == (
        "4f289048b384aef3ad83ff9279d89682c84099101cdcaa10cecf7b2647b77232"
    )
    assert payload["stable_evidence_id"] == (
        "98288cc78212b7629f57b549a59806d122df5918ede8abc063bb9633d44aafd2"
    )
    assert payload["failed_gates"] == [
        "confirmation_median_improvement",
        "each_material_confirmation_median_improvement",
        "worst_confirmation_not_worse",
        "density_monotone_coverage",
    ]
