from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval import phase_projected_thomas_confirmation as p4cm

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/u6_p4cm_phase_projected_thomas_confirmation_v1.json"
EVIDENCE_PATH = (
    ROOT / "docs/evidence/U6_P4CM_PHASE_PROJECTED_THOMAS_CONFIRMATION_RESULT.json"
)


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_contract_and_parent_are_exact() -> None:
    p4cm._validate_contract(ROOT, _contract())


def test_projection_is_deterministic_and_power_exact() -> None:
    contract = _contract()
    bases, scales = p4cm._base_and_scale_fields(contract, 3, seed_offset=17)
    first = p4cm._phase_projected_fields(contract, bases, scales)
    second = p4cm._phase_projected_fields(contract, bases, scales)
    assert np.array_equal(first, second)
    errors = p4cm._projection_errors(contract, bases, first)
    assert errors["maximum_per_sample_power_relative_error"] <= 5e-8
    assert errors["maximum_per_sample_acf_absolute_error"] <= 1e-12


def test_projection_retains_non_gaussian_phase() -> None:
    contract = _contract()
    bases, scales = p4cm._base_and_scale_fields(contract, 16, seed_offset=29)
    projected = p4cm._phase_projected_fields(contract, bases, scales)
    assert np.isfinite(projected).all()
    assert not np.array_equal(p4cm._feature_matrix(projected), p4cm._feature_matrix(bases))


def test_score_prefers_exact_projected_median() -> None:
    rng = np.random.default_rng(7)
    fields = np.asarray([p4cm._normalized(rng.standard_normal((96, 96))) for _ in range(8)])
    features = p4cm._feature_matrix(fields)
    median = np.median(features, axis=0)
    rows = p4cm._score_sources({"source": list(fields)}, median, np.ones(4), median)
    assert rows[0]["gaussian_feature_distance"] == 0.0
    assert rows[0]["projected_feature_distance"] == 0.0


def test_evidence_preserves_exact_projection_and_failed_strength() -> None:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert evidence["two_full_runs_byte_identical"] is True
    assert evidence["projection"]["power_gate"] is True
    assert evidence["projection"]["acf_gate"] is True
    assert evidence["confirmation"]["sources_beating_gaussian_high_order_distance"] == 4
    assert evidence["gates"]["minimum_20_percent_median_feature_improvement"] is False
    assert evidence["decision"] == "FAIL_CLOSED_PHASE_PROJECTED_THOMAS_TRANSFER"
