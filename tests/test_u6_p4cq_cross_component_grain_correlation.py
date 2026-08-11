from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval import cross_component_grain_correlation as p4cq

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4cq_cross_component_grain_correlation_v1.json"
EVIDENCE = ROOT / "docs/evidence/U6_P4CQ_CROSS_COMPONENT_GRAIN_CORRELATION_RESULT.json"


def test_contract_freezes_latest_standard_cross_component_question() -> None:
    contract = p4cq.load_contract(CONTRACT)
    assert set(contract["development"]["source_names"]).isdisjoint(
        contract["confirmation"]["source_names"]
    )
    assert contract["confirmation"]["metadata_lock_sha256"] == (
        "0dfbf2b19000a301af064cfae24a2d363677149863709ea99f7a43b2f471507d"
    )
    assert contract["metrics"]["minimum_sources_beating_independent"] == 2
    assert contract["candidate"]["no_fit_after_confirmation"] is True


def test_patch_correlations_recover_known_cross_component_matrix() -> None:
    contract = p4cq.load_contract(CONTRACT)
    rng = np.random.default_rng(23)
    expected = np.asarray(
        [[1.0, 0.6, -0.2], [0.6, 1.0, 0.1], [-0.2, 0.1, 1.0]]
    )
    factor = np.linalg.cholesky(expected)
    residuals = []
    for _ in range(12):
        values = rng.standard_normal((96 * 96, 3)) @ factor.T
        values -= values.mean(axis=0)
        values /= np.sqrt(np.mean(np.square(values), axis=0))
        residuals.append(values.reshape(96, 96, 3))
    fitted = p4cq._fit_shared_matrix(residuals, contract)
    assert np.max(np.abs(fitted - expected)) < 0.02
    assert p4cq._matrix_error(fitted, expected) < 0.02
    assert p4cq._matrix_error(fitted, np.eye(3)) > 0.35


def test_matrix_fit_rejects_out_of_envelope_correlation() -> None:
    contract = p4cq.load_contract(CONTRACT)
    values = np.ones((96, 96, 3), dtype=np.float64)
    with pytest.raises(p4cq.CrossComponentGrainCorrelationError, match="envelope"):
        p4cq._fit_shared_matrix([values], contract)


def test_patch_selection_is_repeat_exact() -> None:
    contract = p4cq.load_contract(CONTRACT)
    rng = np.random.default_rng(29)
    rgb = np.clip(0.45 + 0.02 * rng.standard_normal((384, 384, 3)), 0.0, 1.0)
    first = p4cq._select_normalized_residuals(rgb, contract)
    second = p4cq._select_normalized_residuals(rgb, contract)
    assert len(first) == contract["patch_observation"]["patches_per_frame"]
    assert all(
        np.array_equal(left, right)
        for left, right in zip(first, second, strict=True)
    )


def test_evidence_matches_frozen_branch_when_present() -> None:
    if not EVIDENCE.is_file():
        pytest.skip("formal P4CQ evidence has not been published")
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["two_full_runs_byte_identical"] is True
    assert evidence["decision"] in {
        "PASS_RETAIN_GENERIC_CROSS_COMPONENT_GRAIN_CORRELATION",
        "FAIL_CLOSED_CROSS_COMPONENT_GRAIN_CORRELATION_TRANSFER",
    }
