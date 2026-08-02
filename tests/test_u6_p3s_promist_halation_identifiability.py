from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.eval.promist_halation_identifiability import (
    ProMistHalationError,
    audit_primary_source,
    build_rows,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p3s_promist_halation_identifiability_v1.json"
DECISION = ROOT / "configs/u6_p3s_promist_halation_identifiability_decision_v1.json"


def test_p3s_contract_freezes_latest_diffusion_family_before_score() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["node"] == "U6.P3S"
    assert contract["status"] == "contract_frozen_before_implementation_or_score"
    assert contract["primary_source"]["arxiv_id"] == "2601.19295v1"
    assert len(contract["candidate_family"]["gaussian_sigma_pixels"]) == 6
    assert contract["candidate_family"]["channel_neutral"]
    assert not contract["training_allowed"]
    assert not contract["production_integration_allowed"]


def test_p3s_contract_keeps_development_and_confirmation_disjoint() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    roles = contract["roles"]
    assert set(roles["development_patterns"]).isdisjoint(
        roles["confirmation_patterns"]
    )
    assert set(roles["development_exposure_scales"]).isdisjoint(
        roles["confirmation_exposure_scales"]
    )
    assert contract["controls"]["correct_topology"].startswith("the exact")
    assert "do not tune" in contract["branch_rule"]["any_gate_fail"]


def test_p3s_primary_source_is_exact_and_contains_all_anchors() -> None:
    contract = load_contract(CONTRACT)
    audit = audit_primary_source(ROOT, contract)
    assert audit["bytes"] == 1_863_685
    assert audit["sha256"].startswith("a0e2dda0")
    assert all(audit["required_text_anchors"].values())


def test_p3s_synthetic_roles_are_finite_exact_and_distinct() -> None:
    contract = load_contract(CONTRACT)
    development = build_rows(contract, "development")
    confirmation = build_rows(contract, "confirmation")
    assert len(development) == 12
    assert len(confirmation) == 12
    assert {row["base_sha256"] for row in development}.isdisjoint(
        {row["base_sha256"] for row in confirmation}
    )
    for row in development + confirmation:
        assert row["exposure"].shape == (128, 128, 3)
        assert row["exposure"].dtype == np.float64
        assert np.all(np.isfinite(row["exposure"]))
        assert np.min(row["exposure"]) >= 0.0


def test_p3s_contract_drift_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["candidate_family"]["gaussian_sigma_pixels"][-1] = 8.0
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProMistHalationError, match="unique and increasing"):
        load_contract(path)


def test_p3s_runner_is_directly_invocable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_u6_p3s_promist_halation_identifiability.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--output" in completed.stdout


def test_p3s_decision_binds_exact_failed_evidence() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["automatic_pass"] is False
    assert decision["report_sha256"] == decision["repeat_report_sha256"]
    assert decision["report_sha256"].startswith("5a5533fa")
    assert decision["metrics"]["six_scale_improvement_over_identity"] > 0.5
    assert decision["metrics"]["six_scale_improvement_over_single_scale"] < 0.0
    assert decision["failed_gates"] == [
        "mist_beats_single_scale",
        "mist_confirmation_error",
        "mist_worst_case",
    ]
