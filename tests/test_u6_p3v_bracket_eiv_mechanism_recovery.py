from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.eval.bracket_eiv_mechanism_recovery import (
    _zero_mean_bounded_logs,
    evaluate,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p3v_bracket_eiv_mechanism_recovery_v1.json"
RUNNER = ROOT / "scripts/run_u6_p3v_bracket_eiv_mechanism_recovery.py"


def test_p3v_contract_freezes_source_only_eiv_before_scoring() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    estimator = contract["estimator"]
    assert contract["status"] == "contract_frozen_implementation_ready"
    assert estimator["iterations"] == 8
    assert "observed source scalars" in estimator["inputs"]
    assert "development target" in estimator["forbidden_inputs"]
    assert estimator["post_fit_changes_allowed"] is False


def test_p3v_contract_keeps_p3u_observations_and_gates_unchanged() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["roles"]["normalizer_reads_targets"] is False
    assert contract["roles"]["mechanism_fit_reads_confirmation"] is False
    assert contract["unchanged_evaluation"]["automatic_gates"] == (
        "exact P3U per-regime and common gates"
    )
    forbidden = " ".join(contract["forbidden"])
    assert "after scoring" in forbidden
    assert "truth gain jitter" in forbidden


def test_p3v_bounded_log_gauge_is_exact_and_bounded() -> None:
    raw = np.log(np.asarray([0.98, 0.999, 1.002, 1.03], dtype=np.float64))
    result = _zero_mean_bounded_logs(raw, np.log(0.99), np.log(1.01))
    assert abs(float(np.mean(result))) <= 1e-14
    assert float(np.min(result)) >= np.log(0.99)
    assert float(np.max(result)) <= np.log(1.01)


def test_p3v_evaluator_is_exact_and_target_blind() -> None:
    first = evaluate(ROOT, CONTRACT)
    second = evaluate(ROOT, CONTRACT)
    assert first == second
    assert first["role_facts"]["normalizer_reads_targets"] is False
    assert all(
        row["normalizer"]["target_read_count"] == 0 for row in first["regime_results"]
    )


def test_p3v_runner_is_directly_invocable() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--output" in completed.stdout
