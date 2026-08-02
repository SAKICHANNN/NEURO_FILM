from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.eval.external_exposure_anchor_sufficiency import _gain_and_anchor_error

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p3w_external_exposure_anchor_sufficiency_v1.json"
RUNNER = ROOT / "scripts/run_u6_p3w_external_exposure_anchor_sufficiency.py"


def test_p3w_contract_freezes_anchor_ladder_before_scoring() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    levels = contract["anchor_precision_levels_ppm"]
    assert contract["status"] == "contract_frozen_implementation_ready"
    assert levels == [0, 100, 250, 500, 1000]
    assert levels == sorted(levels)
    assert contract["required_passing_levels_ppm"] == [0, 100]


def test_p3w_contract_keeps_anchor_independent_and_p3u_gates_unchanged() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["roles"]["anchor_reads_targets"] is False
    assert contract["roles"]["mechanism_fit_reads_confirmation"] is False
    assert "targets" in contract["anchor_observation"]["forbidden_inputs"]
    assert contract["unchanged_evaluation"]["automatic_gates"] == (
        "exact P3U per-regime and common gates"
    )
    forbidden = " ".join(contract["forbidden"])
    assert "after scoring" in forbidden
    assert "favorable error draws" in forbidden


def test_p3w_anchor_error_is_exact_and_within_frozen_bound() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    p3u = json.loads(
        (ROOT / "configs/u6_p3u_mechanism_recovery_uncertainty_v1.json").read_text(
            encoding="utf-8"
        )
    )
    p3s = json.loads(
        (ROOT / "configs/u6_p3s_promist_halation_identifiability_v1.json").read_text(
            encoding="utf-8"
        )
    )
    from src.eval.promist_halation_identifiability import build_rows

    row = build_rows(p3s, "development")[0]
    regime = p3u["regimes"][2]
    first = _gain_and_anchor_error(
        row, "development", regime, p3u, contract["anchor_observation"]["seed"], 100
    )
    second = _gain_and_anchor_error(
        row, "development", regime, p3u, contract["anchor_observation"]["seed"], 100
    )
    assert first == second
    assert abs(first[1]) <= 100e-6


def test_p3w_runner_is_directly_invocable() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--output" in completed.stdout
