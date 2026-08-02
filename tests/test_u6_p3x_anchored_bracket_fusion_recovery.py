from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.eval.anchored_bracket_fusion_recovery import _inverse_variance_fuse

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p3x_anchored_bracket_fusion_recovery_v1.json"
RUNNER = ROOT / "scripts/run_u6_p3x_anchored_bracket_fusion_recovery.py"


def test_p3x_contract_freezes_physical_variance_weighted_fusion() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    fusion = contract["fusion"]
    assert contract["status"] == "contract_frozen_implementation_ready"
    assert fusion["observations_per_group"] == 3
    assert fusion["estimator"] == "per-scalar inverse-variance weighted mean"
    assert fusion["target_reads"] is False
    assert fusion["post_score_changes_allowed"] is False


def test_p3x_contract_keeps_p3u_gates_and_hard_10bit_case() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["anchor_precision_levels_ppm"] == [0, 100]
    assert contract["required_passing_levels_ppm"] == [0, 100]
    assert contract["unchanged_evaluation"]["automatic_gates"] == (
        "exact P3U per-regime and common gates"
    )
    forbidden = " ".join(contract["forbidden"])
    assert "after scoring" in forbidden
    assert "10-bit regime" in forbidden


def test_p3x_inverse_variance_fusion_uses_more_precise_brackets() -> None:
    observations = [
        np.full((1, 1, 1), 0.8),
        np.full((1, 1, 1), 1.0),
        np.full((1, 1, 1), 1.2),
    ]
    fused = _inverse_variance_fuse(observations, [4.0, 1.0, 0.25])
    expected = (0.25 * 0.8 + 1.0 * 1.0 + 4.0 * 1.2) / 5.25
    assert np.array_equal(fused, np.full((1, 1, 1), expected))


def test_p3x_runner_is_directly_invocable() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--output" in completed.stdout
