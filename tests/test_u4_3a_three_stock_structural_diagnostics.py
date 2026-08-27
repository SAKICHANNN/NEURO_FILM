from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.three_stock_structural_diagnostics import (
    ThreeStockStructuralDiagnosticError,
    run_diagnostics,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u4_3a_three_stock_structural_diagnostics_v1.json"


def test_u4_3a_real_outputs_are_deterministic_diagnostics() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    forward = run_diagnostics(config, ROOT)
    reverse = run_diagnostics(config, ROOT, reverse=True)
    assert forward == reverse
    assert forward["status"] == "PASS_DIAGNOSTIC_REVIEW_QUEUES"
    payload = forward["scientific_payload"]
    assert len(payload["rows"]) == 64
    assert len(payload["review_queues"]) == 7
    assert payload["aggregate_scalar_score_present"] is False
    assert payload["automatic_veto_present"] is False
    assert forward["render_calls"] == 0
    assert forward["network_reads"] == 0


def test_u4_3a_rejects_input_identity_drift() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["inputs"]["rf3_d0_report"]["sha256"] = "0" * 64
    with pytest.raises(
        ThreeStockStructuralDiagnosticError, match="input hash mismatch"
    ):
        run_diagnostics(config, ROOT)
