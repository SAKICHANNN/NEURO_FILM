from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.three_stock_target_evidence_equalization import (
    TargetEvidenceError,
    evaluate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/rf3_d1_three_stock_target_evidence_equalization_v1.json"


def test_current_matrix_fails_closed_without_pixel_reads() -> None:
    report = evaluate(CONFIG, ROOT)
    assert not report["automatic_pass"]
    assert (
        report["decision"]
        == "FAIL_CLOSED_COMPARABLE_THREE_STOCK_TARGET_EVIDENCE_UNAVAILABLE"
    )
    assert report["pixel_reads"] == report["operator_fits"] == report["renders"] == 0
    assert not any(report["observed_admission_facts"].values())


def test_replay_is_exact() -> None:
    assert evaluate(CONFIG, ROOT) == evaluate(CONFIG, ROOT)


def test_parent_hash_mismatch_is_invalid(tmp_path: Path) -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    contract["parents"][0]["sha256"] = "0" * 64
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(TargetEvidenceError, match="parent hash mismatch"):
        evaluate(path, ROOT)


def test_required_stock_matrix_is_complete() -> None:
    report = evaluate(CONFIG, ROOT)
    assert set(report["stock_matrix"]) == {
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    }
