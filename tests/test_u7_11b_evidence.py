from __future__ import annotations

import json
from pathlib import Path

from src.inference.render_contract import sha256_file

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / (
    "docs/evidence/U7_11B_INSTALLED_RUNTIME_DESKTOP_BATCH_CONFIRMATION_RESULT.json"
)


def test_u7_11b_evidence_preserves_exact_terminal_negative() -> None:
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert sha256_file(EVIDENCE) == (
        "d8f14c155df6efceb646407b4ceff67e5e8cda82bcff8dc4ba696cb7af6193e4"
    )
    assert report["source_commit"] == "e58cfa4fa7c4f42eb7723b48eff60fd519210789"
    assert report["status"] == (
        "FAIL_CLOSED_U7_11B_INSTALLED_RUNTIME_DESKTOP_BATCH_CONFIRMATION"
    )
    assert report["automatic_pass"] is False
    failed = {
        name for name, passed in report["scientific"]["gates"].items() if not passed
    }
    assert failed == {"installed_child_python_exact"}
    assert report["formal_root_residue_count"] == 0


def test_u7_11b_negative_does_not_erase_mechanical_signal() -> None:
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    cases = report["scientific"]["batch_cases"]
    assert len(cases) == 2
    assert all(row["png16_exact"] for row in cases)
    assert all(row["raw_recipes_exact"] for row in cases)
    assert all(row["raw_batch_json_exact"] for row in cases)
    assert all(row["canonical_receipt_identity_exact"] for row in cases)
    assert all(
        all(row["installed"]["replay_exact"].values()) for row in cases
    )
    runtime = report["scientific"]["runtime"]["python"]["executable"]
    observed = [
        command
        for row in cases
        for command in (
            row["installed"]["worker_python"],
            *row["installed"]["child_argv0"],
        )
    ]
    assert observed and set(observed) == {runtime}


def test_u7_11b_claim_ceiling_remains_look_approximation() -> None:
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    claim = report["scientific"]["claim_ceiling"]
    assert claim["mode"] == "film-inspired"
    assert claim["evidence_grade"] == "look-approximation"
    assert claim["calibrated_stock_response"] is False
    assert claim["physical_film_reproduction"] is False
    assert claim["stock_distinguishability"] is False
    assert claim["native_dialog_automation"] is False
    assert claim["public_release"] is False
