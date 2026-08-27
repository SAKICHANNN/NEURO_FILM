from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rf3_d15_evidence_is_bound_to_formal_replay() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads(
        (
            root
            / "docs/evidence/RF3_D15_THREE_STOCK_AUTONOMOUS_BLIND_SALIENCE_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        evidence["status"]
        == "FAIL_CLOSED_K1_THREE_STOCK_AUTONOMOUS_VISUAL_DISTINGUISHABILITY"
    )
    forward = root / evidence["formal_replay"]["forward_report"]
    reverse = root / evidence["formal_replay"]["reverse_report"]
    assert forward.read_bytes() == reverse.read_bytes()
    assert _sha256(forward) == evidence["formal_replay"]["report_sha256"]
    assert (
        _sha256(root / evidence["observations"]["path"])
        == evidence["observations"]["sha256"]
    )
    assert [
        row["both_rounds_visible_source_count"] for row in evidence["pair_results"]
    ] == [6, 13, 12]
    assert not any(row["pair_gate_pass"] for row in evidence["pair_results"])


def test_rf3_d15_retains_claim_boundaries() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads(
        (
            root
            / "docs/evidence/RF3_D15_THREE_STOCK_AUTONOMOUS_BLIND_SALIENCE_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["observations"]["mapping_read_before_commit"] is False
    assert evidence["observations"]["prior_labeled_exposure_disclosed"] is True
    assert (
        evidence["retained_boundaries"][
            "controlled_or_new_identifying_stock_evidence_still_required"
        ]
        is True
    )
    assert (
        evidence["retained_boundaries"]["product_or_multi_stock_completion_opened"]
        is False
    )
