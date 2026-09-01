from __future__ import annotations

import json
from pathlib import Path

from src.inference.render_contract import sha256_file

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_11A_DESKTOP_SINGLE_LOOK_BATCH_RESULT.json"


def test_u7_11a_evidence_is_exact_formal_report_and_all_gates_pass() -> None:
    report = json.loads(EVIDENCE.read_text("utf-8"))
    assert sha256_file(EVIDENCE) == "fe84f4adb85bac24dbe3bb598cca1266814cb77454f1ddf236d6df945054b2b3"
    assert report["status"] == "PASS_PRIVATE_U7_11A_DESKTOP_SINGLE_LOOK_BATCH"
    assert report["source_commit"] == "662e1c80fc8cb49d04ea09677831edd71ebc3bbf"
    assert report["implementation_commit"] == "41e1c633f4e6be157ed3cad5178c36f3cac82d79"
    assert all(report["scientific"]["gates"].values())
    assert report["scientific"]["canonical_case"]["receipt_claim"] == {
        "calibrated_stock_response": False,
        "evidence_grade": "look-approximation",
        "output_label": "film-inspired / Look Approximation",
        "physical_film_reproduction": False,
        "stock_distinguishability": False,
    }
