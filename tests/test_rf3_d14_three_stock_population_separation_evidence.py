from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/RF3_D14_THREE_STOCK_POPULATION_SEPARATION_RESULT.json"
REPORT_A = ROOT / "outputs/eval/rf3_d14_three_stock_population_separation_v1/report_forward.json"
REPORT_B = ROOT / "outputs/eval/rf3_d14_three_stock_population_separation_v1/report_reverse.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_rf3_d14_evidence_binds_exact_failed_formal_replay() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    forward = REPORT_A.read_bytes()
    reverse = REPORT_B.read_bytes()

    assert forward == reverse
    assert len(forward) == evidence["formal_replay"]["report_bytes"]
    assert _sha256(forward) == evidence["formal_replay"]["report_sha256"]

    report = json.loads(forward)
    assert report["status"] == "FAIL_CLOSED"
    assert report["scientific_identity"] == evidence["formal_replay"]["scientific_identity"]
    assert report["scientific_payload"]["gates"] == evidence["gates"]
    assert evidence["status"] == "FAIL_CLOSED_K1_THREE_STOCK_POPULATION_SEPARATION"
    assert [row["source_pass_count_at_delta_e76_ge_1"] for row in evidence["pair_results"]] == [16, 15, 9]
    assert not evidence["pair_results"][-1]["pair_gate_pass"]
    assert "not blind human distinguishability" in evidence["claim_ceiling"]
