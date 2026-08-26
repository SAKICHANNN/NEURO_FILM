from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_6b_three_stock_file_batch_v1.json"
EVIDENCE = ROOT / "docs/evidence/U7_6B_THREE_STOCK_FILE_BATCH_RESULT.json"


def test_u7_6b_evidence_binds_contract_and_passes_exact_file_batch() -> None:
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert report["config_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert report["decision"] == "PASS"
    assert all(report["gate_results"].values())
    assert report["metrics"]["wall_ratio"] <= 0.95
    assert report["metrics"]["peak_rss_ratio"] <= 1.05
    assert report["metrics"]["candidate_residue_count"] == 0
    assert report["execution_amendment"]["algorithm_or_gate_change"] is False
