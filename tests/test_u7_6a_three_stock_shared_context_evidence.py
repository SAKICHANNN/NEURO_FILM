from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_6a_three_stock_shared_context_v1.json"
EVIDENCE = ROOT / "docs/evidence/U7_6A_THREE_STOCK_SHARED_CONTEXT_RESULT.json"


def test_u7_6a_evidence_binds_config_and_passes_every_gate() -> None:
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert report["config_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert report["decision"] == "PASS"
    assert all(report["gate_results"].values())
    assert report["metrics"]["median_wall_ratio"] <= 0.97
    assert report["metrics"]["median_peak_rss_ratio"] <= 1.05
    assert len(set(report["output_sha256"])) == 3
