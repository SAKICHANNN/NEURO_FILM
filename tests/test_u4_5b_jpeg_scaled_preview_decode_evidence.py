from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U4_5B_JPEG_SCALED_PREVIEW_DECODE_RESULT.json"


def test_u4_5b_evidence_passes_frozen_quality_and_resource_gates() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_JPEG_SCALED_PREVIEW_DECODE_PRODUCT_LATENCY_OPEN"
    assert all(evidence["gates"].values())
    assert evidence["decode"]["decoder_scaled_dimensions"] == [1008, 1512]
    assert evidence["decode"]["preview_dimensions"] == [816, 1224]
    assert evidence["comparison_to_u4_5a_preview_1mp"]["maximum_rss_ratio"] < 0.1
    assert max(row["rgb_rmse_vs_downsampled_full_output"] for row in evidence["fidelity"]) <= 0.03
    assert max(row["rgb_absolute_error_p95"] for row in evidence["fidelity"]) <= 0.08
    assert len(set(evidence["output_hashes"].values())) == 3
