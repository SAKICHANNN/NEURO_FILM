from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_p91_contract_preserves_p90_failure() -> None:
    config = json.loads(
        (ROOT / "configs/p91_canonical_pq_png_partition_invariance_v1.json").read_text(
            encoding="utf-8"
        )
    )
    evidence = ROOT / "docs/evidence/P90_DNG_ACES2_HDR_PQ_PNG_RESULT.json"
    assert config["status"] == "FROZEN_BEFORE_P91_IMPLEMENTATION"
    assert config["uncompressed_feed_bytes"] == 65536
    assert __import__("hashlib").sha256(evidence.read_bytes()).hexdigest() == config[
        "p90_evidence_sha256"
    ]
