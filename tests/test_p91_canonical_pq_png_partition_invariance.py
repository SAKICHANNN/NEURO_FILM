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


def test_p91_evidence_binds_exact_replay() -> None:
    evidence = json.loads(
        (ROOT / "docs/evidence/P91_CANONICAL_PQ_PNG_PARTITION_INVARIANCE_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["status"] == "PASS_PRIVATE_CANONICAL_PQ_PARTITION_INVARIANCE"
    assert evidence["formal_report_sha256"] == (
        "8aa44e4e9968cf58c142728d63c7c4b370b8139a9c0d8e8db9918088f2e36498"
    )
    assert evidence["metrics"]["complete_png_hashes_match_across_partitions"] == 4
    assert evidence["execution"]["p90_failure_preserved"]
