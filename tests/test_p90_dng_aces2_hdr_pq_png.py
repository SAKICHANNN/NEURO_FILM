from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_p90_contract_is_frozen_and_bound() -> None:
    config = json.loads(
        (ROOT / "configs/p90_dng_aces2_hdr_pq_png_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["status"] == "FROZEN_BEFORE_P90_FILE_PUBLICATION"
    assert config["target"] == "hdr_rec2020_pq"
    assert config["required_rows"] == 4
    assert len(config["bindings"]) == 4


def test_p90_evidence_preserves_partition_replay_failure() -> None:
    evidence = json.loads(
        (ROOT / "docs/evidence/P90_DNG_ACES2_HDR_PQ_PNG_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["status"] == "FAIL_CLOSED_PARTITION_BYTE_REPLAY"
    assert evidence["metrics"]["rgb16_sample_hashes_match_across_partitions"] == 4
    assert evidence["metrics"]["png_file_hashes_match_across_partitions"] == 1
    assert evidence["metrics"]["png_file_hashes_differ_across_partitions"] == 3
    assert not evidence["execution"]["forward_reverse_report_exact"]
