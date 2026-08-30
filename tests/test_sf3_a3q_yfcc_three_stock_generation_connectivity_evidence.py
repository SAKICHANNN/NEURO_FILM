from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_sf3_a3q_formal_evidence_is_bound_and_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence_path = (
        root
        / "docs/evidence/SF3_A3Q_YFCC_THREE_STOCK_GENERATION_CONNECTIVITY_RESULT.json"
    )
    payload = evidence_path.read_bytes()
    evidence = json.loads(payload)

    assert hashlib.sha256(payload).hexdigest() == (
        "a46e77852e73ac6ad0c7efbf9615aeea0873dd4186b88c3e89792008aae72f07"
    )
    assert evidence["decision"] == (
        "FAIL_CLOSED_YFCC_THREE_STOCK_CURRENT_PORTRA_AUTHOR_CONNECTIVITY"
    )
    assert evidence["stable_evidence_id"] == (
        "3a91fffee2d8cb3ed88e3d16b0738aeb71d5a7918ebb53f090d73a4962db1b3d"
    )
    assert evidence["connectivity"]["all_three_author_uid_count"] == 7
    assert (
        evidence["portra_generation"][
            "qualifying_current_portra_three_stock_author_uid_count"
        ]
        == 4
    )
    assert evidence["portra_generation"]["required_minimum"] == 5
    assert not evidence["gates"]["minimum_current_portra_three_stock_author_uids"]
    assert all(
        count == 0
        for operation, count in evidence["operation_counts"].items()
        if operation != "source_report_reads"
    )
