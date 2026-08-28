from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/evidence/P307_CAMBRIDGE_HDR_DEGHOST_SOURCE_FEASIBILITY_RESULT.json"
)


def test_p307_evidence_binds_exact_zero_pixel_structure_failure() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["decision"] == (
        "FAIL_CLOSED_CAMBRIDGE_HDR_DEGHOST_SOURCE_STRUCTURE_GAP_NOT_SCIENTIFIC_RESULT"
    )
    assert evidence["formal_reports"]["byte_exact"]
    assert evidence["formal_reports"]["sha256_each"] == (
        "5a5572624ab6d77a67c5b0c1622eaa6bb76233571e72df1d815bc2dd4ab03d7b"
    )
    gates = evidence["gates"]
    assert not gates["complete_paired_roles"]
    assert all(value for name, value in gates.items() if name != "complete_paired_roles")
    incomplete = evidence["archive_results"]["exposure_stacks_part2.zip"][
        "decisive_incomplete_group"
    ]
    assert incomplete == {
        "category": "losm",
        "image_set": "image_set2",
        "ghosted_jpg": 6,
        "ghosted_raw": 6,
        "ground_truth_jpg": 5,
        "ground_truth_raw": 5,
    }
    access = evidence["access_accounting_per_report"]
    assert access["part1_requests"] == 0
    assert access["member_payload_reads"] == 0
    assert access["pixel_decodes"] == 0
    claim = evidence["claim_ceiling"].casefold()
    assert "candidate 3" in claim and "product" in claim
