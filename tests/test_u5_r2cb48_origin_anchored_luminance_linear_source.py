from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "configs/u5_r2cb48_origin_anchored_luminance_linear_source_decision_v1.json"


def test_cb48_source_role_binds_exact_disjoint_population() -> None:
    payload = json.loads(DECISION.read_text(encoding="utf-8"))
    parent = ROOT / payload["parent_decision_path"]
    manifest = ROOT / payload["manifest_path"]
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == payload[
        "parent_decision_sha256"
    ]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == payload[
        "manifest_sha256"
    ]
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    assert [row["id"] for row in rows] == payload["included_source_ids"]
    assert len(rows) == payload["source_count_exact"] == 17
    assert len({row["make"] for row in rows}) == payload["camera_make_count_exact"]
    assert payload["exact_decoded_sha_overlap_with_cb47"] == 0
