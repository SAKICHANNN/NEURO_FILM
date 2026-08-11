from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "configs/u5_r2cb51_analytic_y_chromaticity_source_decision_v1.json"


def test_cb51_source_role_is_exact_and_disjoint_from_cb50() -> None:
    payload = json.loads(DECISION.read_text(encoding="utf-8"))
    parent = ROOT / payload["parent_decision_path"]
    manifest = ROOT / payload["manifest_path"]
    assert (
        hashlib.sha256(parent.read_bytes()).hexdigest()
        == payload["parent_decision_sha256"]
    )
    assert (
        hashlib.sha256(manifest.read_bytes()).hexdigest() == payload["manifest_sha256"]
    )
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in rows}
    selected = [by_id[source_id] for source_id in payload["included_source_ids"]]
    assert len(selected) == payload["source_count_exact"] == 17
    assert len({row["make"] for row in selected}) == payload["camera_make_count_exact"]
    assert payload["exact_decoded_sha_overlap_with_cb50"] == 0
    assert set(payload["excluded_parent_rows"]) == {"fujifilm_s2pro"}
