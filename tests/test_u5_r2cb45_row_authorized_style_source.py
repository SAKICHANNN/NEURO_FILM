from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "configs/u5_r2cb45_row_authorized_style_source_decision_v1.json"


def test_cb45_source_role_binds_exact_parent_and_manifest() -> None:
    payload = json.loads(DECISION.read_text(encoding="utf-8"))
    parent = ROOT / payload["parent_decision_path"]
    manifest = ROOT / payload["manifest_path"]
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == payload[
        "parent_decision_sha256"
    ]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == payload[
        "manifest_sha256"
    ]
    parent_payload = json.loads(parent.read_text(encoding="utf-8"))
    assert parent_payload["decision"] == payload["parent_required_decision"]


def test_cb45_source_role_exactly_matches_manifest_population() -> None:
    payload = json.loads(DECISION.read_text(encoding="utf-8"))
    rows = json.loads((ROOT / payload["manifest_path"]).read_text(encoding="utf-8"))
    assert [row["id"] for row in rows] == payload["included_source_ids"]
    assert len(rows) == payload["source_count_exact"] == 12
    assert len({row["make"] for row in rows}) == payload["camera_make_count_exact"]
    assert payload["previously_consumed_mechanism_development_role"] is True
    assert payload["operator_outputs_inspected_for_this_role"] is False
