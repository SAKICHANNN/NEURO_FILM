from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.local_vision3_source_gate import (
    LocalVision3SourceGateError,
    audit_local_source,
    canonical_json,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bu5_local_vision3_source_gate_v1.json"


def test_contract_rejects_rights_gate_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["gates"]["all_rows_require_explicit_derivative_rights"] = False
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LocalVision3SourceGateError, match="contract drift"):
        load_contract(changed)


def test_audit_is_exact_and_reads_no_pixels() -> None:
    config = load_contract(CONFIG)
    first = audit_local_source(config, ROOT)
    second = audit_local_source(config, ROOT)
    assert canonical_json(first) == canonical_json(second)
    assert first["pixel_file_reads"] == 0
    assert first["selected_row_count"] == 1103


def test_legacy_local_corpus_fails_required_source_evidence() -> None:
    report = audit_local_source(load_contract(CONFIG), ROOT)
    assert report["gate_pass"] is False
    assert report["row_counts_by_style"] == {
        "vision3_50d": 0,
        "vision3_250d": 539,
        "vision3_500t": 564,
    }
    assert report["explicit_rights_row_count"] == 0
    assert report["source_url_row_count"] == 0
    assert report["roll_process_scanner_lineage_row_count"] == 0
    assert report["decision"].startswith("quarantine_")
