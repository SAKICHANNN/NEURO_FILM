from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_u5_r2repid1_corrected_source import evaluate
from tests.test_u5_r2repid0_source_audit import _contract, _fetcher

ROOT = Path(__file__).resolve().parents[1]


def _correction() -> dict:
    return json.loads(
        (ROOT / "configs/u5_r2repid1_corrected_source_audit_v1.json").read_text()
    )


def test_corrected_source_audit_passes_without_image_reads() -> None:
    correction = _correction()
    base = _contract()
    base["official_sources"]["expected_repository"]["last_modified"] = correction[
        "corrections"
    ]["expected_last_modified"]
    fetch = _fetcher(base)
    correction["corrections"]["root_tree_file_facts"] = base["official_sources"][
        "expected_files"
    ]
    report = evaluate(correction, fetch=fetch, base_contract_override=base)
    assert report["automatic_pass"] is True
    assert report["gates"]["parent_failure_evidence_exact"] is True
    assert report["gates"]["exact_root_file_inventory"] is True
    assert report["requests"]["image_members"] == 0
    assert report["requests"]["operator_fits"] == 0
    assert report["decision"] == correction["decision_if_pass"]


def test_corrected_source_audit_fails_if_tree_semantics_drift() -> None:
    correction = _correction()
    correction["corrections"]["root_tree_file_facts"]["README.md"]["size"] += 1
    base = _contract()
    base["official_sources"]["expected_repository"]["last_modified"] = correction[
        "corrections"
    ]["expected_last_modified"]
    fetch = _fetcher(base)
    corrected_tree = dict(base["official_sources"]["expected_files"])
    corrected_tree["README.md"] = dict(corrected_tree["README.md"])
    corrected_tree["README.md"]["size"] += 1
    correction["corrections"]["root_tree_file_facts"] = corrected_tree
    report = evaluate(correction, fetch=fetch, base_contract_override=base)
    assert report["automatic_pass"] is False
    assert report["gates"]["exact_root_file_inventory"] is False
