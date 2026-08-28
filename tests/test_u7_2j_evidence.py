from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_2J_PRODUCT_RECIPE_CATALOG_ENFORCEMENT_RESULT.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u7_2j_evidence_binds_current_sources_and_exact_formal_replay() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS"
    for key in ("contract", "config", "core", "focused_test", "formal_runner"):
        binding = evidence["bindings"][key]
        assert _sha(ROOT / binding["path"]) == binding["sha256"]

    reports = evidence["formal_reports"]
    forward = ROOT / reports["forward_path"]
    reverse = ROOT / reports["reverse_path"]
    assert forward.read_bytes() == reverse.read_bytes()
    assert forward.stat().st_size == reports["bytes_each"]
    assert _sha(forward) == reports["sha256"]
    report = json.loads(forward.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert all(report["gates"].values())
    assert report["owned_runtime_residue_zero"] is True
    assert all(
        row["build_error"] and row["verify_error"] and row["replay_error"]
        for row in report["rejection_results"].values()
    )
    assert all(
        row["output_sha256"] == row["expected_u7_2i_output_sha256"]
        and row["repeat_replay_exact"]
        for row in report["product_results"].values()
    )
