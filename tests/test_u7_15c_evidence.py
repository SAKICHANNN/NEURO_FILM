from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_15C_DESKTOP_BATCH_REPRESENTATIVE_PATH_DISCLOSURE_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u7_15c_evidence_binds_exact_reports_and_sources() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    assert evidence["status"] == (
        "PASS_PRIVATE_U7_15C_DESKTOP_BATCH_REPRESENTATIVE_PATH_DISCLOSURE"
    )
    reports = evidence["formal_reports"]
    forward = ROOT / reports["forward_path"]
    reverse = ROOT / reports["reverse_path"]
    assert forward.read_bytes() == reverse.read_bytes()
    assert len(forward.read_bytes()) == reports["bytes_each"] == 4891
    assert _sha256(forward) == reports["sha256"]
    payload = json.loads(forward.read_text("utf-8"))
    assert payload["formal_source_commit"] == evidence["formal_source_commit"]
    assert payload["stable_identity"] == reports["scientific_identity"]
    assert all(payload["scientific"]["gates"].values())

    for relative, binding in evidence["bindings"].items():
        path = ROOT / relative
        assert path.stat().st_size == binding["bytes"]
        assert _sha256(path) == binding["sha256"]
        blob = subprocess.run(
            ("git", "rev-parse", f"{evidence['formal_source_commit']}:{relative}"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        assert blob == binding["git_blob"]

    result = evidence["result"]
    assert result["visible_values"] == [
        "Automatic · canonical first",
        "001 · same-name.png",
        "002 · same-name.png",
    ]
    assert result["visible_parent_paths_hidden"] is True
    assert result["duplicate_basename_selection_exact"] is True
    assert result["default_explicit_batch_tree_exact"] is True
    assert result["visible_strength_percent"] == 63
    assert result["visible_strength_amount"] == 0.63
    assert all(row["report_created"] is False for row in evidence["excluded_preformal_diagnostics"])
