from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs/evidence/U7_2V_PRODUCT_PHYSICAL_HALATION_PREFLIGHT_RESULT.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u7_2v_evidence_binds_exact_formal_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    reports = evidence["formal_reports"]
    forward = ROOT / reports["forward_path"]
    reverse = ROOT / reports["reverse_path"]

    assert (
        evidence["decision"]
        == "PASS_PRIVATE_U7_2V_PRODUCT_PHYSICAL_HALATION_PREFLIGHT"
    )
    assert evidence["result"]["all_frozen_gates_pass"] is True
    assert all(evidence["gates"].values())
    assert forward.read_bytes() == reverse.read_bytes()
    assert forward.stat().st_size == reports["bytes_each"]
    assert _sha256(forward) == reports["sha256"]


def test_u7_2v_evidence_preserves_claim_and_parent_boundaries() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    ceiling = evidence["claim_ceiling"].casefold()
    result = evidence["result"]

    assert "look approximation" in ceiling
    assert "does not validate physical-halation realism" in ceiling
    assert "not a package" in ceiling
    assert result["expert_rejection_count"] == 3
    assert result["numeric_rejection_count"] == 28
    assert result["valid_locked_control_count"] == 24
    assert len(result["parent_output_sha256"]) == 3
