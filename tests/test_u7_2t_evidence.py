from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_2T_PRODUCT_OUTPUT_EXTENSION_PREFLIGHT_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u7_2t_evidence_binds_exact_formal_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    reports = evidence["formal_reports"]
    forward = ROOT / reports["forward_path"]
    reverse = ROOT / reports["reverse_path"]

    assert evidence["decision"] == "PASS_PRIVATE_U7_2T_PRODUCT_OUTPUT_EXTENSION_PREFLIGHT"
    assert evidence["result"]["all_frozen_gates_pass"] is True
    assert all(evidence["gates"].values())
    assert forward.read_bytes() == reverse.read_bytes()
    assert forward.stat().st_size == reports["bytes_each"]
    assert _sha256(forward) == reports["sha256"]


def test_u7_2t_evidence_preserves_claim_ceiling() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    ceiling = evidence["claim_ceiling"].casefold()

    assert "look approximation" in ceiling
    assert "adds no output format" in ceiling
    assert "not a package" in ceiling
    assert "not" in ceiling and "calibrated stock" in ceiling
