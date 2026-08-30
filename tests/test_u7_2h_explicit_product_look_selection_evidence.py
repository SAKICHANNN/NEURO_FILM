from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_2H_EXPLICIT_PRODUCT_LOOK_SELECTION_RESULT.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u7_2h_evidence_binds_formal_committed_head_result() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS"
    for key in ("contract", "config", "cli", "formal_runner"):
        binding = evidence["bindings"][key]
        assert_historical_evidence_binding(ROOT, binding)
    reports = evidence["formal_reports"]
    forward = ROOT / reports["forward_path"]
    reverse = ROOT / reports["reverse_path"]
    assert forward.read_bytes() == reverse.read_bytes()
    assert len(forward.read_bytes()) == reports["bytes_each"]
    assert _sha(forward) == reports["sha256"]
    report = json.loads(forward.read_text(encoding="utf-8"))
    assert report["implementation_commit"] == evidence["implementation_commit"]
    assert report["status"] == "PASS"
    assert all(report["gates"].values())
    assert report["owned_runtime_residue_zero"] is True


def test_u7_2h_evidence_preserves_claim_ceiling() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    result = evidence["selection_result"]
    retained = evidence["retained_boundaries"]
    assert result["omitted_product_style_rejected_before_input_decode"] is True
    assert result["explicit_available_look_count"] == 3
    assert result["explicit_product_output_hashes_match_prechange"] is True
    assert result["explicit_product_normalized_recipe_hashes_match_prechange"] is True
    assert retained["product_profile_requires_manual_look_selection"] is True
    assert retained["three_colour_algorithms_or_parameters_changed"] is False
    assert retained["ao6_promoted"] is False
    assert retained["stock_separation_or_calibration_claim_opened"] is False
