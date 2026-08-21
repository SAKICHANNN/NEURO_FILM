from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P93_DNG_STANDARD_DEFAULT_CONFORMANCE_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p93_evidence_binds_frozen_tracked_inputs() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL_CLOSED_DNG_STANDARD_DEFAULT_CONFORMANCE"
    for key in (
        "contract",
        "config",
        "v1_implementation",
        "v2_implementation",
        "runner",
    ):
        binding = evidence["bindings"][key]
        assert _sha256(ROOT / binding["path"]) == binding["sha256"]


def test_p93_failure_is_replay_exact_and_scientifically_decisive() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["execution"]["fresh_process_report_count"] == 4
    assert evidence["execution"]["forward_reverse_byte_exact"] is True
    assert evidence["execution"]["two_fresh_process_reports_byte_exact"] is True
    assert evidence["metrics"]["maximum_black_level_error_codes"] == 3
    assert (
        evidence["metrics"]["maximum_black_level_error_row"] == "autel_robotics_xb015"
    )
    assert evidence["metrics"]["pixel_or_rgb_decode_calls"] == 0
    assert evidence["rescue"]["black_level_tolerance_change"] is False
