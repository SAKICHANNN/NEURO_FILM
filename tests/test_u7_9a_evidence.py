from __future__ import annotations

import json
from pathlib import Path

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs"
    / "evidence"
    / "U7_9A_PRIVATE_WINDOWS_PRODUCT_RUNTIME_INSTALL_RESULT.json"
)


def test_u7_9a_evidence_binds_pass_and_sources() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    assert (
        evidence["status"]
        == "PASS_PRIVATE_WINDOWS_REPOSITORY_BOUND_PRODUCT_RUNTIME_INSTALL"
    )
    assert evidence["automatic_pass"] is True
    assert evidence["formal_reports"]["byte_exact"] is True
    assert evidence["runtime"]["wheel_count"] == 12
    assert evidence["runtime"]["wheel_bytes"] == 111666287
    assert evidence["renders"]["direct_launcher_byte_exact"] is True
    for name, value in evidence["gates"].items():
        if name.endswith("_count"):
            assert value == 0
        else:
            assert value is True
    for relative, binding in evidence["source_bindings"].items():
        assert_historical_evidence_binding(ROOT, {"path": relative, **binding})


def test_u7_9a_evidence_preserves_claim_ceiling() -> None:
    claim = json.loads(EVIDENCE.read_text("utf-8"))["claim"]
    assert claim["mode"] == "film-inspired"
    assert claim["evidence_grade"] == "look-approximation"
    assert claim["calibrated_stock_response"] is False
    assert claim["physical_film_reproduction"] is False
    assert claim["standalone_or_portable_installer"] is False
    assert claim["public_release"] is False
