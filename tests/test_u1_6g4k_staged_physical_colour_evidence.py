from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U1_6G4K_STAGED_PHYSICAL_COLOUR_HALATION_RESULT.json"
CONFIG = ROOT / "configs/u1_6g4k_staged_physical_colour_halation_v1.json"
CORE = ROOT / "src/filmfx/staged_colour.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u1_6g4k_evidence_binds_frozen_sources_and_failure() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL_CLOSED_LEGACY_UNPREMULTIPLIED_RGB_DRIFT"
    assert evidence["identities"]["config_sha256"] == _sha256(CONFIG)
    assert evidence["identities"]["core_sha256"] == _sha256(CORE)
    assert (
        evidence["identities"]["forward_report_sha256"]
        == evidence["identities"]["reverse_report_sha256"]
    )
    assert evidence["execution"]["staged_v2_all_policies_pass"] is True
    assert evidence["legacy_compatibility"]["passing_cases"] == 2
    assert evidence["legacy_compatibility"]["total_cases"] == 4
    assert (
        evidence["legacy_compatibility"]["maximum_unpremultiplied_rgb_drift"]
        > evidence["legacy_compatibility"]["frozen_maximum_unpremultiplied_rgb_drift_gate"]
    )
    assert evidence["decision"].startswith("retain implementation as closed")
