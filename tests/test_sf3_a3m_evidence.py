from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/SF3_A3M_IA_EKTAR_FIXED_CAMERA_IDENTIFIABILITY_RESULT.json"
CONFIG = ROOT / "configs/sf3_a3m_ia_ektar_fixed_camera_identifiability_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sf3_a3m_evidence_binds_frozen_inputs_and_failure() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert evidence["status"] == "FAIL_CLOSED"
    assert evidence["operator_fitting_allowed"] is False
    assert evidence["training_allowed"] is False
    assert evidence["bindings"]["config_sha256"] == _sha256(CONFIG)
    assert evidence["bindings"]["source_lock_sha256"] == config["source"]["source_lock_sha256"]
    assert evidence["formal_replay"]["reports_byte_exact"] is True
    assert evidence["formal_replay"]["primary"]["balanced_accuracy"] == 0.875
    assert evidence["formal_replay"]["nuisance_controls"]["capture_date_balanced_accuracy"] == 1.0
    assert evidence["formal_replay"]["primary_minus_strongest_nuisance"] == -0.125
    assert len(evidence["failed_gates"]) == 4
