from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs/evidence/P312_R1FV_DNG_PROFILE_GAIN_TABLE2_STAGE_NO_COPY_INTAKE_RESULT.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p312_evidence_binds_reports_and_preserves_gamma_boundary() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == (
        "PASS_PRIVATE_R1FV_DNG_PROFILE_GAIN_TABLE2_STAGE_NO_COPY_INTAKE"
    )
    for key in ("config", "audit", "forward_report", "reverse_report"):
        binding = evidence["bindings"][key]
        assert _sha(ROOT / binding["path"]) == binding["sha256"]
    assert evidence["formal_result"]["gates_passed"] == 10
    assert evidence["formal_result"]["gamma2_discriminating"] is False
    assert evidence["formal_result"]["general_gamma_claim"] is False
    assert evidence["rights_and_product"]["product_mapping"] is False
