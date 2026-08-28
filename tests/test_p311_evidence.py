from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs/evidence/P311_R1FU_DNG_PER_PROFILE_GAIN_TABLE_NO_COPY_INTAKE_RESULT.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p311_evidence_binds_fail_closed_reports_and_inputs() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == (
        "FAIL_CLOSED_R1FU_DNG_PER_PROFILE_GAIN_TABLE_NO_COPY_INTAKE"
    )
    assert evidence["formal_result"]["failed_gates"] == [
        "parse-and-read-summaries-exact",
        "producer-head-exact",
    ]
    for key in ("config", "audit"):
        binding = evidence["bindings"][key]
        assert _sha(ROOT / binding["path"]) == binding["sha256"]
    for key in ("forward_report", "reverse_report"):
        binding = evidence["bindings"][key]
        assert _sha(ROOT / binding["path"]) == binding["sha256"]
    assert evidence["rights_and_product"]["candidate_3"] is False
    assert evidence["rights_and_product"]["product_mapping"] is False
