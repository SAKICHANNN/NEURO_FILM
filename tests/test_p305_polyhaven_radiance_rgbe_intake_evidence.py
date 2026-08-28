from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P305_POLYHAVEN_RADIANCE_RGBE_INTAKE_RESULT.json"


def test_p305_evidence_binds_private_unlabelled_radiance_primitive() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["decision"] == "PASS_PRIVATE_POLYHAVEN_RADIANCE_RGBE_INTAKE"
    assert all(evidence["passed_gates"].values())
    assert evidence["formal_reports"]["byte_exact"]
    assert evidence["formal_reports"]["sha256_each"] == (
        "b5d1227cacc191d6f83cecf1ccd9e6f093d61d71feeafa2c62d08cdc94022325"
    )
    oracle = evidence["oracle_discrepancy_and_resolution"]
    assert oracle["raw_opencv_difference_components"] == 512 * 1024 * 3
    assert oracle["opencv_plus_exact_half_bin_difference_components"] == 0
    assert oracle["opencv_plus_exact_half_bin_max_abs"] == 0.0
    assert evidence["access_accounting"]["formal_network_requests"] == 0
    claim = evidence["claim_ceiling"].casefold()
    assert "primaries" in claim and "workingimage" in claim and "candidate 3" in claim
