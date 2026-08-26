from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p99_evidence_binds_exact_failed_report() -> None:
    evidence = json.loads(
        Path("docs/evidence/P99_DNG_BASELINE_EXPOSURE_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    report = evidence["bindings"]["raw_report"]
    assert _sha256(Path(report["path"])) == report["sha256"]
    assert evidence["status"] == "FAIL_CLOSED_DNG_BASELINE_EXPOSURE"
    failed = {name for name, passed in evidence["gates"].items() if not passed}
    assert failed == {"no_new_exact_boundary"}
    assert evidence["execution"]["reports_byte_exact"]


def test_p99_claim_remains_private_and_product_closed() -> None:
    evidence = json.loads(
        Path("docs/evidence/P99_DNG_BASELINE_EXPOSURE_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["metrics"]["blackmagic_total_ev"] == 2.65412
    assert evidence["metrics"]["new_exact_boundary_count"] == 1
    assert evidence["metrics"]["float32_oracle_bytes_exact"]
    assert evidence["execution"]["default_loader_changes"] == 0
    assert "no arbitrary DNG" in evidence["claim_ceiling"]
    assert "product" in evidence["claim_ceiling"]
