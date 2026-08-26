from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = json.loads(
    (
        ROOT / "docs/evidence/P253_REFERENCE_PRODUCT_CHAIN_ANDROID_RUNTIME_RESULT.json"
    ).read_text(encoding="utf-8")
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p253_evidence_binds_committed_inputs_and_reports() -> None:
    bindings = EVIDENCE["bindings"]
    for prefix in ("config", "contract", "runner", "fixture"):
        path = ROOT / bindings[f"{prefix}_path"]
        assert path.stat().st_size == bindings[f"{prefix}_bytes"]
        assert _sha256(path) == bindings[f"{prefix}_sha256"]
    normal = ROOT / EVIDENCE["execution"]["normal_report_path"]
    reverse = ROOT / EVIDENCE["execution"]["reverse_report_path"]
    assert normal.read_bytes() == reverse.read_bytes()
    assert normal.stat().st_size == EVIDENCE["execution"]["report_bytes_each"]
    assert _sha256(normal) == EVIDENCE["execution"]["report_sha256"]


def test_p253_report_and_evidence_gates_are_exact() -> None:
    report = json.loads(
        (ROOT / EVIDENCE["execution"]["normal_report_path"]).read_text(encoding="utf-8")
    )
    assert report["status"] == EVIDENCE["status"]
    assert report["stable_evidence_id"] == EVIDENCE["execution"]["stable_evidence_id"]
    assert all(report["gates"].values())
    assert all(EVIDENCE["gates"].values())
    assert len(report["runtime"]["identities"]) == 10
    assert len(report["runtime"]["truth_table"]) == 8
    assert len(report["runtime"]["negative_controls"]) == 4


def test_p253_claim_ceiling_forbids_promotion() -> None:
    assert EVIDENCE["status"].startswith("PASS_PRIVATE_")
    assert EVIDENCE["scientific_candidate_consumed"] is False
    for key in (
        "public_package",
        "schema_mapping",
        "capability_mapping",
        "product_mapping",
        "consumer_promotion",
    ):
        assert EVIDENCE[key] is False
    assert "arm64-v8a link-only" in EVIDENCE["claim_ceiling"]
