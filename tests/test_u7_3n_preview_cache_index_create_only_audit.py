from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_u7_3n_preview_cache_index_create_only import run_audit

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/evidence/U7_3N_PREVIEW_CACHE_INDEX_CREATE_ONLY_RESULT.json"
)


def test_u7_3n_formal_audit_passes_both_orders() -> None:
    forward = run_audit("forward")
    reverse = run_audit("reverse")

    assert forward == reverse
    assert forward["status"] == (
        "PASS_PRIVATE_U7_3N_PREVIEW_CACHE_INDEX_CREATE_ONLY_REPAIR"
    )
    assert all(forward["gates"].values())
    assert {row["case_id"] for row in forward["rows"]} == {
        "success",
        "existing-destination",
        "late-foreign-destination",
        "clean-publication-failure",
        "foreign-stage-replacement",
    }


def test_u7_3n_tracked_evidence_is_exact() -> None:
    payload = EVIDENCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "4d1495adc9fdf0fa9443b48fac42f783a4ec742fd537554c4b43bc5b7018d850"
    )
    evidence = json.loads(payload)
    assert evidence["status"] == (
        "PASS_PRIVATE_U7_3N_PREVIEW_CACHE_INDEX_CREATE_ONLY_REPAIR"
    )
    assert evidence["stable_identity"] == (
        "57fd93ceb99ac9cd660e21dc91363cbd8fca1d080eed84eee1858b680376b4a1"
    )
    assert all(evidence["gates"].values())
