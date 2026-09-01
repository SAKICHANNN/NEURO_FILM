from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_19A_SRW_ARQ_GENERIC_PRODUCT_INGRESS_RESULT.json"


def _git_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)


def _assert_report_pair(
    payload: dict[str, object], prefix: str, expected_identity: str
) -> tuple[dict[str, object], dict[str, object]]:
    reports = payload["reports"]
    assert isinstance(reports, dict)
    forward = (ROOT / str(reports[f"{prefix}_forward_path"])).read_bytes()
    reverse = (ROOT / str(reports[f"{prefix}_reverse_path"])).read_bytes()
    assert forward == reverse
    assert len(forward) == reports[f"{prefix}_bytes"]
    assert hashlib.sha256(forward).hexdigest() == reports[f"{prefix}_sha256"]
    forward_payload = json.loads(forward)
    reverse_payload = json.loads(reverse)
    assert forward_payload["scientific_identity"] == f"sha256:{expected_identity}"
    assert reverse_payload["scientific_identity"] == f"sha256:{expected_identity}"
    return forward_payload, reverse_payload


def test_u7_19a_evidence_binds_exact_product_ingress_result() -> None:
    payload = json.loads(EVIDENCE.read_text("utf-8"))
    assert payload["status"] == "PASS_PRIVATE_U7_19A_SRW_ARQ_GENERIC_PRODUCT_INGRESS"

    preflight, _ = _assert_report_pair(
        payload, "preflight", payload["reports"]["preflight_scientific_identity"]
    )
    formal, _ = _assert_report_pair(
        payload, "formal", payload["reports"]["formal_scientific_identity"]
    )
    assert preflight["passed_extensions"] == [".arq", ".srw"]
    assert all(all(result["gates"].values()) for result in preflight["results"])
    assert all(formal["gates"].values())
    assert formal["admitted_extensions"] == [".arq", ".srw"]
    assert formal["scratch_residue_files"] == 0

    for path, identity in payload["bindings"].items():
        value = _git_bytes(path)
        assert len(value) == identity["bytes"]
        assert hashlib.sha256(value).hexdigest() == identity["sha256"]
        blob = subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{path}"], cwd=ROOT, text=True
        ).strip()
        assert blob == identity["git_blob"]

    assert payload["product_chain"]["strict_replay_byte_exact"] is True
    assert payload["product_chain"]["network_requests"] == 0
    assert payload["claim"]["mode"] == "film-inspired"
    assert payload["claim"]["evidence_grade"] == "look-approximation"
    assert payload["claim"]["general_srw_arq_support"] is False
    assert payload["claim"]["calibrated_stock_response"] is False
    assert payload["claim"]["physical_film_reproduction"] is False
