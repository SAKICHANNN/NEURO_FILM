from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_19B_MULTI_VENDOR_RAW_PRODUCT_INGRESS_RESULT.json"


def _git_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)


def _report(identity: dict[str, object]) -> dict[str, object]:
    data = (ROOT / str(identity["path"])).read_bytes()
    assert len(data) == identity["bytes"]
    assert hashlib.sha256(data).hexdigest() == identity["sha256"]
    return json.loads(data)


def _records_by_extension(report: dict[str, object]) -> dict[str, dict[str, object]]:
    records = report["product_records"]
    assert isinstance(records, list)
    result: dict[str, dict[str, object]] = {}
    for record in records:
        assert isinstance(record, dict)
        extension = record.get("extension", record.get("key"))
        assert isinstance(extension, str)
        result[extension] = record
    return result


def test_u7_19b_evidence_binds_partial_multi_vendor_product_ingress() -> None:
    payload = json.loads(EVIDENCE.read_text("utf-8"))
    assert payload["status"] == (
        "PASS_PRIVATE_U7_19B_PARTIAL_MULTI_VENDOR_RAW_PRODUCT_INGRESS"
    )

    preflight_forward = _report(payload["reports"]["preflight_forward"])
    preflight_reverse = _report(payload["reports"]["preflight_reverse"])
    expected_preflight = f"sha256:{payload['reports']['preflight_scientific_identity']}"
    assert preflight_forward["scientific_identity"] == expected_preflight
    assert preflight_reverse["scientific_identity"] == expected_preflight
    assert set(preflight_forward["passed_extensions"]) == {
        ".3fr",
        ".fff",
        ".mos",
        ".nef",
        ".raf",
        ".x3f",
    }
    assert sum(len(row["records"]) for row in preflight_forward["results"]) == 12
    assert sum(len(row["records"]) for row in preflight_reverse["results"]) == 12

    formal_forward = _report(payload["reports"]["formal_forward"])
    formal_reverse = _report(payload["reports"]["formal_reverse"])
    expected_formal = f"sha256:{payload['reports']['formal_scientific_identity']}"
    assert formal_forward["scientific_identity"] == expected_formal
    assert formal_reverse["scientific_identity"] == expected_formal
    assert formal_forward["status"] == payload["status"]
    assert formal_reverse["status"] == payload["status"]
    assert (
        formal_forward["passed_extensions"] == payload["decision"]["passed_extensions"]
    )
    assert (
        formal_reverse["passed_extensions"] == payload["decision"]["passed_extensions"]
    )
    assert (
        formal_forward["failed_extensions"] == payload["decision"]["failed_extensions"]
    )
    assert (
        formal_reverse["failed_extensions"] == payload["decision"]["failed_extensions"]
    )
    assert all(formal_forward["gates"].values())
    assert all(formal_reverse["gates"].values())
    assert formal_forward["scratch_residue_files"] == 0
    assert formal_reverse["scratch_residue_files"] == 0
    assert all(formal_forward["source_immutability"].values())
    assert all(formal_reverse["source_immutability"].values())

    passing = payload["product_chain"]["passing_records"]
    limit = payload["resource_closures"]["limit_bytes"]
    for report in (formal_forward, formal_reverse):
        records = _records_by_extension(report)
        for extension, expected in passing.items():
            record = records[extension]
            assert record["worker_ok"] is True
            assert record["output"]["sha256"] == expected["output_sha256"]
            assert record["recipe"]["sha256"] == expected["recipe_sha256"]
            assert record["replay_byte_exact"] is True
            assert record["source_unchanged"] is True
            assert record["resource"]["peak_process_tree_rss_bytes"] <= limit
            claim = record["recipe"]["claim"]
            assert claim["render_mode"] == "Style-safe"
            assert claim["output_label"] == "film-inspired"
            assert claim["evidence_grade"] == "look-approximation"
            assert claim["calibrated_reference_allowed"] is False
        for extension in payload["decision"]["failed_extensions"]:
            record = records[extension]
            assert record["worker_ok"] is False
            assert record["error"] == "worker_process_tree_rss_limit_exceeded"
            assert record["resource"]["peak_process_tree_rss_bytes"] > limit

    for path, identity in payload["bindings"].items():
        data = _git_bytes(path)
        assert len(data) == identity["bytes"]
        assert hashlib.sha256(data).hexdigest() == identity["sha256"]
        blob = subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{path}"], cwd=ROOT, text=True
        ).strip()
        assert blob == identity["git_blob"]

    claim = payload["claim"]
    assert claim["mode"] == "film-inspired"
    assert claim["evidence_grade"] == "look-approximation"
    assert claim["generic_vendor_raw_support"] is False
    assert claim["calibrated_stock_response"] is False
    assert claim["physical_film_reproduction"] is False
    assert claim["multi_stock_system_complete"] is False
