from __future__ import annotations

import json
from pathlib import Path

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs"
    / "evidence"
    / "U7_9B_PRIVATE_RUNTIME_CANON_RAW_COMPATIBILITY_RESULT.json"
)


def _evidence() -> dict[str, object]:
    return json.loads(EVIDENCE.read_text("utf-8"))


def test_u7_9b_evidence_binds_formal_pass_and_sources() -> None:
    evidence = _evidence()
    assert (
        evidence["status"]
        == "PASS_PRIVATE_U7_9B_BINARY_RUNTIME_CANON_RAW_COMPATIBILITY"
    )
    assert evidence["automatic_pass"] is True
    assert evidence["formal_reports"] == {
        "forward": (
            "outputs/eval/u7_9b_private_runtime_canon_raw_compatibility_v1/formal.json"
        ),
        "reverse": (
            "outputs/eval/u7_9b_private_runtime_canon_raw_compatibility_v1/"
            "formal_replay.json"
        ),
        "bytes_each": 15475,
        "byte_exact": True,
        "sha256": ("55c5d47e2031c0709581328ca0dd272bdd82cf87e87d3627d5c20b42f23ed805"),
        "scientific_identity_sha256": (
            "7eb3fe64acd7059f88f4a067c2d862d2875832924eb28862278ce53f3b6d1e36"
        ),
    }
    for relative, binding in evidence["source_bindings"].items():
        assert_historical_evidence_binding(ROOT, {"path": relative, **binding})
    assert all(evidence["gates"].values())


def test_u7_9b_evidence_binds_six_exact_replayable_rows() -> None:
    evidence = _evidence()
    records = evidence["records"]
    assert len(records) == 6
    assert {record["source_id"] for record in records} == {
        "canon_eos50d_mraw",
        "canon_eos5d_mark_ii_mraw",
        "canon_eos5d_mark_iii_mraw",
        "canon_eos5d_mark_iv_sraw",
        "canon_eos7d_mark_ii_mraw",
        "canon_eos7d_mraw",
    }
    assert len({record["output_sha256"] for record in records}) == 6
    assert len({record["recipe_sha256"] for record in records}) == 6
    assert all(record["output_bytes"] > 0 for record in records)
    assert all(record["recipe_bytes"] > 0 for record in records)


def test_u7_9b_evidence_preserves_execution_history_and_claim_ceiling() -> None:
    evidence = _evidence()
    attempts = evidence["excluded_preexecution_attempts"]
    assert [attempt["stage"] for attempt in attempts] == [
        "formal-scratch-parent-create",
        "exfat-missing-wheel-control-materialization",
    ]
    assert sum(attempt["product_pixel_decodes"] for attempt in attempts) == 0
    assert sum(attempt["renders"] for attempt in attempts) == 0
    assert sum(attempt["reports"] for attempt in attempts) == 0
    assert sum(attempt["runtime_installations"] for attempt in attempts) == 1

    claim = evidence["claim"]
    assert claim["mode"] == "film-inspired"
    assert claim["evidence_grade"] == "look-approximation"
    assert claim["calibrated_stock_response"] is False
    assert claim["physical_film_reproduction"] is False
    assert claim["general_raw_support"] is False
    assert claim["standalone_or_public_installer"] is False
    assert claim["public_release"] is False
