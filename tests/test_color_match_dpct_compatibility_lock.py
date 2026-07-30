from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs" / "reference_match_dpct_compatibility_v1.json"
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_dpct_compatibility_lock_v1.schema.json"
)


def _lock() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8"))


def test_dpct_compatibility_lock_matches_strict_schema() -> None:
    lock = _lock()
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(lock)


def test_pinned_artifact_hashes_match_independent_audit() -> None:
    lock = _lock()
    assert lock["producer"] == {
        "repository_id": "zhuise-dpct",
        "commit": "3c2e9fdfa5d226d92016c530856a7df94f9c42ab",
        "producer_contracts_sha256": (
            "98a320534eed924a3b8eca3cf822b54dc2eed2bf7c27aa0a037745639f92e68c"
        ),
        "native_reference_sha256": (
            "dab41ef64083ece5aa31b28ce94ab32e4ce2a2bcdfa19a1561cb5f2863c3a628"
        ),
    }
    assert {
        name: value["sha256"]
        for name, value in lock["schemas"].items()
    } == {
        "match_view": (
            "ac422dd80721799a6a0a7e1f65f8f7ee7205981c2ea74343721287806bba7bc7"
        ),
        "transform_bundle": (
            "e1cf6a7e3bf7c5b85d1da1a21a0b3cbf284919759771866e8c273773fae8222f"
        ),
        "diagnostics": (
            "f6c1dec034183017e9668f23ee3a9cee0ad21dec3357e499c602c2ad43b06463"
        ),
        "apply_result": (
            "887e964d7885941a4bd9031fcc9815859f765291a4595b889d2bce5e15c7e3fa"
        ),
    }
    assert lock["conformance"]["fixture_sha256"] == (
        "c9c8c0ff6faf69078b17ea56d870973fa8b5824170ad8f00e8a5869ae9f3e326"
    )


def test_v1_candidate_bridge_is_explicitly_closed() -> None:
    lock = _lock()
    verdict = lock["verdict"]
    assert verdict["status"] == (
        "contract-mapped-candidate-pixel-bridge-closed"
    )
    assert verdict["blockers"] == sorted(set(verdict["blockers"]))
    assert not verdict["consumer_may_invoke"]
    assert not verdict["consumer_may_issue_apply_receipt"]
    assert not verdict["consumer_may_admit_candidate"]
    assert verdict["identity_fallback_required"]


def test_profile_mapping_does_not_claim_implicit_compatibility() -> None:
    mapping = _lock()["profile_mapping"]
    assert mapping["producer_profile_id"].startswith("zhuise.")
    assert mapping["consumer_profile_id"].startswith("neuro-film.")
    assert mapping["producer_profile_id"] != mapping["consumer_profile_id"]
    assert mapping["pixel_hash_mapping"] == "strip-sha256-prefix"
    assert mapping["semantic_state"] == (
        "explicitly-mappable-not-yet-compatible"
    )
