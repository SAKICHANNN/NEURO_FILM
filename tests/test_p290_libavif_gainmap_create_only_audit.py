from __future__ import annotations

import json

from scripts import audit_p290_libavif_gainmap_create_only_api as p290


def test_canonical_is_sorted_and_terminated() -> None:
    assert p290._canonical({"b": 1, "a": 2}) == b'{"a":2,"b":1}\n'


def test_frozen_config_binds_p289_output() -> None:
    config = json.loads(
        (
            p290.ROOT / "configs/p290_libavif_gainmap_create_only_api_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert config["fixture"]["expected_output_bytes"] == 726506
    assert (
        config["fixture"]["expected_output_sha256"]
        == "94556d9f58b54955a07b5d8303da74d3ab350ef3eb061aab655cc5da7af6da70"
    )
    assert config["profile"]["base_cicp"] == [1, 16, 0]
    assert config["profile"]["alternate_cicp"] == [1, 13, 0]


def test_contract_ends_adjacent_wrapper_expansion() -> None:
    text = (
        p290.ROOT / "docs/planning/P290_LIBAVIF_GAINMAP_CREATE_ONLY_API_CONTRACT.md"
    ).read_text(encoding="utf-8")
    assert "terminal P289" in text
    assert "does not choose or" in text
