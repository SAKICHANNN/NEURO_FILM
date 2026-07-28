from __future__ import annotations

import json
from pathlib import Path

from src.eval.global_frontier import sha256_file
from src.eval.physical_neutral_gauged_visual import (
    build_visual_evidence,
    validate_visual_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs"
    / "u6_p7f1_neutral_gauged_visual_confirmation_v1.json"
)


def _config() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_p7f1_contract_binds_two_exact_runs_and_colour_baseline() -> None:
    runtime, parent, colour = validate_visual_contract(ROOT, _config())
    assert parent["stable_evidence_id"] == (
        "5aaec198789de25e283000e9781c3613ba0100339bf6ce0173ffaca937064124"
    )
    assert colour["node"] == "U6.P7B"
    assert set(_config()["visual_ids"]) <= set(runtime.eligible_ids)


def test_p7f1_builds_three_distinct_hash_bound_blind_sheets(
    tmp_path: Path,
) -> None:
    report = build_visual_evidence(
        root=ROOT,
        config=_config(),
        output_dir=tmp_path,
    )
    assert report["status"] == "awaiting_severe_and_blind_scoring"
    assert len(set(report["blind_sheet_sha256"])) == 3
    assert all(
        sha256_file(tmp_path / f"blind_round_{index}.png") == digest
        for index, digest in enumerate(
            report["blind_sheet_sha256"], start=1
        )
    )
    mapping = json.loads(
        (tmp_path / "private_mapping.json").read_text(encoding="utf-8")
    )
    assert len(
        {
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            for value in mapping.values()
        }
    ) == 3
