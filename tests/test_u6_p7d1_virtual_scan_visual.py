from __future__ import annotations

import json
from pathlib import Path

from src.eval.global_frontier import sha256_file
from src.eval.physical_virtual_scan_visual import (
    adjudicate_visual_evidence,
    build_visual_evidence,
    validate_visual_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p7d1_virtual_scan_visual_confirmation_v1.json"
)
ADJUDICATION = (
    ROOT / "configs/u6_p7d1_virtual_scan_visual_adjudication_v1.json"
)


def _config() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_p7d1_contract_binds_two_exact_runs_and_colour_baseline() -> None:
    runtime, parent, colour = validate_visual_contract(ROOT, _config())
    assert parent["stable_evidence_id"] == (
        "90e07ac34ac752db9e4bb0876c290f39082b7e5a67e00e2e9f61863fb4c0b12e"
    )
    assert colour["node"] == "U6.P7B"
    assert set(_config()["visual_ids"]) <= set(runtime.eligible_ids)


def test_p7d1_builds_three_distinct_hash_bound_blind_sheets(
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


def test_p7d1_frozen_scoring_rejects_combined_on_preference() -> None:
    config = json.loads(ADJUDICATION.read_text(encoding="utf-8"))
    result = adjudicate_visual_evidence(root=ROOT, config=config)
    assert result["severe_confirmed_count"] == 0
    assert result["combined_round_wins"] == 0
    assert result["total_choices"] == {
        "colour_only": 27,
        "combined_scan_4000_dpi": 0,
    }
    assert result["decision"] == "preference_fail"
