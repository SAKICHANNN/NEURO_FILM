from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_reference_match_unsupported_media_batch_atomicity_v1 import (
    evaluate_runs,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "reference_match_unsupported_media_batch_atomicity_v1.json"
)


def _run(case: dict, repeat: int, token: str = "same") -> dict:
    return {
        "case_id": case["case_id"],
        "repeat": repeat,
        "exception_type": "ValueError",
        "exception_message": case["expected_rejection_substring"],
        "encode_call_count": 1,
        "input_sha256": {
            "reference": token,
            "valid_source": token,
            "invalid_source": token,
        },
        "target_sha256_before": {"output": token},
        "target_sha256_after": {"output": token},
        "target_hashes_unchanged": True,
        "residual_artifacts": [],
        "automatic_pass": True,
    }


def test_p164_config_freezes_three_exact_second_source_failures() -> None:
    config = load_config(CONFIG)
    assert [row["fixture_kind"] for row in config["cases"]] == [
        "generated-rgba-png",
        "generated-two-page-rgb-tiff",
        "pinned-libultrahdr-reference",
    ]
    assert config["repeat_count_per_case"] == 2
    assert config["gates"]["require_first_valid_source_reaches_render_stage"]


def test_p164_evaluation_requires_exact_atomic_failure_replay() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    passing = [
        _run(case, repeat)
        for case in config["cases"]
        for repeat in (1, 2)
    ]
    assert evaluate_runs(config, passing)["automatic_pass"]

    drift = [
        _run(case, repeat)
        for case in config["cases"]
        for repeat in (1, 2)
    ]
    drift[1]["encode_call_count"] = 0
    drift[1]["automatic_pass"] = False
    assert not evaluate_runs(config, drift)["automatic_pass"]

    residue = [
        _run(case, repeat)
        for case in config["cases"]
        for repeat in (1, 2)
    ]
    residue[-1]["residual_artifacts"] = [".orphan.reference-match-stage.png"]
    residue[-1]["automatic_pass"] = False
    assert not evaluate_runs(config, residue)["automatic_pass"]
