from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_reference_match_output_capability_execution_v1 import (
    evaluate_runs,
    expected_capability_payload,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs" / "reference_match_output_capability_execution_v1.json"
)


def _run(case: dict, repeat: int, token: str = "same") -> dict:
    return {
        "case_id": case["case_id"],
        "repeat": repeat,
        "output_format": case["expected_format"],
        "output_bit_depth": case["output_bit_depth"],
        "output_profile_kind": case["expected_profile_kind"],
        "output_sha256": token,
        "recipe_sha256": token,
        "normalized_report_sha256": token,
        "safety_action": "identity-fallback",
        "residual_artifacts": [],
        "automatic_pass": True,
    }


def test_p166_config_is_the_complete_public_extension_inventory() -> None:
    config = load_config(CONFIG)
    payload = expected_capability_payload(config)
    assert [row["extensions"] for row in payload["capabilities"]] == [
        [".jpeg", ".jpg", ".png", ".tif", ".tiff"],
        [".png", ".tif", ".tiff"],
        [".png"],
    ]
    assert len(config["cases"]) == 9


def test_p166_evaluation_requires_inventory_and_every_exact_replay() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload = expected_capability_payload(config)
    passing = [
        _run(case, repeat)
        for case in config["cases"]
        for repeat in (1, 2)
    ]
    assert evaluate_runs(config, payload, passing)["automatic_pass"]

    drift = [
        _run(case, repeat)
        for case in config["cases"]
        for repeat in (1, 2)
    ]
    drift[1]["output_sha256"] = "changed"
    assert not evaluate_runs(config, payload, drift)["automatic_pass"]

    incomplete_payload = json.loads(json.dumps(payload))
    incomplete_payload["capabilities"][0]["extensions"].pop()
    assert not evaluate_runs(
        config,
        incomplete_payload,
        passing,
    )["automatic_pass"]
