from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_reference_match_target_scale_memory_v1 import (
    evaluate_runs,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "reference_match_target_scale_memory_v1.json"


def _run(peak: int, token: str = "same", wall: float = 40.0) -> dict:
    return {
        "run_pass": wall <= 120.0,
        "monitor": {"peak_process_tree_rss_bytes": peak},
        "worker_result": {
            "output_sha256": token,
            "recipe_sha256": token,
            "normalized_report_sha256": token,
            "safety_action": "identity-fallback",
            "worker_wall_seconds": wall,
        },
    }


def test_p159_config_is_strict_and_target_scale() -> None:
    config = load_config(CONFIG)
    assert config["image"]["width"] * config["image"]["height"] == 24_000_000
    assert config["repeat_count"] == 2
    assert config["gates"]["maximum_peak_process_tree_rss_bytes"] == 3 * 2**30


def test_p159_evaluation_requires_resources_replay_and_identity() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    passing = [_run(2_100_000_000), _run(2_200_000_000)]
    result = evaluate_runs(config, passing)
    assert result["automatic_pass"]
    assert result["resource_gate_pass"]
    assert result["peak_rss_repeat_ratio"] == 2_200_000_000 / 2_100_000_000

    excessive = [_run(3_300_000_000), _run(3_200_000_000)]
    assert not evaluate_runs(config, excessive)["automatic_pass"]

    unstable = [_run(1_000_000_000), _run(1_200_000_000)]
    assert not evaluate_runs(config, unstable)["automatic_pass"]

    drift = [_run(2_000_000_000), _run(2_000_000_000, token="changed")]
    assert not evaluate_runs(config, drift)["automatic_pass"]

    wrong_action = [_run(2_000_000_000), _run(2_000_000_000)]
    wrong_action[1]["worker_result"]["safety_action"] = "applied"
    assert not evaluate_runs(config, wrong_action)["automatic_pass"]
