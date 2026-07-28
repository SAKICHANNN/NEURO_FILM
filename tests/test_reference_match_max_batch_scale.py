from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_reference_match_max_batch_scale_v1 import (
    evaluate_runs,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "reference_match_max_batch_scale_v1.json"
DECISION = (
    ROOT / "configs" / "reference_match_max_batch_scale_decision_v1.json"
)


def _run(peak: int, token: str = "same") -> dict:
    return {
        "run_pass": True,
        "monitor": {"peak_process_tree_rss_bytes": peak},
        "worker_result": {
            "source_file_sha256": [f"source-{index}" for index in range(64)],
            "output_sha256": [f"{token}-{index}" for index in range(64)],
            "recipe_sha256": token,
            "normalized_report_sha256": token,
            "normalized_report": {
                "outputs": [
                    {"source_path": f"<INPUT>/source-{index}.png"}
                    for index in range(64)
                ]
            },
            "safety_actions": ["identity-fallback"] * 64,
        },
    }


def _config() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["_input_sha256"] = {
        "reference": "reference",
        "sources": [f"source-{index}" for index in range(64)],
    }
    return config


def test_p162_config_is_strict_maximum_count() -> None:
    config = load_config(CONFIG)
    assert config["file_match"]["source_count"] == 64
    assert (
        config["image"]["source_seed_stop_exclusive"]
        - config["image"]["source_seed_start"]
        == 64
    )
    assert config["gates"]["maximum_peak_process_tree_rss_bytes"] == int(
        1.5 * 2**30
    )


def test_p162_evaluation_requires_64_ordered_exact_outputs() -> None:
    config = _config()
    passing = [_run(700_000_000), _run(710_000_000)]
    result = evaluate_runs(config, passing)
    assert result["automatic_pass"]
    assert result["ordered_binding_pass"]

    shortened = [_run(700_000_000), _run(700_000_000)]
    shortened[1]["worker_result"]["output_sha256"].pop()
    assert not evaluate_runs(config, shortened)["automatic_pass"]

    reordered = [_run(700_000_000), _run(700_000_000)]
    reordered[1]["worker_result"]["source_file_sha256"][0:2] = [
        "source-1",
        "source-0",
    ]
    assert not evaluate_runs(config, reordered)["automatic_pass"]

    excessive = [_run(1_700_000_000), _run(1_700_000_000)]
    assert not evaluate_runs(config, excessive)["automatic_pass"]


def test_p162_decision_recomputes_frozen_resource_gates() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    peaks = decision["peak_process_tree_rss_bytes"]
    assert decision["candidate_commit"] == config["candidate_commit"]
    assert decision["maximum_peak_process_tree_rss_bytes"] == max(peaks)
    assert decision["minimum_peak_process_tree_rss_bytes"] == min(peaks)
    assert decision["peak_rss_repeat_ratio"] == max(peaks) / min(peaks)
    assert decision["frozen_gates"] == {
        "maximum_peak_process_tree_rss_bytes": config["gates"][
            "maximum_peak_process_tree_rss_bytes"
        ],
        "maximum_peak_rss_repeat_ratio": config["gates"][
            "maximum_peak_rss_repeat_ratio"
        ],
        "worker_wall_seconds_max": config["gates"]["worker_wall_seconds_max"],
        "passed": True,
    }
    assert decision["artifact_identity"][
        "all_64_ordered_outputs_repeat_exact"
    ]
    assert decision["automatic_pass"]
