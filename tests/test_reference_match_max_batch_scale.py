from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_reference_match_max_batch_scale_v1 import (
    evaluate_runs,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "reference_match_max_batch_scale_v1.json"


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
