from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_reference_match_batch_render_lifetime_v1 import (
    evaluate_runs,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "reference_match_batch_render_lifetime_v1.json"


def _run(variant: str, peak: int, wall: float, token: str = "same") -> dict:
    return {
        "variant": variant,
        "run_pass": True,
        "monitor": {"peak_process_tree_rss_bytes": peak},
        "worker_result": {
            "source_file_sha256": ["a", "b", "c"],
            "output_sha256": [f"{token}-{index}" for index in range(3)],
            "recipe_sha256": token,
            "normalized_report_sha256": token,
            "normalized_report": {
                "outputs": [
                    {"source_path": f"<INPUT>/source-{index}.png"}
                    for index in range(3)
                ]
            },
            "safety_actions": ["identity-fallback"] * 3,
            "worker_wall_seconds": wall,
        },
    }


def _config() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["_input_sha256"] = {
        "reference": "reference",
        "sources": ["a", "b", "c"],
    }
    return config


def test_p161_config_freezes_interleaved_exact_comparison() -> None:
    config = load_config(CONFIG)
    assert config["execution_order"] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    assert config["gates"]["minimum_median_rss_reduction_bytes"] == 200 * 2**20
    assert config["gates"]["maximum_candidate_to_baseline_median_rss_ratio"] == 0.92


def test_p161_evaluation_requires_memory_wall_and_exact_artifacts() -> None:
    config = _config()
    passing = [
        _run("baseline", 2_200_000_000, 90.0),
        _run("candidate", 1_850_000_000, 91.0),
        _run("candidate", 1_840_000_000, 90.0),
        _run("baseline", 2_190_000_000, 91.0),
    ]
    result = evaluate_runs(config, passing)
    assert result["automatic_pass"]
    assert result["performance_gate_pass"]

    weak_reduction = [
        _run("baseline", 2_100_000_000, 90.0),
        _run("candidate", 1_950_000_000, 90.0),
        _run("candidate", 1_950_000_000, 90.0),
        _run("baseline", 2_100_000_000, 90.0),
    ]
    assert not evaluate_runs(config, weak_reduction)["automatic_pass"]

    slow = [
        _run("baseline", 2_200_000_000, 90.0),
        _run("candidate", 1_850_000_000, 100.0),
        _run("candidate", 1_840_000_000, 100.0),
        _run("baseline", 2_190_000_000, 90.0),
    ]
    assert not evaluate_runs(config, slow)["automatic_pass"]

    drift = list(passing)
    drift[2] = _run("candidate", 1_840_000_000, 90.0, token="changed")
    assert not evaluate_runs(config, drift)["automatic_pass"]
