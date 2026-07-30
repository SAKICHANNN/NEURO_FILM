from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_reference_match_batch_render_lifetime_v1 import (
    evaluate_runs,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "reference_match_batch_render_lifetime_v1.json"
DECISION = (
    ROOT
    / "configs"
    / "reference_match_batch_render_lifetime_decision_v1.json"
)


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


def test_p161_decision_recomputes_frozen_comparison() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    peaks = decision["peak_process_tree_rss_bytes"]
    walls = decision["worker_wall_seconds"]
    baseline_peaks = [peaks[0], peaks[3]]
    candidate_peaks = [peaks[1], peaks[2]]
    baseline_walls = [walls[0], walls[3]]
    candidate_walls = [walls[1], walls[2]]
    baseline_rss = sum(baseline_peaks) / 2
    candidate_rss = sum(candidate_peaks) / 2
    baseline_wall = sum(baseline_walls) / 2
    candidate_wall = sum(candidate_walls) / 2
    assert decision["implementation_commit"] == config["candidate_commit"]
    assert decision["baseline_commit"] == config["baseline_commit"]
    assert decision["baseline_median_peak_process_tree_rss_bytes"] == baseline_rss
    assert decision["candidate_median_peak_process_tree_rss_bytes"] == candidate_rss
    assert decision["median_peak_rss_reduction_bytes"] == (
        baseline_rss - candidate_rss
    )
    assert decision["candidate_to_baseline_median_peak_rss_ratio"] == (
        candidate_rss / baseline_rss
    )
    assert decision["candidate_to_baseline_median_worker_wall_ratio"] == (
        candidate_wall / baseline_wall
    )
    assert decision["frozen_gates"] == {
        "minimum_median_rss_reduction_bytes": config["gates"][
            "minimum_median_rss_reduction_bytes"
        ],
        "maximum_candidate_to_baseline_median_rss_ratio": config["gates"][
            "maximum_candidate_to_baseline_median_rss_ratio"
        ],
        "maximum_candidate_to_baseline_median_wall_ratio": config["gates"][
            "maximum_candidate_to_baseline_median_wall_ratio"
        ],
        "passed": True,
    }
    assert decision["automatic_pass"]
