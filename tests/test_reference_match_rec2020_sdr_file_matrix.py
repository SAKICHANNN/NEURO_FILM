from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_reference_match_rec2020_sdr_file_matrix_v1 import (
    evaluate_runs,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "reference_match_rec2020_sdr_file_matrix_v1.json"
DECISION = (
    ROOT
    / "configs"
    / "reference_match_rec2020_sdr_file_matrix_decision_v1.json"
)


def _run(case: dict, token: str = "same", peak: int = 500_000_000) -> dict:
    return {
        "case_id": case["case_id"],
        "run_pass": True,
        "monitor": {"peak_process_tree_rss_bytes": peak},
        "worker_result": {
            "decoded": {
                "reference_rail": case["reference_rail"],
                "source_rails": case["source_rails"],
            },
            "output_rails": case["expected_output_rails"],
            "output_profiles": case["expected_output_profiles"],
            "output_formats": ["PNG"] * len(case["source_rails"]),
            "output_bit_depths": [16] * len(case["source_rails"]),
            "output_sha256": [token] * len(case["source_rails"]),
            "recipe_sha256": token,
            "normalized_report_sha256": token,
            "safety_actions": ["identity-fallback"]
            * len(case["source_rails"]),
        },
    }


def _runs(config: dict) -> list[dict]:
    return [
        _run(case, peak=500_000_000 + repeat * 5_000_000)
        for case in config["cases"]
        for repeat in range(2)
    ]


def test_p165_config_freezes_rec2020_and_mixed_sdr_cases() -> None:
    config = load_config(CONFIG)
    assert [row["expected_output_rails"] for row in config["cases"]] == [
        ["linear_rec2020"],
        ["linear_srgb", "linear_rec2020"],
    ]
    assert [row["expected_output_profiles"] for row in config["cases"]] == [
        ["cicp"],
        ["icc", "cicp"],
    ]
    assert config["output_bit_depth"] == 16
    assert config["preflight"]["available_physical_memory_bytes_min"] == 2**33


def test_p165_evaluation_requires_exact_ordered_rail_profile_parity() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    passing = _runs(config)
    assert evaluate_runs(config, passing)["automatic_pass"]

    reordered = _runs(config)
    reordered[-1]["worker_result"]["output_rails"] = [
        "linear_rec2020",
        "linear_srgb",
    ]
    assert not evaluate_runs(config, reordered)["automatic_pass"]

    wrong_profile = _runs(config)
    wrong_profile[0]["worker_result"]["output_profiles"] = ["icc"]
    assert not evaluate_runs(config, wrong_profile)["automatic_pass"]

    excessive = _runs(config)
    excessive[0]["monitor"]["peak_process_tree_rss_bytes"] = 1_700_000_000
    assert not evaluate_runs(config, excessive)["automatic_pass"]


def test_p165_decision_recomputes_frozen_resources_and_semantics() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["candidate_commit"] == config["candidate_commit"]
    assert set(decision["cases"]) == {
        row["case_id"] for row in config["cases"]
    }
    config_cases = {row["case_id"]: row for row in config["cases"]}
    for case_id, result in decision["cases"].items():
        expected = config_cases[case_id]
        peaks = result["peak_process_tree_rss_bytes"]
        assert result["output_rails"] == expected["expected_output_rails"]
        assert result["output_profiles"] == expected[
            "expected_output_profiles"
        ]
        assert result["peak_rss_repeat_ratio"] == max(peaks) / min(peaks)
        assert max(peaks) <= config["gates"][
            "maximum_peak_process_tree_rss_bytes_per_run"
        ]
        assert max(result["worker_wall_seconds"]) <= config["gates"][
            "worker_wall_seconds_max"
        ]
        assert result["safety_actions"] == [
            "identity-fallback"
        ] * len(expected["source_rails"])
        assert result["automatic_pass"]
    assert decision["pre_pixel_contract_attempts"]["pixel_tasks_executed"] == 0
    assert decision["automatic_pass"]
