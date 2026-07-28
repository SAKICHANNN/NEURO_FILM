from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_reference_match_sdr_file_format_matrix_v1 import (
    ROOT as RUNNER_ROOT,
)
from scripts.audit_reference_match_sdr_file_format_matrix_v1 import (
    evaluate_runs,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "reference_match_sdr_file_format_matrix_v1.json"


def _run(case: dict, token: str = "same", peak: int = 700_000_000) -> dict:
    case_id = case["case_id"]
    format_name = case["expected_output_format"]
    return {
        "case_id": case_id,
        "run_pass": True,
        "monitor": {"peak_process_tree_rss_bytes": peak},
        "worker_result": {
            "output_sha256": f"{case_id}-{token}",
            "recipe_sha256": f"{case_id}-{token}",
            "normalized_report_sha256": f"{case_id}-{token}",
            "decoded": {
                role: {
                    "working_space": "linear_srgb",
                    "transfer_state": "display_linear",
                    "bit_depth": case["input_bit_depth"],
                }
                for role in ("reference", "source")
            },
            "output_format": format_name,
            "output_bit_depth": case["output_bit_depth"],
            "output_inspection": {"bit_depth": case["output_bit_depth"]},
            "safety_action": "identity-fallback",
        },
    }


def _runs(config: dict) -> list[dict]:
    return [
        _run(case, peak=700_000_000 + repeat * 5_000_000)
        for case in config["cases"]
        for repeat in range(2)
    ]


def test_p163_config_freezes_exact_three_case_matrix() -> None:
    config = load_config(CONFIG)
    assert [row["case_id"] for row in config["cases"]] == [
        "jpeg8",
        "tiff8",
        "tiff16",
    ]
    assert [row["input_bit_depth"] for row in config["cases"]] == [8, 8, 16]
    assert config["repeat_count_per_case"] == 2
    assert str(RUNNER_ROOT) in __import__("sys").path


def test_p163_evaluation_requires_per_case_exactness_and_semantics() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    passing = _runs(config)
    result = evaluate_runs(config, passing)
    assert result["automatic_pass"]
    assert all(
        row["semantic_pass"] for row in result["case_results"].values()
    )

    drift = _runs(config)
    drift[1]["worker_result"]["output_sha256"] = "changed"
    assert not evaluate_runs(config, drift)["automatic_pass"]

    wrong_depth = _runs(config)
    wrong_depth[-1]["worker_result"]["output_inspection"]["bit_depth"] = 8
    assert not evaluate_runs(config, wrong_depth)["automatic_pass"]

    excessive = _runs(config)
    excessive[0]["monitor"]["peak_process_tree_rss_bytes"] = 1_700_000_000
    assert not evaluate_runs(config, excessive)["automatic_pass"]
