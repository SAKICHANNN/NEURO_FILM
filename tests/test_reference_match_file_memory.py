from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_reference_match_file_memory_v1 import (
    _generated_rgb,
    evaluate_runs,
    load_config,
    normalize_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "reference_match_file_memory_v1.json"


def _run(variant: str, rss: int, token: str = "same") -> dict:
    return {
        "variant": variant,
        "run_pass": True,
        "monitor": {"peak_process_tree_rss_bytes": rss},
        "worker_result": {
            "output_sha256": token,
            "recipe_sha256": token,
            "normalized_report_sha256": token,
            "safety_action": "identity-fallback",
        },
    }


def test_frozen_config_is_strict_and_self_consistent() -> None:
    config = load_config(CONFIG)
    assert config["baseline_commit"] != config["candidate_commit"]
    assert config["image"]["width"] * config["image"]["height"] == 6_000_000
    assert config["execution_order"] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]


def test_generated_rgb_is_exact_and_seed_sensitive() -> None:
    first = _generated_rgb(19, 13, 15401)
    replay = _generated_rgb(19, 13, 15401)
    other = _generated_rgb(19, 13, 15402)
    assert first.shape == (13, 19, 3)
    assert first.dtype.name == "uint8"
    assert first.tobytes() == replay.tobytes()
    assert first.tobytes() != other.tobytes()


def test_report_normalization_changes_only_path_roles() -> None:
    payload = {
        "schema_id": "neuro-film.reference-match-report.v1",
        "reference": {"path": "C:/a/ref.png", "file_sha256": "1" * 64},
        "recipe_file": {"path": "C:/a/look.json", "sha256": "2" * 64},
        "outputs": [
            {
                "source_path": "C:/a/source.png",
                "output_path": "C:/a/output.png",
                "output_sha256": "3" * 64,
                "safety": {"action": "identity-fallback"},
            }
        ],
    }
    normalized = normalize_report(payload)
    assert normalized["reference"]["path"] == "<INPUT>/reference.png"
    assert normalized["recipe_file"]["path"] == "<RUN>/recipe.json"
    assert normalized["outputs"][0]["source_path"] == "<INPUT>/source-0.png"
    assert normalized["outputs"][0]["output_path"] == "<RUN>/output-0.png"
    assert normalized["outputs"][0]["output_sha256"] == "3" * 64
    assert payload["reference"]["path"] == "C:/a/ref.png"


def test_evaluate_runs_requires_both_frozen_memory_gates() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    passing = [
        _run("baseline", 900_000_000),
        _run("candidate", 720_000_000),
        _run("candidate", 724_000_000),
        _run("baseline", 904_000_000),
    ]
    result = evaluate_runs(config, passing)
    assert result["automatic_pass"]
    assert result["artifact_parity_pass"]
    assert result["memory_gate_pass"]

    insufficient_bytes = [
        _run("baseline", 900_000_000),
        _run("candidate", 840_000_000),
        _run("candidate", 840_000_000),
        _run("baseline", 900_000_000),
    ]
    assert not evaluate_runs(config, insufficient_bytes)["memory_gate_pass"]

    insufficient_ratio = [
        _run("baseline", 2_000_000_000),
        _run("candidate", 1_900_000_000),
        _run("candidate", 1_900_000_000),
        _run("baseline", 2_000_000_000),
    ]
    assert not evaluate_runs(config, insufficient_ratio)["memory_gate_pass"]


def test_evaluate_runs_rejects_artifact_or_action_drift() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    runs = [
        _run("baseline", 900_000_000),
        _run("candidate", 700_000_000),
        _run("candidate", 700_000_000, token="changed"),
        _run("baseline", 900_000_000),
    ]
    assert not evaluate_runs(config, runs)["automatic_pass"]

    runs[2] = _run("candidate", 700_000_000)
    runs[2]["worker_result"]["safety_action"] = "applied"
    assert not evaluate_runs(config, runs)["automatic_pass"]
