from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_reference_match_ordered_batch_scale_memory_v1 import (
    evaluate_runs,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "reference_match_ordered_batch_scale_memory_v1.json"


def _run(
    peak: int,
    *,
    token: str = "same",
    sources: list[str] | None = None,
    actions: list[str] | None = None,
) -> dict:
    source_hashes = sources or ["source-a", "source-b", "source-c"]
    return {
        "run_pass": True,
        "monitor": {"peak_process_tree_rss_bytes": peak},
        "worker_result": {
            "source_file_sha256": source_hashes,
            "output_sha256": [f"{token}-{index}" for index in range(3)],
            "recipe_sha256": token,
            "normalized_report_sha256": token,
            "normalized_report": {
                "outputs": [
                    {"source_path": f"<INPUT>/source-{index}.png"}
                    for index in range(3)
                ]
            },
            "safety_actions": actions or ["identity-fallback"] * 3,
        },
    }


def _config() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["_input_sha256"] = {
        "reference": "reference",
        "sources": ["source-a", "source-b", "source-c"],
    }
    return config


def test_p160_config_is_strict_and_ordered() -> None:
    config = load_config(CONFIG)
    assert config["image"]["width"] * config["image"]["height"] == 24_000_000
    assert len(config["image"]["source_seeds"]) == 3
    assert config["file_match"]["source_count"] == 3
    assert config["gates"]["maximum_peak_process_tree_rss_bytes"] == int(
        2.25 * 2**30
    )


def test_p160_evaluation_requires_order_replay_identity_and_resources() -> None:
    config = _config()
    passing = [_run(1_900_000_000), _run(2_000_000_000)]
    result = evaluate_runs(config, passing)
    assert result["automatic_pass"]
    assert result["ordered_binding_pass"]
    assert result["all_identity_fallback_pass"]

    reordered = [
        _run(1_900_000_000),
        _run(
            1_900_000_000,
            sources=["source-b", "source-a", "source-c"],
        ),
    ]
    assert not evaluate_runs(config, reordered)["automatic_pass"]

    drift = [_run(1_900_000_000), _run(1_900_000_000, token="changed")]
    assert not evaluate_runs(config, drift)["automatic_pass"]

    applied = [
        _run(1_900_000_000),
        _run(
            1_900_000_000,
            actions=["identity-fallback", "applied", "identity-fallback"],
        ),
    ]
    assert not evaluate_runs(config, applied)["automatic_pass"]

    excessive = [_run(2_500_000_000), _run(2_500_000_000)]
    assert not evaluate_runs(config, excessive)["automatic_pass"]


def test_p160_config_rejects_non_distinct_sources(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["image"]["source_seeds"][2] = config["image"]["source_seeds"][1]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    try:
        load_config(path)
    except ValueError as error:
        assert "distinct" in str(error)
    else:
        raise AssertionError("duplicate source seeds must be rejected")
