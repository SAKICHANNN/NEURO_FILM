from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_u4_5f_raw_half_size_preview import (
    _verify_execution_bindings,
    evaluate_records,
)
from src.inference.three_stock_preview import preview_dimensions

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u4_5f_raw_half_size_preview_v1.json"


def _record(row: dict, *, wall_ratio: float = 0.8, rss_ratio: float = 0.5) -> dict:
    return {
        "source_id": row["source_id"],
        "source_sha256": row["sha256"],
        "source_unchanged": True,
        "source_width": row["width"],
        "source_height": row["height"],
        "preview_width": 1000,
        "preview_height": 750,
        "candidate": {
            "decoded_width": 2000,
            "decoded_height": 1500,
            "raw_half_size_decode": True,
            "output_hashes": {"a": "1", "b": "2", "c": "3"},
            "peak_process_tree_rss_bytes": 200_000_000,
            "wall_seconds": 5.0,
        },
        "full": {
            "raw_half_size_decode_present": False,
            "output_hashes": {"a": "4", "b": "5", "c": "6"},
        },
        "fidelity": [
            {
                "rgb_rmse": 0.01,
                "rgb_absolute_error_p95": 0.02,
                "new_boundary_fraction": 0.0,
            }
        ],
        "candidate_to_full_rss_ratio": rss_ratio,
        "candidate_to_full_wall_ratio": wall_ratio,
    }


def test_u4_5f_contract_binds_five_existing_p98_sources() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert len(config["sources"]) == 5
    assert len({row["source_id"] for row in config["sources"]}) == 5
    assert len({row["sha256"] for row in config["sources"]}) == 5
    assert config["runtime"] == {
        "platform": "Windows",
        "rawpy": "0.26.1",
        "libraw": [0, 22, 0],
    }
    assert config["gates"]["maximum_rgb_rmse_vs_full_decode"] == 0.03
    assert config["gates"]["maximum_rgb_absolute_error_p95"] == 0.08
    assert config["gates"]["maximum_new_boundary_fraction"] == 0.001
    assert _verify_execution_bindings(config)


def test_evaluate_records_passes_complete_safe_candidate() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    records = []
    for row in config["sources"]:
        record = _record(row)
        record["preview_width"], record["preview_height"] = preview_dimensions(
            row["width"], row["height"], 1_000_000
        )
        records.append(record)
    gates = evaluate_records(config, records)
    assert all(gates.values())


def test_evaluate_records_fails_quality_and_resource_tails_independently() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    records = []
    for row in config["sources"]:
        record = _record(row)
        record["preview_width"], record["preview_height"] = preview_dimensions(
            row["width"], row["height"], 1_000_000
        )
        records.append(record)
    records[0]["fidelity"][0]["rgb_rmse"] = 0.031
    records[1]["candidate_to_full_rss_ratio"] = 0.751
    records[2]["candidate_to_full_wall_ratio"] = 1.051
    gates = evaluate_records(config, records)
    assert gates["rgb_rmse"] is False
    assert gates["candidate_to_full_rss_ratio"] is False
    assert gates["candidate_to_full_wall_ratio"] is False
