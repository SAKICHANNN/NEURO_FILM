#!/usr/bin/env python3
"""Evaluate the frozen U6.P8W exact physical tile-row grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_u6_p8d_cpu_consumer import _monitor  # noqa: E402
from scripts.benchmark_u6_p8v_current_phase_rss import (  # noqa: E402
    _run_worker,
)
from src.eval.global_frontier import sha256_file  # noqa: E402
from src.film_physics.profile_compiler import _canonical_bytes  # noqa: E402


SCHEMA = "neuro_film.u6_p8w_tile_row_grid_contract.v1"


def _load_exact_json(path: Path, expected: str) -> dict[str, Any]:
    if sha256_file(path) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(config: dict[str, Any]) -> dict[str, Any]:
    scenario = config["scenario"]
    measurement = config["measurement"]
    candidates = [int(value) for value in scenario["tile_rows_candidates"]]
    if (
        config.get("schema") != SCHEMA
        or candidates != [64, 32]
        or int(scenario["repeats_per_candidate"]) != 2
        or int(scenario["height"]) * int(scenario["width"]) != 12_000_000
        or int(measurement["p8v_baseline_tile_rows"]) != 128
        or int(
            measurement[
                "p8v_baseline_mean_peak_process_tree_rss_bytes"
            ]
        )
        != 802248704
        or float(
            measurement["minimum_mean_peak_rss_reduction_fraction"]
        )
        != 0.02
        or int(measurement["largest_tile_within_bytes_of_minimum"])
        != 8388608
        or measurement["timing_comparison_valid"]
        or not measurement[
            "timing_and_raw_rss_excluded_from_stable_evidence_id"
        ]
        or measurement["post_result_retuning_allowed"]
        or len(measurement["expected_output_sha256"]) != 64
        or len(set(measurement["phase_order"]))
        != len(measurement["phase_order"])
    ):
        raise ValueError("unsupported U6.P8W contract")
    decision = _load_exact_json(
        ROOT / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    profile = _load_exact_json(
        ROOT / config["profile_contract"],
        config["profile_contract_sha256"],
    )
    if (
        not decision["next_leaf"].startswith("U6.P8W")
        or decision["dominant_phase"] != "physical-spatial"
        or not decision["attribution_complete"]
        or not decision["desktop_python_reference_memory_target_pass"]
        or decision["production_default_changed"]
    ):
        raise ValueError("U6.P8W parent decision drift")
    return profile


def _candidate_summary(
    config: dict[str, Any],
    tile_rows: int,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = config["measurement"]["expected_output_sha256"]
    success = len(runs) == 2 and all(row["success"] for row in runs)
    exact = success and all(
        row["worker"]["output_sha256"] == expected for row in runs
    )
    peaks = [
        int(row["peak_process_tree_rss_bytes"])
        for row in runs
        if row["success"]
    ]
    walls = [
        float(row["wall_seconds"]) for row in runs if row["success"]
    ]
    mean_peak = (
        float(statistics.fmean(peaks)) if len(peaks) == 2 else None
    )
    baseline = float(
        config["measurement"][
            "p8v_baseline_mean_peak_process_tree_rss_bytes"
        ]
    )
    reduction = (
        1.0 - mean_peak / baseline if mean_peak is not None else None
    )
    return {
        "tile_rows": tile_rows,
        "runs": runs,
        "all_success": success,
        "all_output_identity_exact": exact,
        "mean_peak_process_tree_rss_bytes": mean_peak,
        "peak_process_tree_rss_bytes_range": (
            [min(peaks), max(peaks)] if len(peaks) == 2 else None
        ),
        "wall_seconds_range": (
            [min(walls), max(walls)] if len(walls) == 2 else None
        ),
        "mean_peak_rss_reduction_fraction_vs_p8v": reduction,
    }


def evaluate(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    validate_contract(config)
    available = int(psutil.virtual_memory().available)
    if available < int(
        config["safety"]["minimum_available_memory_bytes_before_launch"]
    ):
        raise RuntimeError("available memory is below P8W launch floor")
    scenario = config["scenario"]
    rows: list[dict[str, Any]] = []
    for tile_rows in scenario["tile_rows_candidates"]:
        runs: list[dict[str, Any]] = []
        worker_config = json.loads(json.dumps(config))
        worker_config["scenario"]["tile_rows"] = int(tile_rows)
        for repeat in range(int(scenario["repeats_per_candidate"])):
            worker_output = (
                output_dir
                / "workers"
                / f"tile-{tile_rows}-r{repeat + 1}.json"
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--worker-output",
                str(worker_output),
                "--config",
                str(ROOT / "configs/u6_p8w_tile_row_grid_v1.json"),
                "--tile-rows",
                str(tile_rows),
            ]
            observed = _monitor(
                command,
                output=worker_output,
                timeout_seconds=int(scenario["timeout_seconds"]),
                maximum_rss=int(
                    config["safety"]["maximum_process_tree_rss_bytes"]
                ),
                interval=float(
                    config["measurement"]["sample_interval_seconds"]
                ),
            )
            runs.append({"repeat": repeat + 1, **observed})
            if not observed["success"]:
                break
        rows.append(_candidate_summary(config, int(tile_rows), runs))
    threshold = float(
        config["measurement"]["minimum_mean_peak_rss_reduction_fraction"]
    )
    eligible = [
        row
        for row in rows
        if row["all_success"]
        and row["all_output_identity_exact"]
        and row["mean_peak_rss_reduction_fraction_vs_p8v"] >= threshold
    ]
    selected = int(config["measurement"]["p8v_baseline_tile_rows"])
    candidate_retained = False
    if eligible:
        minimum = min(
            float(row["mean_peak_process_tree_rss_bytes"])
            for row in eligible
        )
        tolerance = int(
            config["measurement"]["largest_tile_within_bytes_of_minimum"]
        )
        selected = max(
            int(row["tile_rows"])
            for row in eligible
            if float(row["mean_peak_process_tree_rss_bytes"])
            <= minimum + tolerance
        )
        candidate_retained = True
    stable = {
        "schema": "neuro_film.u6_p8w_tile_row_grid_stable_evidence.v1",
        "candidate_tile_rows": scenario["tile_rows_candidates"],
        "all_success": all(row["all_success"] for row in rows),
        "all_output_identity_exact": all(
            row["all_output_identity_exact"] for row in rows
        ),
        "selected_tile_rows": selected,
        "candidate_retained": candidate_retained,
    }
    return {
        "schema": "neuro_film.u6_p8w_tile_row_grid_report.v1",
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "rows": rows,
        "stable_evidence": stable,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable)
        ).hexdigest(),
        "decision": (
            "candidate-retained" if candidate_retained else "keep-p8v-128"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p8w_tile_row_grid_v1.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--tile-rows", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.worker:
        if args.worker_output is None or args.tile_rows is None:
            parser.error("--worker-output and --tile-rows are required")
        config["scenario"]["tile_rows"] = int(args.tile_rows)
        _run_worker(config, args.worker_output)
        return 0
    if args.output is None:
        parser.error("--output is required")
    report = evaluate(config, args.output.parent)
    raw = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(f"decision={report['decision']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={hashlib.sha256(raw).hexdigest()}")
    for row in report["rows"]:
        print(
            f"tile_rows={row['tile_rows']}: "
            f"exact={row['all_output_identity_exact']} "
            f"mean_peak={row['mean_peak_process_tree_rss_bytes']} "
            f"reduction={row['mean_peak_rss_reduction_fraction_vs_p8v']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
