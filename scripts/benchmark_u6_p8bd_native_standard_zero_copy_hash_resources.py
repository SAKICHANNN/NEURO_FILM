#!/usr/bin/env python3
"""Rerun P8BB using the exact-receipt zero-copy input hash runtime."""

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

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq  # noqa: E402
import scripts.benchmark_u6_p8aw_native_display_v4_resources as p8aw  # noqa: E402
import scripts.benchmark_u6_p8bb_native_standard_working_image_resources as p8bb  # noqa: E402
from src.film_physics.native_standard_runtime_v2 import (  # noqa: E402
    MemoryBoundNativeStandardRuntime,
)


SCHEMA = (
    "neuro_film.u6_p8bd_native_standard_zero_copy_hash_resources_contract.v1"
)
RESULT_SCHEMA = (
    "neuro_film.u6_p8bd_native_standard_zero_copy_hash_resources_result.v1"
)


def validate_contract(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        config.get("schema") != SCHEMA
        or config["operator_changed"]
        or config["receipt_schema_changed"]
        or config["execution_policy_changed"]
        or config["production_default_changed"]
    ):
        raise ValueError("unsupported P8BD contract")
    parent = p8aq._load_exact_json(
        ROOT / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    base = p8aq._load_exact_json(
        ROOT / config["working_image_contract"],
        config["working_image_contract_sha256"],
    )
    if (
        parent["result"]["status"]
        != "closed-no-eligible-concurrency"
        or not str(parent["next_leaf"]).startswith("U6.P8BD")
        or hashlib.sha256(
            (ROOT / config["candidate_runtime"]).read_bytes()
        ).hexdigest()
        != config["candidate_runtime_sha256"]
    ):
        raise ValueError("P8BD parent or runtime drift")
    return parent, base


def _worker(
    *,
    config_path: Path,
    library_paths: dict[str, Path],
    output: Path,
    staging_path: Path,
) -> None:
    original = p8bb.NativeStandardRuntime
    p8bb.NativeStandardRuntime = MemoryBoundNativeStandardRuntime
    try:
        p8bb._worker(
            config_path=config_path,
            library_paths=library_paths,
            output=output,
            staging_path=staging_path,
        )
    finally:
        p8bb.NativeStandardRuntime = original


def benchmark(
    config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    _, base = validate_contract(config)
    available = int(psutil.virtual_memory().available)
    launch_floor = int(
        base["safety"]["minimum_available_memory_bytes_before_launch"]
    )
    if available < launch_floor:
        raise RuntimeError(
            f"available memory {available} is below launch floor {launch_floor}"
        )
    p8aw._patch_runtime()
    builds = p8aq._build_components(
        base, output_dir / "binaries"
    )
    library_paths = {
        name: Path(row["dll_path"]) for name, row in builds.items()
    }
    runs: list[dict[str, Any]] = []
    script = Path(__file__).resolve()
    base_path = ROOT / config["working_image_contract"]
    for repeat in range(int(base["scenario"]["repeats"])):
        worker_output = (
            output_dir / "workers" / f"run_{repeat + 1}.json"
        )
        staging_path = (
            output_dir / "workers" / f"run_{repeat + 1}.f32.stage"
        )
        command = [
            sys.executable,
            str(script),
            "--worker",
            "--working-image-config",
            str(base_path),
            "--worker-output",
            str(worker_output),
            "--staging-path",
            str(staging_path),
        ]
        for name, path in library_paths.items():
            command.extend([f"--{name}-dll", str(path)])
        run = p8aq._monitor(
            command,
            output=worker_output,
            timeout_seconds=int(
                base["scenario"]["timeout_seconds"]
            ),
            maximum_rss=int(
                base["safety"]["worker_kill_rss_bytes"]
            ),
            interval=float(base["monitor_interval_seconds"]),
        )
        run["repeat"] = repeat + 1
        runs.append(run)
        if not run["success"]:
            break
    successful = len(runs) == 2 and all(
        item["success"] for item in runs
    )
    workers = [
        item["worker"] for item in runs if item["success"]
    ]
    exact = (
        successful
        and all(
            item["output_sha256"] == base["expected_output_sha256"]
            for item in workers
        )
        and len(
            {
                item["working_image_receipt_sha256"]
                for item in workers
            }
        )
        == 1
    )
    cleanup = successful and all(
        item["staging_removed"] for item in workers
    )
    peaks = [
        int(item["peak_process_tree_rss_bytes"])
        for item in runs
        if item["success"]
    ]
    elapsed = [float(item["elapsed_seconds"]) for item in workers]
    gates = base["gates"]
    memory_pass = bool(peaks) and max(peaks) <= int(
        gates["maximum_peak_process_tree_rss_bytes"]
    )
    elapsed_pass = bool(elapsed) and max(elapsed) <= float(
        gates["maximum_worker_elapsed_seconds"]
    )
    passed = (
        successful
        and exact
        and cleanup
        and memory_pass
        and elapsed_pass
    )
    stable_core = {
        "schema": RESULT_SCHEMA,
        "scenario": base["scenario"]["scenario_id"],
        "all_runs_successful": successful,
        "output_and_receipt_exact": exact,
        "output_sha256": (
            base["expected_output_sha256"] if exact else None
        ),
        "staging_cleanup_pass": cleanup,
        "peak_process_tree_rss_bytes": peaks,
        "median_peak_process_tree_rss_bytes": (
            statistics.median(peaks) if peaks else None
        ),
        "memory_gate_bytes": gates[
            "maximum_peak_process_tree_rss_bytes"
        ],
        "memory_pass": memory_pass,
        "elapsed_gate_seconds": gates[
            "maximum_worker_elapsed_seconds"
        ],
        "elapsed_pass": elapsed_pass,
        "pass": passed,
        "operator_changed": False,
        "receipt_schema_changed": False,
        "execution_policy_changed": False,
        "production_default_changed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {
        **stable_core,
        "worker_elapsed_seconds": elapsed,
        "wall_seconds": [
            float(item["wall_seconds"]) for item in runs
        ],
        "runs": runs,
        "stable_evidence_id": hashlib.sha256(
            p8aq._canonical_bytes(stable_core)
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/u6_p8bd_native_standard_zero_copy_hash_resources_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/u6_p8bd_native_standard_zero_copy_hash_resources_v1",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--working-image-config")
    parser.add_argument("--worker-output")
    parser.add_argument("--staging-path")
    for name in (
        "domains",
        "gaussian",
        "adjacency",
        "gauge",
        "context",
        "display",
    ):
        parser.add_argument(f"--{name}-dll")
    arguments = parser.parse_args()
    if arguments.worker:
        _worker(
            config_path=Path(arguments.working_image_config),
            library_paths={
                name: Path(getattr(arguments, f"{name}_dll"))
                for name in (
                    "domains",
                    "gaussian",
                    "adjacency",
                    "gauge",
                    "context",
                    "display",
                )
            },
            output=Path(arguments.worker_output),
            staging_path=Path(arguments.staging_path),
        )
        return
    config_path = Path(arguments.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    output_dir = Path(arguments.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    report = benchmark(
        json.loads(config_path.read_text()), output_dir
    )
    p8aq.write_report(output_dir / "report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
