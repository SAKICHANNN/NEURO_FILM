#!/usr/bin/env python3
"""Select a retained-input native Standard concurrency candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq  # noqa: E402
import scripts.benchmark_u6_p8aw_native_display_v4_resources as p8aw  # noqa: E402
import scripts.benchmark_u6_p8bb_native_standard_working_image_resources as p8bb  # noqa: E402
from src.film_physics.native_standard_runtime import (  # noqa: E402
    NativeStandardRuntime,
)


SCHEMA = (
    "neuro_film.u6_p8bc_native_standard_working_image_concurrency_grid_contract.v1"
)
RESULT_SCHEMA = (
    "neuro_film.u6_p8bc_native_standard_working_image_concurrency_grid_result.v1"
)


def validate_contract(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        config.get("schema") != SCHEMA
        or config["candidate_workers"] != [1, 2, 3]
        or config["repeats_per_candidate"] != 2
        or not config["max_in_flight_equals_workers"]
        or not config["candidate_only_policy_override"]
        or config["production_package_changed"]
    ):
        raise ValueError("unsupported P8BC contract")
    parent = p8aq._load_exact_json(
        ROOT / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    base = p8aq._load_exact_json(
        ROOT / config["working_image_contract"],
        config["working_image_contract_sha256"],
    )
    if (
        parent["result"]["status"] != "closed-memory-gate-failed"
        or parent["result"]["memory_pass"]
        or not str(parent["next_leaf"]).startswith("U6.P8BC")
        or base["scenario"]["repeats"] != 2
    ):
        raise ValueError("P8BC parent or base contract drift")
    return parent, base


def _candidate_worker(
    *,
    workers: int,
    config_path: Path,
    library_paths: dict[str, Path],
    output: Path,
    staging_path: Path,
) -> None:
    if workers not in {1, 2, 3}:
        raise ValueError("unsupported P8BC candidate")

    class CandidateRuntime(NativeStandardRuntime):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.policy = {
                **self.policy,
                "pipeline_workers": workers,
                "max_in_flight": workers,
            }

    original = p8bb.NativeStandardRuntime
    p8bb.NativeStandardRuntime = CandidateRuntime
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
    parent, base = validate_contract(config)
    p8aw._patch_runtime()
    builds = p8aq._build_components(
        base, output_dir / "binaries"
    )
    library_paths = {
        name: Path(row["dll_path"]) for name, row in builds.items()
    }
    candidates: list[dict[str, Any]] = []
    script = Path(__file__).resolve()
    base_path = ROOT / config["working_image_contract"]
    for workers in config["candidate_workers"]:
        runs: list[dict[str, Any]] = []
        for repeat in range(config["repeats_per_candidate"]):
            run_dir = output_dir / f"workers_{workers}"
            worker_output = run_dir / f"run_{repeat + 1}.json"
            staging_path = run_dir / f"run_{repeat + 1}.f32.stage"
            command = [
                sys.executable,
                str(script),
                "--worker",
                "--workers",
                str(workers),
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
        worker_rows = [
            item["worker"] for item in runs if item["success"]
        ]
        peaks = [
            int(item["peak_process_tree_rss_bytes"])
            for item in runs
            if item["success"]
        ]
        elapsed = [
            float(item["elapsed_seconds"]) for item in worker_rows
        ]
        exact = (
            successful
            and all(
                item["output_sha256"]
                == base["expected_output_sha256"]
                for item in worker_rows
            )
            and len(
                {
                    item["working_image_receipt_sha256"]
                    for item in worker_rows
                }
            )
            == 1
        )
        cleanup = successful and all(
            item["staging_removed"] for item in worker_rows
        )
        memory_pass = bool(peaks) and max(peaks) <= int(
            base["gates"]["maximum_peak_process_tree_rss_bytes"]
        )
        elapsed_pass = bool(elapsed) and max(elapsed) <= float(
            base["gates"]["maximum_worker_elapsed_seconds"]
        )
        candidates.append(
            {
                "workers": workers,
                "max_in_flight": workers,
                "runs": runs,
                "all_runs_successful": successful,
                "output_and_receipt_exact": exact,
                "staging_cleanup_pass": cleanup,
                "peak_process_tree_rss_bytes": peaks,
                "median_peak_process_tree_rss_bytes": (
                    statistics.median(peaks) if peaks else None
                ),
                "worker_elapsed_seconds": elapsed,
                "median_worker_elapsed_seconds": (
                    statistics.median(elapsed) if elapsed else None
                ),
                "memory_pass": memory_pass,
                "elapsed_pass": elapsed_pass,
                "pass": (
                    successful
                    and exact
                    and cleanup
                    and memory_pass
                    and elapsed_pass
                ),
            }
        )
    eligible = [row for row in candidates if row["pass"]]
    selected = (
        min(
            eligible,
            key=lambda row: row["median_worker_elapsed_seconds"],
        )
        if eligible
        else None
    )
    stable_candidates = [
        {
            key: value
            for key, value in row.items()
            if key not in {"runs", "worker_elapsed_seconds"}
        }
        for row in candidates
    ]
    stable_core = {
        "schema": RESULT_SCHEMA,
        "scenario": base["scenario"]["scenario_id"],
        "candidate_workers": config["candidate_workers"],
        "candidates": stable_candidates,
        "selected_workers": (
            None if selected is None else selected["workers"]
        ),
        "selected_max_in_flight": (
            None if selected is None else selected["max_in_flight"]
        ),
        "parent_four_worker_peak_process_tree_rss_bytes": parent[
            "result"
        ]["peak_process_tree_rss_bytes"],
        "memory_gate_bytes": base["gates"][
            "maximum_peak_process_tree_rss_bytes"
        ],
        "elapsed_gate_seconds": base["gates"][
            "maximum_worker_elapsed_seconds"
        ],
        "pass": selected is not None,
        "production_package_changed": False,
        "production_default_changed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {
        **stable_core,
        "candidates_with_timings": candidates,
        "stable_evidence_id": hashlib.sha256(
            p8aq._canonical_bytes(stable_core)
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/u6_p8bc_native_standard_working_image_concurrency_grid_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/u6_p8bc_native_standard_working_image_concurrency_grid_v1",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--workers", type=int)
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
        _candidate_worker(
            workers=arguments.workers,
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
