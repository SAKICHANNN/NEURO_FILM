#!/usr/bin/env python3
"""Measure the P8BB WorkingImage adapter with ordered disk staging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq  # noqa: E402
import scripts.benchmark_u6_p8aw_native_display_v4_resources as p8aw  # noqa: E402
from src.film_physics.native_standard_consumer import (  # noqa: E402
    render_native_standard_working_image_to_sink,
)
from src.film_physics.native_standard_package import (  # noqa: E402
    resolve_native_standard_libraries,
)
from src.film_physics.native_standard_runtime import (  # noqa: E402
    NativeStandardRuntime,
)
from src.film_physics.profile_consumer import (  # noqa: E402
    compile_standalone_profile_artifact,
)
from src.preprocess.types import SourceProfile, WorkingImage  # noqa: E402


SCHEMA = (
    "neuro_film.u6_p8bb_native_standard_working_image_resources_contract.v1"
)
RESULT_SCHEMA = (
    "neuro_film.u6_p8bb_native_standard_working_image_resources_result.v1"
)


def validate_contract(config: dict[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema") != SCHEMA
        or int(config["scenario"]["repeats"]) != 2
        or int(config["scenario"]["height"]) != 3000
        or int(config["scenario"]["width"]) != 4000
        or not config["execution"]["full_frame_working_image_retained"]
        or not config["execution"][
            "output_staged_as_ordered_float32_rows"
        ]
    ):
        raise ValueError("unsupported P8BB contract")
    parent = p8aq._load_exact_json(
        ROOT / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if (
        parent.get("node") != "U6.P8BA"
        or not str(parent.get("next_leaf", "")).startswith("U6.P8BB")
    ):
        raise ValueError("P8BB parent decision drift")
    package = p8aq._load_exact_json(
        ROOT / config["package"],
        config["package_file_sha256"],
    )
    if (
        package["execution_policy"]["tile_rows"] != 32
        or package["execution_policy"]["pipeline_workers"] != 4
        or package["execution_policy"]["max_in_flight"] != 4
    ):
        raise ValueError("P8BB package execution policy drift")
    p8aq._load_exact_json(
        ROOT / config["profile_compiler_config"],
        config["profile_compiler_config_sha256"],
    )
    for name, expected in config["component_dll_sha256"].items():
        if package["components"][name]["sha256"] != expected:
            raise ValueError(f"P8BB {name} package identity drift")
    return package


def _worker(
    *,
    config_path: Path,
    library_paths: dict[str, Path],
    output: Path,
    staging_path: Path,
) -> None:
    config = json.loads(config_path.read_text())
    package = validate_contract(config)
    artifact = compile_standalone_profile_artifact(
        root=ROOT,
        config=json.loads(
            (ROOT / config["profile_compiler_config"]).read_text()
        ),
    )
    resolved = resolve_native_standard_libraries(
        package, library_paths
    )
    runtime = NativeStandardRuntime(
        package=package,
        artifact=artifact,
        resolved=resolved,
    )
    scenario = config["scenario"]
    height = int(scenario["height"])
    width = int(scenario["width"])
    pixels = np.ascontiguousarray(
        p8aq._source_rows(
            y0=0,
            y1=height,
            height=height,
            width=width,
        ),
        dtype=np.float32,
    )
    working = WorkingImage(
        pixels=pixels,
        working_space="linear_srgb_d65",
        transfer_state="scene_linear",
        source_transfer_state="scene_linear",
        source_profile=SourceProfile(
            "raw_metadata", "P8BB deterministic synthetic scene"
        ),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=32,
        source_path=Path("p8bb.synthetic.scene-linear"),
    )
    expected_bytes = height * width * 3 * 4
    staged_bytes = 0
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        with staging_path.open("xb") as handle:
            def sink(y0: int, y1: int, rows: np.ndarray) -> None:
                nonlocal staged_bytes
                raw = rows.tobytes()
                handle.write(raw)
                staged_bytes += len(raw)

            receipt = render_native_standard_working_image_to_sink(
                runtime,
                working,
                output_sink=sink,
            )
            handle.flush()
            os.fsync(handle.fileno())
        elapsed = time.perf_counter() - started
        staging_digest = hashlib.sha256()
        with staging_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                staging_digest.update(chunk)
        staging_sha = staging_digest.hexdigest()
        if (
            staged_bytes != expected_bytes
            or staging_sha != receipt["output"]["array_sha256"]
        ):
            raise RuntimeError("P8BB staged output identity drift")
    finally:
        staging_path.unlink(missing_ok=True)
    result = {
        "height": height,
        "width": width,
        "pixels": height * width,
        "elapsed_seconds": elapsed,
        "output_sha256": receipt["output"]["array_sha256"],
        "working_image_receipt_sha256": receipt["receipt_sha256"],
        "runtime_receipt_sha256": receipt["runtime_receipt_sha256"],
        "staged_bytes": staged_bytes,
        "staging_sha256": staging_sha,
        "staging_removed": not staging_path.exists(),
        "full_frame_working_image_retained": True,
        "full_frame_output_retained": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


def benchmark(
    config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    validate_contract(config)
    available = int(psutil.virtual_memory().available)
    launch_floor = int(
        config["safety"]["minimum_available_memory_bytes_before_launch"]
    )
    if available < launch_floor:
        raise RuntimeError(
            f"available memory {available} is below launch floor {launch_floor}"
        )
    p8aw._patch_runtime()
    builds = p8aq._build_components(
        config, output_dir / "binaries"
    )
    library_paths = {
        name: Path(row["dll_path"]) for name, row in builds.items()
    }
    runs: list[dict[str, Any]] = []
    scenario = config["scenario"]
    for repeat in range(int(scenario["repeats"])):
        worker_output = (
            output_dir / "workers" / f"run_{repeat + 1}.json"
        )
        staging_path = (
            output_dir / "workers" / f"run_{repeat + 1}.f32.stage"
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--config",
            str(ROOT / "configs/u6_p8bb_native_standard_working_image_resources_v1.json"),
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
            timeout_seconds=int(scenario["timeout_seconds"]),
            maximum_rss=int(
                config["safety"]["worker_kill_rss_bytes"]
            ),
            interval=float(config["monitor_interval_seconds"]),
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
    output_hashes = [item["output_sha256"] for item in workers]
    exact = (
        successful
        and len(set(output_hashes)) == 1
        and output_hashes[0] == config["expected_output_sha256"]
    )
    cleanup = successful and all(
        item["staging_removed"] for item in workers
    )
    peaks = [
        int(item["peak_process_tree_rss_bytes"])
        for item in runs
        if item["success"]
    ]
    elapsed = [
        float(item["elapsed_seconds"]) for item in workers
    ]
    gates = config["gates"]
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
        "scenario": scenario["scenario_id"],
        "height": int(scenario["height"]),
        "width": int(scenario["width"]),
        "repeat_count": len(runs),
        "all_runs_successful": successful,
        "output_repeat_and_p8ay_exact": exact,
        "output_sha256": output_hashes[0] if exact else None,
        "working_image_receipt_repeat_exact": (
            successful
            and len(
                {
                    item["working_image_receipt_sha256"]
                    for item in workers
                }
            )
            == 1
        ),
        "staging_cleanup_pass": cleanup,
        "peak_process_tree_rss_bytes": peaks,
        "median_peak_process_tree_rss_bytes": (
            statistics.median(peaks) if peaks else None
        ),
        "memory_gate_bytes": int(
            gates["maximum_peak_process_tree_rss_bytes"]
        ),
        "memory_pass": memory_pass,
        "elapsed_gate_seconds": float(
            gates["maximum_worker_elapsed_seconds"]
        ),
        "elapsed_pass": elapsed_pass,
        "pass": passed,
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
        default="configs/u6_p8bb_native_standard_working_image_resources_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/u6_p8bb_native_standard_working_image_resources_v1",
    )
    parser.add_argument("--worker", action="store_true")
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
    config_path = Path(arguments.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if arguments.worker:
        _worker(
            config_path=config_path,
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
    config = json.loads(config_path.read_text())
    output_dir = Path(arguments.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    report = benchmark(config, output_dir)
    p8aq.write_report(output_dir / "report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
