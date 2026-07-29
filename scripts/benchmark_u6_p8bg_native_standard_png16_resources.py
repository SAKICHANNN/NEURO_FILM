#!/usr/bin/env python3
"""Measure the complete native Standard sRGB16 PNG staging transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from src.film_physics.native_standard_factory import (  # noqa: E402
    create_opt_in_native_standard_runtime,
)
from src.film_physics.native_standard_output import (  # noqa: E402
    commit_verified_native_standard_png16,
    verify_native_standard_png16,
)
from src.film_physics.native_standard_staging import (  # noqa: E402
    stage_native_standard_working_image,
)
from src.film_physics.profile_consumer import (  # noqa: E402
    compile_standalone_profile_artifact,
)
from src.preprocess.types import SourceProfile, WorkingImage  # noqa: E402


SCHEMA = (
    "neuro_film.u6_p8bg_native_standard_png16_resources_contract.v1"
)
RESULT_SCHEMA = (
    "neuro_film.u6_p8bg_native_standard_png16_resources_result.v1"
)


def validate_contract(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        config.get("schema") != SCHEMA
        or config["scenario"]["repeats"] != 2
        or config["scenario"]["height"] != 3000
        or config["scenario"]["width"] != 4000
        or not config["execution"]["single_final_quantization"]
        or not config["execution"]["restart_verification"]
        or config["execution"]["decoder_included"]
    ):
        raise ValueError("unsupported P8BG contract")
    parent = p8aq._load_exact_json(
        ROOT / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    base = p8aq._load_exact_json(
        ROOT / config["working_image_contract"],
        config["working_image_contract_sha256"],
    )
    if (
        parent["result"]["status"] != "pass-local-staging-boundary"
        or not str(parent["next_leaf"]).startswith("U6.P8BG")
    ):
        raise ValueError("P8BG parent drift")
    for row in config["implementation"].values():
        if hashlib.sha256(
            (ROOT / row["path"]).read_bytes()
        ).hexdigest() != row["sha256"]:
            raise ValueError("P8BG implementation drift")
    return parent, base


def _worker(
    *,
    config_path: Path,
    library_paths: dict[str, Path],
    worker_output: Path,
    transaction_dir: Path,
) -> None:
    config = json.loads(config_path.read_text())
    _, base = validate_contract(config)
    package = json.loads(
        (ROOT / base["package"]).read_text()
    )
    artifact = compile_standalone_profile_artifact(
        root=ROOT,
        config=json.loads(
            (ROOT / base["profile_compiler_config"]).read_text()
        ),
    )
    runtime, factory_receipt = create_opt_in_native_standard_runtime(
        package=package,
        artifact=artifact,
        library_paths=library_paths,
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
            "raw_metadata", "P8BG deterministic synthetic scene"
        ),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=32,
        source_path=Path("p8bg.synthetic.scene-linear"),
    )
    transaction_dir.mkdir(parents=True, exist_ok=False)
    raw_output = transaction_dir / "render.f32"
    raw_report = transaction_dir / "render.raw.json"
    png_output = transaction_dir / "render.png"
    png_report = transaction_dir / "render.png.json"
    started = time.perf_counter()
    staged = stage_native_standard_working_image(
        runtime,
        working,
        output_path=raw_output,
        report_path=raw_report,
    )
    committed = commit_verified_native_standard_png16(
        staging_report_path=raw_report,
        expected_staging_report_sha256=staged["report_sha256"],
        expected_staging_run_id=staged["run_id"],
        output_path=png_output,
        report_path=png_report,
    )
    verified = verify_native_standard_png16(
        report_path=png_report,
        expected_report_sha256=committed["report_sha256"],
        expected_delivery_id=committed["delivery_id"],
    )
    elapsed = time.perf_counter() - started
    result = {
        "height": height,
        "width": width,
        "pixels": height * width,
        "elapsed_seconds": elapsed,
        "factory_receipt_sha256": factory_receipt["receipt_sha256"],
        "raw_run_id": staged["run_id"],
        "raw_output_sha256": staged["output_sha256"],
        "png_delivery_id": committed["delivery_id"],
        "png_output_sha256": committed["output_sha256"],
        "png_verification_id": verified["verification_id"],
        "png_bytes": png_output.stat().st_size,
        "single_final_quantization": True,
    }
    for path in (png_report, png_output, raw_report, raw_output):
        path.unlink()
    transaction_dir.rmdir()
    result["cleanup_pass"] = not transaction_dir.exists()
    worker_output.parent.mkdir(parents=True, exist_ok=True)
    worker_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


def benchmark(
    config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    _, base = validate_contract(config)
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
        base, output_dir / "binaries"
    )
    library_paths = {
        name: Path(row["dll_path"]) for name, row in builds.items()
    }
    runs: list[dict[str, Any]] = []
    config_path = (
        ROOT / "configs/u6_p8bg_native_standard_png16_resources_v1.json"
    )
    for repeat in range(int(config["scenario"]["repeats"])):
        worker_output = (
            output_dir / "workers" / f"run_{repeat + 1}.json"
        )
        transaction_dir = (
            output_dir / "workers" / f"transaction_{repeat + 1}"
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--config",
            str(config_path),
            "--worker-output",
            str(worker_output),
            "--transaction-dir",
            str(transaction_dir),
        ]
        for name, path in library_paths.items():
            command.extend([f"--{name}-dll", str(path)])
        run = p8aq._monitor(
            command,
            output=worker_output,
            timeout_seconds=int(
                config["scenario"]["timeout_seconds"]
            ),
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
    exact = (
        successful
        and len({item["png_output_sha256"] for item in workers}) == 1
        and len({item["raw_output_sha256"] for item in workers}) == 1
        and len({item["factory_receipt_sha256"] for item in workers})
        == 1
        and all(
            len(item["png_delivery_id"]) == 64
            and len(item["png_verification_id"]) == 64
            for item in workers
        )
    )
    path_bound_ids_distinct = successful and (
        len({item["png_delivery_id"] for item in workers}) == len(workers)
        and len({item["png_verification_id"] for item in workers})
        == len(workers)
    )
    cleanup = successful and all(
        item["cleanup_pass"] for item in workers
    )
    peaks = [
        int(item["peak_process_tree_rss_bytes"])
        for item in runs
        if item["success"]
    ]
    elapsed = [float(item["elapsed_seconds"]) for item in workers]
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
        "scenario": config["scenario"]["scenario_id"],
        "all_runs_successful": successful,
        "png_repeat_exact_and_each_receipt_verified": exact,
        "path_bound_transaction_ids_distinct": path_bound_ids_distinct,
        "png_output_sha256": (
            workers[0]["png_output_sha256"] if exact else None
        ),
        "cleanup_pass": cleanup,
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
        default="configs/u6_p8bg_native_standard_png16_resources_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/u6_p8bg_native_standard_png16_resources_v1",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-output")
    parser.add_argument("--transaction-dir")
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
            worker_output=Path(arguments.worker_output),
            transaction_dir=Path(arguments.transaction_dir),
        )
        return
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
