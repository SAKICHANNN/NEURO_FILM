#!/usr/bin/env python3
"""Resource-bound U6.P8D canonical CPU benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.global_frontier import sha256_file  # noqa: E402
from src.film_physics.profile_compiler import _canonical_bytes  # noqa: E402
from src.film_physics.profile_consumer import (  # noqa: E402
    compile_standalone_profile_artifact,
    render_working_image_row_streamed,
)
from src.preprocess.types import SourceProfile, WorkingImage  # noqa: E402


SCHEMA = "neuro_film.u6_p8d_cpu_resource_characterization_contract.v1"


def _load_exact_json(path: Path, expected: str) -> dict[str, Any]:
    if sha256_file(path) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(config: dict[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema") != SCHEMA
        or not config["execution"]["local_cpu_only"]
        or not config["execution"][
            "exact_output_identity_across_repeats_required"
        ]
        or not config["execution"][
            "timing_excluded_from_stable_evidence_id"
        ]
        or config["execution"]["post_result_retuning_allowed"]
        or int(config["tile_rows"]) <= 0
    ):
        raise ValueError("unsupported U6.P8D contract")
    decision = _load_exact_json(
        ROOT / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    _load_exact_json(
        ROOT / config["parent_contract"],
        config["parent_contract_sha256"],
    )
    profile = _load_exact_json(
        ROOT / config["profile_contract"],
        config["profile_contract_sha256"],
    )
    if (
        not decision["next_leaf"].startswith("U6.P8D")
        or decision["production_default_changed"]
        or decision["native_runtime_opened"]
    ):
        raise ValueError("U6.P8D parent decision drift")
    scenarios = config["scenarios"]
    if (
        not isinstance(scenarios, list)
        or not scenarios
        or len({item["scenario_id"] for item in scenarios})
        != len(scenarios)
        or any(
            int(item["height"]) <= 0
            or int(item["width"]) <= 0
            or int(item["repeats"]) != 2
            or int(item["timeout_seconds"]) <= 0
            for item in scenarios
        )
    ):
        raise ValueError("invalid U6.P8D scenarios")
    return profile


def _analytic_scene(height: int, width: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    output = np.empty((height, width, 3), dtype=np.float32)
    output[..., 0] = x[None, :]
    output[..., 1] = y[:, None]
    output[..., 2] = (
        0.15
        + 0.45 * x[None, :]
        + 0.35 * y[:, None]
    )
    np.clip(output, 0.0, 1.0, out=output)
    return output


def _run_worker(
    profile_config: Path,
    output: Path,
    *,
    height: int,
    width: int,
    tile_rows: int,
) -> None:
    config = json.loads(profile_config.read_text(encoding="utf-8"))
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=config
    )
    scene = _analytic_scene(height, width)
    working = WorkingImage(
        pixels=scene,
        working_space="linear_srgb_d65",
        transfer_state="scene_linear",
        source_transfer_state="scene_linear",
        source_profile=SourceProfile(
            "raw_metadata", "U6.P8D analytic scene"
        ),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=32,
        source_path=Path("u6-p8d-analytic.scene-linear"),
    )
    started = time.perf_counter()
    rendered, receipt = render_working_image_row_streamed(
        artifact, working, tile_rows=tile_rows
    )
    elapsed = time.perf_counter() - started
    result = {
        "height": height,
        "width": width,
        "pixels": height * width,
        "tile_rows": tile_rows,
        "elapsed_seconds": elapsed,
        "output_sha256": hashlib.sha256(
            np.ascontiguousarray(rendered).tobytes()
        ).hexdigest(),
        "receipt_sha256": receipt["receipt_sha256"],
        "output_dtype": rendered.dtype.name,
        "output_shape": list(rendered.shape),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _monitor(
    command: list[str],
    *,
    output: Path,
    timeout_seconds: int,
    maximum_rss: int,
    interval: float,
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    monitored = psutil.Process(process.pid)
    peak_rss = 0
    limit_reason = None
    started = time.perf_counter()
    while process.poll() is None:
        try:
            tree = [monitored, *monitored.children(recursive=True)]
            peak_rss = max(
                peak_rss,
                sum(
                    item.memory_info().rss
                    for item in tree
                    if item.is_running()
                ),
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        elapsed = time.perf_counter() - started
        if elapsed > timeout_seconds:
            limit_reason = "timeout"
            process.kill()
            break
        if peak_rss > maximum_rss:
            limit_reason = "memory-safety-limit"
            process.kill()
            break
        time.sleep(interval)
    stdout, stderr = process.communicate()
    wall = time.perf_counter() - started
    if process.returncode != 0 or limit_reason or not output.exists():
        return {
            "success": False,
            "returncode": process.returncode,
            "limit_reason": limit_reason,
            "wall_seconds": wall,
            "peak_process_tree_rss_bytes": peak_rss,
            "stderr_tail": stderr[-2000:],
            "stdout_tail": stdout[-2000:],
        }
    result = json.loads(output.read_text(encoding="utf-8"))
    return {
        "success": True,
        "returncode": process.returncode,
        "limit_reason": None,
        "wall_seconds": wall,
        "peak_process_tree_rss_bytes": peak_rss,
        "worker": result,
    }


def benchmark(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    validate_contract(config)
    available = int(psutil.virtual_memory().available)
    minimum = int(
        config["safety"]["minimum_available_memory_bytes_before_launch"]
    )
    if available < minimum:
        raise RuntimeError(
            f"available memory {available} is below launch floor {minimum}"
        )
    runs = []
    script = Path(__file__).resolve()
    profile_config = ROOT / config["profile_contract"]
    for scenario in config["scenarios"]:
        for repeat in range(int(scenario["repeats"])):
            worker_output = (
                output_dir
                / "workers"
                / f"{scenario['scenario_id']}-r{repeat + 1}.json"
            )
            command = [
                sys.executable,
                str(script),
                "--worker",
                "--profile-config",
                str(profile_config),
                "--worker-output",
                str(worker_output),
                "--height",
                str(scenario["height"]),
                "--width",
                str(scenario["width"]),
                "--tile-rows",
                str(config["tile_rows"]),
            ]
            observed = _monitor(
                command,
                output=worker_output,
                timeout_seconds=int(scenario["timeout_seconds"]),
                maximum_rss=int(
                    config["safety"]["maximum_process_tree_rss_bytes"]
                ),
                interval=float(
                    config["safety"]["monitor_interval_seconds"]
                ),
            )
            runs.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "repeat": repeat + 1,
                    **observed,
                }
            )
            if not observed["success"]:
                break
    stable_rows = []
    all_success = True
    all_exact = True
    for scenario in config["scenarios"]:
        selected = [
            row
            for row in runs
            if row["scenario_id"] == scenario["scenario_id"]
        ]
        success = (
            len(selected) == int(scenario["repeats"])
            and all(row["success"] for row in selected)
        )
        hashes = {
            row["worker"]["output_sha256"]
            for row in selected
            if row["success"]
        }
        receipts = {
            row["worker"]["receipt_sha256"]
            for row in selected
            if row["success"]
        }
        exact = success and len(hashes) == 1 and len(receipts) == 1
        all_success &= success
        all_exact &= exact
        stable_rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "height": int(scenario["height"]),
                "width": int(scenario["width"]),
                "tile_rows": int(config["tile_rows"]),
                "success": success,
                "repeat_identity_exact": exact,
                "output_sha256": next(iter(hashes)) if exact else None,
                "receipt_sha256": next(iter(receipts)) if exact else None,
            }
        )
    stable_core = {
        "schema": (
            "neuro_film.u6_p8d_cpu_resource_stable_evidence.v1"
        ),
        "rows": stable_rows,
        "all_success": all_success,
        "all_repeat_identity_exact": all_exact,
    }
    report = {
        "schema": (
            "neuro_film.u6_p8d_cpu_resource_characterization_report.v1"
        ),
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpu_count": psutil.cpu_count(),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_memory_bytes": int(psutil.virtual_memory().total),
            "available_memory_bytes_before_launch": available,
        },
        "runs": runs,
        "stable_evidence": stable_core,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable_core)
        ).hexdigest(),
        "decision": (
            "characterization-complete"
            if all_success and all_exact
            else "resource-or-identity-failure"
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u6_p8d_cpu_resource_characterization_v1.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--profile-config", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--tile-rows", type=int)
    args = parser.parse_args()
    if args.worker:
        _run_worker(
            args.profile_config,
            args.worker_output,
            height=args.height,
            width=args.width,
            tile_rows=args.tile_rows,
        )
        return 0
    if args.output is None:
        parser.error("--output is required")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = benchmark(config, args.output.parent)
    raw = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(f"decision={report['decision']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={hashlib.sha256(raw).hexdigest()}")
    for run in report["runs"]:
        print(
            f"{run['scenario_id']}/r{run['repeat']}: "
            f"success={run['success']} "
            f"wall={run['wall_seconds']:.3f}s "
            f"peak={run['peak_process_tree_rss_bytes'] / 2**30:.3f}GiB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
