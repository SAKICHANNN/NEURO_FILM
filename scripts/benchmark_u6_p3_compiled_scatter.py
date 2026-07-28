#!/usr/bin/env python
"""Measure the U6.P3A direct separable baseline in isolated child processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_reference_scatter import load_contract  # noqa: E402
from src.film_physics import (  # noqa: E402
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    apply_compiled_scatter,
    compile_scatter_profile,
)
from src.film_physics.reference_scatter import profile_from_contract  # noqa: E402


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _field(shape: tuple[int, int, int], row_chunk: int = 256) -> np.ndarray:
    height, width, channels = shape
    if channels != 3:
        raise ValueError("benchmark shape must be HxWx3")
    output = np.empty(shape, dtype=np.float32)
    x = np.arange(width, dtype=np.float32) / np.float32(max(width - 1, 1))
    for y0 in range(0, height, row_chunk):
        y1 = min(height, y0 + row_chunk)
        y = np.arange(y0, y1, dtype=np.float32)[:, None] / np.float32(
            max(height - 1, 1)
        )
        wave = np.sin(np.float32(17.0) * x[None, :] + np.float32(11.0) * y)
        block = output[y0:y1]
        block[..., 0] = 0.02 + 0.70 * x + 0.08 * y + 0.02 * wave
        block[..., 1] = 0.03 + 0.28 * x + 0.45 * y - 0.01 * wave
        block[..., 2] = 0.01 + 0.12 * x + 0.60 * y + 0.015 * wave
    return output


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def worker(shape: tuple[int, int, int]) -> dict:
    profile = profile_from_contract(
        load_contract(ROOT / "configs" / "u6_p1_reference_scatter_simulator_v1.json")
    )
    compiled = compile_scatter_profile(profile)
    source_started = perf_counter()
    values = _field(shape)
    source_seconds = perf_counter() - source_started
    state = PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
        PhysicalScale(profile.pixel_pitch_um),
    )
    apply_started = perf_counter()
    result = apply_compiled_scatter(state, compiled)
    apply_seconds = perf_counter() - apply_started
    return {
        "shape": list(shape),
        "source_sha256": _hash_array(values),
        "output_sha256": _hash_array(result.values),
        "source_seconds": source_seconds,
        "apply_seconds": apply_seconds,
        "finite_nonnegative": bool(
            np.all(np.isfinite(result.values)) and np.min(result.values) >= 0.0
        ),
        "logical_input_bytes": int(values.nbytes),
        "logical_output_bytes": int(result.values.nbytes),
    }


def _launch(
    shape: tuple[int, int, int],
    output: Path,
    *,
    interval: float,
    timeout: float,
) -> dict:
    output.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-shape",
        *[str(value) for value in shape],
        "--worker-output",
        str(output),
    ]
    started = perf_counter()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    monitored = psutil.Process(process.pid)
    peak_rss = 0
    timed_out = False
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
        if perf_counter() - started > timeout:
            timed_out = True
            process.kill()
            break
        sleep(interval)
    stdout, stderr = process.communicate()
    return {
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "wall_seconds": perf_counter() - started,
        "peak_process_tree_rss_bytes": peak_rss,
        "stdout": stdout,
        "stderr": stderr,
        "result_exists": output.exists(),
    }


def parent(config_path: Path, output_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runs = []
    for shape_values in config["shapes"]:
        shape = tuple(int(value) for value in shape_values)
        for replay in range(int(config["runs_per_shape"])):
            result_path = output_dir / (
                f"{shape[0]}x{shape[1]}_run{replay + 1}.json"
            )
            monitor = _launch(
                shape,
                result_path,
                interval=float(config["rss_sample_interval_seconds"]),
                timeout=float(config["worker_timeout_seconds"]),
            )
            result = (
                json.loads(result_path.read_text(encoding="utf-8"))
                if result_path.exists()
                else None
            )
            runs.append({"monitor": monitor, "result": result})
            if monitor["exit_code"] != 0 or result is None:
                raise RuntimeError(
                    f"benchmark worker failed for {shape}: {monitor['stderr']}"
                )
    groups: dict[str, list[dict]] = {}
    for run in runs:
        key = "x".join(str(value) for value in run["result"]["shape"])
        groups.setdefault(key, []).append(run)
    repeat_exact = {
        key: len({item["result"]["output_sha256"] for item in items}) == 1
        for key, items in groups.items()
    }
    report = {
        "schema": "neuro_film.u6_p3_scatter_benchmark_report.v1",
        "config": config,
        "runs": runs,
        "repeat_output_exact": repeat_exact,
        "all_pass": bool(
            all(repeat_exact.values())
            and all(item["result"]["finite_nonnegative"] for item in runs)
        ),
    }
    _atomic_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/u6_p3_scatter_benchmark_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-shape", nargs=3, type=int)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker_shape is not None:
        if args.worker_output is None:
            raise ValueError("--worker-output is required in worker mode")
        _atomic_json(args.worker_output, worker(tuple(args.worker_shape)))
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required in parent mode")
    report = parent(args.config, args.output_dir)
    print(f"all_pass={report['all_pass']}")
    for run in report["runs"]:
        result = run["result"]
        monitor = run["monitor"]
        print(
            f"{result['shape'][0]}x{result['shape'][1]} "
            f"apply={result['apply_seconds']:.3f}s "
            f"peak={monitor['peak_process_tree_rss_bytes'] / 2**30:.3f}GiB"
        )


if __name__ == "__main__":
    main()
