#!/usr/bin/env python
"""Fresh-process U6.P3I direct-versus-FFT backing-return benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep
from typing import Any

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.film_physics import (  # noqa: E402
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    apply_compiled_backing_return,
    apply_fft_backing_return,
    backing_return_profile_from_contract,
    compile_backing_return_profile,
)


SCHEMA = "neuro_film.u6_p3i_fft_backing_return_benchmark_contract.v1"


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(
        memoryview(np.ascontiguousarray(array)).cast("B")
    ).hexdigest()


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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def worker(algorithm: str, shape: tuple[int, int, int]) -> dict[str, Any]:
    parent = json.loads(
        (ROOT / "configs/u6_p3d_backing_return_reference_v1.json").read_text(
            encoding="utf-8"
        )
    )
    reference = backing_return_profile_from_contract(parent)
    compiled = compile_backing_return_profile(reference)
    source_started = perf_counter()
    values = _field(shape)
    source_seconds = perf_counter() - source_started
    state = PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
        PhysicalScale(compiled.pixel_pitch_um),
    )
    apply = {
        "direct-separable": apply_compiled_backing_return,
        "fft-2d": apply_fft_backing_return,
    }.get(algorithm)
    if apply is None:
        raise ValueError("unsupported benchmark algorithm")
    apply_started = perf_counter()
    result = apply(state, compiled)
    apply_seconds = perf_counter() - apply_started
    increment = result.values - values
    return {
        "algorithm": algorithm,
        "shape": list(shape),
        "source_sha256": _hash_array(values),
        "output_sha256": _hash_array(result.values),
        "source_seconds": source_seconds,
        "apply_seconds": apply_seconds,
        "finite_nonnegative": bool(
            np.all(np.isfinite(result.values))
            and float(np.min(result.values)) >= 0.0
        ),
        "minimum_direct_increment": float(np.min(increment)),
        "logical_input_bytes": int(values.nbytes),
        "logical_output_bytes": int(result.values.nbytes),
    }


def _launch(
    algorithm: str,
    shape: tuple[int, int, int],
    output: Path,
    *,
    interval: float,
    timeout: float,
) -> dict[str, Any]:
    output.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-algorithm",
        algorithm,
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


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def parent(config_path: Path, output_dir: Path) -> dict[str, Any]:
    raw = config_path.read_bytes()
    config = json.loads(raw.decode("utf-8"))
    if config.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P3I contract")
    runs: list[dict[str, Any]] = []
    order = config["execution_order_per_shape"]
    expected_count = int(config["runs_per_algorithm_shape"])
    for shape_values in config["shapes"]:
        shape = tuple(int(value) for value in shape_values)
        counts = {algorithm: 0 for algorithm in config["algorithms"]}
        for algorithm in order:
            counts[algorithm] += 1
            replay = counts[algorithm]
            result_path = output_dir / (
                f"{shape[0]}x{shape[1]}_{algorithm}_run{replay}.json"
            )
            monitor = _launch(
                algorithm,
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
                    f"benchmark worker failed for {algorithm} {shape}: "
                    f"{monitor['stderr']}"
                )
        if any(value != expected_count for value in counts.values()):
            raise ValueError("execution order does not provide exact replay count")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_shape: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        result = run["result"]
        shape_key = "x".join(str(value) for value in result["shape"])
        groups.setdefault((result["algorithm"], shape_key), []).append(run)
        by_shape.setdefault(shape_key, []).append(run)
    repeat_exact = {
        f"{algorithm}:{shape}": len(
            {item["result"]["output_sha256"] for item in items}
        )
        == 1
        for (algorithm, shape), items in groups.items()
    }
    source_exact = {
        shape: len(
            {item["result"]["source_sha256"] for item in items}
        )
        == 1
        for shape, items in by_shape.items()
    }
    summaries: dict[str, dict[str, Any]] = {}
    for (algorithm, shape), items in groups.items():
        summaries[f"{algorithm}:{shape}"] = {
            "median_apply_seconds": _median(
                [item["result"]["apply_seconds"] for item in items]
            ),
            "median_wall_seconds": _median(
                [item["monitor"]["wall_seconds"] for item in items]
            ),
            "median_peak_process_tree_rss_bytes": _median(
                [
                    float(item["monitor"]["peak_process_tree_rss_bytes"])
                    for item in items
                ]
            ),
            "maximum_peak_process_tree_rss_bytes": max(
                item["monitor"]["peak_process_tree_rss_bytes"] for item in items
            ),
        }
    speedups = {}
    for shape_values in config["shapes"]:
        shape = "x".join(str(value) for value in shape_values)
        direct = summaries[f"direct-separable:{shape}"]
        fft = summaries[f"fft-2d:{shape}"]
        speedups[shape] = (
            direct["median_apply_seconds"] / fft["median_apply_seconds"]
        )

    shape_12 = "3000x4000x3"
    shape_24 = "4000x6000x3"
    direct_24 = summaries[f"direct-separable:{shape_24}"]
    fft_24 = summaries[f"fft-2d:{shape_24}"]
    gates = config["automatic_gates"]
    decisions = {
        "workers": all(
            run["monitor"]["exit_code"] == 0
            and not run["monitor"]["timed_out"]
            and run["monitor"]["result_exists"]
            for run in runs
        ),
        "numeric": all(
            run["result"]["finite_nonnegative"]
            and run["result"]["minimum_direct_increment"] >= -2e-7
            for run in runs
        ),
        "source_identity": all(source_exact.values()),
        "repeat": all(repeat_exact.values()),
        "speedup_12mp": speedups[shape_12]
        >= float(gates["minimum_fft_median_speedup_at_12mp"]),
        "speedup_24mp": speedups[shape_24]
        >= float(gates["minimum_fft_median_speedup_at_24mp"]),
        "fft_time_24mp": fft_24["median_apply_seconds"]
        <= float(gates["maximum_fft_median_apply_seconds_at_24mp"]),
        "fft_memory_24mp": fft_24["maximum_peak_process_tree_rss_bytes"]
        <= int(gates["maximum_fft_peak_process_tree_rss_bytes_at_24mp"]),
        "fft_memory_ratio_24mp": (
            fft_24["median_peak_process_tree_rss_bytes"]
            / direct_24["median_peak_process_tree_rss_bytes"]
        )
        <= float(gates["maximum_fft_to_direct_peak_rss_ratio_at_24mp"]),
    }
    passed = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p3i_fft_backing_return_benchmark_report.v1",
        "node": config["node"],
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "runs": runs,
        "summaries": summaries,
        "fft_median_speedup": speedups,
        "source_hash_exact_across_algorithms": source_exact,
        "repeat_output_exact": repeat_exact,
        "decisions": decisions,
        "automatic_pass": passed,
        "branch": config["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": config["claim_ceiling"],
    }
    evidence_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    report = {**core, "evidence_id": evidence_id}
    _atomic_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/u6_p3i_fft_backing_return_benchmark_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-algorithm")
    parser.add_argument("--worker-shape", nargs=3, type=int)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker_shape is not None:
        if args.worker_output is None or args.worker_algorithm is None:
            raise ValueError("worker algorithm/output are required")
        _atomic_json(
            args.worker_output,
            worker(args.worker_algorithm, tuple(args.worker_shape)),
        )
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required in parent mode")
    report = parent(args.config, args.output_dir)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"evidence_id={report['evidence_id']}")
    for shape, speedup in report["fft_median_speedup"].items():
        direct = report["summaries"][f"direct-separable:{shape}"]
        fft = report["summaries"][f"fft-2d:{shape}"]
        print(
            f"{shape} speedup={speedup:.3f} "
            f"direct={direct['median_apply_seconds']:.3f}s "
            f"fft={fft['median_apply_seconds']:.3f}s "
            f"fft_peak={fft['maximum_peak_process_tree_rss_bytes'] / 2**30:.3f}GiB"
        )


if __name__ == "__main__":
    main()
