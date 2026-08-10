#!/usr/bin/env python
"""Fresh-process U6.P6ZD scanner-glare reference/streaming benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.film_physics.scanner_glare import (
    MultiscaleScannerGlareProfile,
    ScannerGlareComponent,
    apply_scanner_glare,
    compile_scanner_glare_kernel,
)
from src.film_physics.scanner_glare_streaming import (
    apply_scanner_glare_separable_streaming,
    compile_scanner_glare_separable,
)

SCHEMA = "neuro_film.u6_p6zd_scanner_glare_streaming_benchmark_contract.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _profile(root: Path) -> tuple[MultiscaleScannerGlareProfile, int]:
    contract = json.loads(
        (root / "configs/u6_p6zc_scanner_glare_separable_streaming_v1.json").read_text(
            encoding="utf-8"
        )
    )
    raw = contract["profile"]
    return (
        MultiscaleScannerGlareProfile(
            components=tuple(
                ScannerGlareComponent(
                    weight=float(item["weight"]),
                    sigma_pixels=float(item["sigma_pixels"]),
                )
                for item in raw["components"]
            ),
            flare_fraction=float(raw["flare_fraction"]),
            truncate_sigma=float(raw["truncate_sigma"]),
        ),
        int(raw["kernel_size"]),
    )


def _field(shape: tuple[int, int, int], seed: int, row_chunk: int = 128) -> np.ndarray:
    height, width, channels = shape
    if channels != 3 or min(height, width) < 1:
        raise ValueError("benchmark shape must be positive HxWx3")
    output = np.empty(shape, dtype=np.float64)
    x = np.arange(width, dtype=np.float64) / max(width - 1, 1)
    phase = (seed % 1009) / 1009.0
    for start in range(0, height, row_chunk):
        stop = min(start + row_chunk, height)
        y = np.arange(start, stop, dtype=np.float64)[:, None] / max(height - 1, 1)
        wave = np.sin(17.0 * x[None, :] + 11.0 * y + phase)
        block = output[start:stop]
        block[..., 0] = 0.03 + 0.68 * x + 0.08 * y + 0.015 * wave
        block[..., 1] = 0.04 + 0.25 * x + 0.48 * y - 0.012 * wave
        block[..., 2] = 0.02 + 0.15 * x + 0.58 * y + 0.010 * wave
    return np.clip(output, 0.0, 1.0)


def worker(
    algorithm: str, shape: tuple[int, int, int], *, seed: int, row_chunk: int
) -> dict[str, Any]:
    profile, kernel_size = _profile(ROOT)
    values = _field(shape, seed)
    started = perf_counter()
    if algorithm == "fft-2d-reference":
        output = apply_scanner_glare(
            values,
            compile_scanner_glare_kernel(profile, kernel_size=kernel_size),
            flare_fraction=profile.flare_fraction,
        )
    elif algorithm == "separable-row-streaming":
        output = apply_scanner_glare_separable_streaming(
            values,
            compile_scanner_glare_separable(profile, kernel_size=kernel_size),
            flare_fraction=profile.flare_fraction,
            row_chunk=row_chunk,
        )
    else:
        raise ValueError("unsupported benchmark algorithm")
    apply_seconds = perf_counter() - started
    return {
        "algorithm": algorithm,
        "shape": list(shape),
        "source_sha256": _hash_array(values),
        "output_sha256": _hash_array(output),
        "apply_seconds": apply_seconds,
        "finite_bounded": bool(
            np.all(np.isfinite(output))
            and float(np.min(output)) >= 0.0
            and float(np.max(output)) <= 1.0
        ),
        "minimum": float(np.min(output)),
        "maximum": float(np.max(output)),
        "logical_input_bytes": int(values.nbytes),
        "logical_output_bytes": int(output.nbytes),
    }


def _launch(
    algorithm: str,
    shape: tuple[int, int, int],
    output: Path,
    *,
    seed: int,
    row_chunk: int,
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
        "--seed",
        str(seed),
        "--row-chunk",
        str(row_chunk),
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
                sum(item.memory_info().rss for item in tree if item.is_running()),
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


def _ratio(values: list[float]) -> float:
    return max(values) / max(min(values), np.finfo(np.float64).tiny)


def _numerical_probe(seed: int, row_chunk: int) -> dict[str, Any]:
    profile, kernel_size = _profile(ROOT)
    values = _field((321, 513, 3), seed)
    reference = apply_scanner_glare(
        values,
        compile_scanner_glare_kernel(profile, kernel_size=kernel_size),
        flare_fraction=profile.flare_fraction,
    )
    candidate = apply_scanner_glare_separable_streaming(
        values,
        compile_scanner_glare_separable(profile, kernel_size=kernel_size),
        flare_fraction=profile.flare_fraction,
        row_chunk=row_chunk,
    )
    difference = candidate - reference
    return {
        "source_sha256": _hash_array(values),
        "reference_sha256": _hash_array(reference),
        "candidate_sha256": _hash_array(candidate),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference), dtype=np.float64))),
    }


def parent(config_path: Path, output_dir: Path) -> dict[str, Any]:
    raw = config_path.read_bytes()
    config = json.loads(raw.decode("utf-8"))
    if config.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P6ZD contract")
    parent_hashes = {
        key.removesuffix("_path"): _sha256(ROOT / path)
        for key, path in config["parents"].items()
        if key.endswith("_path")
    }
    parent_hashes_exact = all(
        parent_hashes[key.removesuffix("_sha256")] == expected
        for key, expected in config["parents"].items()
        if key.endswith("_sha256")
    )
    runs: list[dict[str, Any]] = []
    for shape_values in config["shapes"]:
        shape = tuple(int(value) for value in shape_values)
        counts = {algorithm: 0 for algorithm in config["algorithms"]}
        for algorithm in config["execution_order_per_shape"]:
            counts[algorithm] += 1
            result_path = output_dir / (
                f"{shape[0]}x{shape[1]}_{algorithm}_run{counts[algorithm]}.json"
            )
            monitor = _launch(
                algorithm,
                shape,
                result_path,
                seed=int(config["fixture_seed"]),
                row_chunk=int(config["row_chunk"]),
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
                    f"worker failed: {algorithm} {shape}: {monitor['stderr']}"
                )
        if any(
            count != int(config["runs_per_algorithm_shape"])
            for count in counts.values()
        ):
            raise ValueError("execution order replay count drift")

    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        result = run["result"]
        key = f"{result['algorithm']}:{'x'.join(map(str, result['shape']))}"
        groups.setdefault(key, []).append(run)
    summaries: dict[str, dict[str, Any]] = {}
    for key, items in groups.items():
        apply_values = [float(item["result"]["apply_seconds"]) for item in items]
        rss_values = [
            float(item["monitor"]["peak_process_tree_rss_bytes"]) for item in items
        ]
        summaries[key] = {
            "median_apply_seconds": float(np.median(apply_values)),
            "median_peak_process_tree_rss_bytes": float(np.median(rss_values)),
            "maximum_peak_process_tree_rss_bytes": int(max(rss_values)),
            "repeat_apply_time_ratio": _ratio(apply_values),
            "repeat_peak_rss_ratio": _ratio(rss_values),
            "repeat_output_exact": len(
                {item["result"]["output_sha256"] for item in items}
            )
            == 1,
        }
    probe = _numerical_probe(int(config["fixture_seed"]), int(config["row_chunk"]))
    shape_12 = "3000x4000x3"
    reference_12 = summaries[f"fft-2d-reference:{shape_12}"]
    streaming_12 = summaries[f"separable-row-streaming:{shape_12}"]
    gates = config["automatic_gates"]
    decisions = {
        "parent_hashes": parent_hashes_exact,
        "workers": all(
            row["monitor"]["exit_code"] == 0
            and not row["monitor"]["timed_out"]
            and row["monitor"]["result_exists"]
            for row in runs
        ),
        "finite_bounded": all(row["result"]["finite_bounded"] for row in runs),
        "source_identity": len({row["result"]["source_sha256"] for row in runs})
        == len(config["shapes"]),
        "repeat_output": all(row["repeat_output_exact"] for row in summaries.values()),
        "numerical_absolute": probe["maximum_absolute_error"]
        <= float(gates["maximum_cross_algorithm_absolute_error"]),
        "numerical_rmse": probe["rmse"] <= float(gates["maximum_cross_algorithm_rmse"]),
        "streaming_memory": streaming_12["maximum_peak_process_tree_rss_bytes"]
        <= int(gates["maximum_streaming_peak_process_tree_rss_bytes_at_12mp"]),
        "streaming_memory_ratio": (
            streaming_12["median_peak_process_tree_rss_bytes"]
            / reference_12["median_peak_process_tree_rss_bytes"]
        )
        <= float(gates["maximum_streaming_to_reference_median_peak_rss_ratio_at_12mp"]),
        "streaming_time": streaming_12["median_apply_seconds"]
        <= float(gates["maximum_streaming_median_apply_seconds_at_12mp"]),
        "streaming_time_ratio": (
            streaming_12["median_apply_seconds"] / reference_12["median_apply_seconds"]
        )
        <= float(
            gates["maximum_streaming_to_reference_median_apply_time_ratio_at_12mp"]
        ),
        "repeat_rss": all(
            row["repeat_peak_rss_ratio"]
            <= float(gates["maximum_repeat_peak_rss_ratio"])
            for row in summaries.values()
        ),
        "repeat_time": all(
            row["repeat_apply_time_ratio"]
            <= float(gates["maximum_repeat_apply_time_ratio"])
            for row in summaries.values()
        ),
    }
    passed = all(decisions.values())
    scientific_identity = {
        "schema": "neuro_film.u6_p6zd_scanner_glare_streaming_scientific_identity.v1",
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "parent_hashes": parent_hashes,
        "probe": probe,
        "output_hashes": sorted(
            {
                f"{row['result']['algorithm']}:{'x'.join(map(str, row['result']['shape']))}": row[
                    "result"
                ]["output_sha256"]
                for row in runs
            }.items()
        ),
        "decisions": decisions,
        "automatic_pass": passed,
    }
    stable_id = hashlib.sha256(
        json.dumps(scientific_identity, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()
    report = {
        "schema": "neuro_film.u6_p6zd_scanner_glare_streaming_benchmark_report.v1",
        "node": config["node"],
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "parent_hashes": parent_hashes,
        "runs": runs,
        "summaries": summaries,
        "numerical_probe": probe,
        "measurements": {
            "streaming_to_reference_median_peak_rss_ratio_at_12mp": (
                streaming_12["median_peak_process_tree_rss_bytes"]
                / reference_12["median_peak_process_tree_rss_bytes"]
            ),
            "streaming_to_reference_median_apply_time_ratio_at_12mp": (
                streaming_12["median_apply_seconds"]
                / reference_12["median_apply_seconds"]
            ),
        },
        "decisions": decisions,
        "automatic_pass": passed,
        "branch": config["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": config["claim_ceiling"],
        "stable_evidence_id": stable_id,
    }
    _atomic_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/u6_p6zd_scanner_glare_streaming_benchmark_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-algorithm")
    parser.add_argument("--worker-shape", nargs=3, type=int)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--seed", type=int, default=6206104)
    parser.add_argument("--row-chunk", type=int, default=64)
    args = parser.parse_args()
    if args.worker_shape is not None:
        if args.worker_output is None or args.worker_algorithm is None:
            raise ValueError("worker algorithm/output are required")
        _atomic_json(
            args.worker_output,
            worker(
                args.worker_algorithm,
                tuple(args.worker_shape),
                seed=args.seed,
                row_chunk=args.row_chunk,
            ),
        )
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required in parent mode")
    report = parent(args.config, args.output_dir)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    for key, summary in report["summaries"].items():
        print(
            f"{key} apply={summary['median_apply_seconds']:.3f}s "
            f"peak={summary['median_peak_process_tree_rss_bytes'] / 2**20:.1f}MiB"
        )


if __name__ == "__main__":
    main()
