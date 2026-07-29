#!/usr/bin/env python
"""Fresh-process U6.P3K 24MP streamed FFT backing-return benchmark."""

from __future__ import annotations

import argparse
import gc
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

from scripts.benchmark_u6_p3i_fft_backing_return import _field  # noqa: E402
from src.film_physics import (  # noqa: E402
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    backing_return_profile_from_contract,
    compile_backing_return_profile,
    iter_fft_backing_return_row_cores,
)


SCHEMA = "neuro_film.u6_p3k_streamed_fft_backing_return_benchmark_contract.v1"


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


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(
        memoryview(np.ascontiguousarray(array)).cast("B")
    ).hexdigest()


def worker(shape: tuple[int, int, int], tile_rows: int) -> dict[str, Any]:
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
    source_sha256 = _hash_array(values)
    state = PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
        PhysicalScale(compiled.pixel_pitch_um),
    )
    del values
    gc.collect()

    output_hash = hashlib.sha256()
    coverage = np.zeros(shape[0], dtype=np.uint8)
    maximum_core_rows = 0
    minimum_output = float("inf")
    minimum_direct_increment = float("inf")
    output_bytes = 0
    stream_started = perf_counter()
    for y0, y1, core in iter_fft_backing_return_row_cores(
        state, compiled, tile_rows=tile_rows, order="forward"
    ):
        core_bytes = memoryview(np.ascontiguousarray(core)).cast("B")
        output_hash.update(core_bytes)
        output_bytes += len(core_bytes)
        coverage[y0:y1] += 1
        maximum_core_rows = max(maximum_core_rows, core.shape[0])
        minimum_output = min(minimum_output, float(np.min(core)))
        minimum_direct_increment = min(
            minimum_direct_increment,
            float(np.min(core - state.values[y0:y1])),
        )
    stream_seconds = perf_counter() - stream_started
    return {
        "shape": list(shape),
        "tile_rows": tile_rows,
        "source_sha256": source_sha256,
        "output_sha256": output_hash.hexdigest(),
        "source_seconds": source_seconds,
        "stream_seconds": stream_seconds,
        "coverage_exactly_once": bool(np.all(coverage == 1)),
        "maximum_live_output_core_rows": maximum_core_rows,
        "minimum_output": minimum_output,
        "minimum_direct_increment": minimum_direct_increment,
        "finite_nonnegative_and_direct_retaining": bool(
            np.isfinite(minimum_output)
            and minimum_output >= 0.0
            and minimum_direct_increment >= -2e-7
        ),
        "logical_input_bytes": int(state.values.nbytes),
        "logical_output_bytes": output_bytes,
        "full_output_allocated": False,
    }


def _launch(
    shape: tuple[int, int, int],
    tile_rows: int,
    output: Path,
    *,
    interval: float,
    timeout: float,
) -> dict[str, Any]:
    output.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-shape",
        *[str(value) for value in shape],
        "--tile-rows",
        str(tile_rows),
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


def parent(config_path: Path, output_dir: Path) -> dict[str, Any]:
    raw = config_path.read_bytes()
    config = json.loads(raw.decode("utf-8"))
    if config.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P3K contract")
    shape = tuple(int(value) for value in config["shape"])
    tile_rows = int(config["tile_rows"])
    runs = []
    for replay in range(int(config["runs"])):
        result_path = output_dir / f"run_{replay + 1}.json"
        monitor = _launch(
            shape,
            tile_rows,
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
            raise RuntimeError(f"stream benchmark worker failed: {monitor['stderr']}")

    source_repeat = len(
        {run["result"]["source_sha256"] for run in runs}
    ) == 1
    output_repeat = len(
        {run["result"]["output_sha256"] for run in runs}
    ) == 1
    stream_seconds = [
        float(run["result"]["stream_seconds"]) for run in runs
    ]
    wall_seconds = [float(run["monitor"]["wall_seconds"]) for run in runs]
    peak_rss = [
        int(run["monitor"]["peak_process_tree_rss_bytes"]) for run in runs
    ]
    median_stream = float(np.median(stream_seconds))
    repeat_wall_ratio = max(wall_seconds) / min(wall_seconds)
    gates = config["automatic_gates"]
    decisions = {
        "workers": all(
            run["monitor"]["exit_code"] == 0
            and not run["monitor"]["timed_out"]
            and run["monitor"]["result_exists"]
            for run in runs
        ),
        "source_repeat": source_repeat,
        "output_repeat": output_repeat,
        "coverage": all(
            run["result"]["coverage_exactly_once"] for run in runs
        ),
        "numeric": all(
            run["result"]["finite_nonnegative_and_direct_retaining"]
            for run in runs
        ),
        "core_bound": max(
            run["result"]["maximum_live_output_core_rows"] for run in runs
        )
        <= int(gates["maximum_live_output_core_rows"]),
        "memory": max(peak_rss)
        <= int(gates["maximum_peak_process_tree_rss_bytes"]),
        "runtime": median_stream
        <= float(gates["maximum_median_stream_seconds"]),
        "repeat_wall": repeat_wall_ratio
        <= float(gates["maximum_repeat_wall_ratio"]),
        "no_full_output": all(
            run["result"]["full_output_allocated"] is False for run in runs
        ),
    }
    passed = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p3k_streamed_fft_backing_return_benchmark_report.v1",
        "node": config["node"],
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "runs": runs,
        "metrics": {
            "median_stream_seconds": median_stream,
            "stream_seconds": stream_seconds,
            "wall_seconds": wall_seconds,
            "repeat_wall_ratio": repeat_wall_ratio,
            "maximum_peak_process_tree_rss_bytes": max(peak_rss),
            "p3i_full_fft_24mp_maximum_rss_bytes": config["parents"][
                "p3i_full_fft_24mp_maximum_rss_bytes"
            ],
            "rss_ratio_to_p3i_full_fft": max(peak_rss)
            / float(
                config["parents"]["p3i_full_fft_24mp_maximum_rss_bytes"]
            ),
        },
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
        default=Path(
            "configs/u6_p3k_streamed_fft_backing_return_benchmark_v1.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-shape", nargs=3, type=int)
    parser.add_argument("--tile-rows", type=int)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker_shape is not None:
        if args.worker_output is None or args.tile_rows is None:
            raise ValueError("worker output/tile rows are required")
        _atomic_json(
            args.worker_output,
            worker(tuple(args.worker_shape), args.tile_rows),
        )
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required in parent mode")
    report = parent(args.config, args.output_dir)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"evidence_id={report['evidence_id']}")
    print(
        f"median_stream={report['metrics']['median_stream_seconds']:.3f}s "
        f"peak={report['metrics']['maximum_peak_process_tree_rss_bytes'] / 2**30:.3f}GiB "
        f"rss_ratio={report['metrics']['rss_ratio_to_p3i_full_fft']:.3f}"
    )


if __name__ == "__main__":
    main()
