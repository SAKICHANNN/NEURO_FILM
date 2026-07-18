"""Measure the frozen U1.6G4G 24MP staged-density workload in child processes."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmfx import execute_staged_density_halation_default  # noqa: E402


def _sha_array(value: np.ndarray) -> str:
    contiguous = value if value.flags.c_contiguous else np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def analytic_highlight_field(
    shape: tuple[int, int, int], *, row_chunk: int
) -> np.ndarray:
    """Construct the fixed field without allocating a full coordinate grid."""
    height, width, channels = shape
    if channels != 3 or height < 2 or width < 2:
        raise ValueError("shape must be HxWx3 with spatial dimensions >=2")
    if isinstance(row_chunk, bool) or not isinstance(row_chunk, int) or row_chunk <= 0:
        raise ValueError("row_chunk must be a positive integer")
    field = np.empty(shape, np.float32)
    x = np.arange(width, dtype=np.float32) / np.float32(width - 1)
    for y0 in range(0, height, row_chunk):
        y1 = min(height, y0 + row_chunk)
        y = np.arange(y0, y1, dtype=np.float32)[:, None] / np.float32(height - 1)
        wave = np.sin(np.float32(19.0) * x[None, :] + np.float32(7.0) * y)
        hot_a = np.exp(
            -(
                ((x[None, :] - np.float32(0.31)) / np.float32(0.035)) ** 2
                + ((y - np.float32(0.42)) / np.float32(0.055)) ** 2
            )
        )
        hot_b = np.exp(
            -(
                ((x[None, :] - np.float32(0.73)) / np.float32(0.055)) ** 2
                + ((y - np.float32(0.68)) / np.float32(0.04)) ** 2
            )
        )
        block = field[y0:y1]
        block[..., 0] = 0.025 + 0.52 * x + 0.10 * y + 0.025 * wave
        block[..., 1] = 0.020 + 0.18 * x + 0.34 * y - 0.018 * wave
        block[..., 2] = 0.018 + 0.12 * x + 0.22 * y + 0.012 * wave
        block += hot_a[..., None] * np.asarray([0.92, 0.78, 0.50], np.float32)
        block += hot_b[..., None] * np.asarray([0.64, 0.82, 1.00], np.float32)
        np.clip(block, 0.0, 1.0, out=block)
    return field


def screen_composite_rows(base: np.ndarray, layer, *, row_chunk: int) -> np.ndarray:
    """Composite one screen layer with full output but bounded row temporaries."""
    if layer.mode != "screen" or layer.rgb is None or layer.alpha is None:
        raise ValueError("a complete screen layer is required")
    if layer.rgb.shape != base.shape:
        raise ValueError("layer RGB shape mismatch")
    alpha = layer.alpha
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    if alpha.shape != (*base.shape[:2], 1):
        raise ValueError("layer alpha shape mismatch")
    output = np.empty_like(base, dtype=np.float32)
    for y0 in range(0, base.shape[0], row_chunk):
        y1 = min(base.shape[0], y0 + row_chunk)
        source = base[y0:y1]
        blend = np.clip(layer.rgb[y0:y1], 0.0, 1.0)
        weight = np.clip(alpha[y0:y1], 0.0, 1.0)
        screened = 1.0 - (1.0 - source) * (1.0 - blend)
        output[y0:y1] = source * (1.0 - weight) + screened * weight
        np.clip(output[y0:y1], 0.0, 1.0, out=output[y0:y1])
    return output


def _metadata_payload(metadata) -> dict:
    payload = dataclasses.asdict(metadata)
    return payload


def _metadata_sha(metadata) -> str:
    encoded = json.dumps(
        _metadata_payload(metadata), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def worker(config: dict, *, inject_failure: bool = False) -> dict:
    total_started = perf_counter()
    shape = tuple(int(value) for value in config["shape"])
    phase_started = perf_counter()
    base = analytic_highlight_field(
        shape, row_chunk=int(config["input_generation_row_chunk"])
    )
    source_seconds = perf_counter() - phase_started
    source_hash = _sha_array(base)
    if inject_failure:
        raise RuntimeError("injected before_executor failure")
    phase_started = perf_counter()
    layer, metadata = execute_staged_density_halation_default(
        base,
        tile_size=int(config["tile_size"]),
        source_row_chunk=int(config["source_row_chunk"]),
        coarse_row_chunk=int(config["coarse_row_chunk"]),
    )
    executor_seconds = perf_counter() - phase_started
    phase_started = perf_counter()
    composite = screen_composite_rows(
        base, layer, row_chunk=int(config["composite_row_chunk"])
    )
    composite_seconds = perf_counter() - phase_started
    bounded = bool(
        np.isfinite(base).all()
        and np.isfinite(layer.rgb).all()
        and np.isfinite(layer.alpha).all()
        and np.isfinite(composite).all()
        and base.min() >= 0.0
        and base.max() <= 1.0
        and layer.rgb.min() >= 0.0
        and layer.rgb.max() <= 1.0
        and layer.alpha.min() >= 0.0
        and layer.alpha.max() <= 0.26
        and composite.min() >= 0.0
        and composite.max() <= 1.0
    )
    return {
        "source_sha256": source_hash,
        "layer_rgb_sha256": _sha_array(layer.rgb),
        "layer_alpha_sha256": _sha_array(layer.alpha),
        "composite_sha256": _sha_array(composite),
        "metadata_sha256": _metadata_sha(metadata),
        "metadata": _metadata_payload(metadata),
        "finite_and_bounded": bounded,
        "timings_seconds": {
            "source_generation": source_seconds,
            "executor": executor_seconds,
            "stream_composite": composite_seconds,
            "worker_total": perf_counter() - total_started,
        },
        "scratch_pixel_bytes": 0,
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _launch_worker(
    config_path: Path,
    output_path: Path,
    *,
    interval: float,
    timeout: float,
    inject_failure: bool,
) -> dict:
    output_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--worker-output",
        str(output_path),
    ]
    if inject_failure:
        command.append("--inject-failure")
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
            peak_rss = max(peak_rss, monitored.memory_info().rss)
        except psutil.NoSuchProcess:
            pass
        if perf_counter() - started > timeout:
            timed_out = True
            process.kill()
            break
        sleep(interval)
    stdout, stderr = process.communicate()
    try:
        peak_rss = max(peak_rss, monitored.memory_info().rss)
    except psutil.NoSuchProcess:
        pass
    orphan = int(psutil.pid_exists(process.pid))
    return {
        "pid": process.pid,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "parent_wall_seconds": perf_counter() - started,
        "peak_child_rss_bytes": peak_rss,
        "stdout": stdout,
        "stderr": stderr,
        "result_exists": output_path.exists(),
        "temporary_exists": output_path.with_name(output_path.name + ".tmp").exists(),
        "orphan_worker_count": orphan,
    }


def _run_pass(worker_result: dict, monitor: dict, gates: dict) -> bool:
    timings = worker_result["timings_seconds"]
    metadata = worker_result["metadata"]
    return bool(
        monitor["exit_code"] == 0
        and not monitor["timed_out"]
        and monitor["peak_child_rss_bytes"] <= gates["peak_child_rss_bytes_max"]
        and timings["worker_total"] <= gates["worker_total_seconds_max"]
        and timings["executor"] <= gates["executor_seconds_max"]
        and timings["stream_composite"] <= gates["stream_composite_seconds_max"]
        and worker_result["finite_and_bounded"]
        and metadata["persistent_derived_full_scalar_bytes"]
        == gates["persistent_derived_full_scalar_bytes"]
        and metadata["scratch_disk_bytes"] == gates["scratch_pixel_bytes"]
        and metadata["max_source_window_shape"][0] < metadata["source_shape"][0]
        and worker_result["scratch_pixel_bytes"] == gates["scratch_pixel_bytes"]
        and not monitor["temporary_exists"]
        and monitor["orphan_worker_count"] == gates["orphan_worker_count"]
    )


def run_parent(config_path: Path, output_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    interval = float(config["rss_sample_interval_seconds"])
    timeout = float(config["worker_timeout_seconds"])
    failure_path = output_dir / "failure_probe.json"
    failure = _launch_worker(
        config_path,
        failure_path,
        interval=interval,
        timeout=timeout,
        inject_failure=True,
    )
    failure_pass = bool(
        failure["exit_code"] != 0
        and not failure["timed_out"]
        and not failure["result_exists"]
        and not failure["temporary_exists"]
        and failure["orphan_worker_count"] == 0
    )
    runs = []
    for index in range(int(config["runs"])):
        worker_path = output_dir / f"worker_{index + 1}.json"
        monitor = _launch_worker(
            config_path,
            worker_path,
            interval=interval,
            timeout=timeout,
            inject_failure=False,
        )
        worker_result = (
            json.loads(worker_path.read_text(encoding="utf-8"))
            if worker_path.exists()
            else None
        )
        passed = bool(worker_result is not None and _run_pass(worker_result, monitor, config["gates"]))
        runs.append({"monitor": monitor, "worker": worker_result, "passed": passed})
    hash_fields = (
        "source_sha256",
        "layer_rgb_sha256",
        "layer_alpha_sha256",
        "composite_sha256",
        "metadata_sha256",
    )
    repeat_hashes = bool(
        len(runs) == 2
        and all(run["worker"] is not None for run in runs)
        and all(
            runs[0]["worker"][field] == runs[1]["worker"][field]
            for field in hash_fields
        )
    )
    return {
        "schema_version": 1,
        "node": config["node"],
        "operator_version": config["operator_version"],
        "failure_probe": {"monitor": failure, "passed": failure_pass},
        "runs": runs,
        "repeat_hashes_pass": repeat_hashes,
        "gate_result": {
            "automatic_pass": bool(
                failure_pass and repeat_hashes and all(run["passed"] for run in runs)
            ),
            "renderer_integration_allowed": False,
        },
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--inject-failure", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.worker_output is not None:
        result = worker(config, inject_failure=args.inject_failure)
        _atomic_write_json(args.worker_output, result)
        return 0
    if args.output is None or args.work_dir is None:
        parser.error("parent mode requires --output and --work-dir")
    report = run_parent(args.config.resolve(), args.work_dir.resolve())
    _atomic_write_json(args.output, report)
    print(json.dumps(report["gate_result"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
