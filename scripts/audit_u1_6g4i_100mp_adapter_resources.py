"""Audit the frozen U1.6G4I 100MP isolated-adapter resource contract."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_u1_6g4g_24mp_resources import (  # noqa: E402
    _atomic_write_json,
    _sha_array,
    analytic_highlight_field,
)
from src.filmfx.staged_density_adapter import (  # noqa: E402
    STAGED_DENSITY_ADAPTER_VERSION,
    render_staged_density_halation_research,
)


def _metadata_payload(metadata: object) -> dict:
    if not dataclasses.is_dataclass(metadata):
        raise TypeError("adapter metadata must be a dataclass")
    return dataclasses.asdict(metadata)


def _json_sha(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def worker(config: dict, *, inject_failure: bool = False) -> dict:
    total_started = perf_counter()
    if inject_failure:
        raise RuntimeError("injected before_input_allocation failure")

    shape = tuple(int(value) for value in config["shape"])
    if int(np.prod(shape[:2])) != int(config["pixel_count"]):
        raise ValueError("shape does not match frozen pixel_count")
    phase_started = perf_counter()
    base = analytic_highlight_field(
        shape, row_chunk=int(config["input_generation_row_chunk"])
    )
    source_seconds = perf_counter() - phase_started
    source_hash = _sha_array(base)

    phase_started = perf_counter()
    output, metadata = render_staged_density_halation_research(
        base,
        tile_size=int(config["tile_size"]),
        source_row_chunk=int(config["source_row_chunk"]),
        coarse_row_chunk=int(config["coarse_row_chunk"]),
        composite_row_chunk=int(config["composite_row_chunk"]),
        output_margin=int(config["output_margin"]),
    )
    adapter_seconds = perf_counter() - phase_started
    metadata_payload = _metadata_payload(metadata)
    bounded = bool(
        base.dtype == np.float32
        and output.dtype == np.float32
        and output.shape == base.shape
        and np.isfinite(base).all()
        and np.isfinite(output).all()
        and base.min() >= 0.0
        and base.max() <= 1.0
        and output.min() >= 0.0
        and output.max() <= 1.0
    )
    return {
        "source_sha256": source_hash,
        "output_sha256": _sha_array(output),
        "metadata_sha256": _json_sha(metadata_payload),
        "metadata": metadata_payload,
        "finite_and_bounded": bounded,
        "timings_seconds": {
            "source_generation": source_seconds,
            "adapter": adapter_seconds,
            "worker_total": perf_counter() - total_started,
        },
        "scratch_pixel_bytes": 0,
    }


def preflight(config: dict, output_dir: Path) -> dict:
    rules = config["preflight"]
    available = int(psutil.virtual_memory().available)
    current = psutil.Process()
    excluded = {current.pid, *(parent.pid for parent in current.parents())}
    competitors = []
    for process in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            if process.pid in excluded:
                continue
            rss = int(process.info["memory_info"].rss)
            if rss > int(rules["competing_process_rss_bytes_max"]):
                competitors.append(
                    {"pid": process.pid, "name": process.info["name"], "rss_bytes": rss}
                )
        except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
            continue
    probe = output_dir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_disk = int(shutil.disk_usage(probe).free)
    passed = bool(
        available >= int(rules["available_physical_memory_bytes_min"])
        and not competitors
        and free_disk >= int(rules["minimum_free_disk_bytes"])
    )
    return {
        "available_physical_memory_bytes": available,
        "free_disk_bytes": free_disk,
        "competing_processes": competitors,
        "passed": passed,
    }


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
    root_process = psutil.Process(process.pid)
    observed_process_ids = {process.pid}
    peak_rss = 0
    timed_out = False
    while process.poll() is None:
        try:
            tree = [root_process, *root_process.children(recursive=True)]
            observed_process_ids.update(item.pid for item in tree)
            tree_rss = 0
            for item in tree:
                try:
                    tree_rss += int(item.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
            peak_rss = max(peak_rss, tree_rss)
        except psutil.NoSuchProcess:
            pass
        if perf_counter() - started > timeout:
            timed_out = True
            for child in root_process.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            process.kill()
            break
        sleep(interval)
    stdout, stderr = process.communicate()
    psutil.wait_procs(
        [psutil.Process(pid) for pid in observed_process_ids if psutil.pid_exists(pid)],
        timeout=2.0,
    )
    orphan_count = sum(int(psutil.pid_exists(pid)) for pid in observed_process_ids)
    return {
        "pid": process.pid,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "parent_wall_seconds": perf_counter() - started,
        "peak_process_tree_rss_bytes": peak_rss,
        "observed_process_ids": sorted(observed_process_ids),
        "stdout": stdout,
        "stderr": stderr,
        "result_exists": output_path.exists(),
        "temporary_exists": output_path.with_name(output_path.name + ".tmp").exists(),
        "orphan_worker_count": orphan_count,
    }


def _run_pass(worker_result: dict, monitor: dict, gates: dict) -> bool:
    timings = worker_result["timings_seconds"]
    metadata = worker_result["metadata"]
    declared_floor = int(
        metadata["input_bytes"]
        + metadata["private_layer_bytes"]
        + metadata["returned_composite_bytes"]
    )
    return bool(
        monitor["exit_code"] == 0
        and not monitor["timed_out"]
        and declared_floor == int(gates["known_live_array_floor_bytes"])
        and monitor["peak_process_tree_rss_bytes"] >= declared_floor
        and monitor["peak_process_tree_rss_bytes"]
        <= int(gates["peak_process_tree_rss_bytes_max"])
        and timings["worker_total"] <= float(gates["worker_total_seconds_max"])
        and timings["adapter"] <= float(gates["adapter_seconds_max"])
        and timings["source_generation"]
        <= float(gates["source_generation_seconds_max"])
        and worker_result["finite_and_bounded"]
        and metadata["version"] == STAGED_DENSITY_ADAPTER_VERSION
        and metadata["public_ndarray_count"] == int(gates["public_ndarray_count"])
        and metadata["scratch_bytes"] == int(gates["scratch_pixel_bytes"])
        and worker_result["scratch_pixel_bytes"] == int(gates["scratch_pixel_bytes"])
        and not monitor["temporary_exists"]
        and monitor["orphan_worker_count"] == int(gates["orphan_worker_count"])
    )


def run_parent(config_path: Path, output_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_result = preflight(config, output_dir)
    if not preflight_result["passed"]:
        return {
            "schema_version": 1,
            "node": config["node"],
            "preflight": preflight_result,
            "failure_probe": None,
            "runs": [],
            "repeat_hashes_pass": False,
            "gate_result": {
                "automatic_pass": False,
                "run_deferred_by_preflight": True,
                "renderer_integration_allowed": False,
            },
            "claim_ceiling": config["claim_ceiling"],
        }

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
        result = (
            json.loads(worker_path.read_text(encoding="utf-8"))
            if worker_path.exists()
            else None
        )
        passed = bool(result and _run_pass(result, monitor, config["gates"]))
        runs.append({"monitor": monitor, "worker": result, "passed": passed})
    fields = ("source_sha256", "output_sha256", "metadata_sha256")
    repeat_hashes = bool(
        len(runs) == 2
        and all(run["worker"] is not None for run in runs)
        and all(
            runs[0]["worker"][field] == runs[1]["worker"][field]
            for field in fields
        )
    )
    return {
        "schema_version": 1,
        "node": config["node"],
        "adapter_version": config["adapter_version"],
        "preflight": preflight_result,
        "failure_probe": {"monitor": failure, "passed": failure_pass},
        "runs": runs,
        "repeat_hashes_pass": repeat_hashes,
        "gate_result": {
            "automatic_pass": bool(
                failure_pass and repeat_hashes and all(run["passed"] for run in runs)
            ),
            "run_deferred_by_preflight": False,
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
    return 0 if report["gate_result"]["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
