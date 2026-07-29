#!/usr/bin/env python
"""Run the frozen U6.P4L fresh-process live performance audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_u5_r2an0_paired_positive_film_recovery import (  # noqa: E402
    _atomic_write,
    _canonical_json,
)
from src.eval.physical_exact_area_live_performance import (  # noqa: E402
    evaluate_live_performance_records,
    load_contract,
)


CONFIG_SHA256 = "31bbdeb1dc494f55b6344f8275de2e9a71ecdd76ef247bb29882363219075700"


def _tree_rss(process: psutil.Process) -> int:
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    total = 0
    for item in processes:
        try:
            total += int(item.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def _terminate_owned_tree(process: psutil.Process) -> None:
    try:
        children = process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []
    for item in reversed(children):
        try:
            item.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        process.terminate()
    except psutil.NoSuchProcess:
        pass
    _, alive = psutil.wait_procs([*children, process], timeout=2.0)
    for item in alive:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass


def _run_worker(
    contract: dict,
    config_path: Path,
    run_index: int,
) -> dict:
    owned = (
        ROOT
        / "outputs/experiments/u6_p4l_exact_area_live_performance_v1"
        / f".owned_run_{run_index}_{process_id()}"
    )
    owned.mkdir(parents=True, exist_ok=False)
    result_path = owned / "worker_result.json"
    stderr_path = owned / "worker.stderr.txt"
    process = None
    started = time.perf_counter()
    peak = 0
    samples = 0
    timed_out = False
    exit_code = -1
    stderr_bytes = -1
    payload: dict = {}
    try:
        with stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_u6_p4l_exact_area_worker.py"),
                    "--config",
                    str(config_path),
                    "--expected-config-sha256",
                    CONFIG_SHA256,
                    "--output",
                    str(result_path),
                ],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=stderr_handle,
            )
            observed = psutil.Process(process.pid)
            maximum = float(contract["worker"]["maximum_runtime_seconds"])
            interval = float(contract["worker"]["sample_interval_seconds"])
            while process.poll() is None:
                peak = max(peak, _tree_rss(observed))
                samples += 1
                if time.perf_counter() - started > maximum:
                    timed_out = True
                    _terminate_owned_tree(observed)
                    break
                time.sleep(interval)
            exit_code = int(process.wait())
            peak = max(peak, _tree_rss(observed))
        stderr_bytes = stderr_path.stat().st_size
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        wall = time.perf_counter() - started
        shutil.rmtree(owned, ignore_errors=False)
    return {
        "run_index": run_index,
        "stream_sha256": payload.get("stream_sha256", ""),
        "output_rows": payload.get("output_rows", 0),
        "process_tree_peak_rss_bytes": peak,
        "wall_seconds": wall,
        "liveness_samples": samples,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "stderr_bytes": stderr_bytes,
        "owned_temp_residue_count": int(owned.exists()),
    }


def process_id() -> int:
    return psutil.Process().pid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u6_p4l_exact_area_live_performance_v1.json",
    )
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/experiments/u6_p4l_exact_area_live_performance_v1/report.json",
    )
    args = parser.parse_args()
    contract, _ = load_contract(
        ROOT, args.config, args.expected_config_sha256
    )
    runs = [
        _run_worker(contract, args.config, run_index)
        for run_index in range(int(contract["worker"]["fresh_process_runs"]))
    ]
    report = evaluate_live_performance_records(contract, runs)
    report["config_sha256"] = CONFIG_SHA256
    report["software_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    report["stable_evidence_id"] = hashlib.sha256(
        _canonical_json(
            {
                "schema": report["schema"],
                "node": report["node"],
                "stream_hashes": [
                    row["stream_sha256"] for row in report["runs"]
                ],
                "checks": report["checks"],
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "claim_ceiling": report["claim_ceiling"],
                "config_sha256": CONFIG_SHA256,
            }
        )
    ).hexdigest()
    _atomic_write(args.output, _canonical_json(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
