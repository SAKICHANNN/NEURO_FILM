#!/usr/bin/env python
"""Run the frozen U6.P4Q integration and fresh-process audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_u5_r2an0_paired_positive_film_recovery import (  # noqa: E402
    _atomic_write,
    _canonical_json,
)
from src.eval.physical_constant_rate_integration_live import (  # noqa: E402
    evaluate_live_records,
    load_contract,
    small_integration_evidence,
)


CONFIG_SHA256 = "a8fbc049027a2ca6a34aca84c8d05e52c81225b9f7bb6bcc2b8844d0a8a6b1ed"


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
    contract: dict[str, Any], config_path: Path, run_index: int
) -> dict[str, Any]:
    owned = (
        ROOT
        / "outputs/experiments/u6_p4q_constant_rate_integration_live_v1"
        / f".owned_run_{run_index}_{os.getpid()}"
    )
    owned.mkdir(parents=True, exist_ok=False)
    result_path = owned / "worker_result.json"
    stderr_path = owned / "worker.stderr.txt"
    started = time.perf_counter()
    peak = 0
    samples = 0
    timed_out = False
    payload: dict[str, Any] = {}
    exit_code = -1
    stderr_bytes = -1
    try:
        with stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_u6_p4q_constant_rate_worker.py"),
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
            maximum = float(
                contract["executor"]["maximum_runtime_seconds"]
            )
            interval = float(
                contract["executor"]["sample_interval_seconds"]
            )
            while process.poll() is None:
                peak = max(peak, _tree_rss(observed))
                samples += 1
                if time.perf_counter() - started > maximum:
                    timed_out = True
                    _terminate_owned_tree(observed)
                    break
                time.sleep(interval)
            exit_code = int(process.wait())
        stderr_bytes = stderr_path.stat().st_size
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        wall = time.perf_counter() - started
        shutil.rmtree(owned, ignore_errors=False)
    return {
        **payload,
        "run_index": run_index,
        "process_tree_peak_rss_bytes": peak,
        "wall_seconds": wall,
        "liveness_samples": samples,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "stderr_bytes": stderr_bytes,
        "owned_temp_residue_count": int(owned.exists()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u6_p4q_constant_rate_integration_live_v1.json",
    )
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/experiments/u6_p4q_constant_rate_integration_live_v1/report.json",
    )
    args = parser.parse_args()
    contract, parent = load_contract(
        ROOT, args.config, args.expected_config_sha256
    )
    small = small_integration_evidence(contract, parent)
    runs = [
        _run_worker(contract, args.config, index)
        for index in range(int(contract["executor"]["fresh_process_runs"]))
    ]
    report = evaluate_live_records(contract, small, runs)
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
                    row.get("stream_sha256", "")
                    for row in report["runs"]
                ],
                "small_integration": report["small_integration"],
                "checks": report["checks"],
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "config_sha256": CONFIG_SHA256,
            }
        )
    ).hexdigest()
    _atomic_write(args.output, _canonical_json(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
