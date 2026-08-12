#!/usr/bin/env python3
"""Measure replayable Native Standard through verified PNG16 at 24MP."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_u6_p4fr_native_standard_replayable_24mp import (
    _input_sha,
    _runtime,
    _source_rows,
)
from src.film_physics.native_standard_output import (
    commit_verified_native_standard_png16,
    verify_native_standard_png16,
)
from src.film_physics.native_standard_replayable_staging import (
    stage_native_standard_replayable_rows,
)


def worker(*, contract: Path, output: Path, transaction: Path) -> None:
    frozen = json.loads(contract.read_text())
    scenario = frozen["scenario"]
    height, width = int(scenario["height"]), int(scenario["width"])
    runtime = _runtime()
    input_sha = _input_sha(height, width, int(runtime.policy["tile_rows"]))
    transaction.mkdir(parents=True, exist_ok=False)
    raw = transaction / "render.f32"
    raw_report = transaction / "render.raw.json"
    png = transaction / "render.png"
    png_report = transaction / "render.png.json"
    started = time.perf_counter()
    try:
        staged = stage_native_standard_replayable_rows(
            runtime,
            height=height,
            width=width,
            source_rows=lambda start, count: _source_rows(
                start, count, height, width
            ),
            expected_input_sha256=input_sha,
            output_path=raw,
            report_path=raw_report,
        )
        committed = commit_verified_native_standard_png16(
            staging_report_path=raw_report,
            expected_staging_report_sha256=staged["report_sha256"],
            expected_staging_run_id=staged["run_id"],
            output_path=png,
            report_path=png_report,
        )
        verified = verify_native_standard_png16(
            report_path=png_report,
            expected_report_sha256=committed["report_sha256"],
            expected_delivery_id=committed["delivery_id"],
        )
        result = {
            "input_sha256": input_sha,
            "raw_output_sha256": staged["output_sha256"],
            "png_output_sha256": committed["output_sha256"],
            "png_verification_id": verified["verification_id"],
            "png_bytes": png.stat().st_size,
            "wall_seconds": time.perf_counter() - started,
            "restart_verified": True,
        }
    finally:
        shutil.rmtree(transaction, ignore_errors=False)
    result["cleanup_pass"] = not transaction.exists()
    output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"))
    )


def _monitor(command: list[str], output: Path, timeout: float, interval: float) -> dict:
    process = subprocess.Popen(command, cwd=ROOT)
    root = psutil.Process(process.pid)
    peak = 0
    started = time.perf_counter()
    try:
        while process.poll() is None:
            if time.perf_counter() - started > timeout:
                raise TimeoutError("P4FU worker timeout")
            try:
                total = sum(
                    item.memory_info().rss
                    for item in [root, *root.children(recursive=True)]
                )
                peak = max(peak, total)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(interval)
        if process.returncode != 0:
            raise RuntimeError(f"P4FU worker exit {process.returncode}")
        return {
            "peak_process_tree_rss_bytes": peak,
            "worker": json.loads(output.read_text()),
        }
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def benchmark(contract: Path, output_dir: Path) -> dict:
    frozen = json.loads(contract.read_text())
    scenario, gate = frozen["scenario"], frozen["gates"]
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for index in range(int(scenario["runs"])):
        output = output_dir / f"worker-{index + 1}.json"
        transaction = output_dir / f"transaction-{index + 1}"
        runs.append(
            _monitor(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--contract",
                    str(contract.resolve()),
                    "--output",
                    str(output),
                    "--transaction",
                    str(transaction),
                ],
                output,
                float(scenario["timeout_seconds"]),
                float(scenario["sample_interval_seconds"]),
            )
        )
    peaks = [row["peak_process_tree_rss_bytes"] for row in runs]
    walls = [row["worker"]["wall_seconds"] for row in runs]
    pngs = {row["worker"]["png_output_sha256"] for row in runs}
    checks = {
        "identity": len(pngs) == 1,
        "peak": max(peaks) <= int(gate["maximum_peak_process_tree_rss_bytes"]),
        "wall": max(walls) <= float(gate["maximum_worker_wall_seconds"]),
        "repeat": max(peaks) / min(peaks)
        <= float(gate["maximum_peak_repeat_ratio"]),
        "verified": all(row["worker"]["restart_verified"] for row in runs),
        "cleanup": all(row["worker"]["cleanup_pass"] for row in runs),
    }
    stable = {
        "contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
        "runs": runs,
        "metrics": {
            "peak_process_tree_rss_bytes": peaks,
            "worker_wall_seconds": walls,
            "peak_repeat_ratio": max(peaks) / min(peaks),
        },
        "gates": checks,
        "decision": frozen["decision_if_pass"]
        if all(checks.values())
        else frozen["decision_if_fail"],
        "claim_ceiling": frozen["claim_ceiling"],
    }
    return {
        "schema": frozen["schema"].replace("_contract", ""),
        "automatic_pass": all(checks.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--transaction", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(
            contract=args.contract,
            output=args.output,
            transaction=args.transaction,
        )
    else:
        print(json.dumps(benchmark(args.contract, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
