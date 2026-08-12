#!/usr/bin/env python3
"""Measure 24MP file-backed scene-linear NPY through verified PNG16."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_u6_p4fr_native_standard_replayable_24mp import (
    _runtime,
    _source_rows,
)
from src.film_physics.native_standard_output import (
    verify_native_standard_png16,
)
from src.film_physics.native_standard_replayable_file import (
    render_native_standard_scene_linear_npy_to_png16,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_input(path: Path, height: int, width: int, tile_rows: int = 32) -> dict:
    mapped = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(height, width, 3),
        fortran_order=False,
    )
    pixel_digest = hashlib.sha256()
    for start in range(0, height, tile_rows):
        count = min(tile_rows, height - start)
        rows = _source_rows(start, count, height, width)
        mapped[start : start + count] = rows
        pixel_digest.update(memoryview(rows).cast("B"))
    mapped.flush()
    del mapped
    return {
        "file_sha256": _sha(path),
        "pixel_sha256": pixel_digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def worker(*, contract: Path, source: Path, source_lock: Path, output: Path, transaction: Path) -> None:
    json.loads(contract.read_text())
    locked = json.loads(source_lock.read_text())
    runtime = _runtime()
    transaction.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        result = render_native_standard_scene_linear_npy_to_png16(
            runtime,
            input_path=source,
            expected_input_file_sha256=locked["file_sha256"],
            expected_input_pixel_sha256=locked["pixel_sha256"],
            raw_output_path=transaction / "render.f32",
            raw_report_path=transaction / "render.raw.json",
            png_output_path=transaction / "render.png",
            png_report_path=transaction / "render.png.json",
        )
        verified = verify_native_standard_png16(
            report_path=transaction / "render.png.json",
            expected_report_sha256=result["png_report_sha256"],
            expected_delivery_id=result["png_delivery_id"],
        )
        receipt = {
            "input_file_sha256": locked["file_sha256"],
            "input_pixel_sha256": locked["pixel_sha256"],
            "png_output_sha256": result["png_output_sha256"],
            "png_verification_output_sha256": verified["output_sha256"],
            "wall_seconds": time.perf_counter() - started,
            "restart_verified": True,
        }
    finally:
        shutil.rmtree(transaction, ignore_errors=False)
    receipt["transaction_cleanup_pass"] = not transaction.exists()
    output.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


def _monitor(command: list[str], output: Path, timeout: float, interval: float) -> dict:
    process = subprocess.Popen(command, cwd=ROOT)
    root = psutil.Process(process.pid)
    peak = 0
    started = time.perf_counter()
    try:
        while process.poll() is None:
            if time.perf_counter() - started > timeout:
                raise TimeoutError("P4FX worker timeout")
            try:
                peak = max(
                    peak,
                    sum(
                        item.memory_info().rss
                        for item in [root, *root.children(recursive=True)]
                    ),
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(interval)
        if process.returncode != 0:
            raise RuntimeError(f"P4FX worker exit {process.returncode}")
        return {"peak_process_tree_rss_bytes": peak, "worker": json.loads(output.read_text())}
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def benchmark(contract: Path, output_dir: Path) -> dict:
    frozen = json.loads(contract.read_text())
    scenario, gate = frozen["scenario"], frozen["gates"]
    output_dir.mkdir(parents=True, exist_ok=True)
    source = output_dir / "scene.npy"
    source_lock_path = output_dir / "scene.lock.json"
    locked = prepare_input(source, int(scenario["height"]), int(scenario["width"]))
    source_lock_path.write_text(json.dumps(locked, sort_keys=True, separators=(",", ":")))
    runs = []
    try:
        for index in range(int(scenario["runs"])):
            output = output_dir / f"worker-{index + 1}.json"
            transaction = output_dir / f"transaction-{index + 1}"
            runs.append(
                _monitor(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", "--contract", str(contract.resolve()), "--source", str(source), "--source-lock", str(source_lock_path), "--output", str(output), "--transaction", str(transaction)],
                    output,
                    float(scenario["timeout_seconds"]),
                    float(scenario["sample_interval_seconds"]),
                )
            )
    finally:
        source.unlink(missing_ok=True)
        source_lock_path.unlink(missing_ok=True)
    peaks = [row["peak_process_tree_rss_bytes"] for row in runs]
    walls = [row["worker"]["wall_seconds"] for row in runs]
    checks = {
        "identity": len({row["worker"]["png_output_sha256"] for row in runs}) == 1,
        "peak": max(peaks) <= int(gate["maximum_peak_process_tree_rss_bytes"]),
        "wall": max(walls) <= float(gate["maximum_worker_wall_seconds"]),
        "repeat": max(peaks) / min(peaks) <= float(gate["maximum_peak_repeat_ratio"]),
        "verified": all(row["worker"]["restart_verified"] for row in runs),
        "cleanup": not source.exists() and not source_lock_path.exists() and all(row["worker"]["transaction_cleanup_pass"] for row in runs),
    }
    stable = {
        "contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
        "source_lock": locked,
        "runs": runs,
        "metrics": {"peak_process_tree_rss_bytes": peaks, "worker_wall_seconds": walls, "peak_repeat_ratio": max(peaks) / min(peaks)},
        "gates": checks,
        "decision": frozen["decision_if_pass"] if all(checks.values()) else frozen["decision_if_fail"],
        "claim_ceiling": frozen["claim_ceiling"],
    }
    return {
        "schema": frozen["schema"].replace("_contract", ""),
        "automatic_pass": all(checks.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--transaction", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(contract=args.contract, source=args.source, source_lock=args.source_lock, output=args.output, transaction=args.transaction)
    else:
        print(json.dumps(benchmark(args.contract, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
