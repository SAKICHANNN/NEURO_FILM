#!/usr/bin/env python3
"""Measure exact packaged Native Standard with replayable 24MP rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq
from src.eval.native_standard_replayable_rows import (
    render_native_standard_replayable_rows,
)
from src.film_physics.native_standard_package import (
    resolve_native_standard_libraries,
)
from src.film_physics.native_standard_runtime import NativeStandardRuntime


def _source_rows(start: int, count: int, height: int, width: int) -> np.ndarray:
    return np.ascontiguousarray(
        p8aq._source_rows(
            y0=start, y1=start + count, height=height, width=width
        ),
        dtype=np.float32,
    )


def _runtime() -> NativeStandardRuntime:
    package = json.loads(
        (ROOT / "configs/u6_p8az_native_standard_package_v1.json").read_text()
    )
    report = json.loads(
        (
            ROOT
            / "outputs/u6_p8b_artifact_only_cpu_consumer_v1/run_a/report.json"
        ).read_text()
    )
    artifact = report["artifact"]
    paths: dict[str, Path] = {}
    expected = {
        row["sha256"]: name for name, row in package["components"].items()
    }
    for dll in (ROOT / "outputs").rglob("*.dll"):
        digest = hashlib.sha256(dll.read_bytes()).hexdigest()
        if digest in expected:
            paths.setdefault(expected[digest], dll)
    resolved = resolve_native_standard_libraries(package, paths)
    return NativeStandardRuntime(
        package=package, artifact=artifact, resolved=resolved
    )


def _input_sha(height: int, width: int, tile_rows: int) -> str:
    digest = hashlib.sha256()
    for start in range(0, height, tile_rows):
        count = min(tile_rows, height - start)
        digest.update(memoryview(_source_rows(start, count, height, width)).cast("B"))
    return digest.hexdigest()


def worker(*, contract: Path, output: Path) -> None:
    frozen = json.loads(contract.read_text())
    scenario = frozen["scenario"]
    height, width = int(scenario["height"]), int(scenario["width"])
    runtime = _runtime()
    input_sha = _input_sha(height, width, int(runtime.policy["tile_rows"]))
    started = time.perf_counter()
    receipt = render_native_standard_replayable_rows(
        runtime,
        height=height,
        width=width,
        source_rows=lambda start, count: _source_rows(
            start, count, height, width
        ),
        expected_input_sha256=input_sha,
        output_sink=lambda _y0, _y1, _rows: None,
    )
    output.write_text(
        json.dumps(
            {"wall_seconds": time.perf_counter() - started, **receipt},
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _monitor(command: list[str], output: Path, timeout: float, interval: float) -> dict:
    process = subprocess.Popen(command, cwd=ROOT)
    root = psutil.Process(process.pid)
    peak = 0
    started = time.perf_counter()
    try:
        while process.poll() is None:
            if time.perf_counter() - started > timeout:
                raise TimeoutError("P4FR worker timeout")
            try:
                observed = [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                observed = []
            for item in observed:
                try:
                    peak = max(peak, sum(row.memory_info().rss for row in observed))
                    break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
            time.sleep(interval)
        if process.returncode != 0:
            raise RuntimeError(f"P4FR worker exit {process.returncode}")
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
                ],
                output,
                float(scenario["timeout_seconds"]),
                float(scenario["sample_interval_seconds"]),
            )
        )
    peaks = [row["peak_process_tree_rss_bytes"] for row in runs]
    walls = [row["worker"]["wall_seconds"] for row in runs]
    identities = {
        (row["worker"]["input_sha256"], row["worker"]["output_sha256"])
        for row in runs
    }
    checks = {
        "runs": len(runs) == int(scenario["runs"]),
        "identity": len(identities) == 1,
        "peak": max(peaks) <= int(gate["maximum_peak_process_tree_rss_bytes"]),
        "wall": max(walls) <= float(gate["maximum_worker_wall_seconds"]),
        "repeat": max(peaks) / min(peaks)
        <= float(gate["maximum_peak_repeat_ratio"]),
        "streamed": all(
            not row["worker"]["full_source_frame_retained"] for row in runs
        ),
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
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(contract=args.contract, output=args.output)
    else:
        print(json.dumps(benchmark(args.contract, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
