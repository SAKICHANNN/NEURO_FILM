#!/usr/bin/env python3
"""Measure 24MP cloud rendering without a retained full source frame."""

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

from src.eval.native_cloud_spatial_partition import (
    _build,
    _configure,
    render_physical_partition,
)
from src.eval.native_msvc import sha256_file
from src.film_physics.native_cloud_scan_runtime_v2 import (
    WindowedNativeCloudScanRuntime,
)
from tests.test_u6_p4fc_opt_in_cloud_scan_runtime_v1 import (
    _profile,
)


def source_rows(start: int, count: int, width: int) -> np.ndarray:
    y = np.arange(start, start + count, dtype=np.uint64)[:, None, None]
    x = np.arange(width, dtype=np.uint64)[None, :, None]
    c = np.arange(3, dtype=np.uint64)[None, None, :]
    return np.ascontiguousarray(((y * 17 + x * 31 + c * 101 + 3) % 997) / 996.0, np.float32)


def worker(*, contract: Path, dll: Path, output: Path) -> None:
    frozen = json.loads(contract.read_text())
    scenario = frozen["scenario"]
    height = int(scenario["height"])
    width = int(scenario["width"])
    lib = _configure(dll)
    p4fb = json.loads(
        (ROOT / "configs/u6_p4fb_native_cloud_spatial_partition_v1.json").read_text()
    )
    p4fb["fixture"]["full_height"] = height
    p4fb["fixture"]["width"] = width

    def provider(get_rows, y0: int, count: int) -> np.ndarray:
        return render_physical_partition(lib, p4fb, get_rows, y0, count)

    runtime = WindowedNativeCloudScanRuntime(
        gaussian_library=dll,
        forward_scatter_profile=_profile(),
        physical_rows=provider,
        physical_component_sha256=sha256_file(
            ROOT / "native/film_physics/nf_cloud_post_spatial_f32_v1.c"
        ),
        tile_rows=int(scenario["tile_rows"]),
    )
    started = time.perf_counter()
    receipt = runtime.render_rows_to_sink(
        height=height,
        width=width,
        source_rows=lambda start, count: source_rows(start, count, width),
        expected_input_sha256=scenario["expected_input_sha256"],
        output_sink=lambda _y0, _y1, _rows: None,
    )
    result = {"wall_seconds": time.perf_counter() - started, **receipt}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")))


def monitor(command: list[str], output: Path, timeout: float, interval: float) -> dict:
    process = subprocess.Popen(command, cwd=ROOT)
    root = psutil.Process(process.pid)
    peak = 0
    started = time.perf_counter()
    try:
        while process.poll() is None:
            if time.perf_counter() - started > timeout:
                raise TimeoutError("P4FL worker timeout")
            try:
                observed = [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                observed = []
            total = 0
            for item in observed:
                try:
                    total += item.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            peak = max(peak, total)
            time.sleep(interval)
        if process.returncode != 0:
            raise RuntimeError(f"P4FL worker exit {process.returncode}")
        return {"peak_process_tree_rss_bytes": peak, "worker": json.loads(output.read_text())}
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def benchmark(contract: Path, output_dir: Path) -> dict:
    contract = contract.resolve()
    output_dir = output_dir.resolve()
    frozen = json.loads(contract.read_text())
    parent = ROOT / frozen["parent"]["path"]
    parent_payload = json.loads(parent.read_text())
    if (
        sha256_file(parent) != frozen["parent"]["sha256"]
        or parent_payload["decision"] != frozen["parent"]["required_decision"]
    ):
        raise RuntimeError("P4FL parent drift")
    dll = _build(ROOT, output_dir / "build", None)
    scenario = frozen["scenario"]
    runs = []
    for index in range(int(scenario["runs"])):
        path = output_dir / f"worker-{index + 1}.json"
        runs.append(
            monitor(
                [sys.executable, str(Path(__file__).resolve()), "--worker", "--contract", str(contract), "--dll", str(dll), "--output", str(path)],
                path,
                float(scenario["timeout_seconds"]),
                float(scenario["sample_interval_seconds"]),
            )
        )
    peaks = [row["peak_process_tree_rss_bytes"] for row in runs]
    walls = [row["worker"]["wall_seconds"] for row in runs]
    gate = frozen["gates"]
    checks = {
        "runs": len(runs) == int(scenario["runs"]),
        "identity": all(
            row["worker"]["input_sha256"] == scenario["expected_input_sha256"]
            and row["worker"]["output_sha256"] == scenario["expected_output_sha256"]
            for row in runs
        ),
        "peak": max(peaks) <= int(gate["maximum_peak_process_tree_rss_bytes"]),
        "wall": max(walls) <= float(gate["maximum_worker_wall_seconds"]),
        "repeat": max(peaks) / min(peaks) <= float(gate["maximum_peak_repeat_ratio"]),
        "streamed": all(
            not row["worker"]["full_source_frame_retained"]
            and not row["worker"]["full_forward_frame_retained"]
            for row in runs
        ),
    }
    stable = {
        "contract_sha256": sha256_file(contract),
        "runs": runs,
        "metrics": {
            "peak_process_tree_rss_bytes": peaks,
            "worker_wall_seconds": walls,
            "peak_repeat_ratio": max(peaks) / min(peaks),
        },
        "gates": checks,
        "decision": frozen["decision_if_pass"] if all(checks.values()) else frozen["decision_if_fail"],
        "claim_ceiling": frozen["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4fl_replayable_cloud_source_24mp.v1",
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
    parser.add_argument("--dll", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(contract=args.contract, dll=args.dll, output=args.output)
    else:
        print(json.dumps(benchmark(args.contract, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
