#!/usr/bin/env python3
"""Measure 24MP correct-domain cloud plus frozen Standard display."""

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
from src.eval.native_cloud_spatial_partition import (
    _build,
    _configure,
    render_physical_partition,
)
from src.eval.native_cloud_standard_display import (
    render_cloud_scan_with_standard_display,
)
from src.film_physics.native_cloud_scan_runtime_v2 import (
    WindowedNativeCloudScanRuntime,
)
from tests.test_u6_p4fc_opt_in_cloud_scan_runtime_v1 import _profile

P4FB = ROOT / "configs/u6_p4fb_native_cloud_spatial_partition_v1.json"


def worker(*, contract: Path, output: Path) -> None:
    contract = contract.resolve()
    output = output.resolve()
    frozen = json.loads(contract.read_text())
    scenario = frozen["scenario"]
    height, width = int(scenario["height"]), int(scenario["width"])
    tile_rows = int(scenario["tile_rows"])
    build = (output.parent / f"native-{output.stem}").resolve()
    dll = _build(ROOT, build, None)
    lib = _configure(dll)
    p4fb = json.loads(P4FB.read_text())
    p4fb["fixture"]["full_height"] = height
    p4fb["fixture"]["width"] = width
    component_sha = hashlib.sha256(
        (ROOT / "native/film_physics/nf_cloud_post_spatial_f32_v1.c").read_bytes()
    ).hexdigest()

    def provider(forward_rows, y0: int, count: int):
        return render_physical_partition(lib, p4fb, forward_rows, y0, count)

    cloud = WindowedNativeCloudScanRuntime(
        gaussian_library=dll,
        forward_scatter_profile=_profile(),
        physical_rows=provider,
        physical_component_sha256=component_sha,
        tile_rows=tile_rows,
    )
    standard = _runtime()
    input_sha = _input_sha(height, width, tile_rows)
    started = time.perf_counter()
    receipt = render_cloud_scan_with_standard_display(
        standard,
        cloud,
        height=height,
        width=width,
        source_rows=lambda start, count: _source_rows(start, count, height, width),
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
                raise TimeoutError("P4FZ worker timeout")
            try:
                observed = [root, *root.children(recursive=True)]
                peak = max(peak, sum(item.memory_info().rss for item in observed))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(interval)
        if process.returncode != 0:
            raise RuntimeError(f"P4FZ worker exit {process.returncode}")
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
        shutil.rmtree(output_dir / f"native-worker-{index + 1}", ignore_errors=True)
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
        "repeat": max(peaks) / min(peaks) <= float(gate["maximum_peak_repeat_ratio"]),
        "streamed": all(not row["worker"]["full_source_frame_retained"] for row in runs),
        "passes": all(row["worker"]["source_passes"] == int(gate["require_source_passes"]) for row in runs),
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
        "decision": frozen["decision_if_pass"] if all(checks.values()) else frozen["decision_if_fail"],
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
