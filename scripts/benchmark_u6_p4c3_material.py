#!/usr/bin/env python
"""Measure streamed U6.P4C3 material fields in isolated child processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from src.film_physics import (  # noqa: E402
    CompoundPoissonProfile,
    render_compound_poisson_region,
)


SCHEMA = "neuro_film.u6_p4c3_streamed_material_benchmark_contract.v1"


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


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P4C3 contract")
    return payload


def _profile(row: dict[str, Any], seed: int) -> CompoundPoissonProfile:
    return CompoundPoissonProfile(
        family=str(row["family"]),
        poisson_rate=float(row["poisson_rate"]),
        correlation_sigma_pixels=float(row["correlation_sigma_pixels"]),
        baseline=float(row["baseline"]),
        scale=float(row["scale"]),
        seed=seed,
    )


def worker(contract: dict[str, Any], chunk_rows: int) -> dict[str, Any]:
    height, width = (int(value) for value in contract["shape"])
    started = perf_counter()
    layer_rows = []
    for layer, row in enumerate(contract["compiled_profiles"]):
        profile = _profile(row, int(contract["seed"]) + layer)
        digest = hashlib.sha256()
        count = 0
        total = 0.0
        total_squares = 0.0
        minimum = math.inf
        maximum = -math.inf
        violations = 0
        layer_started = perf_counter()
        for y0 in range(0, height, chunk_rows):
            y1 = min(height, y0 + chunk_rows)
            values = render_compound_poisson_region(
                profile,
                (height, width),
                origin_yx=(y0, 0),
                shape=(y1 - y0, width),
            )
            digest.update(memoryview(np.ascontiguousarray(values)).cast("B"))
            block = values.astype(np.float64)
            count += int(block.size)
            total += float(np.sum(block, dtype=np.float64))
            total_squares += float(np.sum(np.square(block), dtype=np.float64))
            minimum = min(minimum, float(np.min(block)))
            maximum = max(maximum, float(np.max(block)))
            if profile.family == "density-shot":
                representable_base = float(np.float32(profile.baseline))
                violations += int(
                    np.count_nonzero(block < representable_base)
                )
            else:
                representable_clear_base = float(
                    np.float32(math.exp(-profile.baseline))
                )
                violations += int(np.count_nonzero(block > representable_clear_base))
        mean = total / count
        variance = total_squares / count - mean * mean
        layer_rows.append(
            {
                "layer": layer,
                "family": profile.family,
                "output_sha256": digest.hexdigest(),
                "mean": mean,
                "variance": variance,
                "minimum": minimum,
                "maximum": maximum,
                "finite_domain_valid": bool(
                    math.isfinite(minimum)
                    and math.isfinite(maximum)
                    and minimum > 0.0
                    and (
                        profile.family == "density-shot"
                        or maximum <= 1.0
                    )
                ),
                "brightening_violation_count": violations,
                "apply_seconds": perf_counter() - layer_started,
                "logical_output_bytes": count * 4,
            }
        )
    return {
        "shape": [height, width],
        "chunk_rows": chunk_rows,
        "layers": layer_rows,
        "apply_seconds": perf_counter() - started,
    }


def _launch(
    contract_path: Path,
    chunk_rows: int,
    output: Path,
    *,
    interval: float,
    timeout: float,
) -> dict[str, Any]:
    output.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--contract",
        str(contract_path),
        "--worker-chunk-rows",
        str(chunk_rows),
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


def parent(
    contract_path: Path, output_dir: Path
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    runs = []
    for chunk_rows in (int(value) for value in contract["chunk_rows"]):
        result_path = output_dir / f"chunk_{chunk_rows}.json"
        monitor = _launch(
            contract_path,
            chunk_rows,
            result_path,
            interval=float(contract["rss_sample_interval_seconds"]),
            timeout=float(contract["worker_timeout_seconds"]),
        )
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else None
        )
        runs.append({"monitor": monitor, "result": result})
        if monitor["exit_code"] != 0 or result is None:
            raise RuntimeError(
                f"benchmark worker failed for chunk {chunk_rows}: "
                f"{monitor['stderr']}"
            )
    hash_rows = [
        [layer["output_sha256"] for layer in run["result"]["layers"]]
        for run in runs
    ]
    gates = contract["automatic_gates"]
    decisions = {
        "hashes": len({tuple(row) for row in hash_rows}) == 1,
        "domain": all(
            layer["finite_domain_valid"]
            for run in runs
            for layer in run["result"]["layers"]
        ),
        "brightening": sum(
            layer["brightening_violation_count"]
            for run in runs
            for layer in run["result"]["layers"]
        )
        == int(gates["brightening_violation_count"]),
        "runtime": max(run["result"]["apply_seconds"] for run in runs)
        <= float(gates["worker_apply_seconds_max"]),
        "memory": max(
            run["monitor"]["peak_process_tree_rss_bytes"] for run in runs
        )
        <= int(gates["peak_process_tree_rss_bytes_max"]),
        "timeout": all(
            run["monitor"]["timed_out"] is bool(gates["worker_timeout"])
            for run in runs
        ),
    }
    stable_core = {
        "shape": contract["shape"],
        "chunk_rows": contract["chunk_rows"],
        "output_sha256_by_layer": hash_rows[0],
        "finite_domain_valid": decisions["domain"],
        "brightening_violation_count": sum(
            layer["brightening_violation_count"]
            for run in runs
            for layer in run["result"]["layers"]
        ),
        "decisions": decisions,
    }
    stable_id = hashlib.sha256(
        json.dumps(
            stable_core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    report = {
        "schema": "neuro_film.u6_p4c3_streamed_material_benchmark_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "runs": runs,
        **stable_core,
        "automatic_pass": all(decisions.values()),
        "stable_evidence_id": stable_id,
        "branch": contract["branch_rule"][
            "automatic_pass" if all(decisions.values()) else "automatic_fail"
        ],
    }
    _atomic_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/u6_p4c3_streamed_material_benchmark_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-chunk-rows", type=int)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker_chunk_rows is not None:
        if args.worker_output is None:
            raise ValueError("--worker-output is required in worker mode")
        _atomic_json(
            args.worker_output,
            worker(load_contract(args.contract), args.worker_chunk_rows),
        )
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required in parent mode")
    report = parent(args.contract, args.output_dir)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    for run in report["runs"]:
        print(
            f"chunk={run['result']['chunk_rows']} "
            f"apply={run['result']['apply_seconds']:.3f}s "
            f"peak={run['monitor']['peak_process_tree_rss_bytes'] / 2**30:.3f}GiB"
        )


if __name__ == "__main__":
    main()
