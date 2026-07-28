#!/usr/bin/env python
"""Measure the U6.P6C float32 scanner compiler in isolated workers."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from src.eval.physical_scanner_profile import _profile  # noqa: E402
from src.eval.physical_scanner_standard import (  # noqa: E402
    array_sha256,
    deterministic_gradient,
    evaluate_standard_parity,
)
from src.film_physics import (  # noqa: E402
    apply_scanner_profile_standard_row_tiled,
    compile_scanner_standard_context,
)


SCHEMA = "neuro_film.u6_p6c_scanner_standard_benchmark_contract.v1"


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
        raise ValueError("unsupported U6.P6C contract")
    return payload


def worker(
    contract: dict[str, Any],
    p6a_contract: dict[str, Any],
    tile_rows: int,
) -> dict[str, Any]:
    shape = tuple(int(value) for value in contract["benchmark"]["shape"])
    values = deterministic_gradient(shape)
    profile = _profile(
        p6a_contract["profiles"][contract["candidate"]["profile"]]
    )
    context = compile_scanner_standard_context(values, profile)
    started = perf_counter()
    output = apply_scanner_profile_standard_row_tiled(
        values,
        profile,
        pixel_pitch_um=1.0,
        context=context,
        tile_rows=tile_rows,
    )
    elapsed = perf_counter() - started
    return {
        "shape": list(shape),
        "tile_rows": tile_rows,
        "context_id": context.context_id,
        "output_sha256": array_sha256(output),
        "output_dtype": output.dtype.name,
        "minimum": float(np.min(output)),
        "maximum": float(np.max(output)),
        "finite_bounded": bool(
            np.all(np.isfinite(output))
            and np.all(output >= 0.0)
            and np.all(output <= 1.0)
        ),
        "apply_seconds": elapsed,
        "logical_input_output_bytes": int(values.nbytes + output.nbytes),
    }


def _launch(
    contract_path: Path,
    p6a_contract_path: Path,
    tile_rows: int,
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
        "--p6a-contract",
        str(p6a_contract_path),
        "--worker-tile-rows",
        str(tile_rows),
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
    contract_path: Path,
    p6a_contract_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    p6a_contract = json.loads(p6a_contract_path.read_text(encoding="utf-8"))
    parity = evaluate_standard_parity(contract, p6a_contract)
    runs = []
    benchmark = contract["benchmark"]
    for tile_rows in benchmark["row_partitions"]:
        result_path = output_dir / f"tile_{tile_rows}.json"
        monitor = _launch(
            contract_path,
            p6a_contract_path,
            int(tile_rows),
            result_path,
            interval=float(benchmark["rss_sample_interval_seconds"]),
            timeout=float(benchmark["worker_timeout_seconds"]),
        )
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else None
        )
        runs.append({"monitor": monitor, "result": result})
        if monitor["exit_code"] != 0 or result is None:
            raise RuntimeError(
                f"scanner worker failed for tile {tile_rows}: "
                f"{monitor['stderr']}"
            )
    gates = contract["automatic_gates"]
    hashes = [run["result"]["output_sha256"] for run in runs]
    decisions = {
        "parity": parity["maximum_abs_vs_float64"]
        <= float(gates["parity_max_abs"]),
        "parity_partitions": parity["partitions_exact"],
        "hashes": len(set(hashes)) == 1,
        "domain": all(run["result"]["finite_bounded"] for run in runs),
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
        "shape": benchmark["shape"],
        "row_partitions": benchmark["row_partitions"],
        "parity": parity,
        "output_sha256": hashes[0],
        "finite_bounded": decisions["domain"],
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
        "schema": "neuro_film.u6_p6c_scanner_standard_benchmark_report.v1",
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
        default=ROOT / "configs" / "u6_p6c_scanner_standard_benchmark_v1.json",
    )
    parser.add_argument(
        "--p6a-contract",
        type=Path,
        default=ROOT / "configs" / "u6_p6a_scanner_profile_boundary_v1.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-tile-rows", type=int)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    p6a_contract = json.loads(args.p6a_contract.read_text(encoding="utf-8"))
    if args.worker_tile_rows is not None:
        if args.worker_output is None:
            raise ValueError("--worker-output is required in worker mode")
        _atomic_json(
            args.worker_output,
            worker(contract, p6a_contract, args.worker_tile_rows),
        )
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required in parent mode")
    report = parent(args.contract, args.p6a_contract, args.output_dir)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")


if __name__ == "__main__":
    main()
