#!/usr/bin/env python3
"""Evaluate P4HI in two monitored fresh worker processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.bounded_paired_scanner_photographic_runtime import (
    evaluate,
    load_contract,
)


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _worker(contract_path: Path, output_path: Path, contact_path: Path) -> None:
    contract = load_contract(contract_path)
    started = time.perf_counter()
    result = evaluate(contract, root=ROOT, contact_sheet_path=contact_path)
    _write_json(
        output_path,
        {
            "scientific_result": result,
            "worker_wall_seconds": time.perf_counter() - started,
            "contact_sheet_sha256": _sha256(contact_path),
        },
    )


def _monitor(command: list[str], output_path: Path, *, timeout: float, interval: float) -> dict[str, Any]:
    process = subprocess.Popen(command, cwd=ROOT)
    root_process = psutil.Process(process.pid)
    peak = 0
    started = time.perf_counter()
    try:
        while process.poll() is None:
            if time.perf_counter() - started > timeout:
                raise TimeoutError("P4HI worker timeout")
            try:
                observed = [root_process, *root_process.children(recursive=True)]
            except psutil.NoSuchProcess:
                observed = []
            current = 0
            for item in observed:
                try:
                    current += int(item.memory_info().rss)
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
            peak = max(peak, current)
            time.sleep(interval)
        if process.returncode != 0:
            raise RuntimeError(f"P4HI worker exited {process.returncode}")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return {
            "peak_process_tree_rss_bytes": peak,
            "worker_wall_seconds": payload["worker_wall_seconds"],
            "worker_report_sha256": _sha256(output_path),
            "contact_sheet_sha256": payload["contact_sheet_sha256"],
            "scientific_result": payload["scientific_result"],
        }
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def run(contract_path: Path, output_dir: Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    measurement = contract["measurement"]
    runs: list[dict[str, Any]] = []
    for index in range(int(measurement["formal_processes"])):
        worker_output = output_dir / f"worker-{index + 1}.json"
        contact_output = output_dir / f"contact-{index + 1}.png"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--contract",
            str(contract_path),
            "--worker-output",
            str(worker_output),
            "--contact",
            str(contact_output),
        ]
        runs.append(
            _monitor(
                command,
                worker_output,
                timeout=float(measurement["worker_timeout_seconds"]),
                interval=float(measurement["sample_interval_seconds"]),
            )
        )
    scientific = runs[0]["scientific_result"]
    scientific_exact = all(run["scientific_result"] == scientific for run in runs[1:])
    contact_exact = len({run["contact_sheet_sha256"] for run in runs}) == 1
    gates = contract["automatic_gates"]
    bounded = scientific["stable"]["bounded_execution"]
    checks = {
        "p4he_photographic": bool(scientific["stable"]["automatic_pass"]),
        "repeat_scientific": scientific_exact,
        "repeat_contact": contact_exact,
        "rss": max(run["peak_process_tree_rss_bytes"] for run in runs)
        <= int(gates["maximum_peak_process_tree_rss_bytes"]),
        "wall": max(run["worker_wall_seconds"] for run in runs)
        <= float(gates["maximum_worker_wall_seconds"]),
        "temporary_budget": bounded["maximum_peak_live_temporary_bytes"]
        <= int(gates["maximum_peak_live_temporary_bytes"]),
        "no_full_frame_intermediate": bounded[
            "full_frame_intermediate_count_excluding_input_output"
        ]
        == 0,
        "no_full_frame_visual_accumulation": bounded[
            "full_frame_visual_rows_retained"
        ]
        is False,
    }
    stable = {
        "contract_sha256": hashlib.sha256(_json_bytes(contract)).hexdigest(),
        "scientific_result": scientific,
        "scientific_repeat_exact": scientific_exact,
        "contact_sheet_repeat_exact": contact_exact,
        "contact_sheet_sha256": runs[0]["contact_sheet_sha256"],
        "gates": checks,
        "automatic_pass": all(checks.values()),
        "decision": contract["decision_if_pass"]
        if all(checks.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": contract["schema"].replace("contract", "result"),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_json_bytes(stable)).hexdigest(),
        "resource_measurements_excluded_from_stable_evidence_id": [
            {
                key: value
                for key, value in run.items()
                if key != "scientific_result"
            }
            for run in runs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--contact", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.worker_output is None or args.contact is None:
            parser.error("worker output and contact are required")
        _worker(args.contract, args.worker_output, args.contact)
        return
    if args.output_dir is None or args.report is None:
        parser.error("output dir and report are required")
    result = run(args.contract, args.output_dir)
    _write_json(args.report, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
