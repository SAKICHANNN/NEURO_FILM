"""Run U6.P8BS dual-compiler conformance and 12MP resource evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_field_conformance import (
    canonical_bytes,
    evaluate_conformance,
    profile_from_contract,
)
from src.film_physics.native_thomas_field import (
    load_native_thomas_field_library,
    render_native_thomas_field,
)


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("P8BS contract must be an object")
    return payload


def _worker(contract_path: Path, dll: Path, result_path: Path) -> None:
    contract = _load_contract(contract_path)
    profile = profile_from_contract(contract)
    shape = tuple(int(value) for value in contract["performance"]["shape"])
    library = load_native_thomas_field_library(dll)
    started = time.perf_counter()
    output, raw_mean, workspace_bytes = render_native_thomas_field(
        library, profile, shape
    )
    elapsed = time.perf_counter() - started
    result = {
        "schema": "neuro_film.u6_p8bs_native_thomas_field_worker.v1",
        "shape": list(shape),
        "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        "raw_mean": raw_mean,
        "projected_mean": float(np.mean(output, dtype=np.float64)),
        "variance": float(np.var(output, dtype=np.float64)),
        "workspace_bytes": workspace_bytes,
        "output_bytes": output.nbytes,
        "wall_seconds": elapsed,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(canonical_bytes(result))


def _run_monitored_worker(
    contract_path: Path,
    dll: Path,
    result_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--contract",
        str(contract_path),
        "--dll",
        str(dll),
        "--result",
        str(result_path),
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    root_process = psutil.Process(process.pid)
    peak_rss = 0
    observed: set[int] = set()
    started = time.perf_counter()
    while process.poll() is None:
        if time.perf_counter() - started > timeout_seconds:
            for child in root_process.children(recursive=True):
                child.kill()
            root_process.kill()
            raise TimeoutError("P8BS worker exceeded timeout")
        try:
            tree = [root_process, *root_process.children(recursive=True)]
            total = 0
            for item in tree:
                try:
                    observed.add(item.pid)
                    total += item.memory_info().rss
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
            peak_rss = max(peak_rss, total)
        except psutil.NoSuchProcess:
            pass
        time.sleep(0.01)
    stdout, stderr = process.communicate(timeout=10)
    if process.returncode != 0 or not result_path.is_file():
        raise RuntimeError(
            "P8BS worker failed:\n"
            + stdout.decode(errors="replace")
            + stderr.decode(errors="replace")
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    survivors = [pid for pid in observed if psutil.pid_exists(pid)]
    return {
        "worker": result,
        "peak_process_tree_rss_bytes": peak_rss,
        "stderr_empty": not bool(stderr),
        "surviving_process_count": len(survivors),
    }


def _stable_conformance(report: dict[str, Any]) -> dict[str, Any]:
    stable = json.loads(json.dumps(report))
    for toolchain in stable["toolchains"].values():
        toolchain.pop("dll_path", None)
        toolchain.pop("compiler_output", None)
    return stable


def evaluate(
    contract_path: Path,
    output_dir: Path,
    clang: Path,
) -> dict[str, Any]:
    contract = _load_contract(contract_path)
    build_dir = output_dir / "build"
    conformance = evaluate_conformance(
        ROOT, contract, output_dir=build_dir, clang=clang
    )
    msvc_dll = Path(conformance["toolchains"]["msvc"]["dll_path"])
    runs = [
        _run_monitored_worker(
            contract_path,
            msvc_dll,
            output_dir / f"worker_{index}.json",
            timeout_seconds=30.0,
        )
        for index in (1, 2)
    ]
    walls = [float(run["worker"]["wall_seconds"]) for run in runs]
    peaks = [int(run["peak_process_tree_rss_bytes"]) for run in runs]
    output_hashes = [str(run["worker"]["output_sha256"]) for run in runs]
    performance = contract["performance"]
    wall_ratio = max(walls) / min(walls)
    rss_ratio = max(peaks) / min(peaks)
    gate_results = {
        "conformance": bool(conformance["automatic_pass"]),
        "fresh_output_exact": len(set(output_hashes)) == 1,
        "maximum_wall": max(walls) <= float(performance["maximum_wall_seconds"]),
        "maximum_rss": max(peaks)
        <= int(performance["maximum_process_tree_rss_bytes"]),
        "wall_repeat": wall_ratio
        <= float(performance["maximum_repeat_wall_ratio"]),
        "rss_repeat": rss_ratio <= float(performance["maximum_repeat_rss_ratio"]),
        "worker_cleanup": all(
            run["stderr_empty"] and run["surviving_process_count"] == 0
            for run in runs
        ),
    }
    stable = {
        "schema": "neuro_film.u6_p8bs_native_thomas_field_result.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "conformance": _stable_conformance(conformance),
        "performance": {
            "runs": runs,
            "output_sha256": output_hashes[0],
            "maximum_wall_seconds": max(walls),
            "maximum_peak_process_tree_rss_bytes": max(peaks),
            "wall_repeat_ratio": wall_ratio,
            "rss_repeat_ratio": rss_ratio,
        },
        "gate_results": gate_results,
        "automatic_pass": all(gate_results.values()),
        "decision": (
            contract["decision_if_pass"]
            if all(gate_results.values())
            else contract["decision_if_fail"]
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    identity_payload = {
        "experiment_id": stable["experiment_id"],
        "contract_sha256": stable["contract_sha256"],
        "conformance_id": stable["conformance"]["stable_evidence_id"],
        "output_sha256": stable["performance"]["output_sha256"],
        "gate_results": gate_results,
        "decision": stable["decision"],
    }
    return {
        **stable,
        "stable_evidence_id": hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p8bs_native_thomas_field_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/eval/u6_p8bs_native_thomas_field_v1",
    )
    parser.add_argument(
        "--clang",
        type=Path,
        default=(
            ROOT
            / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"
        ),
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dll", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.dll is None or args.result is None:
            parser.error("--worker requires --dll and --result")
        _worker(args.contract, args.dll, args.result)
        return
    report = evaluate(args.contract, args.output_dir, args.clang)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_bytes(canonical_bytes(report))
    print(json.dumps({
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "automatic_pass": report["automatic_pass"],
        "decision": report["decision"],
        "stable_evidence_id": report["stable_evidence_id"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
