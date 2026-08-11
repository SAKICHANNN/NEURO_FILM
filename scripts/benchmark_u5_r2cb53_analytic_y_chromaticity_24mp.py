"""Fresh-process 24MP resource benchmark for the unchanged CB50-CB52 core."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.analytic_y_chromaticity_transport import (
    select_analytic_y_chromaticity_candidate,
)
from src.eval.fujifilm_characteristic_photographic import _compiled_curve

SCHEMA = "neuro_film.u5_r2cb53_analytic_y_chromaticity_24mp_resources_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb53_analytic_y_chromaticity_24mp_resources_report.v1"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_array(value: np.ndarray) -> str:
    array = value if value.flags.c_contiguous else np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_contract(config: dict) -> None:
    if config.get("schema") != SCHEMA or config.get("experiment_id") != "U5.R2CB53":
        raise ValueError("CB53 contract structure drift")
    parents = config["parents"]
    for path_key, sha_key in (
        ("cb52_decision_path", "cb52_decision_sha256"),
        ("operator_path", "operator_sha256"),
        ("cb11_contract_path", "cb11_contract_sha256"),
        ("cb6_contract_path", "cb6_contract_sha256"),
    ):
        if _hash_file(ROOT / parents[path_key]) != parents[sha_key]:
            raise ValueError(f"CB53 parent drift: {path_key}")
    decision = json.loads((ROOT / parents["cb52_decision_path"]).read_text(encoding="utf-8"))
    if decision.get("decision") != parents["cb52_required_decision"]:
        raise ValueError("CB53 prerequisite decision drift")


def generate_fields(shape: tuple[int, int, int], *, row_chunk: int) -> tuple[np.ndarray, np.ndarray]:
    height, width, channels = shape
    if channels != 3 or height < 2 or width < 2 or row_chunk <= 0:
        raise ValueError("CB53 workload geometry is invalid")
    source = np.empty(shape, dtype=np.float32)
    target = np.empty(shape, dtype=np.float32)
    x = np.arange(width, dtype=np.float32) / np.float32(width - 1)
    for y0 in range(0, height, row_chunk):
        y1 = min(height, y0 + row_chunk)
        y = np.arange(y0, y1, dtype=np.float32)[:, None] / np.float32(height - 1)
        wave = np.sin(np.float32(13.0) * x[None, :] + np.float32(9.0) * y)
        block = source[y0:y1]
        block[..., 0] = np.float32(0.08) + np.float32(0.64) * x + np.float32(0.08) * y + np.float32(0.015) * wave
        block[..., 1] = np.float32(0.06) + np.float32(0.22) * x + np.float32(0.48) * y - np.float32(0.012) * wave
        block[..., 2] = np.float32(0.04) + np.float32(0.30) * x + np.float32(0.30) * y + np.float32(0.009) * wave
        block += np.float32(0.15)
        np.clip(block, np.float32(0.002), np.float32(0.94), out=block)
        target_block = target[y0:y1]
        target_block[..., 0] = np.float32(0.88) * block[..., 0] + np.float32(0.12) * block[..., 1]
        target_block[..., 1] = np.float32(0.88) * block[..., 1] + np.float32(0.12) * block[..., 2]
        target_block[..., 2] = np.float32(0.88) * block[..., 2] + np.float32(0.12) * block[..., 0]
    return source, target


def worker(config: dict) -> dict:
    _validate_contract(config)
    workload = config["workload"]
    started = perf_counter()
    source, target = generate_fields(
        tuple(int(item) for item in workload["shape"]),
        row_chunk=int(workload["generation_row_chunk"]),
    )
    generation_seconds = perf_counter() - started
    cb11 = json.loads((ROOT / config["parents"]["cb11_contract_path"]).read_text(encoding="utf-8"))
    cb6 = json.loads((ROOT / config["parents"]["cb6_contract_path"]).read_text(encoding="utf-8"))
    operator = config["operator"]
    execution_started = perf_counter()
    candidate, scale, luma_error, facts = select_analytic_y_chromaticity_candidate(
        source,
        target,
        curve=_compiled_curve(cb6),
        strength=float(cb11["operator"]["nominal_strength"]),
        boundary_epsilon=float(cb11["operator"]["boundary_epsilon"]),
        dose_grid=list(operator["dose_grid"]),
        maximum_gradient_ratio=float(operator["maximum_gradient_ratio"]),
        maximum_lstar_inversion_fraction=float(operator["maximum_lstar_inversion_fraction"]),
        lstar_order_epsilon=float(operator["lstar_order_epsilon"]),
    )
    execution_seconds = perf_counter() - execution_started
    finite_bounded = bool(
        np.isfinite(candidate).all()
        and candidate.min() >= 0.0
        and candidate.max() <= 1.0
        and np.isfinite(scale).all()
        and np.isfinite(luma_error).all()
    )
    return {
        "source_sha256": _hash_array(source),
        "target_sha256": _hash_array(target),
        "candidate_sha256": _hash_array(candidate),
        "scale_sha256": _hash_array(scale),
        "selected_facts": facts,
        "finite_and_bounded": finite_bounded,
        "maximum_luminance_error": float(np.max(np.abs(luma_error))),
        "timings_seconds": {
            "generation": generation_seconds,
            "operator": execution_seconds,
            "worker_total": perf_counter() - started,
        },
    }


def _kill_tree(root: psutil.Process) -> None:
    children = root.children(recursive=True)
    for process in reversed(children):
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        root.kill()
    except psutil.NoSuchProcess:
        pass


def _launch(
    config_path: Path,
    result_path: Path,
    *,
    interval: float,
    timeout: float,
    maximum_rss: int,
) -> dict:
    result_path.unlink(missing_ok=True)
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "--worker-output", str(result_path)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    root = psutil.Process(process.pid)
    peak = 0
    pids = {process.pid}
    started = perf_counter()
    timed_out = False
    resource_limit_exceeded = False
    while process.poll() is None:
        try:
            tree = [root, *root.children(recursive=True)]
            pids.update(item.pid for item in tree)
            peak = max(peak, sum(item.memory_info().rss for item in tree if item.is_running()))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        if peak > maximum_rss:
            resource_limit_exceeded = True
            _kill_tree(root)
            break
        if perf_counter() - started > timeout:
            timed_out = True
            _kill_tree(root)
            break
        sleep(interval)
    stdout, stderr = process.communicate()
    return {
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "resource_limit_exceeded": resource_limit_exceeded,
        "wall_seconds": perf_counter() - started,
        "peak_process_tree_rss_bytes": peak,
        "stdout": stdout,
        "stderr": stderr,
        "result_exists": result_path.exists(),
        "temporary_exists": result_path.with_name(result_path.name + ".tmp").exists(),
        "orphan_count": sum(int(psutil.pid_exists(pid)) for pid in pids),
    }


def run_parent(config_path: Path, work_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_contract(config)
    workload = config["workload"]
    runs = []
    for index in range(int(workload["runs"])):
        result_path = work_dir / f"worker_{index + 1}.json"
        monitor = _launch(
            config_path.resolve(),
            result_path.resolve(),
            interval=float(workload["rss_sample_interval_seconds"]),
            timeout=float(workload["worker_timeout_seconds"]),
            maximum_rss=int(config["gates"]["maximum_peak_process_tree_rss_bytes"]),
        )
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
        runs.append({"monitor": monitor, "result": result})
        if monitor["timed_out"] or monitor["resource_limit_exceeded"]:
            break
    gates = config["gates"]
    valid = len(runs) == 2 and all(item["result"] is not None for item in runs)
    hash_keys = ("source_sha256", "target_sha256", "candidate_sha256", "scale_sha256")
    hashes_exact = bool(valid and all(runs[0]["result"][key] == runs[1]["result"][key] for key in hash_keys))
    facts_exact = bool(valid and runs[0]["result"]["selected_facts"] == runs[1]["result"]["selected_facts"])
    peaks = [int(item["monitor"]["peak_process_tree_rss_bytes"]) for item in runs]
    walls = [float(item["result"]["timings_seconds"]["worker_total"]) for item in runs] if valid else []
    peak_ratio = max(peaks) / max(1, min(peaks))
    wall_ratio = max(walls) / max(1e-12, min(walls)) if walls else float("inf")
    checks = {
        "workers_completed": bool(valid and all(item["monitor"]["exit_code"] == 0 and not item["monitor"]["timed_out"] for item in runs)),
        "output_hash_exact": hashes_exact,
        "selected_facts_exact": facts_exact,
        "finite_bounded": bool(valid and all(item["result"]["finite_and_bounded"] for item in runs)),
        "luminance_error": bool(valid and max(item["result"]["maximum_luminance_error"] for item in runs) <= float(gates["maximum_luminance_error"])),
        "peak_rss": bool(valid and max(peaks) <= int(gates["maximum_peak_process_tree_rss_bytes"])),
        "worker_time": bool(valid and max(walls) <= float(gates["maximum_worker_seconds"])),
        "peak_repeat": peak_ratio <= float(gates["maximum_peak_rss_repeat_ratio"]),
        "wall_repeat": wall_ratio <= float(gates["maximum_wall_repeat_ratio"]),
        "cleanup": all(not item["monitor"]["temporary_exists"] and item["monitor"]["orphan_count"] == 0 for item in runs),
    }
    automatic_pass = all(checks.values())
    stable = {
        "contract_sha256": _hash_file(config_path),
        "hashes": {key: runs[0]["result"][key] for key in hash_keys} if valid else {},
        "selected_facts": runs[0]["result"]["selected_facts"] if valid else {},
        "checks": checks,
        "automatic_pass": automatic_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": "U5.R2CB53",
        "contract_sha256": _hash_file(config_path),
        "runs": runs,
        "metrics": {
            "maximum_peak_process_tree_rss_bytes": max(peaks),
            "peak_rss_repeat_ratio": peak_ratio,
            "maximum_worker_seconds": max(walls) if walls else None,
            "wall_repeat_ratio": wall_ratio,
        },
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": "pass_cb53_24mp_host_resource_development" if automatic_pass else "close_cb53_open_exact_streaming_optimization",
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.worker_output is not None:
        _atomic_json(args.worker_output, worker(config))
        return 0
    if args.output is None or args.work_dir is None:
        parser.error("parent mode requires --output and --work-dir")
    report = run_parent(args.config.resolve(), args.work_dir.resolve())
    _atomic_json(args.output.resolve(), report)
    print(json.dumps({"automatic_pass": report["automatic_pass"], "decision": report["decision"], "metrics": report["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
