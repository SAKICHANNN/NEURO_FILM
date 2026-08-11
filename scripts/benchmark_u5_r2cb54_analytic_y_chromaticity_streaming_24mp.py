"""Fresh-process CB54 benchmark for the output-exact streamed selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter, sleep

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_u5_r2cb53_analytic_y_chromaticity_24mp import (
    _atomic_json,
    _canonical,
    _hash_array,
    _hash_file,
    _kill_tree,
    generate_fields,
)
from src.eval.analytic_y_chromaticity_streaming import (
    select_analytic_y_chromaticity_candidate_streamed,
)
from src.eval.fujifilm_characteristic_photographic import _compiled_curve

SCHEMA = "neuro_film.u5_r2cb54_analytic_y_chromaticity_streaming_24mp_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb54_analytic_y_chromaticity_streaming_24mp_report.v1"


def _validate(config: dict) -> None:
    if config.get("schema") != SCHEMA or config.get("experiment_id") != "U5.R2CB54":
        raise ValueError("CB54 contract structure drift")
    parents = config["parents"]
    for path_key, sha_key in (
        ("cb53_decision_path", "cb53_decision_sha256"),
        ("cb53_workload_runner_path", "cb53_workload_runner_sha256"),
        ("streaming_operator_path", "streaming_operator_sha256"),
        ("cb11_contract_path", "cb11_contract_sha256"),
        ("cb6_contract_path", "cb6_contract_sha256"),
    ):
        if _hash_file(ROOT / parents[path_key]) != parents[sha_key]:
            raise ValueError(f"CB54 parent drift: {path_key}")
    decision = json.loads((ROOT / parents["cb53_decision_path"]).read_text(encoding="utf-8"))
    if decision.get("decision") != parents["cb53_required_decision"]:
        raise ValueError("CB54 prerequisite decision drift")


def worker(config: dict, *, scratch_root: Path) -> dict:
    _validate(config)
    workload = config["workload"]
    started = perf_counter()
    source, target = generate_fields(
        tuple(int(item) for item in workload["shape"]),
        row_chunk=int(workload["generation_row_chunk"]),
    )
    generation_seconds = perf_counter() - started
    parents = config["parents"]
    cb11 = json.loads((ROOT / parents["cb11_contract_path"]).read_text(encoding="utf-8"))
    cb6 = json.loads((ROOT / parents["cb6_contract_path"]).read_text(encoding="utf-8"))
    operator = config["operator"]
    execution_started = perf_counter()
    candidate, scale, luma_error, facts = select_analytic_y_chromaticity_candidate_streamed(
        source,
        target,
        curve=_compiled_curve(cb6),
        strength=float(cb11["operator"]["nominal_strength"]),
        boundary_epsilon=float(cb11["operator"]["boundary_epsilon"]),
        dose_grid=list(operator["dose_grid"]),
        maximum_gradient_ratio=float(operator["maximum_gradient_ratio"]),
        maximum_lstar_inversion_fraction=float(operator["maximum_lstar_inversion_fraction"]),
        lstar_order_epsilon=float(operator["lstar_order_epsilon"]),
        row_chunk=int(workload["operator_row_chunk"]),
        scratch_root=scratch_root,
    )
    execution_seconds = perf_counter() - execution_started
    return {
        "source_sha256": _hash_array(source),
        "target_sha256": _hash_array(target),
        "candidate_sha256": _hash_array(candidate),
        "scale_sha256": _hash_array(scale),
        "selected_facts": facts,
        "finite_and_bounded": bool(
            candidate.min() >= 0.0
            and candidate.max() <= 1.0
            and scale.min() >= 0.0
            and scale.max() <= 1.0
        ),
        "maximum_luminance_error": float(abs(luma_error).max()),
        "timings_seconds": {
            "generation": generation_seconds,
            "operator": execution_seconds,
            "worker_total": perf_counter() - started,
        },
        "scratch_residue": sorted(path.name for path in scratch_root.iterdir()),
    }


def _launch(config_path: Path, result_path: Path, *, interval: float, timeout: float, maximum_rss: int) -> dict:
    result_path.unlink(missing_ok=True)
    scratch_root = result_path.parent / (result_path.stem + "-scratch")
    scratch_root.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "--worker-output", str(result_path), "--scratch-root", str(scratch_root)],
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
        "scratch_residue": sorted(path.name for path in scratch_root.iterdir()),
        "orphan_count": sum(int(psutil.pid_exists(pid)) for pid in pids),
    }


def run_parent(config_path: Path, work_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate(config)
    work_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for index in range(int(config["workload"]["runs"])):
        result_path = work_dir / f"worker_{index + 1}.json"
        monitor = _launch(
            config_path.resolve(),
            result_path.resolve(),
            interval=float(config["workload"]["rss_sample_interval_seconds"]),
            timeout=float(config["workload"]["worker_timeout_seconds"]),
            maximum_rss=int(config["gates"]["maximum_peak_process_tree_rss_bytes"]),
        )
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
        runs.append({"monitor": monitor, "result": result})
        if monitor["timed_out"] or monitor["resource_limit_exceeded"]:
            break
    valid = len(runs) == 2 and all(item["result"] is not None for item in runs)
    hashes = ("source_sha256", "target_sha256", "candidate_sha256", "scale_sha256")
    peaks = [int(item["monitor"]["peak_process_tree_rss_bytes"]) for item in runs]
    walls = [float(item["result"]["timings_seconds"]["worker_total"]) for item in runs] if valid else []
    gates = config["gates"]
    checks = {
        "workers_completed": bool(valid and all(item["monitor"]["exit_code"] == 0 for item in runs)),
        "hashes_exact": bool(valid and all(runs[0]["result"][key] == runs[1]["result"][key] for key in hashes)),
        "facts_exact": bool(valid and runs[0]["result"]["selected_facts"] == runs[1]["result"]["selected_facts"]),
        "finite_bounded": bool(valid and all(item["result"]["finite_and_bounded"] for item in runs)),
        "luminance_error": bool(valid and max(item["result"]["maximum_luminance_error"] for item in runs) <= float(gates["maximum_luminance_error"])),
        "peak_rss": bool(valid and max(peaks) <= int(gates["maximum_peak_process_tree_rss_bytes"])),
        "worker_time": bool(valid and max(walls) <= float(gates["maximum_worker_seconds"])),
        "peak_repeat": bool(valid and max(peaks) / max(1, min(peaks)) <= float(gates["maximum_peak_rss_repeat_ratio"])),
        "wall_repeat": bool(valid and max(walls) / max(1e-12, min(walls)) <= float(gates["maximum_wall_repeat_ratio"])),
        "cleanup": bool(valid and all(not item["monitor"]["temporary_exists"] and not item["monitor"]["scratch_residue"] for item in runs)),
    }
    automatic_pass = all(checks.values())
    stable = {
        "contract_sha256": _hash_file(config_path),
        "hashes": {key: runs[0]["result"][key] for key in hashes} if valid else {},
        "facts": runs[0]["result"]["selected_facts"] if valid else {},
        "checks": checks,
        "automatic_pass": automatic_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": "U5.R2CB54",
        "contract_sha256": _hash_file(config_path),
        "runs": runs,
        "metrics": {
            "maximum_peak_process_tree_rss_bytes": max(peaks),
            "peak_rss_repeat_ratio": max(peaks) / max(1, min(peaks)),
            "maximum_worker_seconds": max(walls) if walls else None,
            "wall_repeat_ratio": max(walls) / max(1e-12, min(walls)) if walls else None,
        },
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": "pass_cb54_output_exact_streaming_24mp" if automatic_pass else "close_cb54_streaming_without_rescue",
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.worker_output is not None:
        if args.scratch_root is None:
            parser.error("worker mode requires --scratch-root")
        _atomic_json(args.worker_output, worker(config, scratch_root=args.scratch_root))
        return 0
    if args.output is None or args.work_dir is None:
        parser.error("parent mode requires --output and --work-dir")
    report = run_parent(args.config.resolve(), args.work_dir.resolve())
    _atomic_json(args.output.resolve(), report)
    print(json.dumps({"automatic_pass": report["automatic_pass"], "decision": report["decision"], "metrics": report["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
