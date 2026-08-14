from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_evaluator = import_module(
    os.environ.get("NEURO_FILM_P7_VALUE_EVALUATOR", "src.eval.p4hu_ao6_value")
)
evaluate = _evaluator.evaluate
load_contract = _evaluator.load_contract


WORKER_REPORT_SCHEMA = getattr(
    _evaluator,
    "WORKER_REPORT_SCHEMA",
    "neuro-film.u6-p7h-p4hu-ao6-value-worker-report.v1",
)
FINAL_REPORT_SCHEMA = getattr(
    _evaluator,
    "FINAL_REPORT_SCHEMA",
    "neuro-film.u6-p7h-p4hu-ao6-value-run-report.v1",
)
EXPECTED_MEASUREMENT = {
    "formal_processes": 2,
    "sample_interval_seconds": 0.01,
    "worker_timeout_seconds": 900,
    "rss_and_wall_excluded_from_scientific_identity": True,
}
EXPECTED_RESOURCE_GATES = {
    "maximum_peak_process_tree_rss_bytes": 1_610_612_736,
    "maximum_worker_wall_seconds": 600,
}


class InfrastructureInvalid(RuntimeError):
    pass


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_from_root(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _require_exact_execution_contract(contract: dict[str, Any]) -> None:
    if contract.get("measurement") != EXPECTED_MEASUREMENT:
        raise ValueError("measurement must equal the frozen P7H execution contract")
    if contract.get("resource_gates") != EXPECTED_RESOURCE_GATES:
        raise ValueError("resource_gates must equal the frozen P7H limits")
    for key in ("decision_if_pass", "decision_if_fail", "decision_if_infrastructure_invalid"):
        if not isinstance(contract.get(key), str) or not contract[key]:
            raise ValueError(f"{key} must be a non-empty string")


def _write_create_only_json(path: Path, value: object) -> None:
    payload = _canonical_json_bytes(value)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _safe_relative_file(path_text: str, run_dir: Path) -> Path:
    candidate = (run_dir / path_text).resolve()
    try:
        candidate.relative_to(run_dir.resolve())
    except ValueError as error:
        raise InfrastructureInvalid(f"worker output escapes run directory: {path_text}") from error
    if not candidate.is_file():
        raise InfrastructureInvalid(f"worker output is missing: {path_text}")
    return candidate


def _collect_ordered_output_inventory(
    result: dict[str, Any], run_dir: Path
) -> list[dict[str, Any]]:
    rows = result.get("rows")
    if not isinstance(rows, list) or not rows:
        raise InfrastructureInvalid("scientific result must contain non-empty rows")
    inventory: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("outputs"), list):
            raise InfrastructureInvalid("scientific result row is malformed")
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise InfrastructureInvalid("scientific result source_id is malformed")
        for output in row["outputs"]:
            if not isinstance(output, dict):
                raise InfrastructureInvalid("scientific result output is malformed")
            output_path = output.get("output_path")
            expected_sha = output.get("output_sha256")
            arm_id = output.get("arm_id")
            if not all(isinstance(value, str) and value for value in (output_path, expected_sha, arm_id)):
                raise InfrastructureInvalid("scientific result output identity is malformed")
            absolute = _safe_relative_file(output_path, run_dir)
            actual_sha = _sha256_file(absolute)
            if actual_sha != expected_sha:
                raise InfrastructureInvalid(f"worker output hash mismatch: {output_path}")
            inventory.append(
                {
                    "source_id": source_id,
                    "arm_id": arm_id,
                    "output_path": output_path,
                    "output_sha256": actual_sha,
                    "output_size_bytes": absolute.stat().st_size,
                    "output_rgb16_array_sha256": output.get("output_rgb16_array_sha256"),
                    "output_encoded_array_sha256": output.get("output_encoded_array_sha256"),
                }
            )
    return inventory


def _collect_png_inventory(run_dir: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        inventory.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if not inventory:
        raise InfrastructureInvalid("worker produced no PNG artifacts")
    return inventory


def _validate_worker_report_files(report: dict[str, Any], run_dir: Path) -> None:
    result = report.get("scientific_result")
    if not isinstance(result, dict):
        raise InfrastructureInvalid("worker scientific result is malformed")
    if report.get("scientific_stable_evidence_id") != result.get("stable_evidence_id"):
        raise InfrastructureInvalid("worker stable scientific identity is inconsistent")
    ordered_inventory = _collect_ordered_output_inventory(result, run_dir)
    if report.get("ordered_output_inventory") != ordered_inventory:
        raise InfrastructureInvalid("worker ordered output inventory is inconsistent")
    if report.get("ordered_output_inventory_sha256") != _sha256_bytes(
        _canonical_json_bytes(ordered_inventory)
    ):
        raise InfrastructureInvalid("worker ordered output inventory hash is inconsistent")
    png_inventory = _collect_png_inventory(run_dir)
    if report.get("png_inventory") != png_inventory:
        raise InfrastructureInvalid("worker PNG inventory is inconsistent")
    if report.get("png_inventory_sha256") != _sha256_bytes(
        _canonical_json_bytes(png_inventory)
    ):
        raise InfrastructureInvalid("worker PNG inventory hash is inconsistent")


def _worker(config_path: Path, run_dir: Path) -> dict[str, Any]:
    contract = load_contract(config_path)
    _require_exact_execution_contract(contract)
    run_dir.mkdir(parents=True, exist_ok=False)
    build_dir = run_dir / "build"
    started = time.perf_counter()
    result = evaluate(contract, root=ROOT, output_dir=run_dir, build_dir=build_dir)
    wall_seconds = time.perf_counter() - started
    if not isinstance(result, dict):
        raise InfrastructureInvalid("evaluator did not return a JSON object")
    stable_id = result.get("stable_evidence_id")
    if not isinstance(stable_id, str) or not stable_id:
        raise InfrastructureInvalid("scientific result has no stable_evidence_id")
    ordered_inventory = _collect_ordered_output_inventory(result, run_dir)
    png_inventory = _collect_png_inventory(run_dir)
    report = {
        "schema": WORKER_REPORT_SCHEMA,
        "scientific_result": result,
        "scientific_stable_evidence_id": stable_id,
        "ordered_output_inventory": ordered_inventory,
        "ordered_output_inventory_sha256": _sha256_bytes(
            _canonical_json_bytes(ordered_inventory)
        ),
        "png_inventory": png_inventory,
        "png_inventory_sha256": _sha256_bytes(_canonical_json_bytes(png_inventory)),
        "measurement": {
            "worker_wall_seconds": wall_seconds,
            "excluded_from_scientific_identity": True,
        },
    }
    _write_create_only_json(run_dir / "worker_report.json", report)
    return report


def _process_tree_rss_bytes(process: psutil.Process) -> int:
    rss = 0
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    for item in processes:
        try:
            rss += item.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rss


def _terminate_process_tree(process: psutil.Process) -> None:
    try:
        children = process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []
    for item in reversed(children):
        try:
            item.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    try:
        process.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    psutil.wait_procs([*children, process], timeout=10.0)


def _execute_worker(
    *,
    config_path: Path,
    run_dir: Path,
    sample_interval_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--config",
        str(config_path),
        "--output",
        str(run_dir),
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    monitored = psutil.Process(process.pid)
    peak_rss_bytes = 0
    while process.poll() is None:
        peak_rss_bytes = max(peak_rss_bytes, _process_tree_rss_bytes(monitored))
        if time.perf_counter() - started > timeout_seconds:
            _terminate_process_tree(monitored)
            raise InfrastructureInvalid(f"worker exceeded hard timeout of {timeout_seconds}s")
        time.sleep(sample_interval_seconds)
    peak_rss_bytes = max(peak_rss_bytes, _process_tree_rss_bytes(monitored))
    stdout, stderr = process.communicate()
    monitor_wall_seconds = time.perf_counter() - started
    if process.returncode != 0:
        message = stderr.strip() or stdout.strip() or "no worker diagnostic"
        raise InfrastructureInvalid(
            f"worker exited with code {process.returncode}: {message[-2000:]}"
        )
    report_path = run_dir / "worker_report.json"
    if not report_path.is_file():
        raise InfrastructureInvalid("worker report JSON is missing")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InfrastructureInvalid("worker report JSON is unreadable") from error
    if not isinstance(report, dict) or report.get("schema") != WORKER_REPORT_SCHEMA:
        raise InfrastructureInvalid("worker report schema is invalid")
    _validate_worker_report_files(report, run_dir)
    report["monitor"] = {
        "peak_process_tree_rss_bytes": peak_rss_bytes,
        "wall_seconds": monitor_wall_seconds,
        "sample_interval_seconds": sample_interval_seconds,
        "excluded_from_scientific_identity": True,
    }
    report["worker_report_sha256"] = _sha256_file(report_path)
    return report


def _worker_replay_mismatches(
    first: dict[str, Any], second: dict[str, Any]
) -> list[str]:
    comparisons = {
        "scientific_stable_evidence_id": (
            first.get("scientific_stable_evidence_id"),
            second.get("scientific_stable_evidence_id"),
        ),
        "scientific_result": (first.get("scientific_result"), second.get("scientific_result")),
        "ordered_output_inventory": (
            first.get("ordered_output_inventory"),
            second.get("ordered_output_inventory"),
        ),
        "png_inventory": (first.get("png_inventory"), second.get("png_inventory")),
    }
    return [name for name, (left, right) in comparisons.items() if left != right]


def _finalize_report(output_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    report_path = output_dir / "report.json"
    _write_create_only_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    contract = load_contract(config_path)
    _require_exact_execution_contract(contract)
    output_dir.mkdir(parents=True, exist_ok=False)
    common = {
        "schema": FINAL_REPORT_SCHEMA,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "measurement_contract": EXPECTED_MEASUREMENT,
        "resource_gates": EXPECTED_RESOURCE_GATES,
        "resource_measurements_excluded_from_scientific_identity": True,
    }
    workers: list[dict[str, Any]] = []
    try:
        for name in ("run_a", "run_b"):
            workers.append(
                _execute_worker(
                    config_path=config_path,
                    run_dir=output_dir / name,
                    sample_interval_seconds=EXPECTED_MEASUREMENT["sample_interval_seconds"],
                    timeout_seconds=EXPECTED_MEASUREMENT["worker_timeout_seconds"],
                )
            )
    except (
        InfrastructureInvalid,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        psutil.Error,
        subprocess.SubprocessError,
    ) as error:
        invalid = {
            **common,
            "status": "infrastructure_invalid",
            "automatic_pass": False,
            "blind_review_allowed": False,
            "scientific_result_valid": False,
            "decision": contract["decision_if_infrastructure_invalid"],
            "infrastructure_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "workers": workers,
        }
        return _finalize_report(output_dir, invalid)

    mismatches = _worker_replay_mismatches(workers[0], workers[1])
    if mismatches:
        invalid = {
            **common,
            "status": "infrastructure_invalid",
            "automatic_pass": False,
            "blind_review_allowed": False,
            "scientific_result_valid": False,
            "decision": contract["decision_if_infrastructure_invalid"],
            "infrastructure_error": {
                "type": "ReplayMismatch",
                "message": "fresh-worker replay differs",
                "mismatched_fields": mismatches,
            },
            "workers": workers,
        }
        return _finalize_report(output_dir, invalid)

    measurements = [worker["monitor"] for worker in workers]
    rss_pass = all(
        item["peak_process_tree_rss_bytes"]
        <= EXPECTED_RESOURCE_GATES["maximum_peak_process_tree_rss_bytes"]
        for item in measurements
    )
    wall_pass = all(
        item["wall_seconds"] <= EXPECTED_RESOURCE_GATES["maximum_worker_wall_seconds"]
        for item in measurements
    )
    result = workers[0]["scientific_result"]
    scientific_pass = result.get("automatic_pass") is True
    resource_pass = rss_pass and wall_pass
    if not scientific_pass:
        status = "scientific_fail"
    elif not resource_pass:
        status = "resource_fail"
    else:
        status = "scientific_pass"
    automatic_pass = scientific_pass and resource_pass
    final = {
        **common,
        "status": status,
        "automatic_pass": automatic_pass,
        "blind_review_allowed": automatic_pass,
        "scientific_result_valid": True,
        "scientific_pass": scientific_pass,
        "resource_pass": resource_pass,
        "decision": (
            contract["decision_if_pass"] if automatic_pass else contract["decision_if_fail"]
        ),
        "scientific_stable_evidence_id": workers[0]["scientific_stable_evidence_id"],
        "scientific_result": result,
        "fresh_worker_replay": {
            "exact": True,
            "formal_processes": len(workers),
            "ordered_output_inventory_sha256": workers[0][
                "ordered_output_inventory_sha256"
            ],
            "png_inventory_sha256": workers[0]["png_inventory_sha256"],
        },
        "resource_gate_results": {
            "peak_process_tree_rss_pass": rss_pass,
            "worker_wall_pass": wall_pass,
        },
        "workers": workers,
    }
    return _finalize_report(output_dir, final)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen U6.P7H value ablation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = _resolve_from_root(args.config)
    output_path = _resolve_from_root(args.output)
    if args.worker:
        report = _worker(config_path, output_path)
        print(
            json.dumps(
                {
                    "scientific_stable_evidence_id": report["scientific_stable_evidence_id"],
                    "worker_report": str(output_path / "worker_report.json"),
                },
                sort_keys=True,
            )
        )
        return 0
    report = run(config_path, output_path)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "report": report["report_path"],
                "scientific_stable_evidence_id": report.get("scientific_stable_evidence_id"),
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    if report["status"] == "infrastructure_invalid":
        return 3
    return 0 if report["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
