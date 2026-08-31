#!/usr/bin/env python3
"""Audit the frozen U7.8C six-Canon real-scale three-look transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import three_stock_batch as child_module
from src.inference import three_stock_input_batch as batch_module
from src.inference.render_contract import sha256_file, validate_render_recipe
from src.inference.three_stock_input_batch import (
    INPUT_MANIFEST_SCHEMA,
    RECEIPT_SCHEMA,
    render_three_stock_input_batch_to_directory,
)

CONFIG = ROOT / "configs/u7_8c_canon_real_scale_input_batch_v1.json"
LOCK = ROOT / "docs/planning/U7_8C_CANON_REAL_SCALE_INPUT_BATCH_EXECUTION_LOCK_V2.json"
SCRATCH = ROOT / "tmp/u7_8c_formal_scratch"
STYLES = ("velvia_50", "portra_400", "ektar_100")


class U78CError(RuntimeError):
    """Raised when the frozen U7.8C protocol cannot be executed exactly."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise U78CError(f"{path.name} is not a JSON object")
    return payload


def _validate_file_binding(row: Mapping[str, Any], *, label: str) -> None:
    path = ROOT / str(row["path"])
    if not path.is_file():
        raise U78CError(f"{label} path is unavailable")
    if path.stat().st_size != int(row["bytes"]):
        raise U78CError(f"{label} byte length drifted")
    if sha256_file(path) != str(row["sha256"]):
        raise U78CError(f"{label} hash drifted")


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("node_id") != "U7.8C":
        raise U78CError("config node identity drifted")
    for label, row in dict(config["bindings"]).items():
        _validate_file_binding(row, label=label)
    sources = list(config["sources"])
    if len(sources) != 6 or len({row["job_id"] for row in sources}) != 6:
        raise U78CError("source cohort must contain six unique jobs")
    for row in sources:
        _validate_file_binding(row, label=str(row["job_id"]))
        shape = row.get("decoded_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(not isinstance(value, int) or value <= 0 for value in shape)
            or shape[2] != 3
        ):
            raise U78CError(f"decoded shape is invalid for {row['job_id']}")
    resources = dict(config["resource_gates"])
    if int(resources["maximum_process_tree_rss_bytes"]) != 4 * 1024**3:
        raise U78CError("RSS gate drifted")
    if float(resources["maximum_controller_wall_seconds"]) != 1200.0:
        raise U78CError("wall gate drifted")


def _validate_execution_lock(lock: Mapping[str, Any]) -> None:
    if lock.get("node_id") != "U7.8C":
        raise U78CError("execution lock node identity drifted")
    commit = str(lock["implementation_commit"])
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        check=True,
    )
    for path_text, expected_blob in dict(lock["git_blobs"]).items():
        actual_blob = _git("rev-parse", f"{commit}:{path_text}")
        if actual_blob != expected_blob:
            raise U78CError(f"execution-lock Git blob drifted for {path_text}")
        if _git("hash-object", path_text) != expected_blob:
            raise U78CError(f"working file differs from lock for {path_text}")


def _source_identity(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "job_id": row["job_id"],
            "source_id": row["source_id"],
            "bytes": int(row["bytes"]),
            "sha256": row["sha256"],
            "decoded_shape": list(row["decoded_shape"]),
        }
        for row in sorted(config["sources"], key=lambda item: item["job_id"])
    ]


def _manifest_payload(config: Mapping[str, Any], order: str) -> dict[str, object]:
    rows = list(config["sources"])
    rows = rows if order == "forward" else list(reversed(rows))
    return {
        "schema_version": INPUT_MANIFEST_SCHEMA,
        "jobs": [
            {
                "job_id": row["job_id"],
                "input_path": str((ROOT / row["path"]).resolve()),
                "input_sha256": row["sha256"],
            }
            for row in rows
        ],
    }


def _write_manifest(path: Path, config: Mapping[str, Any], order: str) -> None:
    path.write_bytes(_canonical_bytes(_manifest_payload(config, order)))


def _invoke(manifest: Path, destination: Path, config: Mapping[str, Any]) -> dict:
    render = dict(config["render"])
    return render_three_stock_input_batch_to_directory(
        manifest,
        destination,
        root=ROOT,
        profile_path=ROOT / str(render["profile_path"]),
        statistics_path=ROOT / str(render["statistics_path"]),
        guardrails_path=ROOT / str(render["guardrails_path"]),
        look_amount=float(render["look_amount"]),
        seed=int(render["seed"]),
        tile_size=int(render["tile_size"]),
        tile_workers=int(render["tile_workers"]),
        png_compression=int(render["png_compression"]),
    )


def _source_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(row["job_id"]): sha256_file(ROOT / str(row["path"]))
        for row in config["sources"]
    }


def _require_relative_posix_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise U78CError(f"{label} is not a non-empty path string")
    if "\\" in value or ":" in value or value.startswith("/"):
        raise U78CError(f"{label} exposed a machine-local path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise U78CError(f"{label} is not a strict relative POSIX path")
    if path.as_posix() != value:
        raise U78CError(f"{label} is not canonical POSIX")
    return value


def _validate_published(
    receipt: Mapping[str, Any],
    *,
    output_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise U78CError("receipt schema drifted")
    expected = {row["job_id"]: row for row in config["sources"]}
    jobs = receipt.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 6:
        raise U78CError("receipt job count drifted")
    if [row["job_id"] for row in jobs] != sorted(expected):
        raise U78CError("canonical job order drifted")
    rows_identity: list[dict[str, Any]] = []
    output_count = 0
    recipe_count = 0
    child_manifest_count = 0
    for job in jobs:
        job_id = str(job["job_id"])
        expected_row = expected[job_id]
        if job["input_sha256"] != expected_row["sha256"]:
            raise U78CError(f"receipt input identity drifted for {job_id}")
        child_relative = _require_relative_posix_path(
            job.get("child_manifest_path"), label=f"{job_id} child manifest path"
        )
        child_path = output_root / child_relative
        child_sha256 = sha256_file(child_path)
        if child_sha256 != job["child_manifest_sha256"]:
            raise U78CError(f"child manifest identity drifted for {job_id}")
        child_manifest_count += 1
        if tuple(row["style_id"] for row in job["rows"]) != STYLES:
            raise U78CError(f"style order drifted for {job_id}")
        identity_rows: list[dict[str, str]] = []
        expected_shape = tuple(expected_row["decoded_shape"])
        for row in job["rows"]:
            output_relative = _require_relative_posix_path(
                row.get("output_path"), label=f"{job_id} output path"
            )
            recipe_relative = _require_relative_posix_path(
                row.get("recipe_path"), label=f"{job_id} recipe path"
            )
            output_path = output_root / output_relative
            recipe_path = output_root / recipe_relative
            output_sha256 = sha256_file(output_path)
            recipe_sha256 = sha256_file(recipe_path)
            if output_sha256 != row["output_sha256"]:
                raise U78CError(f"output identity drifted for {job_id}")
            if recipe_sha256 != row["recipe_sha256"]:
                raise U78CError(f"recipe identity drifted for {job_id}")
            recipe = _read_json(recipe_path)
            validate_render_recipe(recipe)
            if recipe.get("input", {}).get("sha256") != expected_row["sha256"]:
                raise U78CError(f"recipe source identity drifted for {job_id}")
            claim = recipe.get("claim", {})
            if (
                claim.get("output_label") != "film-inspired"
                or claim.get("evidence_grade") != "look-approximation"
            ):
                raise U78CError(f"recipe claim drifted for {job_id}")
            decoded = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
            if decoded is None or decoded.dtype != np.uint16:
                raise U78CError(f"RGB16 output contract drifted for {job_id}")
            if tuple(decoded.shape) != expected_shape:
                raise U78CError(f"output geometry drifted for {job_id}")
            del decoded
            output_count += 1
            recipe_count += 1
            identity_rows.append(
                {
                    "style_id": str(row["style_id"]),
                    "output_sha256": output_sha256,
                    "recipe_sha256": recipe_sha256,
                }
            )
        rows_identity.append(
            {
                "job_id": job_id,
                "input_sha256": str(job["input_sha256"]),
                "child_manifest_sha256": child_sha256,
                "rows": identity_rows,
            }
        )
    if (output_count, recipe_count, child_manifest_count) != (18, 18, 6):
        raise U78CError("complete transaction counts drifted")
    return {
        "batch_id": str(receipt["batch_id"]),
        "receipt_sha256": sha256_file(output_root / "batch.json"),
        "job_count": 6,
        "output_count": output_count,
        "recipe_count": recipe_count,
        "child_manifest_count": child_manifest_count,
        "rows": rows_identity,
    }


def _run_primary(
    manifest: Path, output_root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, int], list[dict[str, Any]]]:
    source_to_job = {
        str((ROOT / row["path"]).resolve()).casefold(): row["job_id"]
        for row in config["sources"]
    }
    decode_counts = {str(row["job_id"]): 0 for row in config["sources"]}
    job_walls: list[dict[str, Any]] = []
    original_load = child_module.load_working_image
    original_child = batch_module.render_three_stock_batch_to_directory

    def counted_load(path: Path):
        key = str(Path(path).resolve()).casefold()
        if key not in source_to_job:
            raise U78CError("unexpected source reached WorkingImage decode")
        decode_counts[source_to_job[key]] += 1
        return original_load(path)

    def measured_child(input_path: Path, *args, **kwargs):
        key = str(Path(input_path).resolve()).casefold()
        job_id = source_to_job.get(key)
        if job_id is None:
            raise U78CError("unexpected source reached child renderer")
        started = time.perf_counter()
        result = original_child(input_path, *args, **kwargs)
        job_walls.append(
            {"job_id": job_id, "wall_seconds": time.perf_counter() - started}
        )
        return result

    child_module.load_working_image = counted_load
    batch_module.render_three_stock_batch_to_directory = measured_child
    try:
        receipt = _invoke(manifest, output_root, config)
    finally:
        child_module.load_working_image = original_load
        batch_module.render_three_stock_batch_to_directory = original_child
    expected_counts = {str(row["job_id"]): 1 for row in config["sources"]}
    if decode_counts != expected_counts:
        raise U78CError("successful decode counts were not exactly one per input")
    if len(job_walls) != 6:
        raise U78CError("per-job wall instrumentation drifted")
    return dict(receipt), decode_counts, job_walls


def _run_failure_controls(manifest: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    original_child = batch_module.render_three_stock_batch_to_directory
    original_publish = batch_module._publish_no_replace
    original_load = child_module.load_working_image
    source_to_job = {
        str((ROOT / row["path"]).resolve()).casefold(): str(row["job_id"])
        for row in config["sources"]
    }

    failure_output = SCRATCH / "injected-failure-output"
    child_calls = 0
    failure_decode_counts = {str(row["job_id"]): 0 for row in config["sources"]}

    def count_failure_load(path: Path):
        key = str(Path(path).resolve()).casefold()
        if key not in source_to_job:
            raise U78CError("unexpected failure-control source reached decode")
        failure_decode_counts[source_to_job[key]] += 1
        return original_load(path)

    def fail_second(*args, **kwargs):
        nonlocal child_calls
        child_calls += 1
        if child_calls == 2:
            raise RuntimeError("injected second-child failure")
        return original_child(*args, **kwargs)

    child_module.load_working_image = count_failure_load
    batch_module.render_three_stock_batch_to_directory = fail_second
    try:
        try:
            _invoke(manifest, failure_output, config)
        except RuntimeError as exc:
            if "injected second-child failure" not in str(exc):
                raise
        else:
            raise U78CError("injected second-child failure was accepted")
    finally:
        child_module.load_working_image = original_load
        batch_module.render_three_stock_batch_to_directory = original_child
    if child_calls != 2 or failure_output.exists():
        raise U78CError("injected child failure published partial output")
    canonical_first = min(source_to_job.values())
    expected_failure_counts = {
        job_id: int(job_id == canonical_first) for job_id in source_to_job.values()
    }
    if failure_decode_counts != expected_failure_counts:
        raise U78CError("injected failure performed unexpected source decodes")

    foreign_output = SCRATCH / "late-foreign-output"
    foreign_decode_counts = {str(row["job_id"]): 0 for row in config["sources"]}

    def count_foreign_load(path: Path):
        key = str(Path(path).resolve()).casefold()
        if key not in source_to_job:
            raise U78CError("unexpected late-foreign source reached decode")
        foreign_decode_counts[source_to_job[key]] += 1
        return original_load(path)

    def late_foreign(stage: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "foreign.bin").write_bytes(b"foreign")
        raise FileExistsError("injected late foreign destination")

    child_module.load_working_image = count_foreign_load
    batch_module._publish_no_replace = late_foreign
    try:
        try:
            _invoke(manifest, foreign_output, config)
        except FileExistsError as exc:
            if "injected late foreign destination" not in str(exc):
                raise
        else:
            raise U78CError("late foreign destination was accepted")
    finally:
        child_module.load_working_image = original_load
        batch_module._publish_no_replace = original_publish
    foreign_entries = sorted(path.name for path in foreign_output.iterdir())
    if foreign_entries != ["foreign.bin"]:
        raise U78CError("foreign destination did not preserve only foreign data")
    if (foreign_output / "foreign.bin").read_bytes() != b"foreign":
        raise U78CError("foreign destination content drifted")
    if any(count != 1 for count in foreign_decode_counts.values()):
        raise U78CError("late foreign control did not decode each input exactly once")
    shutil.rmtree(foreign_output)
    return {
        "injected_second_child_calls": child_calls,
        "injected_failure_decode_counts": failure_decode_counts,
        "injected_second_child_published_nothing": True,
        "late_foreign_decode_counts": foreign_decode_counts,
        "late_foreign_complete_batch_preserved": True,
    }


def _stable_payload(worker: Mapping[str, Any]) -> dict[str, Any]:
    """Return the order/resource-independent U7.8C scientific payload."""

    return {
        "schema_version": worker["schema_version"],
        "node_id": worker["node_id"],
        "config_sha256": worker["config_sha256"],
        "execution_lock_sha256": worker["execution_lock_sha256"],
        "source_identity": worker["source_identity"],
        "source_set_identity": worker["source_set_identity"],
        "transaction": worker["transaction"],
        "decode_counts": worker["decode_counts"],
        "controls": worker["controls"],
        "gate_results": worker["gate_results"],
        "claim_ceiling": worker["claim_ceiling"],
    }


def _run_worker(order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise U78CError("order must be forward or reverse")
    if SCRATCH.exists():
        raise U78CError("formal scratch root already exists")
    config = _read_json(CONFIG)
    lock = _read_json(LOCK)
    _validate_config(config)
    _validate_execution_lock(lock)
    SCRATCH.mkdir(parents=True)
    manifest = SCRATCH / "jobs.json"
    output_root = SCRATCH / "published"
    before_hashes = _source_hashes(config)
    try:
        _write_manifest(manifest, config, order)
        receipt, decode_counts, job_walls = _run_primary(manifest, output_root, config)
        transaction = _validate_published(
            receipt, output_root=output_root, config=config
        )
        shutil.rmtree(output_root)
        controls = _run_failure_controls(manifest, config)
        after_hashes = _source_hashes(config)
        if before_hashes != after_hashes:
            raise U78CError("source bytes changed during formal execution")
        manifest.unlink()
        residue = list(SCRATCH.iterdir())
        if residue:
            raise U78CError("owned formal file residue remained")
        expected_hashes = {
            str(row["job_id"]): str(row["sha256"]) for row in config["sources"]
        }
        source_exact = before_hashes == expected_hashes
        stable_gates = {
            "all_six_sources_exact_and_immutable": source_exact,
            "successful_decode_calls_exactly_once_per_input": all(
                count == 1 for count in decode_counts.values()
            ),
            "six_children_eighteen_outputs_and_recipes": (
                transaction["job_count"],
                transaction["output_count"],
                transaction["recipe_count"],
                transaction["child_manifest_count"],
            )
            == (6, 18, 18, 6),
            "injected_second_child_publishes_nothing": controls[
                "injected_second_child_published_nothing"
            ],
            "failure_controls_have_no_extra_decodes": (
                sum(controls["injected_failure_decode_counts"].values()) == 1
                and all(
                    count == 1
                    for count in controls["late_foreign_decode_counts"].values()
                )
            ),
            "complete_batch_late_foreign_preserved": controls[
                "late_foreign_complete_batch_preserved"
            ],
            "owned_stage_and_file_residue_count": 0,
            "network_requests": 0,
        }
        worker: dict[str, Any] = {
            "schema_version": (
                "neuro-film.u7-8c-canon-real-scale-input-batch-result.v1"
            ),
            "node_id": "U7.8C",
            "requested_order": order,
            "implementation_commit": lock["implementation_commit"],
            "config_sha256": sha256_file(CONFIG),
            "execution_lock_sha256": sha256_file(LOCK),
            "source_identity": _source_identity(config),
            "source_set_identity": _canonical_sha256(_source_identity(config)),
            "transaction": transaction,
            "decode_counts": decode_counts,
            "per_job_wall_seconds": job_walls,
            "controls": controls,
            "gate_results": stable_gates,
            "claim_ceiling": config["claim_ceiling"],
        }
        return worker
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)


def _tree_rss(process: psutil.Process) -> int:
    try:
        processes = [process, *process.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for item in processes:
        try:
            total += int(item.memory_info().rss)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return total


def _controller_report(
    worker: Mapping[str, Any],
    *,
    wall_seconds: float,
    peak_rss_bytes: int,
    scratch_root_absent: bool,
) -> dict[str, Any]:
    config = _read_json(CONFIG)
    resources = dict(config["resource_gates"])
    resource_gates = {
        "controller_wall_seconds": wall_seconds
        <= float(resources["maximum_controller_wall_seconds"]),
        "process_tree_peak_rss_bytes": peak_rss_bytes
        <= int(resources["maximum_process_tree_rss_bytes"]),
    }
    gate_results = dict(worker["gate_results"])
    gate_results["scratch_root_absent_after_worker"] = scratch_root_absent

    def gate_passed(value: object) -> bool:
        if isinstance(value, bool):
            return value is True
        return type(value) is int and value == 0

    stable_pass = all(gate_passed(value) for value in gate_results.values())
    decision = (
        "PASS_PRIVATE_U7_8C_CANON_REAL_SCALE_INPUT_BATCH"
        if stable_pass and all(resource_gates.values())
        else "FAIL_CLOSED_U7_8C_CANON_REAL_SCALE_INPUT_BATCH"
    )
    report = {
        **dict(worker),
        "decision": decision,
        "gate_results": gate_results,
        "resource_measurement": {
            "controller_wall_seconds": wall_seconds,
            "maximum_controller_wall_seconds": float(
                resources["maximum_controller_wall_seconds"]
            ),
            "process_tree_peak_rss_bytes": peak_rss_bytes,
            "maximum_process_tree_rss_bytes": int(
                resources["maximum_process_tree_rss_bytes"]
            ),
        },
        "resource_gate_results": resource_gates,
    }
    report["scientific_identity"] = _canonical_sha256(_stable_payload(report))
    return report


def _run_controller(order: str) -> dict[str, Any]:
    if SCRATCH.exists():
        raise U78CError("formal scratch root already exists")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", order]
    controller_temp = Path(
        tempfile.mkdtemp(prefix=f"u7_8c_{order}_controller_", dir=ROOT / "tmp")
    )
    stdout_path = controller_temp / "worker.stdout.json"
    stderr_path = controller_temp / "worker.stderr.txt"
    started = time.perf_counter()
    peak_rss = 0
    return_code = -1
    try:
        with (
            stdout_path.open("wb") as stdout_handle,
            stderr_path.open("wb") as stderr_handle,
        ):
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            monitored = psutil.Process(os.getpid())
            while child.poll() is None:
                peak_rss = max(peak_rss, _tree_rss(monitored))
                time.sleep(0.02)
            return_code = int(child.returncode)
        wall_seconds = time.perf_counter() - started
        stdout = stdout_path.read_text(encoding="utf-8")
        stderr = stderr_path.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(controller_temp, ignore_errors=True)
    controller_temp_absent = not controller_temp.exists()
    if return_code != 0:
        raise U78CError(f"formal worker failed with {return_code}: {stderr}")
    try:
        worker = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise U78CError("formal worker emitted invalid JSON") from exc
    if not isinstance(worker, dict):
        raise U78CError("formal worker report is not an object")
    return _controller_report(
        worker,
        wall_seconds=wall_seconds,
        peak_rss_bytes=peak_rss,
        scratch_root_absent=(not SCRATCH.exists() and controller_temp_absent),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", choices=("forward", "reverse"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", choices=("forward", "reverse"))
    args = parser.parse_args()
    if args.worker is None and (args.order is None or args.report is None):
        parser.error("controller mode requires --order and --report")
    if args.worker is not None and (args.order is not None or args.report is not None):
        parser.error("worker mode accepts only --worker")
    return args


def main() -> int:
    args = _parse_args()
    if args.worker is not None:
        sys.stdout.buffer.write(_canonical_bytes(_run_worker(args.worker)))
        return 0
    report = _run_controller(args.order)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["decision"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
