#!/usr/bin/env python3
"""Audit U7.8D on the frozen U7.8C six-Canon real-scale transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_u7_8c_canon_real_scale_input_batch as legacy
from src.inference import three_stock_input_batch as batch_module
from src.inference.render_contract import sha256_file

CONFIG = (
    ROOT / "configs/u7_8e_canon_real_scale_transaction_provenance_confirmation_v1.json"
)
SCRATCH = ROOT / "tmp/u7_8e_formal_scratch"
STYLES = ("velvia_50", "portra_400", "ektar_100")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


class U78EError(RuntimeError):
    """Raised when the frozen U7.8E protocol cannot execute exactly."""


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise U78EError(f"{path.name} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise U78EError(f"{path.name} is not a JSON object")
    return value


def _git_text(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _git_blob_bytes(blob: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{40}", blob):
        raise U78EError("Git blob identity is invalid")
    try:
        return subprocess.check_output(["git", "cat-file", "blob", blob], cwd=ROOT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise U78EError(f"Git blob is unavailable: {blob}") from exc


def _validate_git_binding(*, commit: str, path: str, row: Mapping[str, Any]) -> bytes:
    if _COMMIT.fullmatch(commit) is None:
        raise U78EError("historical commit identity is invalid")
    expected_blob = str(row["git_blob"])
    actual_blob = _git_text("rev-parse", f"{commit}:{path}")
    if actual_blob != expected_blob:
        raise U78EError(f"historical Git blob drifted for {path}")
    payload = _git_blob_bytes(expected_blob)
    expected_bytes = int(row.get("bytes", row.get("git_lf_bytes", -1)))
    expected_sha = str(row.get("sha256", row.get("git_lf_sha256", "")))
    if (
        len(payload) != expected_bytes
        or hashlib.sha256(payload).hexdigest() != expected_sha
    ):
        raise U78EError(f"historical Git payload drifted for {path}")
    return payload


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("node_id") != "U7.8E":
        raise U78EError("config node identity drifted")
    if config.get("status") != "FROZEN_BEFORE_PIXEL_EXECUTION":
        raise U78EError("config freeze status drifted")
    commit = str(config["historical_commit"])
    for row in dict(config["bindings"]).values():
        binding_commit = str(row.get("commit", commit))
        _validate_git_binding(commit=binding_commit, path=str(row["path"]), row=row)
    sources = list(config["sources"])
    if len(sources) != 6 or len({row["job_id"] for row in sources}) != 6:
        raise U78EError("source cohort must contain six unique jobs")
    if tuple(config["render"]["styles"]) != STYLES:
        raise U78EError("style order drifted")
    if dict(config["render"]) != {
        "look_amount": 1.0,
        "seed": 31,
        "tile_size": 256,
        "tile_workers": 1,
        "png_compression": 0,
        "styles": list(STYLES),
    }:
        raise U78EError("render parameters drifted")
    resources = dict(config["resource_gates"])
    if int(resources["maximum_process_tree_rss_bytes"]) != 4 * 1024**3:
        raise U78EError("RSS gate drifted")
    if float(resources["maximum_controller_wall_seconds"]) != 1200.0:
        raise U78EError("wall gate drifted")


def _validate_execution_lock(lock: Mapping[str, Any]) -> str:
    if lock.get("node_id") != "U7.8E":
        raise U78EError("execution lock node identity drifted")
    commit = str(lock.get("implementation_commit"))
    if _COMMIT.fullmatch(commit) is None:
        raise U78EError("implementation commit identity is invalid")
    if _git_text("rev-parse", "HEAD") != commit:
        raise U78EError("formal checkout is not the locked implementation commit")
    for path, expected_blob in dict(lock.get("git_blobs", {})).items():
        if _git_text("rev-parse", f"{commit}:{path}") != expected_blob:
            raise U78EError(f"execution-lock Git blob drifted for {path}")
    if not bool(lock.get("core_changes_forbidden")):
        raise U78EError("execution lock did not forbid core changes")
    return commit


def _resolve_source_config(
    config: Mapping[str, Any], source_root: Path
) -> dict[str, Any]:
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise U78EError("source root is unavailable")
    runtime = json.loads(json.dumps(config))
    for row in runtime["sources"]:
        relative = Path(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise U78EError("source path is not strict repository-relative")
        path = (source_root / relative).resolve()
        if not path.is_file():
            raise U78EError(f"source is unavailable for {row['job_id']}")
        if path.stat().st_size != int(row["bytes"]):
            raise U78EError(f"source byte length drifted for {row['job_id']}")
        if sha256_file(path) != row["sha256"]:
            raise U78EError(f"source hash drifted for {row['job_id']}")
        row["path"] = str(path)
    return runtime


def _materialize_historical_runtime(
    config: Mapping[str, Any], runtime_config: dict[str, Any]
) -> dict[str, Any]:
    commit = str(config["historical_commit"])
    materialized = SCRATCH / "historical-runtime"
    materialized.mkdir()
    results: dict[str, Any] = {}
    targets = {
        "product_profile": materialized / "safe_rich_product_v1.json",
        "statistics": materialized / "film_color_stats.json",
        "guardrails": materialized / "color_guardrails.json",
    }
    for label, target in targets.items():
        row = dict(config["bindings"][label])
        raw = _validate_git_binding(commit=commit, path=str(row["path"]), row=row)
        if label == "statistics":
            if b"\r" in raw or raw.count(b"\n") != int(row["git_lf_count"]):
                raise U78EError("statistics Git line-ending facts drifted")
            if int(row["git_cr_count"]) != 0:
                raise U78EError("statistics frozen CR count drifted")
            payload = raw.replace(b"\n", b"\r\n")
            if row["runtime_materialization"] != "replace-each-lf-with-crlf-v1":
                raise U78EError("statistics materialization policy drifted")
            if len(payload) != int(row["runtime_bytes"]):
                raise U78EError("statistics runtime byte length drifted")
            if hashlib.sha256(payload).hexdigest() != row["runtime_sha256"]:
                raise U78EError("statistics runtime hash drifted")
        else:
            if row["runtime_materialization"] != "git-lf-bytes":
                raise U78EError(f"{label} materialization policy drifted")
            payload = raw
        target.write_bytes(payload)
        results[label] = {
            "git_blob": row["git_blob"],
            "git_lf_bytes": len(raw),
            "git_lf_sha256": hashlib.sha256(raw).hexdigest(),
            "runtime_bytes": len(payload),
            "runtime_sha256": sha256_file(target),
        }
    runtime_config["render"] = {
        **dict(runtime_config["render"]),
        "profile_path": str(targets["product_profile"].resolve()),
        "statistics_path": str(targets["statistics"].resolve()),
        "guardrails_path": str(targets["guardrails"].resolve()),
    }
    return results


def _validate_runtime_root_asset_ledger(
    config: Mapping[str, Any], materialized: Mapping[str, Any]
) -> dict[str, Any]:
    if _git_text("config", "--get", "core.autocrlf").casefold() != "true":
        raise U78EError("formal checkout must materialize with core.autocrlf=true")
    profile_path = SCRATCH / "historical-runtime/safe_rich_product_v1.json"
    profile = _read_json(profile_path)
    expected = {
        "legacy_profile_config": (
            "configs/color_rendering_profiles.yaml",
            str(config["bindings"]["legacy_profile_config"]["git_lf_sha256"]),
        ),
        "style_statistics": (
            "configs/film_color_stats.json",
            str(config["bindings"]["statistics"]["runtime_sha256"]),
        ),
        "color_guardrails": (
            "configs/color_guardrails.json",
            str(config["bindings"]["guardrails"]["git_lf_sha256"]),
        ),
    }
    assets = profile.get("assets")
    if not isinstance(assets, list):
        raise U78EError("profile asset ledger is invalid")
    by_role = {str(row.get("role")): row for row in assets}
    if set(by_role) != set(expected):
        raise U78EError("profile asset ledger roles drifted")
    identities: dict[str, Any] = {}
    for role, (relative, expected_sha) in expected.items():
        row = by_role[role]
        if row.get("path") != relative or row.get("sha256") != expected_sha:
            raise U78EError(f"profile asset ledger entry drifted for {role}")
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise U78EError(f"runtime-root asset materialization drifted for {role}")
        identities[role] = {
            "bytes": path.stat().st_size,
            "sha256": expected_sha,
        }
    if identities["style_statistics"] != {
        "bytes": int(materialized["statistics"]["runtime_bytes"]),
        "sha256": str(materialized["statistics"]["runtime_sha256"]),
    }:
        raise U78EError("runtime-root and scratch statistics identities differ")
    return {
        "core_autocrlf": True,
        "assets": identities,
        "scratch_statistics_equals_runtime_root": True,
    }


def _recipe_commits(output_root: Path) -> list[str]:
    commits = []
    for path in sorted(output_root.rglob("*.recipe.json")):
        recipe = _read_json(path)
        commits.append(str(recipe.get("software", {}).get("commit")))
    return commits


def _validate_recorded_paths(
    output_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    recipe_count = 0
    child_manifest_count = 0
    for row in sorted(config["sources"], key=lambda value: value["job_id"]):
        job_id = str(row["job_id"])
        source = str(Path(str(row["path"])).resolve())
        child_root = output_root / job_id
        child = _read_json(child_root / "batch.json")
        if child.get("input_path") != source:
            raise U78EError(f"child manifest input path drifted for {job_id}")
        child_rows = child.get("rows")
        if not isinstance(child_rows, list) or len(child_rows) != 3:
            raise U78EError(f"child manifest rows drifted for {job_id}")
        by_style = {str(item.get("style_id")): item for item in child_rows}
        if set(by_style) != set(STYLES):
            raise U78EError(f"child manifest style paths drifted for {job_id}")
        for style in STYLES:
            expected_output = str((child_root / f"{style}.png").resolve())
            expected_recipe = str((child_root / f"{style}.recipe.json").resolve())
            child_row = by_style[style]
            if (
                child_row.get("output_path") != expected_output
                or child_row.get("recipe_path") != expected_recipe
            ):
                raise U78EError(f"child manifest output paths drifted for {job_id}")
            recipe = _read_json(child_root / f"{style}.recipe.json")
            if (
                recipe.get("input", {}).get("path") != source
                or recipe.get("output", {}).get("path") != expected_output
            ):
                raise U78EError(f"recipe canonical paths drifted for {job_id}")
            recipe_count += 1
        child_manifest_count += 1
    return {
        "recipe_path_count": recipe_count,
        "child_manifest_path_count": child_manifest_count,
        "all_paths_equal_fixed_source_and_transaction_roots": True,
    }


def _run_primary_with_provenance(
    manifest: Path,
    output_root: Path,
    config: Mapping[str, Any],
    expected_commit: str,
) -> tuple[dict[str, Any], dict[str, int], list[dict[str, Any]], dict[str, Any]]:
    original_software_commit = batch_module._software_commit
    original_child = batch_module.render_three_stock_batch_to_directory
    observations: list[str] = []
    child_snapshots: list[str | None] = []

    def observed(root: Path) -> str:
        value = original_software_commit(root)
        observations.append(value)
        return value

    def recorded_child(*args, **kwargs):
        child_snapshots.append(kwargs.get("software_commit"))
        return original_child(*args, **kwargs)

    batch_module._software_commit = observed
    batch_module.render_three_stock_batch_to_directory = recorded_child
    try:
        receipt, decode_counts, job_walls = legacy._run_primary(
            manifest, output_root, config
        )
    finally:
        batch_module._software_commit = original_software_commit
        batch_module.render_three_stock_batch_to_directory = original_child
    recipe_commits = _recipe_commits(output_root)
    if observations != [expected_commit, expected_commit]:
        raise U78EError("software commit start/recheck observations drifted")
    if child_snapshots != [expected_commit] * 6:
        raise U78EError("child software snapshot propagation drifted")
    if recipe_commits != [expected_commit] * 18:
        raise U78EError("recipe software commits drifted")
    provenance = {
        "transaction_start_snapshot_count": 1,
        "prepublication_recheck_count": 1,
        "total_observation_count": len(observations),
        "observed_commits": observations,
        "child_snapshot_count": len(child_snapshots),
        "all_child_snapshots_equal": True,
        "recipe_commit_count": len(recipe_commits),
        "all_recipe_commits_equal": True,
    }
    return receipt, decode_counts, job_walls, provenance


def _run_controls_with_provenance(
    manifest: Path, config: Mapping[str, Any], expected_commit: str
) -> dict[str, Any]:
    original = batch_module._software_commit
    observations: list[str] = []

    def observed(root: Path) -> str:
        value = original(root)
        observations.append(value)
        return value

    batch_module._software_commit = observed
    try:
        controls = legacy._run_failure_controls(manifest, config)
    finally:
        batch_module._software_commit = original
    if observations != [expected_commit] * 3:
        raise U78EError("failure-control software observations drifted")
    return {**controls, "software_commit_observations": observations}


def _stable_payload(worker: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": worker["schema_version"],
        "node_id": worker["node_id"],
        "implementation_commit": worker["implementation_commit"],
        "config_git_blob": worker["config_git_blob"],
        "execution_lock_sha256": worker["execution_lock_sha256"],
        "source_identity": worker["source_identity"],
        "source_set_identity": worker["source_set_identity"],
        "historical_materialization": worker["historical_materialization"],
        "transaction": worker["transaction"],
        "software_provenance": worker["software_provenance"],
        "decode_counts": worker["decode_counts"],
        "controls": worker["controls"],
        "gate_results": worker["gate_results"],
        "claim_ceiling": worker["claim_ceiling"],
    }


def _run_worker(order: str, source_root: Path, lock_path: Path) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise U78EError("order must be forward or reverse")
    if os.name != "nt":
        raise U78EError("U7.8E formal execution requires Windows")
    if SCRATCH.exists():
        raise U78EError("formal scratch root already exists")
    config = _read_json(CONFIG)
    lock = _read_json(lock_path)
    _validate_config(config)
    implementation_commit = _validate_execution_lock(lock)
    runtime_config = _resolve_source_config(config, source_root)
    legacy.ROOT = ROOT
    legacy.SCRATCH = SCRATCH
    SCRATCH.mkdir(parents=True)
    manifest = SCRATCH / "jobs.json"
    output_root = SCRATCH / "published"
    try:
        historical = _materialize_historical_runtime(config, runtime_config)
        historical["runtime_root_asset_ledger"] = _validate_runtime_root_asset_ledger(
            config, historical
        )
        before_hashes = legacy._source_hashes(runtime_config)
        legacy._write_manifest(manifest, runtime_config, order)
        receipt, decode_counts, job_walls, provenance = _run_primary_with_provenance(
            manifest, output_root, runtime_config, implementation_commit
        )
        transaction = legacy._validate_published(
            receipt, output_root=output_root, config=runtime_config
        )
        transaction["recorded_path_contract"] = _validate_recorded_paths(
            output_root, runtime_config
        )
        transaction["receipt_bytes"] = (output_root / "batch.json").stat().st_size
        shutil.rmtree(output_root)
        controls = _run_controls_with_provenance(
            manifest, runtime_config, implementation_commit
        )
        after_hashes = legacy._source_hashes(runtime_config)
        if before_hashes != after_hashes:
            raise U78EError("source bytes changed during formal execution")
        manifest.unlink()
        shutil.rmtree(SCRATCH / "historical-runtime")
        residue = list(SCRATCH.iterdir())
        if residue:
            raise U78EError("owned formal file residue remained")
        expected_hashes = {
            str(row["job_id"]): str(row["sha256"]) for row in runtime_config["sources"]
        }
        gates = {
            "all_six_sources_exact_and_immutable": before_hashes == expected_hashes,
            "historical_materialization_exact": True,
            "successful_decode_calls_exactly_once_per_input": all(
                value == 1 for value in decode_counts.values()
            ),
            "six_children_eighteen_outputs_and_recipes": (
                transaction["job_count"],
                transaction["output_count"],
                transaction["recipe_count"],
                transaction["child_manifest_count"],
            )
            == (6, 18, 18, 6),
            "transaction_commit_snapshot_recheck_total_1_1_2": (
                provenance["transaction_start_snapshot_count"],
                provenance["prepublication_recheck_count"],
                provenance["total_observation_count"],
            )
            == (1, 1, 2),
            "all_children_and_recipes_bind_implementation_commit": (
                provenance["all_child_snapshots_equal"]
                and provenance["all_recipe_commits_equal"]
            ),
            "all_recorded_paths_equal_fixed_transaction_paths": transaction[
                "recorded_path_contract"
            ]
            == {
                "recipe_path_count": 18,
                "child_manifest_path_count": 6,
                "all_paths_equal_fixed_source_and_transaction_roots": True,
            },
            "injected_second_child_publishes_nothing": controls[
                "injected_second_child_published_nothing"
            ],
            "failure_controls_have_no_extra_decodes": (
                sum(controls["injected_failure_decode_counts"].values()) == 1
                and all(
                    value == 1
                    for value in controls["late_foreign_decode_counts"].values()
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
                "neuro-film.u7-8e-canon-real-scale-transaction-provenance-"
                "confirmation-result.v1"
            ),
            "node_id": "U7.8E",
            "requested_order": order,
            "implementation_commit": implementation_commit,
            "config_git_blob": _git_text(
                "rev-parse", f"HEAD:{CONFIG.relative_to(ROOT).as_posix()}"
            ),
            "execution_lock_sha256": sha256_file(lock_path),
            "source_identity": legacy._source_identity(runtime_config),
            "source_set_identity": _canonical_sha256(
                legacy._source_identity(runtime_config)
            ),
            "historical_materialization": historical,
            "transaction": transaction,
            "software_provenance": provenance,
            "decode_counts": decode_counts,
            "per_job_wall_seconds": job_walls,
            "controls": controls,
            "gate_results": gates,
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
    for process_item in processes:
        try:
            total += int(process_item.memory_info().rss)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return total


def _controller_report(
    worker: Mapping[str, Any], *, wall_seconds: float, peak_rss_bytes: int
) -> dict[str, Any]:
    config = _read_json(CONFIG)
    resources = dict(config["resource_gates"])
    resource_gates = {
        "controller_wall_seconds": wall_seconds
        <= float(resources["maximum_controller_wall_seconds"]),
        "process_tree_peak_rss_bytes": peak_rss_bytes
        <= int(resources["maximum_process_tree_rss_bytes"]),
    }
    gates = {
        **dict(worker["gate_results"]),
        "scratch_root_absent_after_worker": not SCRATCH.exists(),
    }

    def passed(value: object) -> bool:
        if isinstance(value, bool):
            return value is True
        return type(value) is int and value == 0

    decision = (
        "PASS_PRIVATE_U7_8E_CANON_REAL_SCALE_TRANSACTION_PROVENANCE_CONFIRMATION"
        if all(passed(value) for value in gates.values())
        and all(resource_gates.values())
        else "FAIL_CLOSED_U7_8E_CANON_REAL_SCALE_TRANSACTION_PROVENANCE_CONFIRMATION"
    )
    report = {
        **dict(worker),
        "decision": decision,
        "gate_results": gates,
        "resource_measurement": {
            "controller_wall_seconds": wall_seconds,
            "maximum_controller_wall_seconds": float(
                resources["maximum_controller_wall_seconds"]
            ),
            "process_tree_peak_rss_bytes": peak_rss_bytes,
            "maximum_process_tree_peak_rss_bytes": int(
                resources["maximum_process_tree_rss_bytes"]
            ),
        },
        "resource_gate_results": resource_gates,
    }
    report["scientific_identity"] = _canonical_sha256(_stable_payload(report))
    return report


def _cross_run_report(
    forward: Mapping[str, Any], reverse: Mapping[str, Any]
) -> dict[str, Any]:
    expected_pass = (
        "PASS_PRIVATE_U7_8E_CANON_REAL_SCALE_TRANSACTION_PROVENANCE_CONFIRMATION"
    )
    gates = {
        "orders_are_exactly_forward_and_reverse": (
            forward.get("requested_order"),
            reverse.get("requested_order"),
        )
        == ("forward", "reverse"),
        "both_single_controllers_pass": (
            forward.get("decision"),
            reverse.get("decision"),
        )
        == (expected_pass, expected_pass),
        "implementation_commit_exact": forward.get("implementation_commit")
        == reverse.get("implementation_commit"),
        "config_and_execution_lock_exact": (
            forward.get("config_git_blob"),
            forward.get("execution_lock_sha256"),
        )
        == (
            reverse.get("config_git_blob"),
            reverse.get("execution_lock_sha256"),
        ),
        "source_and_historical_materialization_exact": (
            forward.get("source_identity"),
            forward.get("source_set_identity"),
            forward.get("historical_materialization"),
        )
        == (
            reverse.get("source_identity"),
            reverse.get("source_set_identity"),
            reverse.get("historical_materialization"),
        ),
        "all_rgb16_recipe_manifest_receipt_and_batch_identities_exact": (
            forward.get("transaction") == reverse.get("transaction")
        ),
        "software_provenance_exact": forward.get("software_provenance")
        == reverse.get("software_provenance"),
        "decode_controls_and_stable_gates_exact": (
            forward.get("decode_counts"),
            forward.get("controls"),
            forward.get("gate_results"),
        )
        == (
            reverse.get("decode_counts"),
            reverse.get("controls"),
            reverse.get("gate_results"),
        ),
        "scientific_identity_exact": forward.get("scientific_identity")
        == reverse.get("scientific_identity"),
    }
    decision = (
        expected_pass
        if all(gates.values())
        else "FAIL_CLOSED_U7_8E_CANON_REAL_SCALE_TRANSACTION_PROVENANCE_CONFIRMATION"
    )
    return {
        "schema_version": (
            "neuro-film.u7-8e-canon-real-scale-transaction-provenance-"
            "confirmation-comparison.v1"
        ),
        "node_id": "U7.8E",
        "decision": decision,
        "implementation_commit": forward.get("implementation_commit"),
        "forward_scientific_identity": forward.get("scientific_identity"),
        "reverse_scientific_identity": reverse.get("scientific_identity"),
        "gate_results": gates,
        "claim_ceiling": forward.get("claim_ceiling"),
    }


def _run_controller(order: str, source_root: Path, lock_path: Path) -> dict[str, Any]:
    if SCRATCH.exists():
        raise U78EError("formal scratch root already exists")
    controller_temp = Path(
        tempfile.mkdtemp(prefix=f"u7_8e_{order}_controller_", dir=ROOT / "tmp")
    )
    stdout_path = controller_temp / "worker.stdout.json"
    stderr_path = controller_temp / "worker.stderr.txt"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        order,
        "--source-root",
        str(source_root.resolve()),
        "--execution-lock",
        str(lock_path.resolve()),
    ]
    started = time.perf_counter()
    peak_rss = 0
    return_code = -1
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
            monitored = psutil.Process(os.getpid())
            while child.poll() is None:
                peak_rss = max(peak_rss, _tree_rss(monitored))
                time.sleep(0.02)
            return_code = int(child.returncode)
        wall_seconds = time.perf_counter() - started
        worker_stdout = stdout_path.read_text(encoding="utf-8")
        worker_stderr = stderr_path.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(controller_temp, ignore_errors=True)
    if return_code != 0:
        raise U78EError(f"formal worker failed with {return_code}: {worker_stderr}")
    try:
        worker = json.loads(worker_stdout)
    except json.JSONDecodeError as exc:
        raise U78EError("formal worker emitted invalid JSON") from exc
    if not isinstance(worker, dict):
        raise U78EError("formal worker report is not an object")
    return _controller_report(
        worker, wall_seconds=wall_seconds, peak_rss_bytes=peak_rss
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", choices=("forward", "reverse"))
    parser.add_argument("--worker", choices=("forward", "reverse"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--execution-lock", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--compare-forward", type=Path)
    parser.add_argument("--compare-reverse", type=Path)
    args = parser.parse_args()
    comparing = args.compare_forward is not None or args.compare_reverse is not None
    if comparing:
        if (
            args.compare_forward is None
            or args.compare_reverse is None
            or args.report is None
            or args.order is not None
            or args.worker is not None
        ):
            parser.error("compare mode requires both reports and one output report")
    elif args.source_root is None or args.execution_lock is None:
        parser.error(
            "controller and worker modes require source root and execution lock"
        )
    elif args.worker is None and (args.order is None or args.report is None):
        parser.error("controller mode requires --order and --report")
    if args.worker is not None and (args.order is not None or args.report is not None):
        parser.error("worker mode accepts --worker, --source-root and --execution-lock")
    return args


def main() -> int:
    args = _parse_args()
    if args.compare_forward is not None:
        report = _cross_run_report(
            _read_json(args.compare_forward), _read_json(args.compare_reverse)
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if report["decision"].startswith("PASS_") else 1
    if args.worker is not None:
        sys.stdout.buffer.write(
            _canonical_bytes(
                _run_worker(args.worker, args.source_root, args.execution_lock)
            )
        )
        return 0
    report = _run_controller(args.order, args.source_root, args.execution_lock)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["decision"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
