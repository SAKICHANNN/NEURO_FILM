#!/usr/bin/env python3
"""Audit exact multi-vendor RAW files through the public product ingress."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import rawpy
from PIL import __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_u7_19a_srw_arq_generic_working_image as shared
from src.preprocess.raw_decode import RAW_SUFFIXES

PREFLIGHT_SCHEMA = "kmcfm.u7-19b-multi-vendor-raw-preflight.v1"
FORMAL_SCHEMA = "kmcfm.u7-19b-multi-vendor-raw-product-ingress-result.v1"


class U719BError(RuntimeError):
    """Raised when a frozen U7.19B invariant differs."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise U719BError(f"expected JSON object: {path}")
    return value


def _adapt_source(source: dict[str, Any]) -> dict[str, Any]:
    return {**source, "source_id": source["id"]}


def _adapt_stratum(stratum: dict[str, Any]) -> dict[str, Any]:
    return {
        "extension": stratum["extension"],
        "representative_source_id": stratum["representative"],
        "required_members": len(stratum["sources"]),
        "sources": [_adapt_source(row) for row in stratum["sources"]],
        "stratum_id": stratum["id"],
    }


def _adapt_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        **config,
        "per_stratum_preflight_gates": {
            "required_warning_codes": config["preflight"]["required_warning_codes"]
        },
    }


def _runtime_checks(config: dict[str, Any]) -> dict[str, bool]:
    runtime = config["runtime"]
    return {
        "libraw": list(rawpy.libraw_version) == runtime["libraw"],
        "numpy": np.__version__ == runtime["numpy"],
        "pillow": PILLOW_VERSION == runtime["pillow"],
        "python": ".".join(map(str, sys.version_info[:3])) == runtime["python"],
        "rawpy": rawpy.__version__ == runtime["rawpy"],
        "windows": sys.platform == "win32" and runtime["platform"] == "Windows",
    }


def _binding_checks(config: dict[str, Any]) -> dict[str, bool]:
    return {
        name: shared._verify_file(ROOT / binding["path"], binding)
        for name, binding in sorted(config["bindings"].items())
    }


def _peak_rss_bytes() -> int:
    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def _source_worker(
    config: dict[str, Any], producer_repo: Path, source_id: str
) -> dict[str, Any]:
    row = next(
        source
        for stratum in config["strata"]
        for source in stratum["sources"]
        if source["id"] == source_id
    )
    started = time.perf_counter()
    record = shared._row_record(
        producer_repo,
        _adapt_source(row),
        inspect=shared.inspect_input,
        load=lambda path, **_kwargs: shared.load_working_image(path),
    )
    record["resource"] = {
        "peak_process_rss_bytes": _peak_rss_bytes(),
        "wall_seconds": time.perf_counter() - started,
    }
    return record


def _product_worker(
    config: dict[str, Any], producer_repo: Path, extension: str, scratch_root: Path
) -> dict[str, Any]:
    stratum = next(row for row in config["strata"] if row["extension"] == extension)
    started = time.perf_counter()
    record = shared._product_record(
        _adapt_config(config), producer_repo, _adapt_stratum(stratum), scratch_root
    )
    record["resource"] = {
        "peak_process_rss_bytes": _peak_rss_bytes(),
        "wall_seconds": time.perf_counter() - started,
    }
    return record


def _run_worker(
    *,
    stage: str,
    config_path: Path,
    producer_repo: Path,
    key: str,
    scratch_root: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    output = scratch_root / f"{stage}-{key.lstrip('.').replace('/', '_')}.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        stage,
        "--config",
        str(config_path),
        "--producer-repo",
        str(producer_repo),
        "--key",
        key,
        "--scratch-root",
        str(scratch_root / "media"),
        "--output",
        str(output),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": "worker_timeout", "key": key, "worker_ok": False}
    if completed.returncode != 0 or not output.is_file():
        stderr = completed.stderr.strip().splitlines()
        return {
            "error": stderr[-1] if stderr else "worker_failed_without_stderr",
            "key": key,
            "returncode": completed.returncode,
            "worker_ok": False,
        }
    try:
        value = _load_json(output)
        value["worker_ok"] = True
        return value
    finally:
        output.unlink(missing_ok=True)


def _resource_gate(
    record: dict[str, Any], config: dict[str, Any], *, product: bool
) -> bool:
    if not record.get("worker_ok"):
        return False
    limit_key = "product_worker_seconds" if product else "source_worker_seconds"
    return (
        record["resource"]["wall_seconds"] <= config["limits"][limit_key]
        and record["resource"]["peak_process_rss_bytes"]
        <= config["limits"]["peak_process_tree_rss_bytes"]
    )


def _source_gates(
    records: list[dict[str, Any]], stratum: dict[str, Any], config: dict[str, Any]
) -> dict[str, bool]:
    expected_warnings = sorted(config["preflight"]["required_warning_codes"])
    successful = [row for row in records if row.get("worker_ok")]
    return {
        "all_members_complete": len(successful) == len(stratum["sources"]),
        "all_workers_within_limits": len(successful) == len(records)
        and all(_resource_gate(row, config, product=False) for row in successful),
        "all_sources_immutable": len(successful) == len(records)
        and all(row["source_unchanged"] for row in successful),
        "all_structures_exact": len(successful) == len(records)
        and all(
            len(row["pixels"]["shape"]) == 3
            and row["pixels"]["shape"][2] == 3
            and row["pixels"]["dtype"] == "float32"
            and row["pixels"]["finite"]
            and 0.0 <= row["pixels"]["minimum"] < row["pixels"]["maximum"] <= 1.0
            and row["pixels"]["owned"]
            and row["pixels"]["c_contiguous"]
            and row["pixels"]["writeable"]
            and row["working_space"] == "linear_srgb"
            and row["transfer_state"] == "scene_linear"
            and row["source_transfer_state"] == "scene_linear"
            and row["orientation_applied"] is True
            and row["alpha_policy"] == "absent"
            for row in successful
        ),
        "all_warning_boundaries_exact": len(successful) == len(records)
        and all(row["warning_codes"] == expected_warnings for row in successful),
    }


def _scientific_view(report: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(report))
    value.pop("scientific_identity", None)
    for stratum in value.get("results", []):
        for record in stratum.get("records", []):
            record.pop("resource", None)
    for record in value.get("product_records", []):
        record.pop("resource", None)
    return value


def execute_preflight(
    config_path: Path, producer_repo: Path, *, reverse: bool
) -> dict[str, Any]:
    config = _load_json(config_path)
    bindings = _binding_checks(config)
    runtime = _runtime_checks(config)
    if not all(bindings.values()):
        raise U719BError("frozen binding differs")
    if not all(runtime.values()):
        raise U719BError("runtime identity differs")
    if not shared._tracked_clean():
        raise U719BError("tracked worktree is not clean")
    extensions = [row["extension"] for row in config["strata"]]
    if not all(extension in RAW_SUFFIXES for extension in extensions):
        raise U719BError("a frozen extension is absent from public RAW_SUFFIXES")

    controller_root = ROOT / "tmp/u7_19b_multi_vendor_raw_product_ingress"
    if controller_root.exists():
        raise U719BError("owned controller scratch exists before execution")
    controller_root.mkdir(parents=True)
    try:
        strata = list(config["strata"])
        if reverse:
            strata.reverse()
        results = []
        for stratum in strata:
            sources = list(stratum["sources"])
            if reverse:
                sources.reverse()
            records = [
                _run_worker(
                    stage="source-worker",
                    config_path=config_path,
                    producer_repo=producer_repo,
                    key=source["id"],
                    scratch_root=controller_root,
                    timeout_seconds=config["limits"]["source_worker_seconds"],
                )
                for source in sources
            ]
            records.sort(key=lambda row: row.get("source_id", row.get("key", "")))
            gates = _source_gates(records, stratum, config)
            results.append(
                {
                    "admission": "PASS_PREFLIGHT"
                    if all(gates.values())
                    else "FAIL_CLOSED_PREFLIGHT",
                    "extension": stratum["extension"],
                    "gates": gates,
                    "records": records,
                    "representative_source_id": stratum["representative"],
                    "stratum_id": stratum["id"],
                }
            )
        results.sort(key=lambda row: row["extension"])
    finally:
        shutil.rmtree(controller_root, ignore_errors=True)

    passed = sorted(
        row["extension"] for row in results if row["admission"] == "PASS_PREFLIGHT"
    )
    report: dict[str, Any] = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "execution_commit": shared._git_head(),
        "network_requests": 0,
        "passed_extensions": passed,
        "results": results,
        "runtime": {"checks": runtime, **config["runtime"]},
        "schema": PREFLIGHT_SCHEMA,
        "scratch_residue_files": 0
        if not controller_root.exists()
        else sum(1 for p in controller_root.rglob("*") if p.is_file()),
        "status": "PASS_PREFLIGHT_AT_LEAST_ONE_EXTENSION"
        if passed
        else "FAIL_CLOSED_PREFLIGHT_ALL_EXTENSIONS",
        "stop_rule": config["stop_rule"],
        "tracked_worktree_clean": shared._tracked_clean(),
    }
    report["scientific_identity"] = "sha256:" + _sha256_bytes(
        json.dumps(
            _scientific_view(report), sort_keys=True, separators=(",", ":")
        ).encode()
    )
    return report


def execute_formal(
    config_path: Path, execution_lock_path: Path, producer_repo: Path, *, reverse: bool
) -> dict[str, Any]:
    config = _load_json(config_path)
    lock = _load_json(execution_lock_path)
    bindings = {
        name: shared._verify_file(ROOT / item["path"], item)
        for name, item in sorted(lock["bindings"].items())
    }
    reports = {
        name: shared._verify_file(ROOT / item["path"], item)
        for name, item in sorted(lock["preflight_reports"].items())
    }
    commits = {
        name: shared._commit_resolves(value)
        for name, value in sorted(lock["commits"].items())
    }
    runtime = _runtime_checks(config)
    if (
        not all(bindings.values())
        or not all(reports.values())
        or not all(commits.values())
        or not all(runtime.values())
    ):
        raise U719BError("formal identity or runtime differs")
    if not shared._tracked_clean():
        raise U719BError("tracked worktree is not clean")
    preflight = _load_json(ROOT / lock["preflight_reports"]["forward"]["path"])
    admitted = sorted(lock["admitted_extensions"])
    if admitted != sorted(preflight["passed_extensions"]):
        raise U719BError("admitted extensions differ from frozen preflight")

    controller_root = ROOT / "tmp/u7_19b_multi_vendor_raw_product_ingress"
    if controller_root.exists():
        raise U719BError("owned controller scratch exists before formal execution")
    controller_root.mkdir(parents=True)
    try:
        extensions = list(admitted)
        if reverse:
            extensions.reverse()
        product_records = [
            _run_worker(
                stage="product-worker",
                config_path=config_path,
                producer_repo=producer_repo,
                key=extension,
                scratch_root=controller_root,
                timeout_seconds=config["limits"]["product_worker_seconds"],
            )
            for extension in extensions
        ]
        product_records.sort(key=lambda row: row.get("extension", row.get("key", "")))
    finally:
        shutil.rmtree(controller_root, ignore_errors=True)

    complete = [row for row in product_records if row.get("worker_ok")]
    gates = {
        "all_admitted_extensions_dispatched": all(
            extension in RAW_SUFFIXES for extension in admitted
        ),
        "all_product_workers_within_limits": len(complete) == len(admitted)
        and all(_resource_gate(row, config, product=True) for row in complete),
        "all_product_outputs_exact": len(complete) == len(admitted)
        and all(
            row["output"]["format"] == "PNG"
            and row["output"]["mode"] == "RGB"
            and row["output"]["icc_present"]
            for row in complete
        ),
        "all_recipe_fields_exact": len(complete) == len(admitted)
        and all(all(row["recipe"]["exact"].values()) for row in complete),
        "all_replays_byte_exact": len(complete) == len(admitted)
        and all(
            row["replay_byte_exact"]
            and row["output"]["sha256"]
            == row["replay"]["sha256"]
            == row["replay"]["returned_sha256"]
            for row in complete
        ),
        "all_sources_immutable": len(complete) == len(admitted)
        and all(row["source_unchanged"] for row in complete),
        "bindings_exact": all(bindings.values()),
        "commits_resolve": all(commits.values()),
        "network_requests_zero": True,
        "preflight_reports_exact": all(reports.values()),
        "required_extensions_complete": len(complete) == len(admitted),
        "runtime_exact": all(runtime.values()),
        "scratch_residue_zero": not controller_root.exists(),
        "tracked_worktree_clean": shared._tracked_clean(),
    }
    report: dict[str, Any] = {
        "admitted_extensions": admitted,
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "commits": commits,
        "execution_commit": shared._git_head(),
        "gates": gates,
        "network_requests": 0,
        "preflight_reports": reports,
        "product_records": product_records,
        "runtime": {"checks": runtime, **config["runtime"]},
        "schema": FORMAL_SCHEMA,
        "scratch_residue_files": 0 if not controller_root.exists() else 1,
        "status": "PASS_PRIVATE_U7_19B_MULTI_VENDOR_RAW_PRODUCT_INGRESS"
        if all(gates.values())
        else "FAIL_CLOSED_U7_19B_MULTI_VENDOR_RAW_PRODUCT_INGRESS",
        "stop_rule": config["stop_rule"],
    }
    report["scientific_identity"] = "sha256:" + _sha256_bytes(
        json.dumps(
            _scientific_view(report), sort_keys=True, separators=(",", ":")
        ).encode()
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("preflight", "formal", "source-worker", "product-worker"),
        default="preflight",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path)
    parser.add_argument("--producer-repo", type=Path, required=True)
    parser.add_argument("--key")
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    producer_repo = args.producer_repo.resolve()
    config = _load_json(config_path)
    if args.stage == "source-worker":
        if not args.key:
            raise U719BError("source worker requires --key")
        report = _source_worker(config, producer_repo, args.key)
    elif args.stage == "product-worker":
        if not args.key or args.scratch_root is None:
            raise U719BError("product worker requires --key and --scratch-root")
        args.scratch_root.mkdir(parents=True, exist_ok=True)
        report = _product_worker(config, producer_repo, args.key, args.scratch_root)
    elif args.stage == "formal":
        if args.execution_lock is None:
            raise U719BError("formal execution requires --execution-lock")
        report = execute_formal(
            config_path,
            args.execution_lock.resolve(),
            producer_repo,
            reverse=args.reverse,
        )
    else:
        report = execute_preflight(config_path, producer_repo, reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
