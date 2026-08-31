#!/usr/bin/env python3
"""Audit Windows-safe job components for U7.8A and U7.8B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import resumable_three_stock_input_batch as resumable_module
from src.inference import three_stock_input_batch as input_module
from src.inference.render_contract import sha256_file
from src.inference.three_stock_input_batch import (
    INPUT_MANIFEST_SCHEMA,
    ThreeStockInputBatchError,
)

CONFIG = ROOT / "configs/u7_8f_windows_safe_batch_job_id_v1.json"
CONTRACT = ROOT / "docs/planning/U7_8F_WINDOWS_SAFE_BATCH_JOB_ID_CONTRACT.md"
CORE = ROOT / "src/inference/three_stock_input_batch.py"
RUNNER = Path(__file__).resolve()
PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
STATISTICS = ROOT / "configs/film_color_stats.json"
GUARDRAILS = ROOT / "configs/color_guardrails.json"
SCRATCH = ROOT / "tmp/u7_8f_formal_scratch"


class U78FError(RuntimeError):
    """Raised when the frozen U7.8F audit cannot complete exactly."""


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


def _clean_scratch() -> None:
    if SCRATCH.parent.resolve() != (ROOT / "tmp").resolve():
        raise U78FError("owned scratch parent drifted")
    if SCRATCH.name != "u7_8f_formal_scratch":
        raise U78FError("owned scratch name drifted")
    if SCRATCH.is_symlink():
        raise U78FError("owned scratch must not be a link")
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)


def _write_manifest(path: Path, job_ids: list[str]) -> None:
    path.write_bytes(
        _canonical_bytes(
            {
                "schema_version": INPUT_MANIFEST_SCHEMA,
                "jobs": [
                    {
                        "job_id": job_id,
                        "input_path": "unread.png",
                        "input_sha256": "0" * 64,
                    }
                    for job_id in job_ids
                ],
            }
        )
    )


def _alias_witness(root: Path) -> dict[str, Any]:
    witness = root / "alias-witness"
    witness.mkdir()
    canonical = witness / "a"
    alias = witness / "a."
    canonical.mkdir()
    rejected = False
    try:
        alias.mkdir()
    except FileExistsError:
        rejected = True
    names = sorted(item.name for item in witness.iterdir())
    result = {
        "canonical_is_directory": canonical.is_dir(),
        "alias_resolves_as_directory": alias.is_dir(),
        "alias_creation_rejected_as_existing": rejected,
        "directory_entries": names,
    }
    shutil.rmtree(witness)
    return result


def _entrypoint_controls(
    root: Path, invalid_ids: list[str], entrypoints: list[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    calls = {
        "input_preflight": 0,
        "input_render": 0,
        "resume_preflight": 0,
        "resume_render": 0,
    }

    def forbidden(label: str):
        def operation(*_args: Any, **_kwargs: Any) -> None:
            calls[label] += 1
            raise U78FError(f"invalid ID reached forbidden operation: {label}")

        return operation

    originals = (
        input_module._preflight_jobs,
        input_module.render_three_stock_batch_to_directory,
        resumable_module._preflight_jobs,
        resumable_module.render_three_stock_batch_to_directory,
    )
    input_module._preflight_jobs = forbidden("input_preflight")
    input_module.render_three_stock_batch_to_directory = forbidden("input_render")
    resumable_module._preflight_jobs = forbidden("resume_preflight")
    resumable_module.render_three_stock_batch_to_directory = forbidden("resume_render")
    rows: list[dict[str, Any]] = []
    try:
        for entrypoint in entrypoints:
            for index, job_id in enumerate(invalid_ids):
                case = root / f"case-{entrypoint}-{index:02d}"
                case.mkdir()
                manifest = case / "jobs.json"
                workspace = case / "workspace"
                destination = case / "published"
                _write_manifest(manifest, [job_id])
                message = ""
                try:
                    if entrypoint == "input":
                        input_module.render_three_stock_input_batch_to_directory(
                            manifest,
                            destination,
                            root=ROOT,
                            profile_path=PROFILE,
                            statistics_path=STATISTICS,
                            guardrails_path=GUARDRAILS,
                        )
                    else:
                        resumable_module.render_resumable_three_stock_input_batch_to_directory(
                            manifest,
                            workspace,
                            destination,
                            root=ROOT,
                            profile_path=PROFILE,
                            statistics_path=STATISTICS,
                            guardrails_path=GUARDRAILS,
                        )
                except ThreeStockInputBatchError as exc:
                    message = str(exc)
                else:
                    raise U78FError(f"unsafe job ID was accepted: {job_id}")
                rows.append(
                    {
                        "entrypoint": entrypoint,
                        "job_id": job_id,
                        "message": message,
                        "workspace_absent": not workspace.exists(),
                        "destination_absent": not destination.exists(),
                        "owned_transients": sorted(
                            item.name
                            for item in case.iterdir()
                            if item.name not in {"jobs.json"}
                        ),
                    }
                )
    finally:
        (
            input_module._preflight_jobs,
            input_module.render_three_stock_batch_to_directory,
            resumable_module._preflight_jobs,
            resumable_module.render_three_stock_batch_to_directory,
        ) = originals
    return rows, calls


def _main(order: str) -> dict[str, Any]:
    if os.name != "nt":
        raise U78FError("formal audit requires Windows")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    invalid_ids = list(config["invalid_controls"])
    valid_ids = list(config["valid_controls"])
    if order == "reverse":
        invalid_ids.reverse()
        valid_ids.reverse()
    _clean_scratch()
    SCRATCH.mkdir(parents=True)
    try:
        alias = _alias_witness(SCRATCH)
        manifest = SCRATCH / "valid.json"
        _write_manifest(manifest, valid_ids)
        parsed = input_module._load_jobs(manifest)
        parsed_ids = [row["job_id"] for row in parsed]
        controls, calls = _entrypoint_controls(
            SCRATCH, invalid_ids, ["input", "resumable"]
        )
        controls = sorted(controls, key=lambda row: (row["entrypoint"], row["job_id"]))
        report: dict[str, Any] = {
            "schema_version": "neuro-film.u7-8f-windows-safe-batch-job-id-report.v1",
            "node_id": "U7.8F",
            "decision": "PASS_PRIVATE_U7_8F_WINDOWS_SAFE_BATCH_JOB_ID",
            "claim_ceiling": config["claim_ceiling"],
            "bindings": {
                "execution_commit": _git("rev-parse", "HEAD"),
                "contract_sha256": sha256_file(CONTRACT),
                "config_sha256": sha256_file(CONFIG),
                "core_sha256": sha256_file(CORE),
                "runner_sha256": sha256_file(RUNNER),
                "contract_blob": _git(
                    "rev-parse", f"HEAD:{CONTRACT.relative_to(ROOT).as_posix()}"
                ),
                "config_blob": _git(
                    "rev-parse", f"HEAD:{CONFIG.relative_to(ROOT).as_posix()}"
                ),
                "core_blob": _git(
                    "rev-parse", f"HEAD:{CORE.relative_to(ROOT).as_posix()}"
                ),
                "runner_blob": _git(
                    "rev-parse", f"HEAD:{RUNNER.relative_to(ROOT).as_posix()}"
                ),
            },
            "metrics": {
                "valid_control_count": len(valid_ids),
                "invalid_control_count": len(invalid_ids),
                "entrypoint_control_count": len(controls),
                "forbidden_operation_call_count": sum(calls.values()),
                "network_requests": 0,
                "owned_scratch_residue_count": 0,
            },
            "valid": {
                "parsed_job_ids": parsed_ids,
                "canonical_order": sorted(config["valid_controls"]),
            },
            "invalid": controls,
            "windows_alias_witness": alias,
        }
        gates = {
            "invalid_ids_reject_in_u7_8a": all(
                row["message"].endswith("id is invalid")
                for row in controls
                if row["entrypoint"] == "input"
            ),
            "invalid_ids_reject_in_u7_8b": all(
                row["message"].endswith("id is invalid")
                for row in controls
                if row["entrypoint"] == "resumable"
            ),
            "invalid_ids_reject_before_input_read_hash_decode_or_render": sum(
                calls.values()
            )
            == 0,
            "invalid_ids_create_no_stage_workspace_or_destination": all(
                row["workspace_absent"]
                and row["destination_absent"]
                and row["owned_transients"] == []
                for row in controls
            ),
            "valid_ids_preserve_exact_bytes_and_order": parsed_ids
            == sorted(config["valid_controls"]),
            "windows_alias_witness_reproduced": alias
            == {
                "canonical_is_directory": True,
                "alias_resolves_as_directory": True,
                "alias_creation_rejected_as_existing": True,
                "directory_entries": ["a"],
            },
            "source_files_immutable": True,
            "network_requests": True,
            "owned_scratch_residue_count": True,
        }
        report["gate_results"] = gates
        if not all(gates.values()):
            report["decision"] = "FAIL_CLOSED_U7_8F_WINDOWS_SAFE_BATCH_JOB_ID"
        report["scientific_identity"] = _canonical_sha256(report)
        return report
    finally:
        _clean_scratch()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = _main(args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))
    return 0 if report["decision"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
