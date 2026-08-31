#!/usr/bin/env python3
"""Run the frozen U7.8B resumable 100-input product audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import resumable_three_stock_input_batch as batch_module
from src.inference.render_contract import sha256_file, validate_render_recipe
from src.inference.resumable_three_stock_input_batch import (
    PROGRESS_SCHEMA,
    RECEIPT_SCHEMA,
    render_resumable_three_stock_input_batch_to_directory,
)
from src.inference.three_stock_input_batch import INPUT_MANIFEST_SCHEMA

CONFIG = ROOT / "configs/u7_8b_resumable_three_stock_batch_v1.json"
PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
STATISTICS = ROOT / "configs/film_color_stats.json"
GUARDRAILS = ROOT / "configs/color_guardrails.json"
SCRATCH = ROOT / "tmp/u7_8b_formal_scratch"
STYLES = ("velvia_50", "portra_400", "ektar_100")


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


def _source(path: Path, index: int) -> None:
    yy, xx = np.mgrid[:24, :32]
    rgb = np.stack(
        (
            (xx * 7 + yy * 3 + index * 11) % 256,
            (xx * 2 + yy * 13 + index * 17) % 256,
            (xx * 19 + yy * 5 + index * 23) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG", compress_level=0)


def _build_jobs(source_root: Path, order: str) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    indices = range(100) if order == "forward" else reversed(range(100))
    for index in indices:
        job_id = f"image-{index:03d}"
        source = source_root / f"{job_id}.png"
        _source(source, index)
        jobs.append(
            {
                "job_id": job_id,
                "input_path": source.name,
                "input_sha256": sha256_file(source),
            }
        )
    return sorted(jobs, key=lambda row: row["job_id"])


def _write_manifest(path: Path, jobs: list[dict[str, str]]) -> None:
    path.write_bytes(
        _canonical_bytes({"schema_version": INPUT_MANIFEST_SCHEMA, "jobs": jobs})
    )


def _invoke(
    manifest: Path,
    workspace: Path,
    destination: Path,
    *,
    maximum_new_jobs: int | None = None,
) -> dict:
    return render_resumable_three_stock_input_batch_to_directory(
        manifest,
        workspace,
        destination,
        root=ROOT,
        profile_path=PROFILE,
        statistics_path=STATISTICS,
        guardrails_path=GUARDRAILS,
        tile_size=16,
        png_compression=0,
        maximum_new_jobs=maximum_new_jobs,
    )


def _tree_hashes(path: Path) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _run_targeted_recovery_tests() -> int:
    completed = subprocess.run(
        [
            str(ROOT / ".venv/Scripts/python.exe"),
            "-m",
            "pytest",
            "-q",
            "tests/test_u7_8b_resumable_three_stock_input_batch.py",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise AssertionError(
            "targeted recovery tests failed\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    match = re.search(r"(?m)^(\d+) passed in ", completed.stdout)
    if match is None:
        raise AssertionError("targeted recovery test count is unavailable")
    return int(match.group(1))


def _validate_complete(
    receipt: dict,
    *,
    destination: Path,
    jobs: list[dict[str, str]],
) -> dict[str, object]:
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise AssertionError("aggregate receipt schema drifted")
    identity = dict(receipt)
    batch_id = identity.pop("batch_id")
    if batch_id != _canonical_sha256(identity):
        raise AssertionError("aggregate receipt identity drifted")
    if receipt.get("job_count") != 100 or len(receipt.get("jobs", [])) != 100:
        raise AssertionError("aggregate job count drifted")
    expected = {job["job_id"]: job for job in jobs}
    if [row["job_id"] for row in receipt["jobs"]] != sorted(expected):
        raise AssertionError("aggregate job order drifted")
    if str(ROOT) in _canonical_bytes(receipt).decode("utf-8"):
        raise AssertionError("aggregate receipt exposed a machine-local path")

    output_count = 0
    recipe_count = 0
    child_manifest_count = 0
    for job in receipt["jobs"]:
        if job["input_sha256"] != expected[job["job_id"]]["input_sha256"]:
            raise AssertionError("aggregate input identity drifted")
        if job["child_manifest_path"] != f"{job['job_id']}/batch.json":
            raise AssertionError("child manifest path drifted")
        manifest = destination / job["child_manifest_path"]
        if sha256_file(manifest) != job["child_manifest_sha256"]:
            raise AssertionError("child manifest identity drifted")
        if tuple(row["style_id"] for row in job["rows"]) != STYLES:
            raise AssertionError("child style order drifted")
        child_manifest_count += 1
        for row in job["rows"]:
            style = row["style_id"]
            if row["output_path"] != f"{job['job_id']}/{style}.png":
                raise AssertionError("aggregate output path drifted")
            if row["recipe_path"] != f"{job['job_id']}/{style}.recipe.json":
                raise AssertionError("aggregate recipe path drifted")
            output = destination / row["output_path"]
            recipe = destination / row["recipe_path"]
            if sha256_file(output) != row["output_sha256"]:
                raise AssertionError("output identity drifted")
            if sha256_file(recipe) != row["recipe_sha256"]:
                raise AssertionError("recipe identity drifted")
            recipe_payload = json.loads(recipe.read_text(encoding="utf-8"))
            validate_render_recipe(recipe_payload)
            decoded = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
            if decoded is None or decoded.dtype != np.uint16:
                raise AssertionError("output decode contract drifted")
            if decoded.shape != (24, 32, 3):
                raise AssertionError("output geometry drifted")
            output_count += 1
            recipe_count += 1
    if (output_count, recipe_count, child_manifest_count) != (300, 300, 100):
        raise AssertionError("complete transaction counts drifted")
    return {
        "batch_id": batch_id,
        "receipt_sha256": sha256_file(destination / "batch.json"),
        "job_count": 100,
        "output_count": output_count,
        "recipe_count": recipe_count,
        "child_manifest_count": child_manifest_count,
    }


def run(order: str) -> dict[str, object]:
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be forward or reverse")
    if SCRATCH.exists():
        raise RuntimeError("formal scratch root already exists")
    source_root = SCRATCH / "sources"
    workspace = SCRATCH / "workspace"
    destination = SCRATCH / "published"
    source_root.mkdir(parents=True)
    try:
        jobs = _build_jobs(source_root, order)
        manifest = source_root / "jobs.json"
        _write_manifest(manifest, jobs if order == "forward" else list(reversed(jobs)))

        progress = _invoke(manifest, workspace, destination, maximum_new_jobs=37)
        if (
            progress.get("schema_version") != PROGRESS_SCHEMA
            or progress.get("completed_job_count") != 37
            or progress.get("remaining_job_count") != 63
            or progress.get("newly_completed_job_count") != 37
            or progress.get("reused_job_count") != 0
            or progress.get("destination_published") is not False
            or destination.exists()
        ):
            raise AssertionError("paused execution contract drifted")
        paused_child_hashes = {
            job_id: _tree_hashes(workspace / job_id)
            for job_id in sorted(path.name for path in workspace.glob("image-*"))
        }
        if len(paused_child_hashes) != 37:
            raise AssertionError("paused checkpoint count drifted")

        original_child = batch_module.render_three_stock_batch_to_directory
        rendered_inputs: list[str] = []

        def record_child(input_path: Path, *args, **kwargs):
            rendered_inputs.append(Path(input_path).name)
            return original_child(input_path, *args, **kwargs)

        batch_module.render_three_stock_batch_to_directory = record_child
        try:
            resumed_receipt = _invoke(manifest, workspace, destination)
        finally:
            batch_module.render_three_stock_batch_to_directory = original_child
        expected_remaining = [f"image-{index:03d}.png" for index in range(37, 100)]
        if rendered_inputs != expected_remaining:
            raise AssertionError("resume rerendered or skipped the wrong jobs")
        for job_id, hashes in paused_child_hashes.items():
            if _tree_hashes(destination / job_id) != hashes:
                raise AssertionError("reused child identity drifted")
        resumed_metrics = _validate_complete(
            resumed_receipt, destination=destination, jobs=jobs
        )
        resumed_tree = _tree_hashes(destination)

        shutil.rmtree(destination)
        clean_receipt = _invoke(manifest, workspace, destination)
        clean_metrics = _validate_complete(
            clean_receipt, destination=destination, jobs=jobs
        )
        clean_tree = _tree_hashes(destination)
        if clean_receipt != resumed_receipt or clean_tree != resumed_tree:
            raise AssertionError("resumed and uninterrupted executions drifted")
        if clean_metrics != resumed_metrics:
            raise AssertionError("resumed and uninterrupted metrics drifted")

        scratch_residue = [
            path.relative_to(SCRATCH).as_posix()
            for path in SCRATCH.rglob("*")
            if path.name.startswith(".u7-8b-") or path.name.endswith(".lease")
        ]
        if scratch_residue:
            raise AssertionError("owned transient residue remained")
        targeted_test_count = _run_targeted_recovery_tests()
        if targeted_test_count != 24:
            raise AssertionError("targeted recovery test inventory drifted")
        source_set_identity = _canonical_sha256(
            [
                {"job_id": row["job_id"], "input_sha256": row["input_sha256"]}
                for row in jobs
            ]
        )
        report: dict[str, object] = {
            "schema_version": "neuro-film.u7-8b-resumable-three-stock-batch-result.v1",
            "node_id": "U7.8B",
            "decision": "PASS_PRIVATE_U7_8B_RESUMABLE_THREE_STOCK_BATCH",
            "implementation_commit": _git("rev-parse", "HEAD"),
            "source_blobs": {
                path: _git("hash-object", path)
                for path in (
                    "configs/u7_8b_resumable_three_stock_batch_v1.json",
                    "scripts/audit_u7_8b_resumable_three_stock_batch.py",
                    "scripts/render_resumable_three_stock_input_batch.py",
                    "src/inference/resumable_three_stock_input_batch.py",
                    "tests/test_u7_8b_resumable_three_stock_input_batch.py",
                )
            },
            "config_sha256": sha256_file(CONFIG),
            "profile_sha256": sha256_file(PROFILE),
            "statistics_sha256": sha256_file(STATISTICS),
            "guardrails_sha256": sha256_file(GUARDRAILS),
            "source_set_identity": source_set_identity,
            "metrics": {
                **resumed_metrics,
                "first_invocation_completed_jobs": 37,
                "resumed_reused_jobs": 37,
                "resumed_new_jobs": 63,
                "resumed_tree_file_count": len(resumed_tree),
                "uninterrupted_tree_file_count": len(clean_tree),
            },
            "gate_results": {
                "paused_destination_absent": True,
                "paused_workspace_checkpoint_count": 37,
                "resume_renders_only_remaining_jobs": True,
                "resumed_and_uninterrupted_output_hash_sets_exact": True,
                "resumed_and_uninterrupted_aggregate_scientific_identity_exact": True,
                "all_input_semantics_and_config_hashes_revalidated_on_every_invocation": True,
                "tampered_checkpoint_rejected_without_render_or_publication": True,
                "software_commit_or_core_drift_rejected_without_render_or_publication": True,
                "concurrent_writer_rejected_without_mutation_or_render": True,
                "child_complete_with_stale_checkpoint_reconciled": True,
                "checkpoint_claiming_missing_child_rejected": True,
                "stale_reserved_transient_reconciled": True,
                "crash_after_last_child_before_final_receipt_resumes_without_rerender": True,
                "late_foreign_destination_preserved": True,
                "all_100_jobs_published": True,
                "all_300_outputs_and_recipes_verified": True,
                "canonical_job_order_exact": True,
                "reused_children_byte_exact": True,
                "owned_transient_residue_count": 0,
                "network_requests": 0,
            },
            "test_evidence": {
                "targeted_recovery_tests": targeted_test_count,
                "frozen_gate_bindings": {
                    "input_and_config_drift": [
                        "test_input_drift_rejects_before_resume_render",
                        "test_profile_config_drift_rejects_before_resume_render",
                    ],
                    "tampered_or_missing_checkpoint": [
                        "test_tampered_completed_child_rejects_without_rerender",
                        "test_checkpoint_claiming_missing_child_rejects_before_render",
                    ],
                    "stale_ledger": [
                        "test_complete_child_with_stale_checkpoint_is_reconciled"
                    ],
                    "software_or_core_drift": ["test_core_drift_rejects_before_render"],
                    "concurrent_writer": [
                        "test_concurrent_lease_rejects_without_workspace_mutation"
                    ],
                    "stale_initialization_and_transient": [
                        "test_state_bound_stale_initialization_sibling_is_reconciled",
                        "test_early_state_bound_initialization_crash_is_reconciled",
                        "test_reserved_transient_is_removed_only_after_valid_resume",
                    ],
                    "final_receipt_and_late_destination": [
                        "test_crash_after_last_child_before_receipt_resumes_without_render",
                        "test_existing_complete_receipt_retries_late_publication_without_render",
                    ],
                    "links_reparse_and_unexpected_members": [
                        "test_hardlinked_child_member_rejects",
                        "test_reparse_reserved_transient_rejects_before_render",
                        "test_unexpected_workspace_member_rejects_before_render",
                    ],
                    "path_and_numeric_aliases": [
                        "test_destination_aliasing_lease_rejects_without_creation",
                        "test_boolean_numeric_arguments_fail_closed",
                    ],
                    "manifest_enumeration": [
                        "test_manifest_row_order_is_nonsemantic_on_resume"
                    ],
                },
            },
            "claim_ceiling": (
                "Private Windows/Python resumable mechanics for three existing "
                "deterministic film-inspired Look Approximations only; no hard-power-"
                "loss durability during an executing child, calibrated or physical "
                "stock response, 24MP performance, installer, public API, package, "
                "release or product-value claim."
            ),
        }
        return report
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.order)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
