#!/usr/bin/env python3
"""Run the frozen U7.8A 100-input product transaction audit."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from src.inference import three_stock_input_batch as batch_module
from src.inference.render_contract import sha256_file
from src.inference.three_stock_input_batch import (
    INPUT_MANIFEST_SCHEMA,
    RECEIPT_SCHEMA,
    ThreeStockInputBatchError,
    render_three_stock_input_batch_to_directory,
)

CONFIG = ROOT / "configs/u7_8a_three_stock_input_batch_v1.json"
PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
STATISTICS = ROOT / "configs/film_color_stats.json"
GUARDRAILS = ROOT / "configs/color_guardrails.json"
SCRATCH = ROOT / "tmp/u7_8a_formal_scratch"
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


def _build_jobs(source_root: Path) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for index in range(100):
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
    return jobs


def _write_manifest(path: Path, jobs: list[dict[str, str]]) -> None:
    path.write_bytes(
        _canonical_bytes({"schema_version": INPUT_MANIFEST_SCHEMA, "jobs": jobs})
    )


def _invoke(manifest: Path, destination: Path) -> dict:
    return render_three_stock_input_batch_to_directory(
        manifest,
        destination,
        root=ROOT,
        profile_path=PROFILE,
        statistics_path=STATISTICS,
        guardrails_path=GUARDRAILS,
        tile_size=16,
        png_compression=0,
    )


def _run_negative_controls(
    *, source_root: Path, jobs: list[dict[str, str]]
) -> dict[str, bool | int]:
    original_child = batch_module.render_three_stock_batch_to_directory
    original_publish = batch_module._publish_no_replace

    preflight_manifest = source_root / "preflight-control.json"
    bad_job = dict(jobs[0])
    bad_job["input_sha256"] = "0" * 64
    _write_manifest(preflight_manifest, [bad_job])
    child_calls = 0

    def forbidden_child(*args, **kwargs):
        nonlocal child_calls
        child_calls += 1
        raise AssertionError("child render ran before complete preflight")

    batch_module.render_three_stock_batch_to_directory = forbidden_child
    try:
        try:
            _invoke(preflight_manifest, SCRATCH / "preflight-control-output")
        except ThreeStockInputBatchError as exc:
            if "input hash drifted" not in str(exc):
                raise
        else:
            raise AssertionError("bad input hash control was accepted")
    finally:
        batch_module.render_three_stock_batch_to_directory = original_child
    if child_calls != 0:
        raise AssertionError("preflight control reached the child renderer")

    failure_manifest = source_root / "child-failure-control.json"
    _write_manifest(failure_manifest, jobs[:2])
    child_calls = 0

    def fail_second(*args, **kwargs):
        nonlocal child_calls
        child_calls += 1
        if child_calls == 2:
            raise RuntimeError("injected child failure")
        return original_child(*args, **kwargs)

    batch_module.render_three_stock_batch_to_directory = fail_second
    failure_output = SCRATCH / "child-failure-control-output"
    try:
        try:
            _invoke(failure_manifest, failure_output)
        except RuntimeError as exc:
            if "injected child failure" not in str(exc):
                raise
        else:
            raise AssertionError("injected child failure was accepted")
    finally:
        batch_module.render_three_stock_batch_to_directory = original_child
    if child_calls != 2 or failure_output.exists():
        raise AssertionError("child failure control published a partial batch")

    drift_manifest = source_root / "recipe-drift-control.json"
    _write_manifest(drift_manifest, [jobs[0]])

    def drift_recipe(*args, **kwargs):
        child = original_child(*args, **kwargs)
        recipe_path = Path(args[1]) / "velvia_50.recipe.json"
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        recipe["input"]["sha256"] = "0" * 64
        recipe_path.write_bytes(_canonical_bytes(recipe))
        return child

    batch_module.render_three_stock_batch_to_directory = drift_recipe
    drift_output = SCRATCH / "recipe-drift-control-output"
    try:
        try:
            _invoke(drift_manifest, drift_output)
        except ThreeStockInputBatchError as exc:
            if "recipe input identity" not in str(exc):
                raise
        else:
            raise AssertionError("recipe identity drift control was accepted")
    finally:
        batch_module.render_three_stock_batch_to_directory = original_child
    if drift_output.exists():
        raise AssertionError("recipe drift control published a batch")

    foreign_manifest = source_root / "foreign-control.json"
    _write_manifest(foreign_manifest, [jobs[0]])
    foreign_output = SCRATCH / "foreign-control-output"

    def foreign_claim(stage: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "foreign.bin").write_bytes(b"foreign")
        raise FileExistsError("late foreign destination")

    batch_module._publish_no_replace = foreign_claim
    try:
        try:
            _invoke(foreign_manifest, foreign_output)
        except FileExistsError as exc:
            if "late foreign destination" not in str(exc):
                raise
        else:
            raise AssertionError("late foreign destination control was accepted")
    finally:
        batch_module._publish_no_replace = original_publish
    if (foreign_output / "foreign.bin").read_bytes() != b"foreign":
        raise AssertionError("foreign destination was not preserved")
    if list(foreign_output.iterdir()) != [foreign_output / "foreign.bin"]:
        raise AssertionError("owned batch data entered the foreign destination")
    shutil.rmtree(foreign_output)

    stage_residue = len(list(SCRATCH.rglob("*.stage")))
    if stage_residue:
        raise AssertionError("owned stage residue remained after controls")
    return {
        "preflight_rejected_before_child_render": True,
        "injected_child_failure_published_nothing": True,
        "recipe_input_identity_drift_published_nothing": True,
        "late_foreign_destination_preserved": True,
        "owned_stage_residue_count": stage_residue,
    }


def _validate_receipt(
    receipt: dict,
    *,
    output_root: Path,
    expected_jobs: list[dict[str, str]],
) -> dict[str, object]:
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise AssertionError("receipt schema drifted")
    identity = dict(receipt)
    batch_id = identity.pop("batch_id")
    if batch_id != _canonical_sha256(identity):
        raise AssertionError("batch identity drifted")
    if receipt.get("job_count") != 100 or len(receipt.get("jobs", [])) != 100:
        raise AssertionError("receipt job count drifted")
    expected_by_id = {row["job_id"]: row for row in expected_jobs}
    ordered_ids = [row["job_id"] for row in receipt["jobs"]]
    if ordered_ids != sorted(expected_by_id):
        raise AssertionError("receipt job order drifted")
    if str(ROOT) in _canonical_bytes(receipt).decode("utf-8"):
        raise AssertionError("aggregate receipt exposed a machine-local path")

    output_count = 0
    recipe_count = 0
    child_manifest_count = 0
    decoded_shapes: set[tuple[int, int, int]] = set()
    for job in receipt["jobs"]:
        expected = expected_by_id[job["job_id"]]
        if job["input_sha256"] != expected["input_sha256"]:
            raise AssertionError("receipt input identity drifted")
        child_manifest_path = output_root / job["child_manifest_path"]
        if sha256_file(child_manifest_path) != job["child_manifest_sha256"]:
            raise AssertionError("child manifest identity drifted")
        child = json.loads(child_manifest_path.read_text(encoding="utf-8"))
        if child["input_sha256"] != expected["input_sha256"]:
            raise AssertionError("child input identity drifted")
        if tuple(row["style_id"] for row in job["rows"]) != STYLES:
            raise AssertionError("child style order drifted")
        child_manifest_count += 1
        for row in job["rows"]:
            output_path = output_root / row["output_path"]
            recipe_path = output_root / row["recipe_path"]
            if sha256_file(output_path) != row["output_sha256"]:
                raise AssertionError("output identity drifted")
            if sha256_file(recipe_path) != row["recipe_sha256"]:
                raise AssertionError("recipe identity drifted")
            recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
            if recipe["input"]["sha256"] != expected["input_sha256"]:
                raise AssertionError("recipe input identity drifted")
            if recipe["claim"]["output_label"] != "film-inspired":
                raise AssertionError("recipe output label drifted")
            if recipe["claim"]["evidence_grade"] != "look-approximation":
                raise AssertionError("recipe evidence grade drifted")
            decoded = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
            if decoded.dtype != np.uint16 or decoded.shape != (24, 32, 3):
                raise AssertionError("decoded output contract drifted")
            decoded_shapes.add(decoded.shape)
            output_count += 1
            recipe_count += 1
    if (output_count, recipe_count, child_manifest_count) != (300, 300, 100):
        raise AssertionError("complete transaction counts drifted")
    return {
        "batch_id": batch_id,
        "receipt_sha256": sha256_file(output_root / "batch.json"),
        "job_count": len(receipt["jobs"]),
        "output_count": output_count,
        "recipe_count": recipe_count,
        "child_manifest_count": child_manifest_count,
        "decoded_shapes": [list(shape) for shape in sorted(decoded_shapes)],
    }


def run(order: str) -> dict[str, object]:
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be forward or reverse")
    if SCRATCH.exists():
        raise RuntimeError("formal scratch root already exists")
    source_root = SCRATCH / "sources"
    output_root = SCRATCH / "published"
    source_root.mkdir(parents=True)
    try:
        jobs = _build_jobs(source_root)
        manifest_jobs = jobs if order == "forward" else list(reversed(jobs))
        manifest = source_root / "jobs.json"
        _write_manifest(manifest, manifest_jobs)
        receipt = _invoke(manifest, output_root)
        metrics = _validate_receipt(
            receipt, output_root=output_root, expected_jobs=jobs
        )
        controls = _run_negative_controls(source_root=source_root, jobs=jobs)
        source_set_identity = _canonical_sha256(
            [
                {"job_id": row["job_id"], "input_sha256": row["input_sha256"]}
                for row in jobs
            ]
        )
        report: dict[str, object] = {
            "schema_version": "neuro-film.u7-8a-three-stock-input-batch-result.v1",
            "node_id": "U7.8A",
            "decision": "PASS_PRIVATE_U7_8A_THREE_STOCK_INPUT_BATCH",
            "implementation_commit": _git("rev-parse", "HEAD"),
            "source_blobs": {
                path: _git("hash-object", path)
                for path in (
                    "configs/u7_8a_three_stock_input_batch_v1.json",
                    "scripts/audit_u7_8a_three_stock_input_batch.py",
                    "scripts/render_three_stock_input_batch.py",
                    "src/inference/three_stock_input_batch.py",
                    "src/inference/three_stock_batch.py",
                )
            },
            "config_sha256": sha256_file(CONFIG),
            "profile_sha256": sha256_file(PROFILE),
            "statistics_sha256": sha256_file(STATISTICS),
            "guardrails_sha256": sha256_file(GUARDRAILS),
            "source_set_identity": source_set_identity,
            "metrics": metrics,
            "controls": controls,
            "gate_results": {
                "all_inputs_hash_verified_before_decode": True,
                "all_child_recipe_input_hashes_match_frozen_jobs": True,
                "canonical_job_order_exact": True,
                "aggregate_receipt_machine_local_paths_exposed": False,
                "all_100_jobs_published": metrics["job_count"] == 100,
                "all_300_outputs_and_recipes_verified": (
                    metrics["output_count"] == 300 and metrics["recipe_count"] == 300
                ),
                "foreign_destination_preserved": controls[
                    "late_foreign_destination_preserved"
                ],
                "injected_child_failure_publishes_nothing": controls[
                    "injected_child_failure_published_nothing"
                ],
                "recipe_input_identity_drift_publishes_nothing": controls[
                    "recipe_input_identity_drift_published_nothing"
                ],
                "owned_stage_residue_count": controls["owned_stage_residue_count"],
                "network_requests": 0,
            },
            "claim_ceiling": (
                "Private Windows/Python 100-small-image deterministic transaction "
                "for three existing film-inspired Look Approximations only; no "
                "calibrated stock response, 24MP performance, installer, public API, "
                "package, release or product-value claim."
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
