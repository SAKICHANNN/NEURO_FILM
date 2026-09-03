#!/usr/bin/env python3
"""Audit final full-scope validation of desktop batch directory publication."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.inference.product_desktop as desktop_module
from src.inference.product_desktop import ProductDesktopError, ProductDesktopWorkflow

CONFIG = ROOT / "configs/u7_21e_desktop_batch_aggregate_publication_scope_v1.json"


def _source(path: Path, offset: int) -> None:
    yy, xx = np.mgrid[:43, :61]
    rgb = np.stack(
        (
            (xx * 11 + yy * 3 + offset) % 251,
            (xx * 5 + yy * 13 + 17 + offset * 2) % 251,
            (xx * 7 + yy * 19 + 41 + offset * 3) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _workflow(root: Path) -> ProductDesktopWorkflow:
    scratch = root / "scratch"
    scratch.mkdir()
    return ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        max_preview_pixels=3_000,
        tile_size=256,
        tile_workers=1,
        png_compression=6,
    )


def _case(root: Path, case: str, config: dict[str, object]) -> dict[str, object]:
    case_root = root / case
    case_root.mkdir()
    first = case_root / "B strange SOURCE.png"
    second = case_root / "a-source.png"
    _source(first, 1)
    _source(second, 2)
    workflow = _workflow(case_root)
    _, bound = workflow.render_batch_previews((first, second), float(config["look_amount"]))
    destination = case_root / "batch"
    aggregate_staged = False
    post_aggregate_validations = 0
    original_write = desktop_module._write_bound_json
    original_validate = desktop_module._validate_runtime_source_scope

    def write_then_mark(path: Path, payload: dict[str, object]):
        nonlocal aggregate_staged
        seal = original_write(path, payload)
        if path.name == "batch.json":
            aggregate_staged = True
        return seal

    def validate(root_path: Path, commit: str, scope: tuple[str, ...]) -> None:
        nonlocal post_aggregate_validations
        if aggregate_staged:
            post_aggregate_validations += 1
            if case == "post-aggregate-drift":
                raise ProductDesktopError("runtime source scope changed")
        original_validate(root_path, commit, scope)

    error = None
    receipt = None
    with (
        mock.patch.object(desktop_module, "_write_bound_json", write_then_mark),
        mock.patch.object(desktop_module, "_validate_runtime_source_scope", validate),
    ):
        try:
            receipt = workflow.export_batch(
                bound, str(config["look_id"]), destination
            )
        except ProductDesktopError as exc:
            error = str(exc)
    row = {
        "case": case,
        "aggregate_staged": aggregate_staged,
        "post_aggregate_validations": post_aggregate_validations,
        "error": error,
        "destination_exists": destination.exists(),
        "stage_residue": sorted(
            path.name for path in case_root.glob(".batch.u7-11a-*.stage")
        ),
        "preview_cleanup": workflow.close(),
    }
    if receipt is not None:
        row.update(
            {
                "job_count": receipt.job_count,
                "claim": receipt.receipt["claim"],
                "member_names": sorted(path.name for path in destination.iterdir()),
            }
        )
    return row


def audit(scratch_root: Path, order: str) -> dict[str, object]:
    if os.name != "nt":
        raise RuntimeError("U7.21E formal directory publication requires Windows")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    requested = list(config["cases"])
    if order == "reverse":
        requested.reverse()
    with tempfile.TemporaryDirectory(prefix="u7-21e-", dir=scratch_root) as temporary:
        cases = [_case(Path(temporary), str(case), config) for case in requested]
    cases.sort(key=lambda row: str(row["case"]))
    rows = {str(row["case"]): row for row in cases}
    success = rows["success"]
    drift = rows["post-aggregate-drift"]
    gates = {
        "success_aggregate_staged": success["aggregate_staged"] is True,
        "success_final_scope_validation_once": success["post_aggregate_validations"] == 1,
        "success_published": success["destination_exists"] is True,
        "success_receipt_complete": success.get("job_count") == config["input_count"],
        "drift_aggregate_staged": drift["aggregate_staged"] is True,
        "drift_final_scope_validation_once": drift["post_aggregate_validations"] == 1,
        "drift_rejected": drift["error"] == "runtime source scope changed",
        "drift_publication_zero": drift["destination_exists"] is False,
        "stage_residue_zero": all(not row["stage_residue"] for row in cases),
        "preview_cleanup": all(row["preview_cleanup"] is True for row in cases),
        "claim_ceiling_exact": success.get("claim") == {
            "output_label": "film-inspired / Look Approximation",
            "evidence_grade": "look-approximation",
            "calibrated_stock_response": False,
            "physical_film_reproduction": False,
            "stock_distinguishability": False,
        },
    }
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    return {
        "schema": "kmcfm.u7-21e-desktop-batch-aggregate-publication-scope-result.v1",
        "status": (
            "PASS_PRIVATE_U7_21E_DESKTOP_BATCH_AGGREGATE_PUBLICATION_SCOPE"
            if all(gates.values())
            else "FAIL_CLOSED_U7_21E_DESKTOP_BATCH_AGGREGATE_PUBLICATION_SCOPE"
        ),
        "source_commit": source_commit,
        "cases": cases,
        "gates": gates,
        "claim_ceiling": config["claim"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.scratch_root.resolve(strict=True), args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    return 0 if str(report["status"]).startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
