#!/usr/bin/env python3
"""Run the frozen U7.3G full-input direct-preview comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.three_stock_preview import preview_fidelity_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise RuntimeError(f"failed to decode RGB image: {path}")
    maximum = float(np.iinfo(bgr.dtype).max)
    return np.ascontiguousarray(bgr[..., ::-1].astype(np.float32) / maximum)


def _run_preview(config: dict[str, Any], destination: Path) -> tuple[dict[str, Any], float]:
    render = config["render"]
    started = time.perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_three_stock_preview.py"),
            str(ROOT / config["source"]["path"]),
            str(destination),
            "--max-preview-pixels",
            str(render["max_preview_pixels"]),
            "--look-amount",
            str(render["look_amount"]),
            "--seed",
            str(render["seed"]),
            "--tile-size",
            str(render["tile_size"]),
            "--tile-workers",
            str(render["tile_workers"]),
            "--png-compression",
            str(render["png_compression"]),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wall_seconds = time.perf_counter() - started
    return json.loads(completed.stdout), wall_seconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u7_3g_three_stock_direct_preview_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_path = ROOT / config["source"]["path"]
    if _sha256(source_path) != config["source"]["sha256"]:
        raise RuntimeError("source SHA-256 mismatch")
    for row in config["reference_outputs"].values():
        if _sha256(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError("reference output SHA-256 mismatch")

    ROOT.joinpath("tmp").mkdir(exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="u7_3g_", dir=ROOT / "tmp"))
    try:
        first, first_wall = _run_preview(config, scratch / "forward")
        second, second_wall = _run_preview(config, scratch / "reverse")
        first_hashes = {row["style_id"]: row["output_sha256"] for row in first["rows"]}
        second_hashes = {
            row["style_id"]: row["output_sha256"] for row in second["rows"]
        }
        rows: list[dict[str, Any]] = []
        for style in sorted(config["reference_outputs"]):
            candidate = _load_rgb(scratch / "forward" / f"{style}.preview.png")
            reference = _load_rgb(ROOT / config["reference_outputs"][style]["path"])
            reference = np.ascontiguousarray(
                cv2.resize(
                    reference,
                    (first["preview_width"], first["preview_height"]),
                    interpolation=cv2.INTER_AREA,
                ),
                dtype=np.float32,
            )
            rows.append(
                {
                    "style_id": style,
                    "candidate_sha256": first_hashes[style],
                    **preview_fidelity_metrics(candidate, reference),
                }
            )

        gates_config = config["gates"]
        development_gates = {
            "preview_pixel_range": (
                gates_config["minimum_preview_pixels"]
                <= first["preview_pixels"]
                <= gates_config["maximum_preview_pixels"]
            ),
            "rgb_rmse": max(row["rgb_rmse"] for row in rows)
            <= gates_config["maximum_rgb_rmse_vs_downsampled_full_output"],
            "rgb_absolute_error_p95": max(
                row["rgb_absolute_error_p95"] for row in rows
            )
            <= gates_config["maximum_rgb_absolute_error_p95"],
            "new_boundary_fraction": max(
                row["new_boundary_fraction"] for row in rows
            )
            <= gates_config["maximum_new_boundary_fraction"],
            "fresh_process_file_identity": first_hashes == second_hashes,
            "all_three_distinct": len(set(first_hashes.values())) == 3,
            "development_wall": max(first_wall, second_wall)
            <= gates_config["maximum_three_look_development_wall_seconds"],
        }
        product_latency_gate = (
            max(first_wall, second_wall)
            <= gates_config["product_warm_preview_target_seconds"]
        )
        development_pass = all(development_gates.values())
        status = (
            "PASS_DEVELOPMENT_DIRECT_PREVIEW_PRODUCT_LATENCY_OPEN"
            if development_pass and not product_latency_gate
            else "PASS_PRODUCT_DIRECT_PREVIEW"
            if development_pass
            else "FAIL_CLOSED_DIRECT_PREVIEW"
        )
        report = {
            "schema": "kmcfm.u7-3g-three-stock-direct-preview-result.v1",
            "status": status,
            "contract_sha256": _sha256(args.config),
            "source": config["source"],
            "preview": {
                "width": first["preview_width"],
                "height": first["preview_height"],
                "pixels": first["preview_pixels"],
                "basis": first["preview_basis"],
            },
            "runs": [
                {"label": "forward", "wall_seconds": first_wall},
                {"label": "reverse", "wall_seconds": second_wall},
            ],
            "rows": rows,
            "gates": {
                **development_gates,
                "product_warm_preview_target": product_latency_gate,
            },
            "output_hashes": first_hashes,
            "claim_ceiling": config["claim_ceiling"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if development_pass else 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
