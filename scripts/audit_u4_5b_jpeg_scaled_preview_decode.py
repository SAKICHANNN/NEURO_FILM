#!/usr/bin/env python3
"""Measure the frozen decoder-scaled JPEG three-stock preview candidate."""

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
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.three_stock_preview import (
    preview_dimensions,
    preview_fidelity_metrics,
)


class U45BError(RuntimeError):
    """Raised when the frozen scaled-decode audit cannot be executed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise U45BError(f"failed to decode RGB image: {path}")
    maximum = float(np.iinfo(bgr.dtype).max)
    return np.ascontiguousarray(bgr[..., ::-1].astype(np.float32) / maximum)


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


def _run_one(config: dict[str, Any], destination: Path) -> dict[str, Any]:
    render = config["render"]
    command = [
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
        "--jpeg-scaled-decode",
    ]
    started = time.perf_counter()
    child = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    monitored = psutil.Process(child.pid)
    peak_rss = 0
    while child.poll() is None:
        peak_rss = max(peak_rss, _tree_rss(monitored))
        time.sleep(0.02)
    stdout, stderr = child.communicate()
    wall_seconds = time.perf_counter() - started
    if child.returncode != 0:
        raise U45BError(f"preview worker failed with {child.returncode}: {stderr}")
    try:
        manifest = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise U45BError("preview worker emitted invalid JSON") from exc
    output_hashes = {
        str(row["style_id"]): str(row["output_sha256"])
        for row in manifest.get("rows", [])
    }
    if len(output_hashes) != 3:
        raise U45BError("preview worker did not emit three stock rows")
    rows: list[dict[str, Any]] = []
    for style in sorted(config["reference_outputs"]):
        candidate = _load_rgb(destination / f"{style}.preview.png")
        reference = _load_rgb(ROOT / config["reference_outputs"][style]["path"])
        reference = np.ascontiguousarray(
            cv2.resize(
                reference,
                (int(manifest["preview_width"]), int(manifest["preview_height"])),
                interpolation=cv2.INTER_AREA,
            ),
            dtype=np.float32,
        )
        rows.append(
            {
                "style_id": style,
                "output_sha256": output_hashes[style],
                **preview_fidelity_metrics(candidate, reference),
            }
        )
    return {
        "wall_seconds": wall_seconds,
        "peak_process_tree_rss_bytes": peak_rss,
        "source_width": int(manifest["source_width"]),
        "source_height": int(manifest["source_height"]),
        "decoded_width": int(manifest["decoded_width"]),
        "decoded_height": int(manifest["decoded_height"]),
        "preview_width": int(manifest["preview_width"]),
        "preview_height": int(manifest["preview_height"]),
        "preview_pixels": int(manifest["preview_pixels"]),
        "jpeg_scaled_decode": manifest["jpeg_scaled_decode"],
        "preview_basis": manifest["preview_basis"],
        "rows": rows,
        "output_hashes": output_hashes,
    }


def evaluate_runs(config: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate two measured runs against the frozen quality/resource gates."""

    if len(runs) != 2:
        raise U45BError("exactly two fresh-process runs are required")
    gates = config["gates"]
    expected_width, expected_height = preview_dimensions(
        int(config["source"]["width"]),
        int(config["source"]["height"]),
        int(config["render"]["max_preview_pixels"]),
    )
    wall_values = [float(run["wall_seconds"]) for run in runs]
    peak_values = [int(run["peak_process_tree_rss_bytes"]) for run in runs]
    repeat_ratio = max(wall_values) / min(wall_values)
    rows = [row for run in runs for row in run["rows"]]
    decisions = {
        "exact_repeat_output_hashes": runs[0]["output_hashes"]
        == runs[1]["output_hashes"],
        "three_distinct_stock_outputs": all(
            len(set(run["output_hashes"].values())) == 3 for run in runs
        ),
        "exact_preview_dimensions": all(
            run["preview_width"] == expected_width
            and run["preview_height"] == expected_height
            for run in runs
        ),
        "decoder_scaled_before_float_expansion": all(
            run["jpeg_scaled_decode"] is True
            and run["decoded_width"] < run["source_width"]
            and run["decoded_height"] < run["source_height"]
            for run in runs
        ),
        "rgb_rmse": max(float(row["rgb_rmse"]) for row in rows)
        <= float(gates["maximum_rgb_rmse_vs_downsampled_full_output"]),
        "rgb_absolute_error_p95": max(
            float(row["rgb_absolute_error_p95"]) for row in rows
        )
        <= float(gates["maximum_rgb_absolute_error_p95"]),
        "new_boundary_fraction": max(
            float(row["new_boundary_fraction"]) for row in rows
        )
        <= float(gates["maximum_new_boundary_fraction"]),
        "process_tree_rss": max(peak_values)
        <= int(gates["maximum_process_tree_rss_bytes"]),
        "wall_seconds": max(wall_values) <= float(gates["maximum_wall_seconds"]),
        "repeat_wall_ratio": repeat_ratio <= float(gates["maximum_repeat_wall_ratio"]),
    }
    return {
        "maximum_wall_seconds": max(wall_values),
        "maximum_process_tree_rss_bytes": max(peak_values),
        "repeat_wall_ratio": repeat_ratio,
        "gates": decisions,
        "pass": all(decisions.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u4_5b_jpeg_scaled_preview_decode_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = ROOT / config["source"]["path"]
    if not source.is_file() or _sha256(source) != config["source"]["sha256"]:
        raise U45BError("frozen source identity mismatch")
    for row in config["reference_outputs"].values():
        path = ROOT / row["path"]
        if not path.is_file() or _sha256(path) != row["sha256"]:
            raise U45BError("frozen reference output identity mismatch")

    scratch_parent = ROOT / "tmp"
    scratch_parent.mkdir(exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="u4_5b_", dir=scratch_parent))
    try:
        runs = []
        for index in range(int(config["render"]["runs"])):
            destination = scratch / f"run_{index}"
            runs.append(_run_one(config, destination))
            shutil.rmtree(destination)
        assessment = evaluate_runs(config, runs)
        residue_empty = not any(scratch.iterdir())
        gates = {**assessment["gates"], "owned_residue_empty": residue_empty}
        passed = all(gates.values())
        report = {
            "schema": "kmcfm.u4-5b-jpeg-scaled-preview-decode-result.v1",
            "status": (
                "PASS_JPEG_SCALED_PREVIEW_DECODE"
                if passed
                else "FAIL_CLOSED_JPEG_SCALED_PREVIEW_DECODE"
            ),
            "contract_sha256": _sha256(args.config),
            "source": config["source"],
            "platform": {
                "sys_platform": sys.platform,
                "python": sys.version.split()[0],
            },
            "runs": runs,
            "summary": {
                "maximum_wall_seconds": assessment["maximum_wall_seconds"],
                "maximum_process_tree_rss_bytes": assessment[
                    "maximum_process_tree_rss_bytes"
                ],
                "repeat_wall_ratio": assessment["repeat_wall_ratio"],
            },
            "gates": gates,
            "claim_ceiling": config["claim_ceiling"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if passed else 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
