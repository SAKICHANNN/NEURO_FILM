#!/usr/bin/env python3
"""Measure the unchanged three-look Python CPU path at 1, 12 and 24 MP."""

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

import psutil

ROOT = Path(__file__).resolve().parents[1]


class U45AError(RuntimeError):
    """Raised when the frozen scale-matrix contract cannot be executed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _run_one(
    *, config: dict[str, Any], tier: dict[str, Any], destination: Path
) -> dict[str, Any]:
    render = config["render"]
    command = [
        sys.executable,
        str(ROOT / "scripts/render_three_stock_preview.py"),
        str(ROOT / config["source"]["path"]),
        str(destination),
        "--max-preview-pixels",
        str(tier["maximum_pixels"]),
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
        raise U45AError(
            f"{tier['tier_id']} worker failed with {child.returncode}: {stderr}"
        )
    try:
        manifest = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise U45AError(f"{tier['tier_id']} worker emitted invalid JSON") from exc
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 3:
        raise U45AError(f"{tier['tier_id']} worker did not emit three rows")
    output_hashes = {str(row["style_id"]): str(row["output_sha256"]) for row in rows}
    return {
        "wall_seconds": wall_seconds,
        "peak_process_tree_rss_bytes": peak_rss,
        "width": int(manifest["preview_width"]),
        "height": int(manifest["preview_height"]),
        "pixels": int(manifest["preview_pixels"]),
        "output_hashes": output_hashes,
        "three_outputs_distinct": len(set(output_hashes.values())) == 3,
    }


def summarize_tier(
    tier: dict[str, Any], runs: list[dict[str, Any]], maximum_repeat_ratio: float
) -> dict[str, Any]:
    """Reduce one tier without hiding per-run measurements."""

    if len(runs) != 2:
        raise U45AError("each tier requires exactly two fresh-process runs")
    walls = [float(run["wall_seconds"]) for run in runs]
    peaks = [int(run["peak_process_tree_rss_bytes"]) for run in runs]
    repeat_ratio = max(walls) / min(walls)
    mechanics = {
        "exact_repeat_output_hashes": runs[0]["output_hashes"]
        == runs[1]["output_hashes"],
        "three_outputs_distinct": all(run["three_outputs_distinct"] for run in runs),
        "expected_dimensions": all(
            run["pixels"] <= int(tier["maximum_pixels"])
            and run["pixels"] > 0
            and run["width"] > 0
            and run["height"] > 0
            for run in runs
        ),
        "repeat_wall_ratio": repeat_ratio <= maximum_repeat_ratio,
    }
    product_targets = {
        "wall_seconds": max(walls)
        <= float(tier["provisional_maximum_wall_seconds"]),
        "process_tree_rss_bytes": max(peaks)
        <= int(tier["provisional_maximum_process_tree_rss_bytes"]),
    }
    return {
        "tier_id": tier["tier_id"],
        "maximum_pixels": tier["maximum_pixels"],
        "runs": runs,
        "maximum_wall_seconds": max(walls),
        "maximum_process_tree_rss_bytes": max(peaks),
        "repeat_wall_ratio": repeat_ratio,
        "mechanical_gates": mechanics,
        "provisional_product_targets": product_targets,
        "mechanical_pass": all(mechanics.values()),
        "provisional_product_targets_pass": all(product_targets.values()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u4_5a_three_stock_windows_cpu_scale_matrix_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = ROOT / config["source"]["path"]
    if not source.is_file() or _sha256(source) != config["source"]["sha256"]:
        raise U45AError("frozen source identity mismatch")
    scratch_parent = ROOT / "tmp"
    scratch_parent.mkdir(exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="u4_5a_", dir=scratch_parent))
    summaries: list[dict[str, Any]] = []
    try:
        for tier in config["tiers"]:
            runs = []
            for run_index in range(int(config["render"]["runs_per_tier"])):
                destination = scratch / f"{tier['tier_id']}_{run_index}"
                runs.append(_run_one(config=config, tier=tier, destination=destination))
                shutil.rmtree(destination)
            summaries.append(
                summarize_tier(
                    tier,
                    runs,
                    float(config["gates"]["maximum_repeat_wall_ratio"]),
                )
            )
        residue_empty = not any(scratch.iterdir())
        all_mechanical = all(row["mechanical_pass"] for row in summaries)
        all_targets = all(
            row["provisional_product_targets_pass"] for row in summaries
        )
        status = (
            "PASS_CURRENT_WINDOWS_CPU_SCALE_MATRIX"
            if all_mechanical and all_targets
            else "PASS_MECHANICS_PRODUCT_TARGETS_OPEN"
            if all_mechanical
            else "FAIL_CLOSED_WINDOWS_CPU_SCALE_MATRIX"
        )
        report = {
            "schema": "kmcfm.u4-5a-three-stock-windows-cpu-scale-matrix-result.v1",
            "status": status,
            "contract_sha256": _sha256(args.config),
            "source": config["source"],
            "platform": {
                "sys_platform": sys.platform,
                "python": sys.version.split()[0],
            },
            "tiers": summaries,
            "gates": {
                "all_mechanical": all_mechanical,
                "all_provisional_product_targets": all_targets,
                "owned_residue_empty": residue_empty,
            },
            "claim_ceiling": config["claim_ceiling"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if all_mechanical and residue_empty else 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
