#!/usr/bin/env python3
"""Audit exact 24MP three-stock tile-parallel files, time, and process-tree RSS."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
STYLES = ("velvia_50", "portra_400", "ektar_100")
FROZEN_EVIDENCE = ROOT / "docs/evidence/U7_2A_THREE_STOCK_EXACT_GAMUT_PARALLEL_RESULT.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=256)
    return parser.parse_args()


def _tree_rss(process: psutil.Process) -> int:
    try:
        values = [process, *process.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for value in values:
        try:
            total += int(value.memory_info().rss)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return total


def _run_one(
    *, source: Path, style: str, output: Path, workers: int, tile_size: int
) -> dict[str, object]:
    if output.exists():
        raise RuntimeError(f"scratch output already exists: {output}")
    command = [
        sys.executable,
        str(ROOT / "scripts/render_film.py"),
        str(source),
        "--style",
        style,
        "--use-render-profile",
        "--output-bit-depth",
        "16",
        "--tile-size",
        str(tile_size),
        "--tile-workers",
        str(workers),
        "--output",
        str(output),
    ]
    started = time.perf_counter()
    child = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    monitored = psutil.Process(child.pid)
    peak_rss = 0
    while child.poll() is None:
        peak_rss = max(peak_rss, _tree_rss(monitored))
        time.sleep(0.02)
    _, stderr = child.communicate()
    elapsed = time.perf_counter() - started
    if child.returncode != 0:
        raise RuntimeError(f"{style} render failed: {stderr}")
    peak_rss = max(peak_rss, _tree_rss(monitored) if psutil.pid_exists(child.pid) else 0)
    result = {
        "style": style,
        "wall_seconds": elapsed,
        "peak_process_tree_rss_bytes": peak_rss,
        "png_bytes": output.stat().st_size,
        "png_sha256": _sha256_file(output),
    }
    output.unlink()
    result["owned_output_removed"] = not output.exists()
    return result


def main() -> int:
    args = _parse_args()
    if args.workers < 2 or args.tile_size < 1:
        raise ValueError("workers must be at least 2 and tile-size must be positive")
    frozen = json.loads(FROZEN_EVIDENCE.read_text(encoding="utf-8"))
    product = frozen["product_scale_cli_replay"]
    source = ROOT / product["source_path"]
    expected = {row["style"]: row for row in product["rows"]}
    args.scratch.mkdir(parents=True, exist_ok=True)
    rows = [
        _run_one(
            source=source,
            style=style,
            output=args.scratch / f"{style}.png",
            workers=args.workers,
            tile_size=args.tile_size,
        )
        for style in STYLES
    ]
    for row in rows:
        reference = expected[row["style"]]
        row["exact_frozen_file"] = (
            row["png_sha256"] == reference["png_sha256"]
            and row["png_bytes"] == reference["png_bytes"]
        )
        row["wall_ratio_to_original_full_frame_serial"] = (
            row["wall_seconds"] / reference["baseline_seconds"]
        )
        row["wall_ratio_to_full_frame_gamut_parallel"] = (
            row["wall_seconds"] / reference["candidate_seconds"]
        )
    if not all(row["exact_frozen_file"] and row["owned_output_removed"] for row in rows):
        raise RuntimeError("exact output or cleanup gate failed")
    report = {
        "schema": "neuro-film.u7-2b-three-stock-parallel-tile-resource-audit.v1",
        "implementation_commit": "3364bd30",
        "source_path": product["source_path"],
        "source_sha256": _sha256_file(source),
        "width": product["width"],
        "height": product["height"],
        "pixels": product["pixels"],
        "tile_size": args.tile_size,
        "workers": args.workers,
        "rows": rows,
        "all_frozen_files_exact": True,
        "maximum_peak_process_tree_rss_bytes": max(
            row["peak_process_tree_rss_bytes"] for row in rows
        ),
        "total_wall_seconds": sum(row["wall_seconds"] for row in rows),
        "scratch_empty": not any(args.scratch.iterdir()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
