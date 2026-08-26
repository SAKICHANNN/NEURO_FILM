#!/usr/bin/env python3
"""Audit the explicit lossless 24MP PNG16 fast-export tier."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u7_2d_fast_png16_export_v1.json"
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
STYLE_PATHS = {
    "velvia_50": ROOT / "outputs/eval/u7_2_three_stock_24mp_smoke/velvia_50.png",
    "portra_400": ROOT / "outputs/eval/u7_2_three_stock_24mp_smoke/portra_400.png",
    "ektar_100": ROOT / "outputs/eval/u7_2_three_stock_24mp_smoke/ektar_100.png",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


def _decoded_rgb16(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None or value.dtype != np.uint16 or value.ndim != 3:
        raise RuntimeError(f"invalid RGB16 PNG: {path}")
    return value[..., ::-1]


def _icc_sha256(path: Path) -> str:
    with Image.open(path) as image:
        profile = image.info.get("icc_profile", b"")
    return hashlib.sha256(profile).hexdigest()


def _run_render(
    *,
    source: Path,
    style: str,
    output: Path,
    compression: int | None,
) -> dict[str, object]:
    if output.exists():
        raise RuntimeError(f"scratch output exists: {output}")
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
        "512",
        "--tile-workers",
        "8",
        "--output",
        str(output),
    ]
    if compression is not None:
        command.extend(
            ["--png-compression", str(compression), "--write-recipe"]
        )
    started = time.perf_counter()
    child = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    process = psutil.Process(child.pid)
    peak = 0
    while child.poll() is None:
        peak = max(peak, _tree_rss(process))
        time.sleep(0.02)
    _, stderr = child.communicate()
    wall = time.perf_counter() - started
    if child.returncode != 0:
        raise RuntimeError(f"render failed for {style}: {stderr}")
    result: dict[str, object] = {
        "style": style,
        "compression": 6 if compression is None else compression,
        "wall_seconds": wall,
        "peak_process_tree_rss_bytes": peak,
        "png_bytes": output.stat().st_size,
        "png_sha256": _sha256_file(output),
        "icc_sha256": _icc_sha256(output),
    }
    recipe_path = output.with_suffix(".recipe.json")
    if compression is not None:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        result["recipe_schema_id"] = recipe["schema_id"]
        result["recipe_png_compression"] = recipe["output"]["png_compression"]
        result["recipe_sha256"] = _sha256_file(recipe_path)
    result["samples"] = _decoded_rgb16(output)
    output.unlink()
    recipe_path.unlink(missing_ok=True)
    output.with_suffix(".metrics.json").unlink(missing_ok=True)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["experiment_id"] != "U7.2D":
        raise RuntimeError("U7.2D contract drifted")
    source = ROOT / contract["source"]["path"]
    if _sha256_file(source) != contract["source"]["sha256"]:
        raise RuntimeError("U7.2D source identity drifted")
    args.scratch.mkdir(parents=True, exist_ok=True)
    if any(args.scratch.iterdir()):
        raise RuntimeError("U7.2D scratch must start empty")

    sequence = [None, 0, 0, None]
    if args.order == "reverse":
        sequence.reverse()
    performance_rows = []
    candidate_hashes: list[str] = []
    baseline_hashes: list[str] = []
    frozen_velvia = STYLE_PATHS["velvia_50"]
    frozen_velvia_samples = _decoded_rgb16(frozen_velvia)
    for index, compression in enumerate(sequence):
        row = _run_render(
            source=source,
            style="velvia_50",
            output=args.scratch / f"performance_{index}.png",
            compression=compression,
        )
        samples = row.pop("samples")
        row["decoded_samples_exact"] = np.array_equal(
            samples, frozen_velvia_samples
        )
        if compression is None:
            baseline_hashes.append(str(row["png_sha256"]))
        else:
            candidate_hashes.append(str(row["png_sha256"]))
        performance_rows.append(row)

    stock_rows = []
    for style in ("portra_400", "ektar_100"):
        frozen = STYLE_PATHS[style]
        row = _run_render(
            source=source,
            style=style,
            output=args.scratch / f"{style}.png",
            compression=0,
        )
        samples = row.pop("samples")
        row["decoded_samples_exact"] = np.array_equal(
            samples, _decoded_rgb16(frozen)
        )
        row["baseline_png_bytes"] = frozen.stat().st_size
        row["file_size_ratio"] = row["png_bytes"] / frozen.stat().st_size
        stock_rows.append(row)

    baseline_rows = [row for row in performance_rows if row["compression"] == 6]
    candidate_rows = [row for row in performance_rows if row["compression"] == 0]
    baseline_median = statistics.median(
        float(row["wall_seconds"]) for row in baseline_rows
    )
    candidate_median = statistics.median(
        float(row["wall_seconds"]) for row in candidate_rows
    )
    velvia_size_ratio = (
        int(candidate_rows[0]["png_bytes"]) / frozen_velvia.stat().st_size
    )
    all_rows = [*candidate_rows, *stock_rows]
    gates = {
        "all_three_decoded_rgb16_samples_exact": all(
            bool(row["decoded_samples_exact"])
            for row in [candidate_rows[0], *stock_rows]
        ),
        "all_three_srgb_icc_profiles_exact": len(
            {str(row["icc_sha256"]) for row in all_rows}
        )
        == 1,
        "candidate_repeat_output_bytes_exact": len(set(candidate_hashes)) == 1,
        "wall_ratio": candidate_median / baseline_median
        <= contract["gates"]["maximum_candidate_to_baseline_median_wall_ratio"],
        "file_size_ratio": max(
            [velvia_size_ratio, *[float(row["file_size_ratio"]) for row in stock_rows]]
        )
        <= contract["gates"]["maximum_candidate_to_baseline_file_size_ratio"],
        "default_encoder_output_bytes_unchanged": len(set(baseline_hashes)) == 1
        and baseline_hashes[0] == _sha256_file(frozen_velvia),
        "recipe_records_exact_png_compression": all(
            row.get("recipe_schema_id") == "kmcfm.render-recipe.v3"
            and row.get("recipe_png_compression") == 0
            for row in all_rows
        ),
        "scratch_empty": not any(args.scratch.iterdir()),
    }
    stable_core = {
        "experiment_id": contract["experiment_id"],
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": contract["source"]["sha256"],
        "candidate_hashes": sorted(set(candidate_hashes)),
        "baseline_hashes": sorted(set(baseline_hashes)),
        "stock_candidate_hashes": {
            row["style"]: row["png_sha256"] for row in stock_rows
        },
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL_CLOSED",
    }
    report = {
        "schema": "neuro-film.u7-2d-fast-png16-export-audit.v1",
        "order": args.order,
        "contract_sha256": _sha256_file(CONTRACT),
        "performance_rows": performance_rows,
        "stock_rows": stock_rows,
        "performance": {
            "baseline_median_seconds": baseline_median,
            "candidate_median_seconds": candidate_median,
            "candidate_to_baseline_wall_ratio": candidate_median / baseline_median,
            "velvia_file_size_ratio": velvia_size_ratio,
        },
        **stable_core,
        "stable_identity": hashlib.sha256(
            json.dumps(
                stable_core,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
        "claim_ceiling": contract["claim_ceiling"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
