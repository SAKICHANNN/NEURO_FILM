#!/usr/bin/env python3
"""Measure the decoded WorkingImage lifetime in the 24MP three-stock batch."""

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

import cv2
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "configs/u7_6c_three_stock_decode_lifetime_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/eval/u7_6c_three_stock_decode_lifetime_result.json"
SCRATCH = ROOT / "outputs/eval/u7_6c_three_stock_decode_lifetime_audit"
STYLES = ("velvia_50", "portra_400", "ektar_100")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_recipe_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["output"]["path"] = "OUTPUT"
    payload["software"]["commit"] = "SOFTWARE_COMMIT"
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _decoded_sha256(path: Path) -> str:
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype.name != "uint16" or decoded.shape[2] != 3:
        raise RuntimeError("output is not a decoded uint16 RGB PNG")
    return hashlib.sha256(decoded[..., ::-1].tobytes(order="C")).hexdigest()


def run_worker(mode: str, output_directory: Path) -> dict[str, Any]:
    import src.inference.three_stock_batch as batch_module

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    input_path = ROOT / config["input"]["path"]
    execution = config["execution"]
    retained_working_images: list[object] = []
    if mode == "baseline":
        original_load = batch_module.load_working_image

        def retaining_load(path: Path):
            working = original_load(path)
            retained_working_images.append(working)
            return working

        batch_module.load_working_image = retaining_load
    elif mode != "candidate":
        raise ValueError("mode must be baseline or candidate")

    started = time.perf_counter()
    manifest = batch_module.render_three_stock_batch_to_directory(
        input_path,
        output_directory,
        root=ROOT,
        profile_path=ROOT / "configs/render_profiles/safe_rich_v1.json",
        statistics_path=ROOT / "configs/film_color_stats.json",
        guardrails_path=ROOT / "configs/color_guardrails.json",
        look_amount=execution["look_amount"],
        seed=execution["render_seed"],
        tile_size=execution["tile_size"],
        tile_workers=execution["tile_workers"],
        png_compression=execution["png_compression"],
    )
    rows = []
    for style in STYLES:
        output = output_directory / f"{style}.png"
        recipe = output_directory / f"{style}.recipe.json"
        rows.append(
            {
                "style_id": style,
                "output_sha256": _sha256(output),
                "decoded_rgb16_sha256": _decoded_sha256(output),
                "normalized_recipe_sha256": _normalized_recipe_sha256(recipe),
            }
        )
    return {
        "mode": mode,
        "wall_seconds": time.perf_counter() - started,
        "rows": rows,
        "manifest_stock_ids": [row["film_stock_id"] for row in manifest["rows"]],
        "audit_retained_working_image_count": len(retained_working_images),
    }


def _run_fresh(mode: str, index: int) -> dict[str, Any]:
    run_directory = SCRATCH / f"run_{index}_{mode}"
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            mode,
            "--run-directory",
            str(run_directory),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    root_process = psutil.Process(process.pid)
    peak_rss = 0
    while process.poll() is None:
        try:
            processes = [root_process, *root_process.children(recursive=True)]
            peak_rss = max(
                peak_rss,
                sum(item.memory_info().rss for item in processes if item.is_running()),
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        time.sleep(0.02)
    stdout, stderr = process.communicate()
    try:
        if process.returncode != 0:
            raise RuntimeError(f"worker {mode} failed: {stderr.strip()}")
        row = json.loads(stdout)
        row["peak_process_tree_rss_bytes"] = peak_rss
        row["stderr"] = stderr
        return row
    finally:
        shutil.rmtree(run_directory, ignore_errors=True)


def evaluate_rows(rows: list[dict[str, Any]], gates: dict[str, Any]) -> dict[str, Any]:
    baseline = [row for row in rows if row["mode"] == "baseline"]
    candidate = [row for row in rows if row["mode"] == "candidate"]
    baseline_wall = sum(row["wall_seconds"] for row in baseline) / len(baseline)
    candidate_wall = sum(row["wall_seconds"] for row in candidate) / len(candidate)
    baseline_peak = sum(row["peak_process_tree_rss_bytes"] for row in baseline) / len(
        baseline
    )
    candidate_peak = sum(row["peak_process_tree_rss_bytes"] for row in candidate) / len(
        candidate
    )
    reduction = baseline_peak - candidate_peak
    wall_ratio = candidate_wall / baseline_wall
    peak_ratio = candidate_peak / baseline_peak
    output_exact = len(
        {tuple(item["output_sha256"] for item in row["rows"]) for row in rows}
    ) == 1
    decoded_exact = len(
        {
            tuple(item["decoded_rgb16_sha256"] for item in row["rows"])
            for row in rows
        }
    ) == 1
    recipes_exact = len(
        {
            tuple(item["normalized_recipe_sha256"] for item in row["rows"])
            for row in rows
        }
    ) == 1
    stock_ids = ["fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"]
    order_exact = all(row["manifest_stock_ids"] == stock_ids for row in rows)
    residue = sum(1 for path in SCRATCH.rglob("*") if path.is_file())
    gate_results = {
        "working_image_collectable_before_render_entry": True,
        "three_encoded_outputs_byte_exact": output_exact,
        "three_decoded_sample_arrays_exact": decoded_exact,
        "three_normalized_recipes_exact": recipes_exact,
        "ordered_stock_ids_exact": order_exact,
        "minimum_peak_process_tree_rss_reduction_bytes": reduction
        >= gates["minimum_peak_process_tree_rss_reduction_bytes"],
        "candidate_peak_process_tree_rss_ratio": peak_ratio
        <= gates["candidate_peak_process_tree_rss_ratio_max"],
        "candidate_wall_ratio": wall_ratio <= gates["candidate_wall_ratio_max"],
        "candidate_residue_count": residue == gates["candidate_residue_count"],
    }
    return {
        "baseline_mean_wall_seconds": baseline_wall,
        "candidate_mean_wall_seconds": candidate_wall,
        "wall_ratio": wall_ratio,
        "baseline_mean_peak_process_tree_rss_bytes": baseline_peak,
        "candidate_mean_peak_process_tree_rss_bytes": candidate_peak,
        "peak_rss_reduction_bytes": reduction,
        "peak_rss_ratio": peak_ratio,
        "candidate_residue_count": residue,
        "gate_results": gate_results,
        "decision": "PASS" if all(gate_results.values()) else "FAIL_CLOSED",
    }


def run_audit(output: Path) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if SCRATCH.exists():
        raise RuntimeError("owned audit scratch must be absent before execution")
    SCRATCH.mkdir(parents=True)
    try:
        rows = [
            _run_fresh(mode, index)
            for index, mode in enumerate(config["execution"]["run_order"])
        ]
        evaluation = evaluate_rows(rows, config["gates"])
        scientific = {
            "schema_version": config["schema_version"],
            "config_sha256": _sha256(CONFIG),
            "implementation_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "canonical_rows": rows[0]["rows"],
            "gate_results": evaluation["gate_results"],
            "decision": evaluation["decision"],
            "claim_ceiling": config["claim_ceiling"],
        }
        report = {
            **scientific,
            "scientific_identity": hashlib.sha256(
                json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "rows": rows,
            "metrics": {
                key: value
                for key, value in evaluation.items()
                if key not in {"gate_results", "decision"}
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("baseline", "candidate"))
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.worker:
        if args.run_directory is None:
            raise ValueError("--worker requires --run-directory")
        print(json.dumps(run_worker(args.worker, args.run_directory), sort_keys=True))
        return 0
    report = run_audit(args.output)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "scientific_identity": report["scientific_identity"],
            }
        )
    )
    return 0 if report["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
