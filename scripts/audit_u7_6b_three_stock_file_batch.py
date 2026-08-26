#!/usr/bin/env python3
"""Audit the 24MP one-decode three-stock file batch against three CLI renders."""

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
CONFIG = ROOT / "configs/u7_6b_three_stock_file_batch_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/eval/u7_6b_three_stock_file_batch_result.json"
SCRATCH = ROOT / "outputs/eval/u7_6b_three_stock_file_batch_audit"
STYLES = ("velvia_50", "portra_400", "ektar_100")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_recipe_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["output"]["path"] = "OUTPUT"
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _decoded_sha256(path: Path) -> str:
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype.name != "uint16" or decoded.shape[2] != 3:
        raise RuntimeError("output is not a decoded uint16 RGB PNG")
    return hashlib.sha256(decoded[..., ::-1].tobytes(order="C")).hexdigest()


def run_worker(mode: str, output_directory: Path) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    input_path = ROOT / config["input"]["path"]
    execution = config["execution"]
    started = time.perf_counter()
    if mode == "baseline":
        output_directory.mkdir(parents=True)
        for style in STYLES:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/render_film.py"),
                    str(input_path),
                    "--style",
                    style,
                    "--use-render-profile",
                    "--seed",
                    str(execution["render_seed"]),
                    "--tile-size",
                    str(execution["tile_size"]),
                    "--tile-workers",
                    str(execution["tile_workers"]),
                    "--output-bit-depth",
                    "16",
                    "--png-compression",
                    str(execution["png_compression"]),
                    "--write-recipe",
                    "--output",
                    str(output_directory / f"{style}.png"),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
    elif mode == "candidate":
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/render_three_stock_batch.py"),
                str(input_path),
                str(output_directory),
                "--seed",
                str(execution["render_seed"]),
                "--tile-size",
                str(execution["tile_size"]),
                "--tile-workers",
                str(execution["tile_workers"]),
                "--png-compression",
                str(execution["png_compression"]),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        raise ValueError("mode must be baseline or candidate")
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
    }


def _run_fresh(mode: str, index: int) -> dict[str, Any]:
    run_directory = SCRATCH / f"run_{index}_{mode}"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        mode,
        "--run-directory",
        str(run_directory),
    ]
    process = subprocess.Popen(
        command,
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
    output_exact = len(
        {
            tuple(item["output_sha256"] for item in row["rows"])
            for row in rows
        }
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
    order_exact = all([item["style_id"] for item in row["rows"]] == list(STYLES) for row in rows)
    wall_ratio = candidate_wall / baseline_wall
    peak_ratio = candidate_peak / baseline_peak
    residue = sum(1 for _ in SCRATCH.rglob("*") if _.is_file()) if SCRATCH.exists() else 0
    gate_results = {
        "three_encoded_outputs_byte_exact": output_exact,
        "three_decoded_sample_arrays_exact": decoded_exact,
        "three_normalized_recipes_exact": recipes_exact,
        "ordered_stock_ids_exact": order_exact,
        "median_wall_ratio": wall_ratio <= gates["median_wall_ratio_max"],
        "candidate_peak_process_tree_rss_ratio": peak_ratio
        <= gates["candidate_peak_process_tree_rss_ratio_max"],
        "candidate_residue_count": residue == gates["candidate_residue_count"],
    }
    return {
        "baseline_mean_wall_seconds": baseline_wall,
        "candidate_mean_wall_seconds": candidate_wall,
        "wall_ratio": wall_ratio,
        "baseline_mean_peak_process_tree_rss_bytes": baseline_peak,
        "candidate_mean_peak_process_tree_rss_bytes": candidate_peak,
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
            for index, mode in enumerate(("baseline", "candidate", "candidate", "baseline"))
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
        scientific_id = hashlib.sha256(
            json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        report = {
            **scientific,
            "scientific_identity": scientific_id,
            "rows": rows,
            "metrics": {key: value for key, value in evaluation.items() if key != "gate_results"},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    print(json.dumps({"decision": report["decision"], "scientific_identity": report["scientific_identity"]}))
    return 0 if report["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
