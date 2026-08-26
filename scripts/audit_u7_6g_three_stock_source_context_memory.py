#!/usr/bin/env python3
"""Measure scratch-backed safe-Lab source-context construction at 24MP."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_engine.safe_lab_rgb_context import (
    build_safe_lab_source_context,
    build_safe_lab_source_context_bounded,
)
from src.preprocess import normalized_icc_profile_sha256

CONFIG = ROOT / "configs/u7_6g_three_stock_source_context_memory_v1.json"
DEFAULT_OUTPUT = (
    ROOT / "outputs/eval/u7_6g_three_stock_source_context_memory_result.json"
)
SCRATCH = ROOT / "outputs/eval/u7_6g_three_stock_source_context_memory_audit"
STYLES = ("velvia_50", "portra_400", "ektar_100")
STOCK_IDS = ("fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decoded_sha256(path: Path) -> str:
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype.name != "uint16" or decoded.shape[2] != 3:
        raise RuntimeError("output is not a decoded uint16 RGB PNG")
    return hashlib.sha256(decoded[..., ::-1].tobytes(order="C")).hexdigest()


def _icc_fingerprint(path: Path) -> str:
    with Image.open(path) as image:
        profile = image.info.get("icc_profile")
        if not isinstance(profile, bytes):
            raise TypeError("output has no byte-valued ICC profile")
    return normalized_icc_profile_sha256(profile)


def _normalized_recipe_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["output"]["path"] = "OUTPUT"
    payload["output"]["sha256"] = "OUTPUT_SHA"
    payload["software"]["commit"] = "SOFTWARE_COMMIT"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_worker(mode: str, run_directory: Path) -> dict[str, Any]:
    import src.inference.three_stock_look as look_module
    from src.inference.three_stock_batch import render_three_stock_batch_to_directory

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    execution = config["execution"]
    context_rows: list[dict[str, Any]] = []
    context_scratch = run_directory / "context_scratch"

    if mode == "baseline":

        def builder(source):
            context = build_safe_lab_source_context(source)
            context_rows.append(asdict(context))
            return context

    elif mode == "candidate":

        def builder(source):
            context = build_safe_lab_source_context_bounded(
                source,
                scratch_directory=context_scratch,
                row_chunk=execution["lab_row_chunk"],
            )
            context_rows.append(asdict(context))
            return context

    else:
        raise ValueError("mode must be baseline or candidate")

    look_module.build_safe_lab_source_context = builder
    output_directory = run_directory / "output"
    started = time.perf_counter()
    manifest = render_three_stock_batch_to_directory(
        ROOT / config["input"]["path"],
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
    wall_seconds = time.perf_counter() - started
    if len(context_rows) != 1:
        raise RuntimeError("three-stock batch did not build exactly one source context")
    context_residue_count = (
        sum(1 for path in context_scratch.rglob("*") if path.is_file())
        if context_scratch.exists()
        else 0
    )
    rows = []
    for style in STYLES:
        output = output_directory / f"{style}.png"
        rows.append(
            {
                "style_id": style,
                "output_sha256": _sha256(output),
                "decoded_rgb16_sha256": _decoded_sha256(output),
                "icc_fingerprint_sha256": _icc_fingerprint(output),
                "normalized_recipe_sha256": _normalized_recipe_sha256(
                    output_directory / f"{style}.recipe.json"
                ),
            }
        )
    return {
        "mode": mode,
        "wall_seconds": wall_seconds,
        "source_context": context_rows[0],
        "context_residue_count": context_residue_count,
        "rows": rows,
        "manifest_stock_ids": [row["film_stock_id"] for row in manifest["rows"]],
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

    def identities(key: str) -> set[tuple[str, ...]]:
        return {tuple(item[key] for item in row["rows"]) for row in rows}

    context_identities = {
        json.dumps(row["source_context"], sort_keys=True, separators=(",", ":"))
        for row in rows
    }
    residue = sum(row["context_residue_count"] for row in candidate)
    gate_results = {
        "source_context_cross_execution_exact": len(context_identities) == 1,
        "three_encoded_outputs_cross_execution_exact": len(
            identities("output_sha256")
        )
        == 1,
        "three_decoded_sample_arrays_cross_execution_exact": len(
            identities("decoded_rgb16_sha256")
        )
        == 1,
        "three_icc_fingerprints_cross_execution_exact": len(
            identities("icc_fingerprint_sha256")
        )
        == 1,
        "three_normalized_recipe_semantics_cross_execution_exact": len(
            identities("normalized_recipe_sha256")
        )
        == 1,
        "ordered_stock_ids_exact": all(
            row["manifest_stock_ids"] == list(STOCK_IDS) for row in rows
        ),
        "minimum_peak_process_tree_rss_reduction_bytes": reduction
        >= gates["minimum_peak_process_tree_rss_reduction_bytes"],
        "candidate_peak_process_tree_rss_ratio": peak_ratio
        <= gates["candidate_peak_process_tree_rss_ratio_max"],
        "candidate_wall_ratio": wall_ratio <= gates["candidate_wall_ratio_max"],
        "candidate_residue_count": residue == gates["candidate_residue_count"],
    }
    return {
        "decision": "PASS" if all(gate_results.values()) else "FAIL_CLOSED",
        "gate_results": gate_results,
        "baseline_mean_wall_seconds": baseline_wall,
        "candidate_mean_wall_seconds": candidate_wall,
        "wall_ratio": wall_ratio,
        "baseline_mean_peak_process_tree_rss_bytes": baseline_peak,
        "candidate_mean_peak_process_tree_rss_bytes": candidate_peak,
        "peak_rss_reduction_bytes": reduction,
        "peak_rss_ratio": peak_ratio,
        "candidate_residue_count": residue,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker", choices=("baseline", "candidate"))
    parser.add_argument("--run-directory", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.run_directory is None:
            parser.error("--worker requires --run-directory")
        print(json.dumps(run_worker(args.worker, args.run_directory), sort_keys=True))
        return 0
    if args.run_directory is not None:
        parser.error("--run-directory requires --worker")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    decoded = cv2.imread(str(ROOT / config["input"]["path"]), cv2.IMREAD_UNCHANGED)
    dimensions = config["input"]["expected_dimensions"]
    if decoded is None or [decoded.shape[1], decoded.shape[0]] != dimensions:
        raise RuntimeError("frozen input dimensions drifted")
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    try:
        rows = [
            _run_fresh(mode, index)
            for index, mode in enumerate(config["execution"]["run_order"])
        ]
        evaluation = evaluate_rows(rows, config["gates"])
        identity_payload = {
            "decision": evaluation["decision"],
            "gate_results": evaluation["gate_results"],
            "execution_rows": [
                {
                    "mode": row["mode"],
                    "source_context": row["source_context"],
                    "rows": row["rows"],
                    "manifest_stock_ids": row["manifest_stock_ids"],
                }
                for row in rows
            ],
        }
        identity = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        report = {
            "schema_version": "neuro-film.u7-6g-three-stock-source-context-memory-result.v1",
            "node_id": "U7.6G",
            "config_sha256": _sha256(CONFIG),
            "execution_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "runs": rows,
            **evaluation,
            "scientific_identity": identity,
            "claim_ceiling": config["claim_ceiling"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"decision": report["decision"], "scientific_identity": identity}))
        return 0 if report["decision"] == "PASS" else 1
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
