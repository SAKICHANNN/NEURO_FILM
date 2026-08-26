#!/usr/bin/env python3
"""Audit exact and resource-bounded three-stock source-context reuse."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_color_baseline import load_guardrail_config
from src.inference.render_contract import load_render_profile
from src.inference.three_stock_look import (
    iter_three_stock_look_rgb_shared_context,
    list_three_stock_looks,
    render_three_stock_look_rgb,
)

CONFIG = ROOT / "configs/u7_6a_three_stock_shared_context_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/eval/u7_6a_three_stock_shared_context_result.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _inputs(config: dict[str, Any]):
    height, width, _ = config["input"]["shape"]
    source = np.random.default_rng(config["input"]["seed"]).random(
        (height, width, 3), dtype=np.float32
    )
    profile = load_render_profile(
        ROOT / "configs/render_profiles/safe_rich_v1.json", root=ROOT
    )
    payload = json.loads(
        (ROOT / "configs/film_color_stats.json").read_text(encoding="utf-8")
    )
    statistics_by_style: dict[str, dict[str, Any]] = {}
    guardrails_by_style: dict[str, dict[str, Any]] = {}
    for row in list_three_stock_looks():
        style = row["style_id"]
        statistics_by_style[style] = payload["styles"][style]
        guardrails_by_style[style] = load_guardrail_config(
            ROOT / "configs/color_guardrails.json", style
        )
    return source, profile, statistics_by_style, guardrails_by_style


def run_worker(mode: str) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source, profile, statistics_by_style, guardrails_by_style = _inputs(config)
    execution = config["execution"]
    hashes: list[str] = []
    bounded: list[bool] = []
    started = time.perf_counter()
    if mode == "baseline":
        for row in list_three_stock_looks():
            style = row["style_id"]
            output = render_three_stock_look_rgb(
                source,
                profile=profile,
                film_stock_id=row["film_stock_id"],
                look_amount=execution["look_amount"],
                style_statistics=statistics_by_style[style],
                guardrails=guardrails_by_style[style],
                seed=execution["render_seed"],
                tile_size=execution["tile_size"],
                tile_workers=execution["tile_workers"],
            )
            hashes.append(_array_sha256(output))
            bounded.append(
                bool(
                    np.isfinite(output).all()
                    and np.all((output >= 0.0) & (output <= 1.0))
                )
            )
            del output
            gc.collect()
    elif mode == "candidate":
        for _, output in iter_three_stock_look_rgb_shared_context(
            source,
            profile=profile,
            look_amount=execution["look_amount"],
            style_statistics=statistics_by_style,
            guardrails=guardrails_by_style,
            seed=execution["render_seed"],
            tile_size=execution["tile_size"],
            tile_workers=execution["tile_workers"],
        ):
            hashes.append(_array_sha256(output))
            bounded.append(
                bool(
                    np.isfinite(output).all()
                    and np.all((output >= 0.0) & (output <= 1.0))
                )
            )
            del output
            gc.collect()
    else:
        raise ValueError("mode must be baseline or candidate")
    return {
        "mode": mode,
        "wall_seconds": time.perf_counter() - started,
        "output_sha256": hashes,
        "finite_bounded": bounded,
    }


def _run_fresh(mode: str) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", mode]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    root = psutil.Process(process.pid)
    peak_rss = 0
    while process.poll() is None:
        try:
            processes = [root, *root.children(recursive=True)]
            peak_rss = max(
                peak_rss,
                sum(item.memory_info().rss for item in processes if item.is_running()),
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        time.sleep(0.02)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"worker {mode} failed: {stderr.strip()}")
    row = json.loads(stdout)
    row["peak_process_tree_rss_bytes"] = peak_rss
    row["stderr"] = stderr
    return row


def evaluate_rows(rows: list[dict[str, Any]], gates: dict[str, Any]) -> dict[str, Any]:
    baseline = [row for row in rows if row["mode"] == "baseline"]
    candidate = [row for row in rows if row["mode"] == "candidate"]
    baseline_wall = statistics.median(row["wall_seconds"] for row in baseline)
    candidate_wall = statistics.median(row["wall_seconds"] for row in candidate)
    baseline_peak = statistics.median(
        row["peak_process_tree_rss_bytes"] for row in baseline
    )
    candidate_peak = statistics.median(
        row["peak_process_tree_rss_bytes"] for row in candidate
    )
    exact = len({tuple(row["output_sha256"]) for row in rows}) == 1
    bounded = all(all(row["finite_bounded"]) for row in rows)
    complete = len(rows) == 4 and all(not row["stderr"] for row in rows)
    wall_ratio = candidate_wall / baseline_wall
    peak_ratio = candidate_peak / baseline_peak
    candidate_wall_repeat = max(row["wall_seconds"] for row in candidate) / min(
        row["wall_seconds"] for row in candidate
    )
    candidate_peak_repeat = max(
        row["peak_process_tree_rss_bytes"] for row in candidate
    ) / min(row["peak_process_tree_rss_bytes"] for row in candidate)
    gate_results = {
        "all_three_output_arrays_byte_exact": exact,
        "fresh_process_reports_complete": complete,
        "median_wall_ratio": wall_ratio <= gates["median_wall_ratio_max"],
        "candidate_peak_process_tree_rss_ratio": peak_ratio
        <= gates["candidate_peak_process_tree_rss_ratio_max"],
        "candidate_repeat_wall_ratio": candidate_wall_repeat
        <= gates["candidate_repeat_wall_ratio_max"],
        "candidate_repeat_peak_rss_ratio": candidate_peak_repeat
        <= gates["candidate_repeat_peak_rss_ratio_max"],
        "finite_bounded_outputs": bounded,
    }
    return {
        "baseline_median_wall_seconds": baseline_wall,
        "candidate_median_wall_seconds": candidate_wall,
        "median_wall_ratio": wall_ratio,
        "baseline_median_peak_process_tree_rss_bytes": baseline_peak,
        "candidate_median_peak_process_tree_rss_bytes": candidate_peak,
        "median_peak_rss_ratio": peak_ratio,
        "candidate_repeat_wall_ratio": candidate_wall_repeat,
        "candidate_repeat_peak_rss_ratio": candidate_peak_repeat,
        "gate_results": gate_results,
        "decision": "PASS" if all(gate_results.values()) else "FAIL_CLOSED",
    }


def run_audit(output: Path) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = [_run_fresh(mode) for mode in config["execution"]["fresh_process_order"]]
    evaluation = evaluate_rows(rows, config["gates"])
    scientific = {
        "schema_version": config["schema_version"],
        "config_sha256": _sha256(CONFIG),
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "output_sha256": rows[0]["output_sha256"],
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("baseline", "candidate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(run_worker(args.worker), sort_keys=True))
        return 0
    report = run_audit(args.output)
    print(json.dumps({"decision": report["decision"], "scientific_identity": report["scientific_identity"]}))
    return 0 if report["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
