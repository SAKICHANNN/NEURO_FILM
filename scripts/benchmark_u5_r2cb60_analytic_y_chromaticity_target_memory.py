#!/usr/bin/env python3
"""Measure the output-exact CB60 target materializer in the complete image path."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from functools import partial
from pathlib import Path
from time import perf_counter, sleep

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.nonexpansive_fraction_transport_external_sort import (
    nonexpansive_fraction_transport_target_external_sorted,
)
from src.eval.nonexpansive_fraction_transport_statistics_streaming import (
    nonexpansive_fraction_transport_target_statistics_streamed,
)
from src.eval.nonexpansive_fraction_transport_streaming import (
    nonexpansive_fraction_transport_target_row_materialized,
)
from src.filmfx import composite_layers
from src.inference.analytic_y_chromaticity_profile import (
    load_analytic_y_chromaticity_profile,
    render_analytic_y_chromaticity_profile,
)
from src.preprocess import load_working_image, save_srgb16_png


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _worker(config: dict, output: Path, scratch: Path) -> dict:
    started = perf_counter()
    working = load_working_image(Path(config["baseline"]["input_path"]))
    runtime = load_analytic_y_chromaticity_profile(
        ROOT / "configs/render_profiles/analytic_y_chromaticity_cb56_v1.json",
        root=ROOT,
    )
    materializer = config["candidate"].get("target_materializer", "row_bounded_hue_v1")
    target_builder = {
        "row_bounded_hue_v1": nonexpansive_fraction_transport_target_row_materialized,
        "statistics_streamed_v1": nonexpansive_fraction_transport_target_statistics_streamed,
        "external_sorted_v1": partial(
            nonexpansive_fraction_transport_target_external_sorted,
            scratch_root=scratch,
        ),
    }.get(materializer)
    if target_builder is None:
        raise ValueError("unsupported CB memory target materializer")
    candidate, facts = render_analytic_y_chromaticity_profile(
        working,
        runtime,
        scratch_root=scratch,
        target_builder=target_builder,
    )
    image = composite_layers(candidate, [], output_margin=0)
    save_srgb16_png(image, output)
    return {
        "output_sha256": _sha256(output),
        "selector_facts": facts,
        "worker_seconds": perf_counter() - started,
        "scratch_residue": sorted(path.name for path in scratch.iterdir()),
    }


def _launch(config_path: Path, work_dir: Path, index: int) -> dict:
    output = work_dir / f"candidate-{index}.png"
    result_path = work_dir / f"candidate-{index}.json"
    scratch = work_dir / f"scratch-{index}"
    scratch.mkdir(parents=True, exist_ok=True)
    environment = dict(__import__("os").environ)
    environment.update(
        {"TEMP": str(scratch), "TMP": str(scratch), "PYTHONDONTWRITEBYTECODE": "1"}
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(config_path),
            "--worker-output",
            str(result_path),
            "--image-output",
            str(output),
            "--scratch-root",
            str(scratch),
        ],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    root = psutil.Process(process.pid)
    peak = 0
    started = perf_counter()
    while process.poll() is None:
        try:
            tree = [root, *root.children(recursive=True)]
            peak = max(
                peak,
                sum(item.memory_info().rss for item in tree if item.is_running()),
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
        if perf_counter() - started > 360.0:
            for child in reversed(root.children(recursive=True)):
                child.kill()
            root.kill()
            raise TimeoutError("CB60 worker exceeded 360 seconds")
        sleep(0.02)
    stdout, stderr = process.communicate()
    result = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.exists()
        else None
    )
    return {
        "exit_code": process.returncode,
        "wall_seconds": perf_counter() - started,
        "peak_process_tree_rss_bytes": peak,
        "stdout": stdout,
        "stderr": stderr,
        "result": result,
    }


def _parent(config_path: Path, output: Path, work_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    work_dir.mkdir(parents=True, exist_ok=True)
    runs = [_launch(config_path.resolve(), work_dir, index) for index in (1, 2)]
    baseline = config["baseline"]
    gates = config["gates"]
    peaks = [row["peak_process_tree_rss_bytes"] for row in runs]
    walls = [row["wall_seconds"] for row in runs]
    results = [row["result"] for row in runs]
    complete = all(
        row["exit_code"] == 0 and result is not None
        for row, result in zip(runs, results)
    )
    checks = {
        "workers_completed": complete,
        "output_sha256_exact": complete
        and all(
            result["output_sha256"] == baseline["output_sha256"] for result in results
        ),
        "selector_facts_exact": complete
        and all(
            result["selector_facts"] == baseline["selector_facts"] for result in results
        ),
        "repeat_exact": complete
        and results[0]["output_sha256"] == results[1]["output_sha256"]
        and results[0]["selector_facts"] == results[1]["selector_facts"],
        "peak_rss": max(peaks) <= gates["maximum_peak_process_tree_rss_bytes"],
        "peak_rss_ratio": max(peaks) / baseline["peak_process_tree_rss_bytes"]
        <= gates["maximum_peak_rss_ratio_to_baseline"],
        "wall_ratio": max(walls) / baseline["wall_seconds"]
        <= gates["maximum_wall_ratio_to_baseline"],
        "scratch_cleanup": complete
        and all(
            len(result["scratch_residue"]) == gates["scratch_residue_count"]
            for result in results
        ),
    }
    report = {
        "schema": config.get(
            "report_schema",
            "neuro_film.u5_r2cb60_analytic_y_chromaticity_target_memory_report.v1",
        ),
        "experiment_id": config["experiment_id"],
        "contract_sha256": _sha256(config_path),
        "runs": runs,
        "metrics": {
            "maximum_peak_process_tree_rss_bytes": max(peaks),
            "peak_rss_ratio_to_baseline": max(peaks)
            / baseline["peak_process_tree_rss_bytes"],
            "maximum_wall_seconds": max(walls),
            "wall_ratio_to_baseline": max(walls) / baseline["wall_seconds"],
        },
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "claim_ceiling": config["claim_ceiling"],
    }
    _write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--image-output", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.worker_output is not None:
        result = _worker(config, args.image_output, args.scratch_root)
        _write_json(args.worker_output, result)
        return 0
    if args.output is None or args.work_dir is None:
        parser.error("parent mode requires --output and --work-dir")
    report = _parent(args.config, args.output, args.work_dir)
    print(
        json.dumps(
            {"automatic_pass": report["automatic_pass"], "metrics": report["metrics"]},
            sort_keys=True,
        )
    )
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
