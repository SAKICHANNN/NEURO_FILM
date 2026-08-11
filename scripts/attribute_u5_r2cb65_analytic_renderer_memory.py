#!/usr/bin/env python3
"""Attribute complete analytic renderer RSS to coarse execution phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.analytic_y_chromaticity_streaming import (
    select_analytic_y_chromaticity_candidate_streamed,
)
from src.eval.fixed_global_policy_confirmation import render_fixed_pair
from src.eval.fujifilm_characteristic_luma_chroma import (
    apply_characteristic_luma_chroma,
)
from src.eval.nonexpansive_fraction_transport_streaming import (
    nonexpansive_fraction_transport_target_row_materialized,
)
from src.filmfx import composite_layers
from src.inference.analytic_y_chromaticity_profile import (
    load_analytic_y_chromaticity_profile,
)
from src.preprocess import load_working_image, save_srgb16_png


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mark(path: Path, phase: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(phase, encoding="ascii")
    temporary.replace(path)


def _worker(input_path: Path, output: Path, scratch: Path, marker: Path) -> dict:
    started = perf_counter()
    _mark(marker, "load")
    working = load_working_image(input_path)
    runtime = load_analytic_y_chromaticity_profile(
        ROOT / "configs/render_profiles/analytic_y_chromaticity_cb61_v2.json",
        root=ROOT,
    )
    source = np.asarray(working.pixels, dtype=np.float32)
    operator = runtime.cb11["operator"]
    weights = np.asarray(operator["luminance_weights"], dtype=np.float64)
    epsilon = float(operator["boundary_epsilon"])

    _mark(marker, "ao6")
    ao6 = render_fixed_pair(source, runtime.artifact, runtime.ao6_config["component"])[
        runtime.ao6_config["arm_id"]
    ]
    _mark(marker, "safe_base")
    safe_base, _, _ = apply_characteristic_luma_chroma(
        source,
        runtime.curve,
        weights=weights,
        strength=float(operator["nominal_strength"]),
        boundary_epsilon=epsilon,
    )
    config = runtime.cb52
    _mark(marker, "target")
    target = nonexpansive_fraction_transport_target_row_materialized(
        safe_base,
        ao6,
        weights=weights,
        minimum_valid_fraction=float(config["operator"]["minimum_valid_fraction"]),
        fraction_knots=int(config["operator"]["fraction_knots"]),
        maximum_fraction_slope=float(config["operator"]["maximum_fraction_slope"]),
    )
    _mark(marker, "selector")
    candidate, _, _, facts = select_analytic_y_chromaticity_candidate_streamed(
        source,
        target,
        curve=runtime.curve,
        strength=float(operator["nominal_strength"]),
        boundary_epsilon=epsilon,
        dose_grid=config["operator"]["dose_grid"],
        maximum_gradient_ratio=float(
            config["automatic_gates"]["maximum_p999_gradient_ratio_vs_source"]
        ),
        maximum_lstar_inversion_fraction=float(
            config["automatic_gates"][
                "maximum_adjacent_lstar_gradient_sign_inversion_fraction"
            ]
        ),
        lstar_order_epsilon=float(config["operator"]["lstar_order_epsilon"]),
        row_chunk=int(runtime.profile["execution"]["row_chunk"]),
        scratch_root=scratch,
    )
    _mark(marker, "composite")
    image = composite_layers(candidate, [], output_margin=0)
    _mark(marker, "encode")
    save_srgb16_png(image, output)
    _mark(marker, "complete")
    return {
        "output_sha256": _sha256(output),
        "selector_facts": facts,
        "worker_seconds": perf_counter() - started,
        "scratch_residue": sorted(path.name for path in scratch.iterdir()),
    }


def _parent(input_path: Path, output: Path, work_dir: Path) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    marker = work_dir / "phase.txt"
    worker_result = work_dir / "worker.json"
    image_output = work_dir / "output.png"
    scratch = work_dir / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {"TEMP": str(scratch), "TMP": str(scratch), "PYTHONDONTWRITEBYTECODE": "1"}
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--input",
            str(input_path),
            "--worker-result",
            str(worker_result),
            "--image-output",
            str(image_output),
            "--scratch-root",
            str(scratch),
            "--marker",
            str(marker),
        ],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    root = psutil.Process(process.pid)
    peaks: dict[str, int] = {}
    started = perf_counter()
    while process.poll() is None:
        phase = "startup"
        try:
            if marker.exists():
                phase = marker.read_text(encoding="ascii")
            tree = [root, *root.children(recursive=True)]
            rss = sum(item.memory_info().rss for item in tree if item.is_running())
            peaks[phase] = max(peaks.get(phase, 0), rss)
        except (OSError, psutil.AccessDenied, psutil.NoSuchProcess):
            pass
        if perf_counter() - started > 360.0:
            for child in reversed(root.children(recursive=True)):
                child.kill()
            root.kill()
            raise TimeoutError("CB65 worker exceeded 360 seconds")
        sleep(0.02)
    stdout, stderr = process.communicate()
    result = (
        json.loads(worker_result.read_text(encoding="utf-8"))
        if worker_result.exists()
        else None
    )
    report = {
        "schema": "neuro_film.u5_r2cb65_analytic_renderer_memory_attribution_report.v1",
        "experiment_id": "U5.R2CB65",
        "exit_code": process.returncode,
        "wall_seconds": perf_counter() - started,
        "phase_peak_process_tree_rss_bytes": peaks,
        "stdout": stdout,
        "stderr": stderr,
        "result": result,
        "claim_ceiling": "Local coarse phase attribution only; not a product benchmark.",
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--worker-result", type=Path)
    parser.add_argument("--image-output", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--marker", type=Path)
    args = parser.parse_args()
    if args.worker_result is not None:
        result = _worker(args.input, args.image_output, args.scratch_root, args.marker)
        args.worker_result.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 0
    if args.output is None or args.work_dir is None:
        parser.error("parent mode requires --output and --work-dir")
    report = _parent(args.input, args.output, args.work_dir)
    print(json.dumps(report["phase_peak_process_tree_rss_bytes"], sort_keys=True))
    return 0 if report["exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
