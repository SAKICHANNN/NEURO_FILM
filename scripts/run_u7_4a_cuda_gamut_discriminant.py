#!/usr/bin/env python3
"""Run the frozen U7.4A isolated CUDA gamut discriminant."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_color_baseline import (
    compress_to_srgb_gamut_parallel,
    lab_to_rgb_no_clip,
    load_guardrail_config,
)
from src.color_engine.safe_lab import (
    apply_safe_lab_transform,
    safe_lab_context_from_lab,
)
from src.eval.cuda_srgb_gamut import compress_to_srgb_gamut_cuda
from src.inference import load_render_profile


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    contract = json.loads(raw)
    if contract.get("schema") != "neuro-film.u7-4a-cuda-gamut-discriminant-contract.v1":
        raise ValueError("U7.4A contract schema is invalid")
    for binding in contract["bindings"]:
        if _sha256(ROOT / binding["path"]) != binding["sha256"]:
            raise ValueError(f"U7.4A binding drift: {binding['path']}")
    if _sha256(ROOT / contract["source"]["path"]) != contract["source"]["sha256"]:
        raise ValueError("U7.4A source drift")
    return contract, hashlib.sha256(raw).hexdigest()


def _style_pair(
    source: np.ndarray,
    style: str,
    profile: dict[str, Any],
    statistics: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    lab = np.asarray(rgb2lab(source), dtype=np.float32)
    context = safe_lab_context_from_lab(lab)
    params = profile["style_parameters"][style]
    guard = load_guardrail_config(ROOT / "configs/color_guardrails.json", style)
    styled = apply_safe_lab_transform(
        lab,
        source_context=context,
        destination_mean=np.asarray(statistics[style]["mean"], dtype=np.float32),
        destination_std=np.asarray(statistics[style]["std"], dtype=np.float32),
        style=style,
        strength=float(params["strength"]),
        luma_strength=float(params["luma_strength"]),
        tone_rolloff=float(params["tone_rolloff"]),
        shadow_floor_l=float(params["shadow_floor_l"]),
        highlight_ceiling_l=float(params["highlight_ceiling_l"]),
        preserve_luma_detail_strength=float(params["preserve_luma_detail"]),
        chroma_curve_strength=float(params["chroma_curve_strength"]),
        neutral_protect=float(guard["neutral_protect"]),
        skin_protect=float(guard["skin_protect"]),
        max_chroma_gain=guard.get("max_chroma_gain"),
        max_chroma_boost=guard.get("max_chroma_boost"),
        max_chroma_absolute=guard.get("max_chroma_absolute"),
    )
    return lab, styled


def evaluate(contract_path: Path, *, order: str) -> dict[str, Any]:
    contract, contract_sha = _load_contract(contract_path)
    width, height = (int(value) for value in contract["dimensions"])
    with Image.open(ROOT / contract["source"]["path"]) as image:
        rgb8 = np.asarray(
            image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
    source = np.ascontiguousarray(rgb8.astype(np.float32) / 255.0)
    profile = load_render_profile(ROOT / "configs/render_profiles/safe_rich_v1.json", root=ROOT)
    statistics_payload = json.loads((ROOT / "configs/film_color_stats.json").read_text())
    statistics_map = statistics_payload["styles"]
    styles = list(contract["styles"])
    if order == "reverse":
        styles.reverse()
    execution = contract["execution"]
    gates = contract["gates"]
    rows: list[dict[str, Any]] = []
    for style in styles:
        source_lab, target_lab = _style_pair(source, style, profile, statistics_map)
        cpu_times: list[float] = []
        cuda_times: list[float] = []
        cpu = None
        cuda = None
        for index in range(int(execution["warmups"]) + int(execution["measured_repeats"])):
            started = time.perf_counter()
            cpu = compress_to_srgb_gamut_parallel(
                source_lab,
                target_lab,
                iterations=int(execution["iterations"]),
                workers=int(execution["cpu_workers"]),
            )
            cpu_wall = time.perf_counter() - started
            started = time.perf_counter()
            cuda = compress_to_srgb_gamut_cuda(
                source_lab,
                target_lab,
                iterations=int(execution["iterations"]),
                device_index=int(execution["cuda_device"]),
            )
            cuda_wall = time.perf_counter() - started
            if index >= int(execution["warmups"]):
                cpu_times.append(cpu_wall)
                cuda_times.append(cuda_wall)
        assert cpu is not None and cuda is not None
        error = np.abs(cuda.output_lab - cpu)
        cpu_rgb16 = np.rint(lab_to_rgb_no_clip(cpu) * 65535.0).astype(np.uint16)
        cuda_rgb16 = np.rint(lab_to_rgb_no_clip(cuda.output_lab) * 65535.0).astype(np.uint16)
        code_delta = np.abs(cuda_rgb16.astype(np.int32) - cpu_rgb16.astype(np.int32))
        repeat = compress_to_srgb_gamut_cuda(
            source_lab,
            target_lab,
            iterations=int(execution["iterations"]),
            device_index=int(execution["cuda_device"]),
        )
        row = {
            "style": style,
            "maximum_lab_absolute_error": float(error.max()),
            "lab_p99_absolute_error": float(np.quantile(error, 0.99)),
            "maximum_rgb16_code_delta": int(code_delta.max()),
            "rgb16_mismatch_fraction": float(np.count_nonzero(code_delta) / code_delta.size),
            "cpu_median_seconds": float(statistics.median(cpu_times)),
            "cuda_median_seconds": float(statistics.median(cuda_times)),
            "cuda_to_cpu_wall_ratio": float(statistics.median(cuda_times) / statistics.median(cpu_times)),
            "peak_cuda_allocated_bytes": cuda.peak_allocated_bytes,
            "cuda_repeat_exact": bool(np.array_equal(cuda.output_lab, repeat.output_lab)),
            "cpu_lab_sha256": hashlib.sha256(cpu.tobytes()).hexdigest(),
            "cuda_lab_sha256": hashlib.sha256(cuda.output_lab.tobytes()).hexdigest(),
        }
        row["passes"] = bool(
            row["maximum_lab_absolute_error"] <= gates["maximum_lab_absolute_error"]
            and row["lab_p99_absolute_error"] <= gates["maximum_lab_p99_absolute_error"]
            and row["maximum_rgb16_code_delta"] <= gates["maximum_rgb16_code_delta"]
            and row["rgb16_mismatch_fraction"] <= gates["maximum_rgb16_mismatch_fraction"]
            and row["cuda_to_cpu_wall_ratio"] <= gates["maximum_median_cuda_to_cpu_wall_ratio"]
            and row["peak_cuda_allocated_bytes"] <= gates["maximum_peak_cuda_allocated_bytes"]
            and row["cuda_repeat_exact"]
        )
        rows.append(row)
    canonical_rows = sorted(rows, key=lambda row: row["style"])
    numerical_pass = bool(
        all(
            row["maximum_lab_absolute_error"] <= gates["maximum_lab_absolute_error"]
            and row["lab_p99_absolute_error"] <= gates["maximum_lab_p99_absolute_error"]
            and row["maximum_rgb16_code_delta"] <= gates["maximum_rgb16_code_delta"]
            and row["rgb16_mismatch_fraction"] <= gates["maximum_rgb16_mismatch_fraction"]
            and row["cuda_repeat_exact"]
            for row in rows
        )
    )
    scientific = {
        "schema": "neuro-film.u7-4a-cuda-gamut-discriminant-report.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": contract_sha,
        "source_resized_rgb8_sha256": hashlib.sha256(rgb8.tobytes()).hexdigest(),
        "rows": [
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "cpu_median_seconds",
                    "cuda_median_seconds",
                    "cuda_to_cpu_wall_ratio",
                    "peak_cuda_allocated_bytes",
                    "passes",
                }
            }
            for row in canonical_rows
        ],
        "numerical_pass": numerical_pass,
    }
    automatic_pass = bool(all(row["passes"] for row in rows))
    report = {
        **scientific,
        "automatic_pass": automatic_pass,
        "order": order,
        "performance_rows": rows,
        "stable_scientific_id": _stable_id(scientific),
        "decision": contract["decision_if_pass"] if automatic_pass else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=ROOT / "configs/u7_4a_cuda_gamut_discriminant_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    args = parser.parse_args()
    report = evaluate(args.contract, order=args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
