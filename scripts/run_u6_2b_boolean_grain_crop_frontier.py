#!/usr/bin/env python
"""Run the frozen U6.2B existing-image Boolean-grain crop frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmfx.boolean_grain_composite import (  # noqa: E402
    linear_luminance,
    render_boolean_grain_composite,
)
from src.filmfx.compositor import composite_layers  # noqa: E402
from src.filmfx.effects import grain_residual_layer  # noqa: E402
from src.filmfx.fast_blur import gaussian_filter_safe  # noqa: E402
from src.roll2film.adaptive_density_strength import (  # noqa: E402
    linear_to_srgb8,
    new_hard_clipping_fraction_rgb8,
    srgb8_to_linear,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return _sha256(np.ascontiguousarray(values).tobytes())


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _load_crop(sample: dict[str, Any], crop_size: int) -> np.ndarray:
    path = ROOT / sample["source_path"]
    if _file_sha256(path) != sample["source_sha256"]:
        raise ValueError(f"source hash mismatch for {sample['sample_id']}")
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if list(rgb.size) != sample["source_size"]:
            raise ValueError(f"source size mismatch for {sample['sample_id']}")
        x, y, width, height = (int(value) for value in sample["crop_xywh"])
        if width != crop_size or height != crop_size:
            raise ValueError("configured crop is not the frozen crop size")
        if x < 0 or y < 0 or x + width > rgb.width or y + height > rgb.height:
            raise ValueError("configured crop lies outside the source")
        return np.asarray(rgb.crop((x, y, x + width, y + height)), dtype=np.uint8)


def _metrics(
    source_rgb8: np.ndarray,
    output_linear: np.ndarray,
    *,
    lowpass_sigma: float,
    endpoint_epsilon: float,
) -> dict[str, Any]:
    source_linear = srgb8_to_linear(source_rgb8)
    output_rgb8 = linear_to_srgb8(output_linear)
    source_luma = linear_luminance(source_linear)
    output_luma = linear_luminance(output_linear)
    residual = output_luma - source_luma
    lowpass_residual = gaussian_filter_safe(
        residual.astype(np.float32),
        sigma=float(lowpass_sigma),
    )
    return {
        "linear_output_minimum": float(np.min(output_linear)),
        "linear_output_maximum": float(np.max(output_linear)),
        "absolute_mean_linear_luma_drift": abs(float(np.mean(residual))),
        "linear_luma_residual_rms": float(
            np.sqrt(np.mean(residual * residual, dtype=np.float64))
        ),
        "lowpass_linear_luma_rmse": float(
            np.sqrt(
                np.mean(
                    lowpass_residual.astype(np.float64) ** 2,
                    dtype=np.float64,
                )
            )
        ),
        "new_hard_endpoint_fraction": new_hard_clipping_fraction_rgb8(
            source_rgb8,
            output_rgb8,
            epsilon=float(endpoint_epsilon),
        ),
        "output_rgb8_sha256": _array_sha256(output_rgb8),
    }


def _save_rgb8(values: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values, mode="RGB").save(path, format="PNG", compress_level=9)
    return _file_sha256(path)


def _labelled_panel(label: str, rgb8: np.ndarray) -> Image.Image:
    header = 28
    panel = Image.new("RGB", (rgb8.shape[1], rgb8.shape[0] + header), "white")
    panel.paste(Image.fromarray(rgb8, mode="RGB"), (0, header))
    ImageDraw.Draw(panel).text((8, 8), label, fill="black")
    return panel


def _comparison_page(
    sample_id: str,
    source: np.ndarray,
    candidates: dict[str, np.ndarray],
    *,
    candidate_order: list[str],
    output_path: Path,
) -> str:
    panels = [_labelled_panel("source", source)]
    panels.extend(
        _labelled_panel(candidate_id, candidates[candidate_id])
        for candidate_id in candidate_order
    )
    page = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
        "white",
    )
    cursor = 0
    for panel in panels:
        page.paste(panel, (cursor, 0))
        cursor += panel.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(output_path, format="PNG", compress_level=9)
    return _file_sha256(output_path)


def run_frontier(
    config: dict[str, Any],
    *,
    config_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    crop_size = int(config["crop_size"])
    render = config["boolean_render"]
    metric_config = config["automatic_metrics"]
    gates = config["gates"]
    boolean_reports: dict[str, dict[str, Any]] = {
        candidate["candidate_id"]: {} for candidate in config["boolean_candidates"]
    }
    legacy_reports: dict[str, dict[str, Any]] = {}
    visual_assets: dict[str, Any] = {}
    output_arrays: dict[str, dict[str, np.ndarray]] = {}

    for sample in config["samples"]:
        sample_id = str(sample["sample_id"])
        source_rgb8 = _load_crop(sample, crop_size)
        source_linear = srgb8_to_linear(source_rgb8)
        outputs_for_page: dict[str, np.ndarray] = {}

        legacy = config["legacy_comparator"]
        legacy_layer = grain_residual_layer(
            source_rgb8.astype(np.float32) / 255.0,
            strength=float(legacy["strength"]),
            seed=int(sample["seed"]),
            color=bool(legacy["color"]),
        )
        legacy_encoded = composite_layers(
            source_rgb8.astype(np.float32) / 255.0,
            [legacy_layer],
        )
        legacy_rgb8 = np.rint(legacy_encoded * 255.0).astype(np.uint8)
        outputs_for_page[legacy["candidate_id"]] = legacy_rgb8
        legacy_reports[sample_id] = {
            "output_sha256": _save_rgb8(
                legacy_rgb8,
                output_root / "renders" / legacy["candidate_id"] / f"{sample_id}.png",
            )
        }

        for candidate in config["boolean_candidates"]:
            candidate_id = candidate["candidate_id"]
            kwargs = {
                "input_shape": tuple(int(value) for value in render["input_shape"]),
                "output_zoom": int(render["output_zoom"]),
                "radius_input_pixels": float(candidate["radius_input_pixels"]),
                "luma_residual_strength": float(candidate["luma_residual_strength"]),
                "monte_carlo_samples": int(render["monte_carlo_samples"]),
                "gaussian_filter_sigma_output_pixels": float(
                    render["gaussian_filter_sigma_output_pixels"]
                ),
                "maximum_input_intensity": float(
                    render["maximum_input_intensity"]
                ),
                "epsilon": float(render["epsilon"]),
                "seed": int(sample["seed"]),
            }
            first = render_boolean_grain_composite(source_linear, **kwargs)
            second = render_boolean_grain_composite(source_linear, **kwargs)
            repeat_error = float(
                np.max(np.abs(first.linear_rgb - second.linear_rgb))
            )
            metrics = _metrics(
                source_rgb8,
                first.linear_rgb,
                lowpass_sigma=float(metric_config["lowpass_sigma_pixels"]),
                endpoint_epsilon=float(
                    metric_config["hard_endpoint_epsilon"]
                ),
            )
            output_rgb8 = linear_to_srgb8(first.linear_rgb)
            outputs_for_page[candidate_id] = output_rgb8
            output_arrays.setdefault(candidate_id, {})[sample_id] = output_rgb8
            metrics.update(
                {
                    "source_sha256": sample["source_sha256"],
                    "crop_xywh": sample["crop_xywh"],
                    "seed": int(sample["seed"]),
                    "grain_count": first.context.grain_count,
                    "context_fingerprint": first.context.fingerprint(),
                    "grain_luminance_sha256": _array_sha256(
                        first.grain_luminance
                    ),
                    "linear_output_sha256": _array_sha256(first.linear_rgb),
                    "repeat_maximum_absolute_error": repeat_error,
                    "output_file_sha256": _save_rgb8(
                        output_rgb8,
                        output_root / "renders" / candidate_id / f"{sample_id}.png",
                    ),
                }
            )
            boolean_reports[candidate_id][sample_id] = metrics

        page_order = [legacy["candidate_id"]] + [
            candidate["candidate_id"] for candidate in config["boolean_candidates"]
        ]
        random.Random(
            int(config["visual"]["blind_page_candidate_order_seed"])
            + int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16)
        ).shuffle(page_order)
        page_path = output_root / "visual" / f"comparison_{sample_id}.png"
        visual_assets[sample_id] = {
            "candidate_order": page_order,
            "page": str(page_path.relative_to(output_root)),
            "page_sha256": _comparison_page(
                sample_id,
                source_rgb8,
                outputs_for_page,
                candidate_order=page_order,
                output_path=page_path,
            ),
        }

    candidate_checks: dict[str, dict[str, bool]] = {}
    for candidate_id, sample_reports in boolean_reports.items():
        values = list(sample_reports.values())
        checks = {
            "finite_and_structurally_bounded": all(
                np.isfinite(report["linear_output_minimum"])
                and np.isfinite(report["linear_output_maximum"])
                and report["linear_output_minimum"] >= 0.0
                and report["linear_output_maximum"] <= 1.0
                for report in values
            ),
            "mean_luma_drift": max(
                report["absolute_mean_linear_luma_drift"] for report in values
            )
            <= gates["maximum_absolute_mean_linear_luma_drift"],
            "visible_residual_floor": min(
                report["linear_luma_residual_rms"] for report in values
            )
            >= gates["minimum_linear_luma_residual_rms"],
            "residual_ceiling": max(
                report["linear_luma_residual_rms"] for report in values
            )
            <= gates["maximum_linear_luma_residual_rms"],
            "lowpass_drift": max(
                report["lowpass_linear_luma_rmse"] for report in values
            )
            <= gates["maximum_lowpass_linear_luma_rmse"],
            "endpoint_budget": max(
                report["new_hard_endpoint_fraction"] for report in values
            )
            <= gates["maximum_new_hard_endpoint_fraction"],
            "repeat_exact": max(
                report["repeat_maximum_absolute_error"] for report in values
            )
            <= gates["repeat_maximum_absolute_error"],
        }
        candidate_checks[candidate_id] = checks

    automatic_survivors = [
        candidate_id
        for candidate_id, checks in candidate_checks.items()
        if all(checks.values())
    ]
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": _git_commit(),
        "config_sha256": config_sha256,
        "sample_count": len(config["samples"]),
        "boolean_reports": boolean_reports,
        "legacy_reports": legacy_reports,
        "candidate_checks": candidate_checks,
        "automatic_survivors": automatic_survivors,
        "automatic_gate_passed": bool(automatic_survivors),
        "visual_assets": visual_assets,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_2b_boolean_grain_crop_frontier_v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/u6_2b_boolean_grain_crop_frontier_v1",
    )
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    report = run_frontier(
        json.loads(config_bytes),
        config_sha256=_sha256(config_bytes),
        output_root=args.output_root,
    )
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(report_path)
    print(
        json.dumps(
            {
                "output": str(report_path),
                "sha256": _sha256(encoded),
                "automatic_gate_passed": report["automatic_gate_passed"],
                "automatic_survivors": report["automatic_survivors"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["automatic_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
