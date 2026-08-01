"""Photographic stress test for the fixed KCI Velvia display-tone residual."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import PchipInterpolator

from src.color_engine.gamut import compress_source_to_working_gamut
from src.color_engine.lab import lab_to_linear_rgb, linear_rgb_to_lab
from src.color_engine.srgb_transfer import encoded_srgb_to_linear, linear_srgb_to_encoded


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rgb(path: Path, *, maximum_long_edge: int) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError("BR2 requires exact RGB source PNGs")
        image.load()
        if max(image.size) > maximum_long_edge:
            scale = maximum_long_edge / max(image.size)
            size = tuple(max(1, int(round(value * scale))) for value in image.size)
            image = image.resize(size, Image.Resampling.LANCZOS)
        encoded = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    return np.asarray(encoded_srgb_to_linear(encoded), dtype=np.float32)


def _curve_from_report(parent_report: dict[str, Any]) -> PchipInterpolator:
    if (
        not parent_report.get("automatic_pass")
        or parent_report.get("selected_model") != "monotone_pchip"
        or parent_report.get("decision")
        != "retain_controlled_monotone_tone_residual_for_photo_stress"
    ):
        raise ValueError("BR1 parent did not authorize photographic stress")
    parameters = parent_report["full_fit_monotone_pchip"]
    source = np.asarray(parameters["source_knots"], dtype=np.float64)
    target = np.asarray(parameters["target_knots"], dtype=np.float64)
    if (
        source.shape != (8,)
        or target.shape != (8,)
        or np.any(np.diff(source) <= 0.0)
        or np.any(np.diff(target) < 0.0)
        or source[0] != 0.0
        or source[-1] != 100.0
        or target[0] != 0.0
        or target[-1] != 100.0
    ):
        raise ValueError("BR1 full-fit curve identity drift")
    return PchipInterpolator(source, target, extrapolate=False)


def apply_fixed_tone(
    source_linear: np.ndarray, curve: PchipInterpolator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply fixed L-star curve with source-segment gamut safety."""

    source = np.asarray(source_linear)
    if (
        source.dtype != np.float32
        or source.ndim != 3
        or source.shape[2] != 3
        or not np.isfinite(source).all()
        or np.min(source) < 0.0
        or np.max(source) > 1.0
    ):
        raise ValueError("BR2 source must be finite in-gamut HxWx3 float32")
    source_before = source.copy()
    source_lab = linear_rgb_to_lab(source, working_space="linear_srgb")
    target_lab = source_lab.copy()
    target_lab[..., 0] = np.asarray(curve(source_lab[..., 0]), dtype=np.float32)
    if (
        not np.isfinite(target_lab).all()
        or np.min(target_lab[..., 0]) < 0.0
        or np.max(target_lab[..., 0]) > 100.0
    ):
        raise ValueError("BR2 fixed curve escaped its frozen L-star domain")
    output_lab = compress_source_to_working_gamut(
        source_lab,
        target_lab,
        working_space="linear_srgb",
        iterations=24,
    )
    output = lab_to_linear_rgb(output_lab, working_space="linear_srgb")
    if (
        not np.isfinite(output).all()
        or np.min(output) < -2e-6
        or np.max(output) > 1.0 + 2e-6
        or not np.array_equal(source, source_before)
    ):
        raise ValueError("BR2 gamut-safe output invariant failed")
    tone_delta = target_lab[..., 0] - source_lab[..., 0]
    applied_delta = output_lab[..., 0] - source_lab[..., 0]
    scale = np.ones_like(tone_delta, dtype=np.float32)
    material = np.abs(tone_delta) > np.float32(1e-6)
    scale[material] = applied_delta[material] / tone_delta[material]
    return (
        np.asarray(np.clip(output, 0.0, 1.0), dtype=np.float32),
        source_lab,
        output_lab,
        scale,
    )


def _gradient_inversion_fraction(
    source_lstar: np.ndarray, output_lstar: np.ndarray, *, epsilon: float
) -> float:
    inversions = 0
    comparisons = 0
    for axis in (0, 1):
        source_delta = np.diff(source_lstar, axis=axis)
        output_delta = np.diff(output_lstar, axis=axis)
        material = np.abs(source_delta) > epsilon
        comparisons += int(np.count_nonzero(material))
        inversions += int(
            np.count_nonzero(
                material & (source_delta * output_delta < -(epsilon * epsilon))
            )
        )
    return 0.0 if comparisons == 0 else inversions / comparisons


def _save_rgb(linear: np.ndarray, path: Path) -> str:
    encoded = linear_srgb_to_encoded(linear)
    if not np.isfinite(encoded).all() or np.min(encoded) < 0.0 or np.max(encoded) > 1.0:
        raise ValueError("BR2 encoded output escaped sRGB")
    quantized = np.rint(encoded * np.float32(255.0)).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(quantized, mode="RGB").save(path, compress_level=6)
    return sha256_file(path)


def _sheet(rows: list[dict[str, Any]], path: Path) -> str:
    cell_width = 360
    cell_height = 260
    header = 28
    canvas = Image.new("RGB", (cell_width * 3, header + cell_height * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(("source", "candidate", "absolute difference x4")):
        draw.text((column * cell_width + 8, 8), label, fill="black")
    for row_index, row in enumerate(rows):
        for column, key in enumerate(("source", "candidate", "difference")):
            with Image.open(row["visual_paths"][key]) as image:
                image.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
                x = column * cell_width + (cell_width - image.width) // 2
                y = header + row_index * cell_height + (cell_height - image.height) // 2
                canvas.paste(image, (x, y))
        draw.text((5, header + row_index * cell_height + 5), row["id"], fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, compress_level=6)
    return sha256_file(path)


def evaluate_photographic_stress(
    *,
    root: Path,
    config: dict[str, Any],
    parent_report: dict[str, Any],
    visual_root: Path,
) -> dict[str, Any]:
    manifest_path = root / config["population"]["manifest_path"]
    if sha256_file(manifest_path) != config["population"]["manifest_sha256"]:
        raise ValueError("BR2 source manifest identity drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest) != config["population"]["source_count_exact"]:
        raise ValueError("BR2 source count drift")
    curve = _curve_from_report(parent_report)
    gates = config["automatic_gates"]
    scale_threshold = float(config["evaluation"]["gamut_scale_material_threshold"])
    boundary_epsilon = float(config["evaluation"]["hard_boundary_epsilon"])
    gradient_epsilon = float(config["evaluation"]["gradient_sign_epsilon_lstar"])
    rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    for manifest_row in manifest:
        source_path = root / manifest_row["decoded_path"]
        if sha256_file(source_path) != manifest_row["decoded_sha256"]:
            raise ValueError("BR2 source image identity drift")
        source = _load_rgb(
            source_path,
            maximum_long_edge=int(config["population"]["maximum_long_edge"]),
        )
        output, source_lab, output_lab, scale = apply_fixed_tone(source, curve)
        source_encoded = linear_srgb_to_encoded(source)
        output_encoded = linear_srgb_to_encoded(output)
        effect = np.abs(output_lab[..., 0] - source_lab[..., 0])
        source_boundary = (source_encoded <= boundary_epsilon) | (
            source_encoded >= 1.0 - boundary_epsilon
        )
        output_boundary = (output_encoded <= boundary_epsilon) | (
            output_encoded >= 1.0 - boundary_epsilon
        )
        row = {
            "id": manifest_row["id"],
            "make": manifest_row["make"],
            "model": manifest_row["model"],
            "source_sha256": manifest_row["decoded_sha256"],
            "shape": list(source.shape),
            "p95_absolute_lstar_effect": float(np.quantile(effect, 0.95)),
            "maximum_absolute_lstar_effect": float(np.max(effect)),
            "median_gamut_scale": float(np.median(scale)),
            "fraction_gamut_scale_below_0p99": float(np.mean(scale < scale_threshold)),
            "new_hard_boundary_fraction": float(
                np.mean(output_boundary & ~source_boundary)
            ),
            "adjacent_lstar_gradient_sign_inversion_fraction": _gradient_inversion_fraction(
                source_lab[..., 0], output_lab[..., 0], epsilon=gradient_epsilon
            ),
            "output_linear_minimum": float(np.min(output)),
            "output_linear_maximum": float(np.max(output)),
        }
        rows.append(row)
        visual_rows.append(
            {
                "id": row["id"],
                "source": source,
                "candidate": output,
                "difference": np.clip(np.abs(output - source) * 4.0, 0.0, 1.0),
            }
        )
    metrics = {
        "source_count": len(rows),
        "camera_make_count": len({row["make"] for row in rows}),
        "population_median_per_image_p95_lstar_effect": float(
            np.median([row["p95_absolute_lstar_effect"] for row in rows])
        ),
        "maximum_per_image_p95_lstar_effect": max(
            row["p95_absolute_lstar_effect"] for row in rows
        ),
        "population_median_gamut_scale": float(
            np.median([row["median_gamut_scale"] for row in rows])
        ),
        "maximum_per_image_fraction_gamut_scale_below_0p99": max(
            row["fraction_gamut_scale_below_0p99"] for row in rows
        ),
        "maximum_new_hard_boundary_fraction": max(
            row["new_hard_boundary_fraction"] for row in rows
        ),
        "maximum_adjacent_lstar_gradient_sign_inversion_fraction": max(
            row["adjacent_lstar_gradient_sign_inversion_fraction"] for row in rows
        ),
        "output_linear_minimum": min(row["output_linear_minimum"] for row in rows),
        "output_linear_maximum": max(row["output_linear_maximum"] for row in rows),
    }
    checks = {
        "source_count": metrics["source_count"] == gates["source_count_exact"],
        "camera_make_count": metrics["camera_make_count"]
        == gates["camera_make_count_exact"],
        "visible_effect": metrics["population_median_per_image_p95_lstar_effect"]
        >= gates["minimum_population_median_per_image_p95_lstar_effect"],
        "bounded_effect": metrics["maximum_per_image_p95_lstar_effect"]
        <= gates["maximum_per_image_p95_lstar_effect"],
        "median_gamut_scale": metrics["population_median_gamut_scale"]
        >= gates["minimum_population_median_gamut_scale"],
        "gamut_scale_tail": metrics[
            "maximum_per_image_fraction_gamut_scale_below_0p99"
        ]
        <= gates["maximum_per_image_fraction_gamut_scale_below_0p99"],
        "new_boundaries": metrics["maximum_new_hard_boundary_fraction"]
        <= gates["maximum_new_hard_boundary_fraction"],
        "gradient_order": metrics[
            "maximum_adjacent_lstar_gradient_sign_inversion_fraction"
        ]
        <= gates["maximum_adjacent_lstar_gradient_sign_inversion_fraction"],
        "finite_bounded_rgb": metrics["output_linear_minimum"] >= 0.0
        and metrics["output_linear_maximum"] <= 1.0,
    }
    automatic_pass = all(checks.values())
    visual_assets: dict[str, Any] = {}
    if automatic_pass:
        for visual in visual_rows:
            paths = {
                key: visual_root / visual["id"] / f"{key}.png"
                for key in ("source", "candidate", "difference")
            }
            hashes = {
                key: _save_rgb(visual[key], paths[key])
                for key in ("source", "candidate", "difference")
            }
            visual_assets[visual["id"]] = hashes
            visual["visual_paths"] = {key: str(value) for key, value in paths.items()}
        visual_assets["direct_severe_sheet_sha256"] = _sheet(
            visual_rows, visual_root / "direct_severe_sheet.png"
        )
    return {
        "rows": rows,
        "metrics": metrics,
        "automatic_checks": checks,
        "automatic_pass": automatic_pass,
        "visual_review_allowed": automatic_pass,
        "visual_review_status": "pending" if automatic_pass else "forbidden",
        "visual_assets": visual_assets,
        "decision": (
            "open_autonomous_severe_visual_review"
            if automatic_pass
            else "close_fixed_tone_photographic_stress_without_rescue"
        ),
    }


__all__ = ["apply_fixed_tone", "evaluate_photographic_stress", "sha256_file"]
