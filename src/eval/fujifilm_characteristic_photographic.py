"""U5.R2CB7 photographic test for the retained reversal characteristic shape."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar

from src.color_engine.gamut import compress_source_to_working_gamut
from src.color_engine.lab import lab_to_linear_rgb, linear_rgb_to_lab
from src.color_engine.srgb_transfer import linear_srgb_to_encoded
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json, hash_file
from src.eval.kci_velvia_tone_photographic_stress import (
    _gradient_inversion_fraction,
    _load_rgb,
    _save_rgb,
)

SCHEMA = "neuro_film.u5_r2cb7_fujifilm_characteristic_photographic_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb7_fujifilm_characteristic_photographic_report.v1"
EXPERIMENT_ID = "U5.R2CB7"
CONTRACT_SHA256 = "6871ea260a8aaa1463fd48e8cddaf32224e6532908940cadfb66ee34f4a2c5e0"


class FujifilmCharacteristicPhotographicError(RuntimeError):
    """Raised when a CB7 contract, source or execution invariant fails."""


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise FujifilmCharacteristicPhotographicError(
            "CB7 paths must be repository-relative"
        )
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise FujifilmCharacteristicPhotographicError("CB7 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise FujifilmCharacteristicPhotographicError("CB7 contract structure drift")
    for section, keys in (
        ("parent", ("decision_path", "contract_path")),
        (
            "population",
            ("decision_path", "manifest_path", "visual_review_path"),
        ),
    ):
        for key in keys:
            _relative_path(payload[section][key])
    return payload


def _load_exact_json(root: Path, relative: str, expected_hash: str) -> Any:
    path = root / _relative_path(relative)
    if not path.is_file() or hash_file(path) != expected_hash:
        raise FujifilmCharacteristicPhotographicError(
            f"CB7 input identity mismatch: {relative}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _compiled_curve(parent_contract: Mapping[str, Any]) -> PchipInterpolator:
    source = parent_contract["source_observation"]
    exposure = np.asarray(source["log_exposure"], dtype=np.float64)
    coordinate = (exposure - exposure[0]) / (exposure[-1] - exposure[0])
    densities = np.stack(
        [
            np.asarray(source["density"][name], dtype=np.float64)
            for name in (
                "blue_sensitive_yellow_dye",
                "green_sensitive_magenta_dye",
                "red_sensitive_cyan_dye",
            )
        ],
        axis=0,
    )
    transmission = np.power(10.0, -densities)
    normalized = (transmission - transmission[:, :1]) / (
        transmission[:, -1:] - transmission[:, :1]
    )
    mean = np.mean(normalized, axis=0)
    if (
        coordinate.shape != mean.shape
        or coordinate[0] != 0.0
        or coordinate[-1] != 1.0
        or not np.all(np.diff(coordinate) > 0.0)
        or not np.all(np.diff(mean) >= 0.0)
        or not np.isclose(mean[0], 0.0)
        or not np.isclose(mean[-1], 1.0)
    ):
        raise FujifilmCharacteristicPhotographicError(
            "CB7 compiled characteristic curve is invalid"
        )
    return PchipInterpolator(coordinate, mean, extrapolate=False)


def apply_characteristic_tone(
    source_linear: np.ndarray,
    curve: PchipInterpolator,
    *,
    strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(source_linear)
    if (
        source.dtype != np.float32
        or source.ndim != 3
        or source.shape[2] != 3
        or not np.isfinite(source).all()
        or np.min(source) < 0.0
        or np.max(source) > 1.0
        or not np.isfinite(strength)
        or strength <= 0.0
        or strength > 1.0
    ):
        raise FujifilmCharacteristicPhotographicError(
            "CB7 source or strength is invalid"
        )
    source_before = source.copy()
    source_lab = linear_rgb_to_lab(source, working_space="linear_srgb")
    coordinate = np.asarray(source_lab[..., 0], dtype=np.float64) / 100.0
    mapped = np.asarray(curve(np.clip(coordinate, 0.0, 1.0)), dtype=np.float32)
    target_lab = source_lab.copy()
    target_lab[..., 0] = np.float32(100.0) * (
        np.float32(1.0 - strength) * np.asarray(coordinate, dtype=np.float32)
        + np.float32(strength) * mapped
    )
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
        raise FujifilmCharacteristicPhotographicError(
            "CB7 gamut-safe output invariant failed"
        )
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


def _sample_indices(count: int, maximum: int) -> np.ndarray:
    if count <= 0 or maximum <= 0:
        raise FujifilmCharacteristicPhotographicError("CB7 sample count is invalid")
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _gamma_residual(
    source_lstar: np.ndarray,
    output_lstar: np.ndarray,
    sample_indices: np.ndarray,
    bounds: tuple[float, float],
) -> tuple[float, float]:
    source = (
        np.asarray(source_lstar, dtype=np.float64).reshape(-1)[sample_indices] / 100.0
    )
    target = (
        np.asarray(output_lstar, dtype=np.float64).reshape(-1)[sample_indices] / 100.0
    )
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise FujifilmCharacteristicPhotographicError("CB7 gamma sample is invalid")
    result = minimize_scalar(
        lambda gamma: float(np.mean((np.power(source, gamma) - target) ** 2)),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 256},
    )
    if not result.success or not np.isfinite(result.x):
        raise FujifilmCharacteristicPhotographicError("CB7 gamma fit failed")
    residual = np.abs(np.power(source, result.x) - target) * 100.0
    return float(result.x), float(np.median(residual))


def _thumbnail(encoded: np.ndarray, width: int, height: int) -> Image.Image:
    image = Image.fromarray(
        np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB"
    )
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    return image


def _contact_sheet(rows: list[dict[str, Any]], output: Path) -> str:
    width, height, header = 360, 260, 28
    canvas = Image.new("RGB", (width * 3, header + height * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(("source", "characteristic", "difference x4")):
        draw.text((column * width + 8, 8), label, fill="black")
    for index, row in enumerate(rows):
        for column, key in enumerate(("source", "candidate", "difference")):
            with Image.open(row["visual_paths"][key]) as image:
                image.load()
                thumb = image.copy()
            thumb.thumbnail((width, height), Image.Resampling.LANCZOS)
            x = column * width + (width - thumb.width) // 2
            y = header + index * height + (height - thumb.height) // 2
            canvas.paste(thumb, (x, y))
        draw.text((5, header + index * height + 5), row["id"], fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, compress_level=6)
    return hash_file(output)


def _validate_inputs(
    config: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parent = config["parent"]
    decision = _load_exact_json(
        root, parent["decision_path"], parent["decision_sha256"]
    )
    if (
        decision.get("decision") != parent["required_decision"]
        or decision.get("stable_evidence_id") != parent["required_stable_evidence_id"]
    ):
        raise FujifilmCharacteristicPhotographicError("CB7 parent is not retained")
    parent_contract = _load_exact_json(
        root, parent["contract_path"], parent["contract_sha256"]
    )
    population = config["population"]
    source_decision = _load_exact_json(
        root, population["decision_path"], population["decision_sha256"]
    )
    manifest = _load_exact_json(
        root, population["manifest_path"], population["manifest_sha256"]
    )
    visual = _load_exact_json(
        root, population["visual_review_path"], population["visual_review_sha256"]
    )
    if (
        source_decision.get("result", {}).get("decision")
        != population["required_decision"]
        or visual.get("decision") != population["required_visual_decision"]
    ):
        raise FujifilmCharacteristicPhotographicError(
            "CB7 source population is not eligible"
        )
    excluded = set(population["exclude_ids"])
    eligible = tuple(visual.get("eligible_ids", ()))
    rows = [
        dict(row)
        for row in manifest
        if row.get("id") in eligible and row.get("id") not in excluded
    ]
    if len(rows) != int(population["source_count_exact"]) or len(
        {row["make"] for row in rows}
    ) != int(population["camera_make_count_exact"]):
        raise FujifilmCharacteristicPhotographicError("CB7 source inventory drift")
    return parent_contract, rows


def evaluate_photographic(
    config: Mapping[str, Any], root: Path, output_dir: Path
) -> dict[str, Any]:
    parent_contract, population = _validate_inputs(config, root)
    curve = _compiled_curve(parent_contract)
    strength = float(config["operator"]["strength"])
    gates = config["automatic_gates"]
    evaluation = config["evaluation"]
    gamma_bounds = tuple(float(value) for value in evaluation["gamma_bounds"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    for source_row in population:
        source_path = root / _relative_path(source_row["decoded_path"])
        if (
            not source_path.is_file()
            or hash_file(source_path) != source_row["decoded_sha256"]
        ):
            raise FujifilmCharacteristicPhotographicError("CB7 source image drift")
        source = _load_rgb(
            source_path,
            maximum_long_edge=int(config["population"]["maximum_long_edge"]),
        )
        output, source_lab, output_lab, scale = apply_characteristic_tone(
            source, curve, strength=strength
        )
        indices = _sample_indices(
            source.shape[0] * source.shape[1],
            int(evaluation["colour_metric_max_samples"]),
        )
        gamma, gamma_residual = _gamma_residual(
            source_lab[..., 0], output_lab[..., 0], indices, gamma_bounds
        )
        source_encoded = linear_srgb_to_encoded(source)
        output_encoded = linear_srgb_to_encoded(output)
        epsilon = float(evaluation["hard_boundary_epsilon"])
        source_boundary = (source_encoded <= epsilon) | (
            source_encoded >= 1.0 - epsilon
        )
        output_boundary = (output_encoded <= epsilon) | (
            output_encoded >= 1.0 - epsilon
        )
        effect = np.abs(output_lab[..., 0] - source_lab[..., 0])
        output_name = f"{source_row['id']}__characteristic.png"
        output_sha = _save_rgb(output, output_dir / output_name)
        row = {
            "id": source_row["id"],
            "make": source_row["make"],
            "source_sha256": source_row["decoded_sha256"],
            "shape": list(source.shape),
            "output_path": output_name,
            "output_sha256": output_sha,
            "p95_absolute_lstar_effect": float(np.quantile(effect, 0.95)),
            "maximum_absolute_lstar_effect": float(np.max(effect)),
            "best_gamma": gamma,
            "median_gamma_residual_delta_lstar": gamma_residual,
            "median_gamut_scale": float(np.median(scale)),
            "fraction_gamut_scale_below_0p99": float(
                np.mean(scale < float(evaluation["gamut_scale_material_threshold"]))
            ),
            "new_hard_boundary_fraction": float(
                np.mean(output_boundary & ~source_boundary)
            ),
            "adjacent_lstar_gradient_sign_inversion_fraction": _gradient_inversion_fraction(
                source_lab[..., 0],
                output_lab[..., 0],
                epsilon=float(evaluation["gradient_sign_epsilon_lstar"]),
            ),
            "output_linear_minimum": float(np.min(output)),
            "output_linear_maximum": float(np.max(output)),
        }
        rows.append(row)
        visual_rows.append(
            {
                "id": source_row["id"],
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
        "population_median_per_image_gamma_residual_delta_lstar": float(
            np.median([row["median_gamma_residual_delta_lstar"] for row in rows])
        ),
        "rows_gamma_residual_delta_lstar_ge_0p75": sum(
            row["median_gamma_residual_delta_lstar"] >= 0.75 for row in rows
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
        "source_count": metrics["source_count"] == int(gates["source_count_exact"]),
        "camera_make_count": metrics["camera_make_count"]
        == int(gates["camera_make_count_exact"]),
        "visible_effect": metrics["population_median_per_image_p95_lstar_effect"]
        >= float(gates["minimum_population_median_per_image_p95_lstar_effect"]),
        "bounded_effect": metrics["maximum_per_image_p95_lstar_effect"]
        <= float(gates["maximum_per_image_p95_lstar_effect"]),
        "non_gamma": metrics["population_median_per_image_gamma_residual_delta_lstar"]
        >= float(
            gates["minimum_population_median_per_image_gamma_residual_delta_lstar"]
        )
        and metrics["rows_gamma_residual_delta_lstar_ge_0p75"]
        >= int(gates["minimum_rows_gamma_residual_delta_lstar_ge_0p75"]),
        "median_gamut_scale": metrics["population_median_gamut_scale"]
        >= float(gates["minimum_population_median_gamut_scale"]),
        "gamut_scale_tail": metrics["maximum_per_image_fraction_gamut_scale_below_0p99"]
        <= float(gates["maximum_per_image_fraction_gamut_scale_below_0p99"]),
        "new_boundaries": metrics["maximum_new_hard_boundary_fraction"]
        <= float(gates["maximum_new_hard_boundary_fraction"]),
        "gradient_order": metrics[
            "maximum_adjacent_lstar_gradient_sign_inversion_fraction"
        ]
        <= float(gates["maximum_adjacent_lstar_gradient_sign_inversion_fraction"]),
        "finite_bounded_rgb": metrics["output_linear_minimum"]
        >= float(gates["output_linear_minimum"])
        and metrics["output_linear_maximum"] <= float(gates["output_linear_maximum"]),
    }
    automatic_pass = all(checks.values())
    visual_assets: dict[str, Any] = {}
    if automatic_pass:
        for visual in visual_rows:
            paths = {
                key: output_dir / visual["id"] / f"{key}.png"
                for key in ("source", "candidate", "difference")
            }
            visual_assets[visual["id"]] = {
                key: _save_rgb(visual[key], paths[key])
                for key in ("source", "candidate", "difference")
            }
            visual["visual_paths"] = {key: str(value) for key, value in paths.items()}
        visual_assets["contact_sheet_sha256"] = _contact_sheet(
            visual_rows, output_dir / "contact_sheet.png"
        )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "parent_stable_evidence_id": config["parent"]["required_stable_evidence_id"],
        "rows": rows,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "visual_review_allowed": automatic_pass,
        "visual_review_status": "pending" if automatic_pass else "forbidden",
        "visual_assets": visual_assets,
        "decision": (
            "open_autonomous_severe_visual_review"
            if automatic_pass
            else "close_characteristic_photographic_compiler_without_rescue"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


def write_report(report: Mapping[str, Any], output: Path) -> str:
    payload = canonical_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CONTRACT_SHA256",
    "FujifilmCharacteristicPhotographicError",
    "_compiled_curve",
    "_gamma_residual",
    "apply_characteristic_tone",
    "evaluate_photographic",
    "load_contract",
    "write_report",
]
