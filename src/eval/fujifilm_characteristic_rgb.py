"""U5.R2CB8 intrinsic RGB-cube execution of the retained characteristic shape."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import PchipInterpolator

from src.color_engine.lab import linear_rgb_to_lab
from src.eval.fujifilm_characteristic_photographic import (
    _compiled_curve,
    _load_exact_json,
)
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json, hash_file
from src.eval.fujifilm_e6_dye_operator_photographic import (
    _gradient_p999_ratio,
    _median_delta_e76,
    _new_boundary_fraction,
)
from src.eval.kci_velvia_tone_photographic_stress import _load_rgb, _save_rgb
from src.roll2film.baselines import fit_joint_basic_adjustment

SCHEMA = "neuro_film.u5_r2cb8_fujifilm_characteristic_rgb_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb8_fujifilm_characteristic_rgb_report.v1"
EXPERIMENT_ID = "U5.R2CB8"
CONTRACT_SHA256 = "3886642bd944732f6c1ea4a32d45a6cabc4862a7c4e1a199d993bb0e78512862"


class FujifilmCharacteristicRgbError(RuntimeError):
    """Raised when a CB8 contract, source or intrinsic-cube invariant fails."""


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise FujifilmCharacteristicRgbError("CB8 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise FujifilmCharacteristicRgbError("CB8 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise FujifilmCharacteristicRgbError("CB8 contract structure drift")
    for section, keys in (
        (
            "parents",
            (
                "cb6_decision_path",
                "cb6_contract_path",
                "cb7_decision_path",
            ),
        ),
        (
            "population",
            ("decision_path", "manifest_path", "visual_review_path"),
        ),
    ):
        for key in keys:
            _relative_path(payload[section][key])
    return payload


def apply_characteristic_rgb(
    source_linear: np.ndarray,
    curve: PchipInterpolator,
    *,
    strength: float,
) -> np.ndarray:
    source = np.asarray(source_linear)
    if (
        source.dtype != np.float32
        or source.ndim < 2
        or source.shape[-1] != 3
        or not np.isfinite(source).all()
        or np.min(source) < 0.0
        or np.max(source) > 1.0
        or not np.isfinite(strength)
        or strength <= 0.0
        or strength > 1.0
    ):
        raise FujifilmCharacteristicRgbError("CB8 source or strength is invalid")
    source_before = source.copy()
    mapped = np.asarray(curve(np.asarray(source, dtype=np.float64)), dtype=np.float32)
    output = np.float32(1.0 - strength) * source + np.float32(strength) * mapped
    if (
        not np.array_equal(source, source_before)
        or not np.isfinite(output).all()
        or np.min(output) < 0.0
        or np.max(output) > 1.0
    ):
        raise FujifilmCharacteristicRgbError("CB8 intrinsic cube invariant failed")
    return np.asarray(output, dtype=np.float32)


def _minimum_derivative(
    curve: PchipInterpolator, *, strength: float, count: int
) -> float:
    if count < 3 or strength <= 0.0 or strength > 1.0:
        raise FujifilmCharacteristicRgbError("CB8 derivative probe is invalid")
    coordinate = np.linspace(0.0, 1.0, count, dtype=np.float64)
    derivative = (1.0 - strength) + strength * np.asarray(
        curve.derivative()(coordinate), dtype=np.float64
    )
    if not np.all(np.isfinite(derivative)):
        raise FujifilmCharacteristicRgbError("CB8 derivative is non-finite")
    return float(np.min(derivative))


def _sample_indices(count: int, maximum: int) -> np.ndarray:
    if count <= 0 or maximum <= 0:
        raise FujifilmCharacteristicRgbError("CB8 sample count is invalid")
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


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
            np.count_nonzero(material & (source_delta * output_delta < -(epsilon**2)))
        )
    return 0.0 if comparisons == 0 else inversions / comparisons


def _validate_inputs(
    config: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parents = config["parents"]
    cb6 = _load_exact_json(
        root, parents["cb6_decision_path"], parents["cb6_decision_sha256"]
    )
    cb7 = _load_exact_json(
        root, parents["cb7_decision_path"], parents["cb7_decision_sha256"]
    )
    if (
        cb6.get("decision") != parents["cb6_required_decision"]
        or cb6.get("stable_evidence_id") != parents["cb6_required_stable_evidence_id"]
        or cb7.get("decision") != parents["cb7_required_decision"]
    ):
        raise FujifilmCharacteristicRgbError("CB8 parent decisions are not exact")
    cb6_contract = _load_exact_json(
        root, parents["cb6_contract_path"], parents["cb6_contract_sha256"]
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
        raise FujifilmCharacteristicRgbError("CB8 source population is not eligible")
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
        raise FujifilmCharacteristicRgbError("CB8 source inventory drift")
    return cb6_contract, rows


def _contact_sheet(rows: list[dict[str, Any]], output: Path) -> str:
    width, height, header = 420, 300, 28
    canvas = Image.new("RGB", (width * 3, header + height * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(("source", "characteristic RGB", "difference x4")):
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


def evaluate_characteristic_rgb(
    config: Mapping[str, Any], root: Path, output_dir: Path
) -> dict[str, Any]:
    cb6_contract, population = _validate_inputs(config, root)
    curve = _compiled_curve(cb6_contract)
    strength = float(config["operator"]["strength"])
    evaluation = config["evaluation"]
    gates = config["automatic_gates"]
    derivative = _minimum_derivative(
        curve,
        strength=strength,
        count=int(evaluation["derivative_probe_count"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    for source_row in population:
        source_path = root / _relative_path(source_row["decoded_path"])
        if (
            not source_path.is_file()
            or hash_file(source_path) != source_row["decoded_sha256"]
        ):
            raise FujifilmCharacteristicRgbError("CB8 source image drift")
        source = _load_rgb(
            source_path,
            maximum_long_edge=int(config["population"]["maximum_long_edge"]),
        )
        output = apply_characteristic_rgb(source, curve, strength=strength)
        source_lab = linear_rgb_to_lab(source, working_space="linear_srgb")
        output_lab = linear_rgb_to_lab(output, working_space="linear_srgb")
        indices = _sample_indices(
            source.shape[0] * source.shape[1],
            int(evaluation["colour_metric_max_samples"]),
        )
        source_sample = source.reshape(-1, 3)[indices]
        output_sample = output.reshape(-1, 3)[indices]
        basic = fit_joint_basic_adjustment(source_sample, output_sample)
        basic_sample = np.clip(basic.apply(source_sample), 0.0, 1.0)
        style = _median_delta_e76(source_sample, output_sample)
        non_basic = _median_delta_e76(basic_sample, output_sample)
        output_name = f"{source_row['id']}__characteristic_rgb.png"
        output_sha = _save_rgb(output, output_dir / output_name)
        row = {
            "id": source_row["id"],
            "make": source_row["make"],
            "source_sha256": source_row["decoded_sha256"],
            "shape": list(source.shape),
            "output_path": output_name,
            "output_sha256": output_sha,
            "style_delta_e76_median": style,
            "joint_basic_residual_delta_e76_median": non_basic,
            "new_hard_boundary_fraction": _new_boundary_fraction(
                source, output, float(evaluation["hard_boundary_epsilon"])
            ),
            "p999_gradient_ratio_vs_source": _gradient_p999_ratio(source, output),
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
        "minimum_curve_derivative": derivative,
        "population_median_style_delta_e76": float(
            np.median([row["style_delta_e76_median"] for row in rows])
        ),
        "population_median_joint_basic_residual_delta_e76": float(
            np.median([row["joint_basic_residual_delta_e76_median"] for row in rows])
        ),
        "rows_joint_basic_residual_delta_e76_ge_0p25": sum(
            row["joint_basic_residual_delta_e76_median"] >= 0.25 for row in rows
        ),
        "maximum_new_hard_boundary_fraction": max(
            row["new_hard_boundary_fraction"] for row in rows
        ),
        "maximum_p999_gradient_ratio_vs_source": max(
            row["p999_gradient_ratio_vs_source"] for row in rows
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
        "finite_bounded_rgb": bool(
            all(
                np.isfinite(row["output_linear_minimum"])
                and np.isfinite(row["output_linear_maximum"])
                for row in rows
            )
        )
        is bool(gates["all_outputs_finite"])
        and metrics["output_linear_minimum"] >= float(gates["output_linear_minimum"])
        and metrics["output_linear_maximum"] <= float(gates["output_linear_maximum"]),
        "positive_jacobian": metrics["minimum_curve_derivative"]
        >= float(gates["minimum_curve_derivative"]),
        "new_boundaries": metrics["maximum_new_hard_boundary_fraction"]
        <= float(gates["maximum_new_hard_boundary_fraction"]),
        "gradient_magnitude": metrics["maximum_p999_gradient_ratio_vs_source"]
        <= float(gates["maximum_p999_gradient_ratio_vs_source"]),
        "gradient_order": metrics[
            "maximum_adjacent_lstar_gradient_sign_inversion_fraction"
        ]
        <= float(gates["maximum_adjacent_lstar_gradient_sign_inversion_fraction"]),
        "visible_style": metrics["population_median_style_delta_e76"]
        >= float(gates["minimum_population_median_style_delta_e76"]),
        "non_basic": metrics["population_median_joint_basic_residual_delta_e76"]
        >= float(gates["minimum_population_median_joint_basic_residual_delta_e76"])
        and metrics["rows_joint_basic_residual_delta_e76_ge_0p25"]
        >= int(gates["minimum_rows_joint_basic_residual_delta_e76_ge_0p25"]),
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
        "parent_cb6_stable_evidence_id": config["parents"][
            "cb6_required_stable_evidence_id"
        ],
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
            else "close_intrinsic_rgb_characteristic_without_rescue"
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
    "FujifilmCharacteristicRgbError",
    "_minimum_derivative",
    "apply_characteristic_rgb",
    "evaluate_characteristic_rgb",
    "load_contract",
    "write_report",
]
