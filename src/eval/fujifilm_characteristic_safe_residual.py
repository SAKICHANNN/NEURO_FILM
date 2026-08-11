"""U5.R2CB9 analytical safe execution of the retained characteristic residual."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from src.color_engine.lab import linear_rgb_to_lab
from src.eval.fujifilm_characteristic_photographic import (
    _compiled_curve,
    _load_exact_json,
)
from src.eval.fujifilm_characteristic_rgb import (
    _contact_sheet,
    _gradient_inversion_fraction,
    _sample_indices,
    _validate_inputs,
)
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json, hash_file
from src.eval.fujifilm_e6_dye_operator_photographic import (
    _gradient_p999_ratio,
    _median_delta_e76,
    _new_boundary_fraction,
)
from src.eval.kci_velvia_tone_photographic_stress import _load_rgb, _save_rgb
from src.roll2film.baselines import fit_joint_basic_adjustment

SCHEMA = "neuro_film.u5_r2cb9_fujifilm_characteristic_safe_residual_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb9_fujifilm_characteristic_safe_residual_report.v1"
EXPERIMENT_ID = "U5.R2CB9"
CONTRACT_SHA256 = "7081cb8079af12a0cb6b2e657046290d01fbdd806de105e0d244af27031bd60e"


class FujifilmCharacteristicSafeResidualError(RuntimeError):
    """Raised when the CB9 contract or analytical execution invariant fails."""


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise FujifilmCharacteristicSafeResidualError(
            "CB9 paths must be repository-relative"
        )
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise FujifilmCharacteristicSafeResidualError("CB9 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise FujifilmCharacteristicSafeResidualError("CB9 contract structure drift")
    for section, keys in (
        ("parents", ("cb6_contract_path", "cb8_decision_path")),
        ("population", ("decision_path", "manifest_path", "visual_review_path")),
    ):
        for key in keys:
            _relative_path(payload[section][key])
    return payload


def apply_characteristic_safe_residual(
    source_linear: np.ndarray,
    curve: PchipInterpolator,
    *,
    nominal_strength: float,
    boundary_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one maximum-safe scalar per pixel along the exact CB8 RGB residual."""

    source = np.asarray(source_linear)
    if (
        source.dtype != np.float32
        or source.ndim < 2
        or source.shape[-1] != 3
        or not np.isfinite(source).all()
        or np.min(source) < 0.0
        or np.max(source) > 1.0
        or not np.isfinite(nominal_strength)
        or nominal_strength <= 0.0
        or nominal_strength > 1.0
        or not np.isfinite(boundary_epsilon)
        or boundary_epsilon <= 0.0
        or boundary_epsilon >= 0.5
    ):
        raise FujifilmCharacteristicSafeResidualError("CB9 input is invalid")
    source_before = source.copy()
    source64 = np.asarray(source, dtype=np.float64)
    mapped = np.asarray(curve(source64), dtype=np.float64)
    residual = float(nominal_strength) * (mapped - source64)
    alpha = np.ones(source.shape[:-1], dtype=np.float64)
    lower = float(np.nextafter(np.float32(boundary_epsilon), np.float32(1.0)))
    upper = float(
        np.nextafter(np.float32(1.0 - boundary_epsilon), np.float32(0.0))
    )
    for channel in range(3):
        value = source64[..., channel]
        delta = residual[..., channel]
        outward_low = (value <= boundary_epsilon) & (delta < 0.0)
        outward_high = (value >= 1.0 - boundary_epsilon) & (delta > 0.0)
        alpha[outward_low | outward_high] = 0.0
        moving_low = (value > boundary_epsilon) & (delta < 0.0)
        moving_high = (value < 1.0 - boundary_epsilon) & (delta > 0.0)
        low_limit = np.divide(
            value - lower,
            -delta,
            out=np.full_like(value, np.inf),
            where=moving_low,
        )
        high_limit = np.divide(
            upper - value,
            delta,
            out=np.full_like(value, np.inf),
            where=moving_high,
        )
        alpha = np.minimum(alpha, np.where(moving_low, low_limit, np.inf))
        alpha = np.minimum(alpha, np.where(moving_high, high_limit, np.inf))
    alpha = np.clip(alpha, 0.0, 1.0)
    output = np.asarray(source64 + alpha[..., None] * residual, dtype=np.float32)
    if (
        not np.array_equal(source, source_before)
        or not np.isfinite(output).all()
        or np.min(output) < 0.0
        or np.max(output) > 1.0
        or _new_boundary_fraction(source, output, boundary_epsilon) != 0.0
    ):
        raise FujifilmCharacteristicSafeResidualError(
            "CB9 analytical boundary invariant failed"
        )
    return output, np.asarray(alpha, dtype=np.float32)


def _validate_cb9_inputs(
    config: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parents = config["parents"]
    decision = _load_exact_json(
        root, parents["cb8_decision_path"], parents["cb8_decision_sha256"]
    )
    if (
        decision.get("decision") != parents["cb8_required_decision"]
        or decision.get("stable_evidence_id")
        != parents["cb8_required_stable_evidence_id"]
    ):
        raise FujifilmCharacteristicSafeResidualError("CB9 parent decision drift")
    cb8_shape = dict(config)
    cb8_shape["parents"] = {
        "cb6_decision_path": "configs/u5_r2cb6_fujifilm_characteristic_forward_proxy_decision_v1.json",
        "cb6_decision_sha256": "5bf79d2ebaa40f3187cd3bc62686b1ee90a6b3d8142400d663e076eb37d24297",
        "cb6_required_decision": "retain_characteristic_constrained_forward_mechanism",
        "cb6_required_stable_evidence_id": "f4e3bec72b673379103617ac2b247eed88cd500c8159a984a4f3237eb5379bae",
        "cb6_contract_path": parents["cb6_contract_path"],
        "cb6_contract_sha256": parents["cb6_contract_sha256"],
        "cb7_decision_path": "configs/u5_r2cb7_fujifilm_characteristic_photographic_decision_v1.json",
        "cb7_decision_sha256": "1abeab7e5eeae89114b50588f7c1f9deb85d20a6de571ca358f12c967d4e5cb2",
        "cb7_required_decision": "close_characteristic_photographic_compiler_without_rescue",
    }
    return _validate_inputs(cb8_shape, root)


def evaluate_characteristic_safe_residual(
    config: Mapping[str, Any], root: Path, output_dir: Path
) -> dict[str, Any]:
    cb6_contract, population = _validate_cb9_inputs(config, root)
    curve = _compiled_curve(cb6_contract)
    strength = float(config["operator"]["nominal_strength"])
    evaluation = config["evaluation"]
    epsilon = float(evaluation["hard_boundary_epsilon"])
    gates = config["automatic_gates"]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    for source_row in population:
        source_path = root / _relative_path(source_row["decoded_path"])
        if not source_path.is_file() or hash_file(source_path) != source_row["decoded_sha256"]:
            raise FujifilmCharacteristicSafeResidualError("CB9 source image drift")
        source = _load_rgb(
            source_path,
            maximum_long_edge=int(config["population"]["maximum_long_edge"]),
        )
        output, alpha = apply_characteristic_safe_residual(
            source,
            curve,
            nominal_strength=strength,
            boundary_epsilon=epsilon,
        )
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
        output_name = f"{source_row['id']}__characteristic_safe_residual.png"
        row = {
            "id": source_row["id"],
            "make": source_row["make"],
            "source_sha256": source_row["decoded_sha256"],
            "shape": list(source.shape),
            "output_path": output_name,
            "output_sha256": _save_rgb(output, output_dir / output_name),
            "style_delta_e76_median": _median_delta_e76(source_sample, output_sample),
            "joint_basic_residual_delta_e76_median": _median_delta_e76(
                basic_sample, output_sample
            ),
            "median_applied_alpha": float(np.median(alpha)),
            "fraction_alpha_below_0p99": float(np.mean(alpha < 0.99)),
            "new_hard_boundary_fraction": _new_boundary_fraction(source, output, epsilon),
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
        "population_median_style_delta_e76": float(
            np.median([row["style_delta_e76_median"] for row in rows])
        ),
        "population_median_joint_basic_residual_delta_e76": float(
            np.median([row["joint_basic_residual_delta_e76_median"] for row in rows])
        ),
        "rows_joint_basic_residual_delta_e76_ge_0p25": sum(
            row["joint_basic_residual_delta_e76_median"] >= 0.25 for row in rows
        ),
        "population_median_applied_alpha": float(
            np.median([row["median_applied_alpha"] for row in rows])
        ),
        "population_median_fraction_alpha_below_0p99": float(
            np.median([row["fraction_alpha_below_0p99"] for row in rows])
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
        "finite_bounded_rgb": all(
            np.isfinite(row["output_linear_minimum"])
            and np.isfinite(row["output_linear_maximum"])
            for row in rows
        )
        and metrics["output_linear_minimum"] >= float(gates["output_linear_minimum"])
        and metrics["output_linear_maximum"] <= float(gates["output_linear_maximum"]),
        "new_boundaries": metrics["maximum_new_hard_boundary_fraction"]
        <= float(gates["maximum_new_hard_boundary_fraction"]),
        "gradient_magnitude": metrics["maximum_p999_gradient_ratio_vs_source"]
        <= float(gates["maximum_p999_gradient_ratio_vs_source"]),
        "gradient_order": metrics["maximum_adjacent_lstar_gradient_sign_inversion_fraction"]
        <= float(gates["maximum_adjacent_lstar_gradient_sign_inversion_fraction"]),
        "visible_style": metrics["population_median_style_delta_e76"]
        >= float(gates["minimum_population_median_style_delta_e76"]),
        "non_basic": metrics["population_median_joint_basic_residual_delta_e76"]
        >= float(gates["minimum_population_median_joint_basic_residual_delta_e76"])
        and metrics["rows_joint_basic_residual_delta_e76_ge_0p25"]
        >= int(gates["minimum_rows_joint_basic_residual_delta_e76_ge_0p25"]),
        "effect_retention": metrics["population_median_applied_alpha"]
        >= float(gates["minimum_population_median_applied_alpha"])
        and metrics["population_median_fraction_alpha_below_0p99"]
        <= float(gates["maximum_population_median_fraction_alpha_below_0p99"]),
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
        "parent_cb8_stable_evidence_id": config["parents"][
            "cb8_required_stable_evidence_id"
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
            else "close_analytical_safe_residual_without_rescue"
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
    "FujifilmCharacteristicSafeResidualError",
    "apply_characteristic_safe_residual",
    "evaluate_characteristic_safe_residual",
    "load_contract",
    "write_report",
]
