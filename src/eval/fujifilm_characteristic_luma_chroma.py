"""U5.R2CB11 exact-luminance, analytically bounded chroma execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from src.color_engine.lab import linear_rgb_to_lab
from src.eval.fujifilm_characteristic_minmax import _anchored_curve
from src.eval.fujifilm_characteristic_photographic import (
    _compiled_curve,
    _load_exact_json,
)
from src.eval.fujifilm_characteristic_rgb import (
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

SCHEMA = "neuro_film.u5_r2cb11_fujifilm_characteristic_luma_chroma_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb11_fujifilm_characteristic_luma_chroma_report.v1"
EXPERIMENT_ID = "U5.R2CB11"
CONTRACT_SHA256 = "fd6127a0c336fdfc661870d8636e8c7a8bbe741f096465e52c73a5a327213b5f"


class FujifilmCharacteristicLumaChromaError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise FujifilmCharacteristicLumaChromaError("CB11 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise FujifilmCharacteristicLumaChromaError("CB11 contract structure drift")
    return payload


def apply_characteristic_luma_chroma(
    source_linear: np.ndarray,
    curve: PchipInterpolator,
    *,
    weights: np.ndarray,
    strength: float,
    boundary_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(source_linear)
    w = np.asarray(weights, dtype=np.float64)
    if (
        source.dtype != np.float32
        or source.ndim < 2
        or source.shape[-1] != 3
        or not np.isfinite(source).all()
        or np.min(source) < 0.0
        or np.max(source) > 1.0
        or w.shape != (3,)
        or not np.isfinite(w).all()
        or np.min(w) <= 0.0
        or abs(float(np.sum(w)) - 1.0) > 1e-12
        or strength <= 0.0
        or strength > 1.0
    ):
        raise FujifilmCharacteristicLumaChromaError("CB11 input is invalid")
    before = source.copy()
    x = source.astype(np.float64)
    luminance = np.sum(x * w, axis=-1)
    mapped_luminance = _anchored_curve(
        luminance, curve, strength=strength, epsilon=boundary_epsilon
    )
    chroma = x - luminance[..., None]
    scale = np.ones(luminance.shape, dtype=np.float64)
    lower_target = float(
        np.float32(boundary_epsilon)
        + np.float32(4.0) * np.spacing(np.float32(boundary_epsilon))
    )
    upper_edge = np.float32(1.0 - boundary_epsilon)
    upper_target = float(upper_edge - np.float32(4.0) * np.spacing(upper_edge))
    for channel in range(3):
        c = chroma[..., channel]
        value = x[..., channel]
        lower = np.where(value > boundary_epsilon, lower_target, 0.0)
        upper = np.where(value < 1.0 - boundary_epsilon, upper_target, 1.0)
        positive = c > 0.0
        negative = c < 0.0
        upper_limit = np.divide(
            upper - mapped_luminance,
            c,
            out=np.full_like(c, np.inf),
            where=positive,
        )
        lower_limit = np.divide(
            mapped_luminance - lower,
            -c,
            out=np.full_like(c, np.inf),
            where=negative,
        )
        scale = np.minimum(scale, np.where(positive, upper_limit, np.inf))
        scale = np.minimum(scale, np.where(negative, lower_limit, np.inf))
    scale = np.clip(scale, 0.0, 1.0)
    limited = scale < 1.0
    scale[limited] = np.nextafter(scale[limited], 0.0)
    output = np.asarray(
        mapped_luminance[..., None] + scale[..., None] * chroma, dtype=np.float32
    )
    reconstructed = np.sum(output.astype(np.float64) * w, axis=-1)
    if (
        not np.array_equal(source, before)
        or not np.isfinite(output).all()
        or np.min(output) < 0.0
        or np.max(output) > 1.0
        or _new_boundary_fraction(source, output, boundary_epsilon) != 0.0
    ):
        raise FujifilmCharacteristicLumaChromaError("CB11 intrinsic invariant failed")
    return output, scale.astype(np.float32), reconstructed - mapped_luminance


def _inputs(config: Mapping[str, Any], root: Path):
    parent = config["parents"]
    decision = _load_exact_json(
        root, parent["cb10_decision_path"], parent["cb10_decision_sha256"]
    )
    if decision.get("decision") != parent["cb10_required_decision"]:
        raise FujifilmCharacteristicLumaChromaError("CB11 parent decision drift")
    shape = dict(config)
    shape["parents"] = {
        "cb6_decision_path": "configs/u5_r2cb6_fujifilm_characteristic_forward_proxy_decision_v1.json",
        "cb6_decision_sha256": "5bf79d2ebaa40f3187cd3bc62686b1ee90a6b3d8142400d663e076eb37d24297",
        "cb6_required_decision": "retain_characteristic_constrained_forward_mechanism",
        "cb6_required_stable_evidence_id": "f4e3bec72b673379103617ac2b247eed88cd500c8159a984a4f3237eb5379bae",
        "cb6_contract_path": parent["cb6_contract_path"],
        "cb6_contract_sha256": parent["cb6_contract_sha256"],
        "cb7_decision_path": "configs/u5_r2cb7_fujifilm_characteristic_photographic_decision_v1.json",
        "cb7_decision_sha256": "1abeab7e5eeae89114b50588f7c1f9deb85d20a6de571ca358f12c967d4e5cb2",
        "cb7_required_decision": "close_characteristic_photographic_compiler_without_rescue",
    }
    return _validate_inputs(shape, root)


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    cb6, population = _inputs(config, root)
    curve = _compiled_curve(cb6)
    operator = config["operator"]
    weights = np.asarray(operator["luminance_weights"], dtype=np.float64)
    strength = float(operator["nominal_strength"])
    epsilon = float(operator["boundary_epsilon"])
    gates = config["automatic_gates"]
    evaluation = config["evaluation"]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source_row in population:
        source_path = root / source_row["decoded_path"]
        if hash_file(source_path) != source_row["decoded_sha256"]:
            raise FujifilmCharacteristicLumaChromaError("CB11 source drift")
        source = _load_rgb(
            source_path,
            maximum_long_edge=int(config["population"]["maximum_long_edge"]),
        )
        output, scale, luma_error = apply_characteristic_luma_chroma(
            source,
            curve,
            weights=weights,
            strength=strength,
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
        name = f"{source_row['id']}__characteristic_luma_chroma.png"
        rows.append(
            {
                "id": source_row["id"],
                "make": source_row["make"],
                "source_sha256": source_row["decoded_sha256"],
                "output_path": name,
                "output_sha256": _save_rgb(output, output_dir / name),
                "style_delta_e76_median": _median_delta_e76(
                    source_sample, output_sample
                ),
                "joint_basic_residual_delta_e76_median": _median_delta_e76(
                    basic_sample, output_sample
                ),
                "median_chroma_scale": float(np.median(scale)),
                "fraction_chroma_scale_below_0p5": float(np.mean(scale < 0.5)),
                "maximum_luminance_reconstruction_error": float(
                    np.max(np.abs(luma_error))
                ),
                "new_hard_boundary_fraction": _new_boundary_fraction(
                    source, output, epsilon
                ),
                "p999_gradient_ratio_vs_source": _gradient_p999_ratio(source, output),
                "adjacent_lstar_gradient_sign_inversion_fraction": _gradient_inversion_fraction(
                    source_lab[..., 0],
                    output_lab[..., 0],
                    epsilon=float(evaluation["gradient_sign_epsilon_lstar"]),
                ),
            }
        )
    metrics = {
        "source_count": len(rows),
        "camera_make_count": len({r["make"] for r in rows}),
        "maximum_luminance_reconstruction_error": max(
            r["maximum_luminance_reconstruction_error"] for r in rows
        ),
        "maximum_new_hard_boundary_fraction": max(
            r["new_hard_boundary_fraction"] for r in rows
        ),
        "maximum_p999_gradient_ratio_vs_source": max(
            r["p999_gradient_ratio_vs_source"] for r in rows
        ),
        "maximum_adjacent_lstar_gradient_sign_inversion_fraction": max(
            r["adjacent_lstar_gradient_sign_inversion_fraction"] for r in rows
        ),
        "population_median_chroma_scale": float(
            np.median([r["median_chroma_scale"] for r in rows])
        ),
        "population_median_fraction_chroma_scale_below_0p5": float(
            np.median([r["fraction_chroma_scale_below_0p5"] for r in rows])
        ),
        "population_median_style_delta_e76": float(
            np.median([r["style_delta_e76_median"] for r in rows])
        ),
        "population_median_joint_basic_residual_delta_e76": float(
            np.median([r["joint_basic_residual_delta_e76_median"] for r in rows])
        ),
        "rows_joint_basic_residual_delta_e76_ge_0p25": sum(
            r["joint_basic_residual_delta_e76_median"] >= 0.25 for r in rows
        ),
    }
    checks = {
        "source_count": metrics["source_count"] == int(gates["source_count_exact"]),
        "camera_make_count": metrics["camera_make_count"]
        == int(gates["camera_make_count_exact"]),
        "luminance_exact": metrics["maximum_luminance_reconstruction_error"]
        <= float(gates["maximum_luminance_reconstruction_error"]),
        "new_boundaries": metrics["maximum_new_hard_boundary_fraction"]
        <= float(gates["maximum_new_hard_boundary_fraction"]),
        "gradient_magnitude": metrics["maximum_p999_gradient_ratio_vs_source"]
        <= float(gates["maximum_p999_gradient_ratio_vs_source"]),
        "gradient_order": metrics[
            "maximum_adjacent_lstar_gradient_sign_inversion_fraction"
        ]
        <= float(gates["maximum_adjacent_lstar_gradient_sign_inversion_fraction"]),
        "chroma_retention": metrics["population_median_chroma_scale"]
        >= float(gates["minimum_population_median_chroma_scale"])
        and metrics["population_median_fraction_chroma_scale_below_0p5"]
        <= float(gates["maximum_population_median_fraction_chroma_scale_below_0p5"]),
        "visible_style": metrics["population_median_style_delta_e76"]
        >= float(gates["minimum_population_median_style_delta_e76"]),
        "non_basic": metrics["population_median_joint_basic_residual_delta_e76"]
        >= float(gates["minimum_population_median_joint_basic_residual_delta_e76"])
        and metrics["rows_joint_basic_residual_delta_e76_ge_0p25"]
        >= int(gates["minimum_rows_joint_basic_residual_delta_e76_ge_0p25"]),
    }
    passed = all(checks.values())
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "rows": rows,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": passed,
        "visual_review_allowed": passed,
        "visual_review_status": "pending" if passed else "forbidden",
        "decision": "open_autonomous_severe_visual_review"
        if passed
        else "close_characteristic_luma_chroma_without_rescue",
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


def write_report(report: Mapping[str, Any], output: Path) -> str:
    payload = canonical_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
