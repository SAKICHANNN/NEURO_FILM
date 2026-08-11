"""U5.R2CB10 continuous min-max execution of the retained characteristic shape."""

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

SCHEMA = "neuro_film.u5_r2cb10_fujifilm_characteristic_minmax_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb10_fujifilm_characteristic_minmax_report.v1"
EXPERIMENT_ID = "U5.R2CB10"
CONTRACT_SHA256 = "e89d3a1d279bfd0580c08feefe46006665833e88fe4e61314ebc0ac2e2d8096c"


class FujifilmCharacteristicMinmaxError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise FujifilmCharacteristicMinmaxError("CB10 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise FujifilmCharacteristicMinmaxError("CB10 contract structure drift")
    return payload


def _anchored_curve(
    values: np.ndarray,
    curve: PchipInterpolator,
    *,
    strength: float,
    epsilon: float,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    result = x.copy()
    interior = (x > epsilon) & (x < 1.0 - epsilon)
    coordinate = (x[interior] - epsilon) / (1.0 - 2.0 * epsilon)
    mapped = (1.0 - strength) * coordinate + strength * np.asarray(
        curve(coordinate), dtype=np.float64
    )
    lower = float(np.nextafter(np.float32(epsilon), np.float32(1.0)))
    upper = float(np.nextafter(np.float32(1.0 - epsilon), np.float32(0.0)))
    result[interior] = np.clip(
        epsilon + (1.0 - 2.0 * epsilon) * mapped, lower, upper
    )
    return result


def apply_characteristic_minmax(
    source_linear: np.ndarray,
    curve: PchipInterpolator,
    *,
    strength: float,
    boundary_epsilon: float,
) -> np.ndarray:
    source = np.asarray(source_linear)
    if (
        source.dtype != np.float32
        or source.ndim < 2
        or source.shape[-1] != 3
        or not np.isfinite(source).all()
        or np.min(source) < 0.0
        or np.max(source) > 1.0
        or strength <= 0.0
        or strength > 1.0
        or boundary_epsilon <= 0.0
        or boundary_epsilon >= 0.5
    ):
        raise FujifilmCharacteristicMinmaxError("CB10 input is invalid")
    before = source.copy()
    x = np.asarray(source, dtype=np.float64)
    low = np.maximum(np.min(x, axis=-1), boundary_epsilon)
    high = np.minimum(np.max(x, axis=-1), 1.0 - boundary_epsilon)
    mapped_low = _anchored_curve(
        low, curve, strength=strength, epsilon=boundary_epsilon
    )
    mapped_high = _anchored_curve(
        high, curve, strength=strength, epsilon=boundary_epsilon
    )
    span = high - low
    coordinate = np.divide(
        x - low[..., None],
        span[..., None],
        out=np.zeros_like(x),
        where=span[..., None] > 0.0,
    )
    interior_output = mapped_low[..., None] + coordinate * (
        mapped_high - mapped_low
    )[..., None]
    neutral = span == 0.0
    interior_output[neutral] = mapped_low[neutral, None]
    output64 = np.where(
        (x <= boundary_epsilon) | (x >= 1.0 - boundary_epsilon),
        x,
        interior_output,
    )
    output = np.asarray(output64, dtype=np.float32)
    if (
        not np.array_equal(source, before)
        or not np.isfinite(output).all()
        or np.min(output) < 0.0
        or np.max(output) > 1.0
        or _new_boundary_fraction(source, output, boundary_epsilon) != 0.0
    ):
        raise FujifilmCharacteristicMinmaxError("CB10 intrinsic invariant failed")
    return output


def _inputs(config: Mapping[str, Any], root: Path):
    parent = config["parents"]
    decision = _load_exact_json(
        root, parent["cb9_decision_path"], parent["cb9_decision_sha256"]
    )
    if decision.get("decision") != parent["cb9_required_decision"]:
        raise FujifilmCharacteristicMinmaxError("CB10 parent decision drift")
    cb8_shape = dict(config)
    cb8_shape["parents"] = {
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
    return _validate_inputs(cb8_shape, root)


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    cb6, population = _inputs(config, root)
    curve = _compiled_curve(cb6)
    strength = float(config["operator"]["nominal_strength"])
    epsilon = float(config["operator"]["boundary_epsilon"])
    gates = config["automatic_gates"]
    evaluation = config["evaluation"]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    visuals = []
    for source_row in population:
        source_path = root / source_row["decoded_path"]
        if hash_file(source_path) != source_row["decoded_sha256"]:
            raise FujifilmCharacteristicMinmaxError("CB10 source drift")
        source = _load_rgb(
            source_path, maximum_long_edge=int(config["population"]["maximum_long_edge"])
        )
        output = apply_characteristic_minmax(
            source, curve, strength=strength, boundary_epsilon=epsilon
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
        name = f"{source_row['id']}__characteristic_minmax.png"
        rows.append(
            {
                "id": source_row["id"],
                "make": source_row["make"],
                "source_sha256": source_row["decoded_sha256"],
                "output_path": name,
                "output_sha256": _save_rgb(output, output_dir / name),
                "style_delta_e76_median": _median_delta_e76(source_sample, output_sample),
                "joint_basic_residual_delta_e76_median": _median_delta_e76(
                    basic_sample, output_sample
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
        visuals.append((source_row["id"], source, output))
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
        "maximum_new_hard_boundary_fraction": max(
            row["new_hard_boundary_fraction"] for row in rows
        ),
        "maximum_p999_gradient_ratio_vs_source": max(
            row["p999_gradient_ratio_vs_source"] for row in rows
        ),
        "maximum_adjacent_lstar_gradient_sign_inversion_fraction": max(
            row["adjacent_lstar_gradient_sign_inversion_fraction"] for row in rows
        ),
    }
    checks = {
        "source_count": metrics["source_count"] == int(gates["source_count_exact"]),
        "camera_make_count": metrics["camera_make_count"]
        == int(gates["camera_make_count_exact"]),
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
    }
    passed = all(checks.values())
    visual_assets = {}
    if passed:
        for row_id, source, output in visuals:
            visual_assets[row_id] = {
                "source_sha256": _save_rgb(source, output_dir / row_id / "source.png"),
                "candidate_sha256": _save_rgb(
                    output, output_dir / row_id / "candidate.png"
                ),
                "difference_sha256": _save_rgb(
                    np.clip(np.abs(output - source) * 4.0, 0.0, 1.0),
                    output_dir / row_id / "difference.png",
                ),
            }
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
        "visual_assets": visual_assets,
        "decision": "open_autonomous_severe_visual_review"
        if passed
        else "close_characteristic_minmax_without_rescue",
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


def write_report(report: Mapping[str, Any], output: Path) -> str:
    payload = canonical_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
