"""CB15 exact characteristic-luminance plus AO6-chroma factorization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from src.color_engine.lab import linear_rgb_to_lab
from src.eval.characteristic_vs_ao6_fresh import _sheet
from src.eval.fixed_global_policy_confirmation import render_fixed_pair
from src.eval.fujifilm_characteristic_forward_proxy import load_contract as load_cb6
from src.eval.fujifilm_characteristic_luma_chroma import (
    apply_characteristic_luma_chroma,
)
from src.eval.fujifilm_characteristic_photographic import (
    _compiled_curve,
    _load_exact_json,
)
from src.eval.fujifilm_characteristic_rgb import (
    _gradient_inversion_fraction,
    _sample_indices,
)
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json, hash_file
from src.eval.fujifilm_e6_dye_operator_photographic import (
    _gradient_p999_ratio,
    _median_delta_e76,
    _new_boundary_fraction,
)
from src.eval.kci_velvia_tone_photographic_stress import _load_rgb, _save_rgb
from src.film_physics.profile_consumer import validate_standalone_profile_artifact

SCHEMA = "neuro_film.u5_r2cb15_characteristic_ao6_factorized_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb15_characteristic_ao6_factorized_report.v1"
EXPERIMENT_ID = "U5.R2CB15"


class CharacteristicAo6FactorizedError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise CharacteristicAo6FactorizedError("CB15 contract structure drift")
    return payload


def apply_characteristic_ao6_factorized(
    source_linear: np.ndarray,
    ao6_linear: np.ndarray,
    curve: PchipInterpolator,
    *,
    weights: np.ndarray,
    strength: float,
    boundary_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(source_linear)
    ao6 = np.asarray(ao6_linear)
    source_before = source.copy()
    ao6_before = ao6.copy()
    w = np.asarray(weights, dtype=np.float64)
    if (
        source.dtype != np.float32
        or ao6.dtype != np.float32
        or source.shape != ao6.shape
        or source.ndim < 2
        or source.shape[-1] != 3
        or not np.isfinite(source).all()
        or not np.isfinite(ao6).all()
        or np.min(source) < 0.0
        or np.max(source) > 1.0
        or np.min(ao6) < 0.0
        or np.max(ao6) > 1.0
    ):
        raise CharacteristicAo6FactorizedError("CB15 input is invalid")
    characteristic, _, _ = apply_characteristic_luma_chroma(
        source,
        curve,
        weights=w,
        strength=strength,
        boundary_epsilon=boundary_epsilon,
    )
    mapped_luma = np.sum(characteristic.astype(np.float64) * w, axis=-1)
    ao6_f64 = ao6.astype(np.float64)
    ao6_luma = np.sum(ao6_f64 * w, axis=-1)
    chroma = ao6_f64 - ao6_luma[..., None]
    scale = np.ones(mapped_luma.shape, dtype=np.float64)
    lower_target = float(
        np.float32(boundary_epsilon)
        + np.float32(4.0) * np.spacing(np.float32(boundary_epsilon))
    )
    upper_edge = np.float32(1.0 - boundary_epsilon)
    upper_target = float(upper_edge - np.float32(4.0) * np.spacing(upper_edge))
    for channel in range(3):
        c = chroma[..., channel]
        source_value = source[..., channel]
        lower = np.where(source_value > boundary_epsilon, lower_target, 0.0)
        upper = np.where(source_value < 1.0 - boundary_epsilon, upper_target, 1.0)
        positive = c > 0.0
        negative = c < 0.0
        scale = np.minimum(
            scale,
            np.where(
                positive,
                np.divide(
                    upper - mapped_luma,
                    c,
                    out=np.full_like(c, np.inf),
                    where=positive,
                ),
                np.inf,
            ),
        )
        scale = np.minimum(
            scale,
            np.where(
                negative,
                np.divide(
                    mapped_luma - lower,
                    -c,
                    out=np.full_like(c, np.inf),
                    where=negative,
                ),
                np.inf,
            ),
        )
    scale = np.clip(scale, 0.0, 1.0)
    limited = scale < 1.0
    scale[limited] = np.nextafter(scale[limited], 0.0)
    output = np.asarray(
        mapped_luma[..., None] + scale[..., None] * chroma, dtype=np.float32
    )
    luma_error = np.sum(output.astype(np.float64) * w, axis=-1) - mapped_luma
    if (
        not np.isfinite(output).all()
        or np.min(output) < 0.0
        or np.max(output) > 1.0
        or _new_boundary_fraction(source, output, boundary_epsilon) != 0.0
        or not np.array_equal(source, source_before)
        or not np.array_equal(ao6, ao6_before)
    ):
        raise CharacteristicAo6FactorizedError("CB15 intrinsic invariant failed")
    return output, scale.astype(np.float32), luma_error


def _inputs(config: Mapping[str, Any], root: Path):
    parents = config["parents"]
    cb14 = _load_exact_json(
        root, parents["cb14_decision_path"], parents["cb14_decision_sha256"]
    )
    if cb14.get("status") != parents["cb14_required_status"]:
        raise CharacteristicAo6FactorizedError("CB14 decision drift")
    cb11 = _load_exact_json(
        root, parents["cb11_contract_path"], parents["cb11_contract_sha256"]
    )
    cb12 = _load_exact_json(
        root, parents["cb12_contract_path"], parents["cb12_contract_sha256"]
    )
    cb6 = load_cb6(root / cb11["parents"]["cb6_contract_path"])
    curve = _compiled_curve(cb6)

    ao6_config = cb12["ao6"]
    artifact_report = _load_exact_json(
        root,
        ao6_config["frozen_artifact_report_path"],
        ao6_config["frozen_artifact_report_sha256"],
    )
    artifact = artifact_report["artifact"]
    if (
        artifact_report.get("artifact_canonical_sha256")
        != ao6_config["frozen_artifact_canonical_sha256"]
        or artifact.get("bundle_sha256") != ao6_config["frozen_bundle_sha256"]
    ):
        raise CharacteristicAo6FactorizedError("AO6 artifact drift")
    validate_standalone_profile_artifact(artifact)

    population = config["population"]
    decision = _load_exact_json(
        root, population["decision_path"], population["decision_sha256"]
    )
    rows = _load_exact_json(
        root, population["manifest_path"], population["manifest_sha256"]
    )
    if (
        decision.get("status") != population["required_status"]
        or len(rows) != population["source_count_exact"]
        or len({row["make"] for row in rows}) != population["camera_make_count_exact"]
    ):
        raise CharacteristicAo6FactorizedError("CB15 population drift")
    for row in rows:
        if hash_file(root / row["decoded_path"]) != row["decoded_sha256"]:
            raise CharacteristicAo6FactorizedError(f"CB15 source drift: {row['id']}")
    return cb11, ao6_config, artifact, rows, curve


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    cb11, ao6_config, artifact, source_rows, curve = _inputs(config, root)
    if output_dir.exists():
        raise FileExistsError("CB15 is create-only")
    output_dir.mkdir(parents=True)
    op = cb11["operator"]
    w = np.asarray(op["luminance_weights"], dtype=np.float64)
    epsilon = float(op["boundary_epsilon"])
    rows = []
    for source_row in source_rows:
        source = _load_rgb(
            root / source_row["decoded_path"],
            maximum_long_edge=int(config["population"]["maximum_long_edge"]),
        )
        ao6 = render_fixed_pair(source, artifact, ao6_config["component"])[
            ao6_config["arm_id"]
        ]
        candidate, scale, luma_error = apply_characteristic_ao6_factorized(
            source,
            ao6,
            curve,
            weights=w,
            strength=float(op["nominal_strength"]),
            boundary_epsilon=epsilon,
        )
        row_dir = output_dir / "renders" / source_row["id"]
        source_path = row_dir / "source.png"
        ao6_path = row_dir / "ao6.png"
        candidate_path = row_dir / "candidate.png"
        source_lab = linear_rgb_to_lab(source, working_space="linear_srgb")
        candidate_lab = linear_rgb_to_lab(candidate, working_space="linear_srgb")
        flat_source = source.reshape(-1, 3)
        flat_ao6 = ao6.reshape(-1, 3)
        flat_candidate = candidate.reshape(-1, 3)
        indices = _sample_indices(
            flat_source.shape[0],
            int(config["evaluation"]["maximum_colour_metric_samples_per_source"]),
        )
        rows.append(
            {
                "id": source_row["id"],
                "make": source_row["make"],
                "source_path": str(source_path),
                "source_sha256": _save_rgb(source, source_path),
                "ao6_path": str(ao6_path),
                "ao6_sha256": _save_rgb(ao6, ao6_path),
                "candidate_path": str(candidate_path),
                "candidate_sha256": _save_rgb(candidate, candidate_path),
                "candidate_style_delta_e76_median": _median_delta_e76(
                    flat_source[indices], flat_candidate[indices]
                ),
                "ao6_style_delta_e76_median": _median_delta_e76(
                    flat_source[indices], flat_ao6[indices]
                ),
                "candidate_vs_ao6_delta_e76_median": _median_delta_e76(
                    flat_candidate[indices], flat_ao6[indices]
                ),
                "candidate_new_boundary_fraction": _new_boundary_fraction(
                    source, candidate, epsilon
                ),
                "ao6_new_boundary_fraction": _new_boundary_fraction(
                    source, ao6, epsilon
                ),
                "candidate_p999_gradient_ratio_vs_source": _gradient_p999_ratio(
                    source, candidate
                ),
                "candidate_lstar_inversion_fraction": _gradient_inversion_fraction(
                    source_lab[..., 0], candidate_lab[..., 0], epsilon=0.0001
                ),
                "median_chroma_scale": float(np.median(scale)),
                "fraction_chroma_scale_below_0p5": float(np.mean(scale < 0.5)),
                "maximum_luminance_reconstruction_error": float(
                    np.max(np.abs(luma_error))
                ),
            }
        )
    metrics = {
        "source_count": len(rows),
        "camera_make_count": len({row["make"] for row in rows}),
        "maximum_luminance_reconstruction_error": max(
            row["maximum_luminance_reconstruction_error"] for row in rows
        ),
        "maximum_new_hard_boundary_fraction": max(
            row["candidate_new_boundary_fraction"] for row in rows
        ),
        "maximum_p999_gradient_ratio_vs_source": max(
            row["candidate_p999_gradient_ratio_vs_source"] for row in rows
        ),
        "maximum_adjacent_lstar_gradient_sign_inversion_fraction": max(
            row["candidate_lstar_inversion_fraction"] for row in rows
        ),
        "population_median_chroma_scale": float(
            np.median([row["median_chroma_scale"] for row in rows])
        ),
        "population_median_fraction_chroma_scale_below_0p5": float(
            np.median([row["fraction_chroma_scale_below_0p5"] for row in rows])
        ),
        "population_median_style_delta_e76": float(
            np.median([row["candidate_style_delta_e76_median"] for row in rows])
        ),
        "population_median_delta_e76_vs_ao6": float(
            np.median([row["candidate_vs_ao6_delta_e76_median"] for row in rows])
        ),
    }
    gates = config["automatic_gates"]
    checks = {
        "source_count": metrics["source_count"]
        == config["population"]["source_count_exact"],
        "camera_make_count": metrics["camera_make_count"]
        == config["population"]["camera_make_count_exact"],
        "luminance_exact": metrics["maximum_luminance_reconstruction_error"]
        <= gates["maximum_luminance_reconstruction_error"],
        "new_boundaries": metrics["maximum_new_hard_boundary_fraction"]
        <= gates["maximum_new_hard_boundary_fraction"],
        "gradient_magnitude": metrics["maximum_p999_gradient_ratio_vs_source"]
        <= gates["maximum_p999_gradient_ratio_vs_source"],
        "gradient_order": metrics[
            "maximum_adjacent_lstar_gradient_sign_inversion_fraction"
        ]
        <= gates["maximum_adjacent_lstar_gradient_sign_inversion_fraction"],
        "chroma_retention": metrics["population_median_chroma_scale"]
        >= gates["minimum_population_median_chroma_scale"]
        and metrics["population_median_fraction_chroma_scale_below_0p5"]
        <= gates["maximum_population_median_fraction_chroma_scale_below_0p5"],
        "visible_style": metrics["population_median_style_delta_e76"]
        >= gates["minimum_population_median_style_delta_e76"],
        "material_vs_ao6": metrics["population_median_delta_e76_vs_ao6"]
        >= gates["minimum_population_median_delta_e76_vs_ao6"],
    }
    automatic = all(checks.values())
    sheets = []
    mappings = {}
    if automatic:
        for round_index in range(config["blind_protocol"]["rounds"]):
            path = output_dir / "blind" / f"round_{round_index + 1}.png"
            digest, mapping = _sheet(
                rows, path, seed=20260811 + 1500, round_index=round_index
            )
            sheets.append(
                {
                    "round": round_index + 1,
                    "path": path.relative_to(output_dir).as_posix(),
                    "sha256": digest,
                }
            )
            mappings[str(round_index + 1)] = mapping
    for row in rows:
        for key in ("source_path", "ao6_path", "candidate_path"):
            row[key] = Path(row[key]).relative_to(output_dir).as_posix()
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": hash_file(
            root / "configs/u5_r2cb15_characteristic_ao6_factorized_v1.json"
        ),
        "rows": rows,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": automatic,
        "blind_sheets": sheets,
        "sealed_mappings": mappings,
        "decision": "open_severe_review_then_blind_development_adjudication"
        if automatic
        else "close_factorized_candidate_without_rescue",
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
    "CharacteristicAo6FactorizedError",
    "apply_characteristic_ao6_factorized",
    "evaluate",
    "load_contract",
    "write_report",
]
