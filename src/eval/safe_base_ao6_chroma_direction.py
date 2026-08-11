"""CB17 safe CB11 base with AO6 chroma direction and CB11 magnitude."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.color_engine.lab import linear_rgb_to_lab
from src.eval.characteristic_ao6_factorized import _inputs
from src.eval.characteristic_vs_ao6_fresh import _sheet
from src.eval.fixed_global_policy_confirmation import render_fixed_pair
from src.eval.fujifilm_characteristic_luma_chroma import (
    apply_characteristic_luma_chroma,
)
from src.eval.fujifilm_characteristic_photographic import _load_exact_json
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

SCHEMA = "neuro_film.u5_r2cb17_safe_base_ao6_chroma_direction_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb17_safe_base_ao6_chroma_direction_report.v1"
EXPERIMENT_ID = "U5.R2CB17"


class SafeBaseAo6DirectionError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise SafeBaseAo6DirectionError("CB17 contract structure drift")
    return payload


def ao6_direction_target(
    safe_base_linear: np.ndarray,
    ao6_linear: np.ndarray,
    *,
    weights: np.ndarray,
    boundary_epsilon: float | None = None,
) -> np.ndarray:
    del boundary_epsilon
    base = np.asarray(safe_base_linear)
    ao6 = np.asarray(ao6_linear)
    w = np.asarray(weights, dtype=np.float64)
    if base.dtype != np.float32 or ao6.dtype != np.float32 or base.shape != ao6.shape:
        raise SafeBaseAo6DirectionError("CB17 direction input drift")
    base64 = base.astype(np.float64)
    ao664 = ao6.astype(np.float64)
    base_luma = np.sum(base64 * w, axis=-1)
    ao6_luma = np.sum(ao664 * w, axis=-1)
    base_chroma = base64 - base_luma[..., None]
    ao6_chroma = ao664 - ao6_luma[..., None]
    base_norm = np.linalg.norm(base_chroma, axis=-1)
    ao6_norm = np.linalg.norm(ao6_chroma, axis=-1)
    target_chroma = base_chroma.copy()
    valid = ao6_norm > 1e-12
    target_chroma[valid] = (
        base_norm[valid, None] * ao6_chroma[valid] / ao6_norm[valid, None]
    )
    target = np.asarray(base_luma[..., None] + target_chroma, dtype=np.float32)
    if not np.isfinite(target).all():
        raise SafeBaseAo6DirectionError("CB17 direction target is nonfinite")
    return target


def apply_safe_base_direction_target(
    source_linear: np.ndarray,
    safe_base_linear: np.ndarray,
    target_linear: np.ndarray,
    *,
    weights: np.ndarray,
    boundary_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(source_linear)
    base = np.asarray(safe_base_linear)
    target = np.asarray(target_linear)
    w = np.asarray(weights, dtype=np.float64)
    before = (source.copy(), base.copy(), target.copy())
    if (
        source.dtype != np.float32
        or base.dtype != np.float32
        or target.dtype != np.float32
        or source.shape != base.shape
        or source.shape != target.shape
        or source.ndim < 2
        or source.shape[-1] != 3
        or not np.isfinite(source).all()
        or not np.isfinite(base).all()
        or not np.isfinite(target).all()
        or np.min(source) < 0.0
        or np.max(source) > 1.0
        or np.min(base) < 0.0
        or np.max(base) > 1.0
    ):
        raise SafeBaseAo6DirectionError("CB17 apply input drift")
    base64 = base.astype(np.float64)
    residual = target.astype(np.float64) - base64
    base_luma = np.sum(base64 * w, axis=-1)
    scale = np.ones(base_luma.shape, dtype=np.float64)
    lower_target = float(
        np.float32(boundary_epsilon)
        + np.float32(4.0) * np.spacing(np.float32(boundary_epsilon))
    )
    upper_edge = np.float32(1.0 - boundary_epsilon)
    upper_target = float(upper_edge - np.float32(4.0) * np.spacing(upper_edge))
    for channel in range(3):
        delta = residual[..., channel]
        source_value = source[..., channel]
        lower = np.where(source_value > boundary_epsilon, lower_target, 0.0)
        upper = np.where(source_value < 1.0 - boundary_epsilon, upper_target, 1.0)
        positive = delta > 0.0
        negative = delta < 0.0
        scale = np.minimum(
            scale,
            np.where(
                positive,
                np.divide(
                    upper - base64[..., channel],
                    delta,
                    out=np.full_like(delta, np.inf),
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
                    base64[..., channel] - lower,
                    -delta,
                    out=np.full_like(delta, np.inf),
                    where=negative,
                ),
                np.inf,
            ),
        )
    scale = np.clip(scale, 0.0, 1.0)
    limited = scale < 1.0
    scale[limited] = np.nextafter(scale[limited], 0.0)
    output = np.asarray(base64 + scale[..., None] * residual, dtype=np.float32)
    luma_error = np.sum(output.astype(np.float64) * w, axis=-1) - base_luma
    if (
        not np.isfinite(output).all()
        or np.min(output) < 0.0
        or np.max(output) > 1.0
        or _new_boundary_fraction(source, output, boundary_epsilon) != 0.0
        or not np.array_equal(source, before[0])
        or not np.array_equal(base, before[1])
        or not np.array_equal(target, before[2])
    ):
        raise SafeBaseAo6DirectionError("CB17 intrinsic invariant failed")
    return output, scale.astype(np.float32), luma_error


def evaluate_direction_candidate(
    config: Mapping[str, Any],
    root: Path,
    output_dir: Path,
    *,
    target_builder: Any,
    report_schema: str,
    experiment_id: str,
    contract_filename: str,
    blind_seed: int = 20260811 + 1700,
) -> dict[str, Any]:
    decision = _load_exact_json(
        root,
        config["parents"]["cb16_decision_path"],
        config["parents"]["cb16_decision_sha256"],
    )
    if decision.get("decision") != config["parents"]["cb16_required_status"]:
        raise SafeBaseAo6DirectionError("CB16 decision drift")
    cb11, ao6_config, artifact, source_rows, curve = _inputs(config, root)
    if output_dir.exists():
        raise FileExistsError("CB17 is create-only")
    output_dir.mkdir(parents=True)
    op = cb11["operator"]
    weights = np.asarray(op["luminance_weights"], dtype=np.float64)
    epsilon = float(op["boundary_epsilon"])
    rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        source = _load_rgb(
            root / source_row["decoded_path"],
            maximum_long_edge=int(config["population"]["maximum_long_edge"]),
        )
        ao6 = render_fixed_pair(source, artifact, ao6_config["component"])[
            ao6_config["arm_id"]
        ]
        safe_base, _, _ = apply_characteristic_luma_chroma(
            source,
            curve,
            weights=weights,
            strength=float(op["nominal_strength"]),
            boundary_epsilon=epsilon,
        )
        target = target_builder(
            safe_base,
            ao6,
            weights=weights,
            boundary_epsilon=epsilon,
        )
        candidate, scale, luma_error = apply_safe_base_direction_target(
            source,
            safe_base,
            target,
            weights=weights,
            boundary_epsilon=epsilon,
        )
        row_dir = output_dir / "renders" / source_row["id"]
        source_path = row_dir / "source.png"
        ao6_path = row_dir / "ao6.png"
        base_path = row_dir / "cb11.png"
        candidate_path = row_dir / "candidate.png"
        source_lab = linear_rgb_to_lab(source, working_space="linear_srgb")
        candidate_lab = linear_rgb_to_lab(candidate, working_space="linear_srgb")
        flat_source = source.reshape(-1, 3)
        flat_base = safe_base.reshape(-1, 3)
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
                "cb11_path": str(base_path),
                "cb11_sha256": _save_rgb(safe_base, base_path),
                "candidate_path": str(candidate_path),
                "candidate_sha256": _save_rgb(candidate, candidate_path),
                "candidate_style_delta_e76_median": _median_delta_e76(
                    flat_source[indices], flat_candidate[indices]
                ),
                "candidate_vs_cb11_delta_e76_median": _median_delta_e76(
                    flat_candidate[indices], flat_base[indices]
                ),
                "candidate_new_boundary_fraction": _new_boundary_fraction(
                    source, candidate, epsilon
                ),
                "candidate_p999_gradient_ratio_vs_source": _gradient_p999_ratio(
                    source, candidate
                ),
                "candidate_lstar_inversion_fraction": _gradient_inversion_fraction(
                    source_lab[..., 0], candidate_lab[..., 0], epsilon=0.0001
                ),
                "median_direction_scale": float(np.median(scale)),
                "fraction_direction_scale_below_0p5": float(np.mean(scale < 0.5)),
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
        "population_median_direction_scale": float(
            np.median([row["median_direction_scale"] for row in rows])
        ),
        "population_median_fraction_direction_scale_below_0p5": float(
            np.median([row["fraction_direction_scale_below_0p5"] for row in rows])
        ),
        "population_median_style_delta_e76": float(
            np.median([row["candidate_style_delta_e76_median"] for row in rows])
        ),
        "population_median_delta_e76_vs_cb11": float(
            np.median([row["candidate_vs_cb11_delta_e76_median"] for row in rows])
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
        "direction_retention": metrics["population_median_direction_scale"]
        >= gates["minimum_population_median_direction_scale"]
        and metrics["population_median_fraction_direction_scale_below_0p5"]
        <= gates["maximum_population_median_fraction_direction_scale_below_0p5"],
        "visible_style": metrics["population_median_style_delta_e76"]
        >= gates["minimum_population_median_style_delta_e76"],
        "material_vs_cb11": metrics["population_median_delta_e76_vs_cb11"]
        >= gates["minimum_population_median_delta_e76_vs_cb11"],
    }
    automatic = all(checks.values())
    sheets: list[dict[str, Any]] = []
    mappings: dict[str, Any] = {}
    if automatic:
        for round_index in range(config["blind_protocol"]["rounds"]):
            path = output_dir / "blind" / f"round_{round_index + 1}.png"
            digest, mapping = _sheet(
                rows, path, seed=blind_seed, round_index=round_index
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
        for key in ("source_path", "ao6_path", "cb11_path", "candidate_path"):
            row[key] = Path(row[key]).relative_to(output_dir).as_posix()
    report: dict[str, Any] = {
        "schema": report_schema,
        "experiment_id": experiment_id,
        "contract_sha256": hash_file(root / f"configs/{contract_filename}"),
        "rows": rows,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": automatic,
        "blind_sheets": sheets,
        "sealed_mappings": mappings,
        "decision": (
            "open_severe_review_then_blind_development_adjudication"
            if automatic
            else "close_safe_base_direction_without_rescue"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    return evaluate_direction_candidate(
        config,
        root,
        output_dir,
        target_builder=ao6_direction_target,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb17_safe_base_ao6_chroma_direction_v1.json",
    )


def write_report(report: Mapping[str, Any], output: Path) -> str:
    payload = canonical_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "SafeBaseAo6DirectionError",
    "ao6_direction_target",
    "apply_safe_base_direction_target",
    "evaluate",
    "evaluate_direction_candidate",
    "load_contract",
    "write_report",
]
