"""U6.P4U bounded photographic integration of P4D/P4T material kernels."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.global_frontier import (
    new_hard_clipping_fraction,
    sha256_file,
)
from src.eval.physical_density_conditioned_structure import (
    profiles_from_contract,
)
from src.eval.physical_joint_ablation import (
    _canonicalize_endpoint_roundoff,
)
from src.eval.physical_neutral_gauged_chain import (
    apply_gauge_to_intermediate,
    validate_contract as validate_p7f_contract,
)
from src.eval.physical_nonspatial_attribution import _metric_sample
from src.eval.physical_virtual_scan_sampling import (
    _median_delta_e76,
    compile_virtual_scan_profile,
)
from src.eval.real_uniform_grain_physical import (
    render_anisotropic_structure,
)
from src.film_physics import (
    apply_dye_diffusion,
    apply_forward_scatter,
    apply_scanner_mtf,
)
from src.film_physics.density_conditioned_structure import (
    render_density_conditioned_structure,
)
from src.real_film.gold_matrix_transplant import style_and_basic_residual


SCHEMA = "neuro_film.u6_p4u_photographic_material_stress_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4u_photographic_material_stress_report.v1"


class PhotographicMaterialStressError(RuntimeError):
    """Raised when the frozen P4U contract or render leaves its domain."""


def _load_exact_json(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = root / binding["path"]
    if sha256_file(path) != binding["sha256"]:
        raise PhotographicMaterialStressError(
            f"parent hash mismatch: {binding['path']}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PhotographicMaterialStressError("parent must be a JSON object")
    return payload


def validate_contract(
    root: Path, config: dict[str, Any]
) -> tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    if (
        config.get("schema") != SCHEMA
        or config["execution"].get("post_result_retuning_allowed")
        or config["execution"].get("production_integration_allowed")
        or config["execution"].get("closed_p8bp_reopened")
        or config["chain"].get("target_density_clipping_allowed")
        or config["chain"].get("material_density_clipping_allowed")
        or config["chain"].get("display_rgb_noise_allowed")
    ):
        raise PhotographicMaterialStressError("unsupported P4U contract")
    parents = config["parents"]
    p7f_contract = _load_exact_json(root, parents["p7f_contract"])
    p7f_decision = _load_exact_json(root, parents["p7f_decision"])
    p7f_report = _load_exact_json(root, parents["p7f_report"])
    p4d = _load_exact_json(root, parents["p4d_contract"])
    p4t = _load_exact_json(root, parents["p4t_contract"])
    p4t_decision = _load_exact_json(root, parents["p4t_decision"])
    runtime, gauge = validate_p7f_contract(root, p7f_contract)
    expected_arms = [
        "colour_only",
        "no_material_control",
        "p4d_isotropic_shared_seed",
        "p4t_anisotropic_shared_seed",
    ]
    if (
        config["chain"]["arms"] != expected_arms
        or int(config["chain"]["sampling_dpi"]) != 4000
        or len(runtime.eligible_ids)
        != int(config["population"]["expected_images"])
        or len(config["population"]["visual_ids"]) != 9
        or not set(config["population"]["visual_ids"]).issubset(
            runtime.eligible_ids
        )
        or p7f_decision["visual_result"]["decision"] != "complete_pass"
        or p7f_decision["production_default_changed"]
        or p7f_report["selected_candidate_arm_id"]
        != "gauged_spatial_4000"
        or p4t_decision["result"]["status"]
        != "automatic_and_synthetic_severe_pass"
        or not p4t_decision["next_leaf"].startswith("U6.P4U")
    ):
        raise PhotographicMaterialStressError("P4U parent or population drift")
    p4d_profiles = p4d["profiles"]
    p4t_candidate = p4t["candidate"]
    if (
        [float(row["grain_optical_density"]) for row in p4d_profiles]
        != [
            float(value)
            for value in p4t_candidate["grain_optical_density_by_rgb_layer"]
        ]
        or float(p4d["synthetic_evaluation"]["maximum_input_density"])
        != float(p4t_candidate["maximum_target_density"])
        or config["chain"]["material_seed_policy"]
        != (
            "use exact P4T layer seeds for both material arms so the "
            "comparison changes kernel geometry rather than random identity"
        )
    ):
        raise PhotographicMaterialStressError("P4D/P4T comparison drift")
    return runtime, gauge, p7f_report, p4d, p4t


def _material_profiles(
    p4d: dict[str, Any], p4t: dict[str, Any]
) -> tuple[Any, ...]:
    seeds = [int(value) for value in p4t["candidate"]["layer_seeds"]]
    profiles = profiles_from_contract(p4d)
    return tuple(
        replace(profile, seed=seeds[index])
        for index, profile in enumerate(profiles)
    )


def _finish_physical(
    density: np.ndarray, runtime: Any, gauge: Any
) -> np.ndarray:
    developed = runtime.apply_adjacency(density, runtime.profile)
    developed = apply_dye_diffusion(developed, runtime.profile)
    interpreted = _canonicalize_endpoint_roundoff(
        runtime.print_operator.interpretation.apply(developed)
    )
    scanned = _canonicalize_endpoint_roundoff(
        apply_scanner_mtf(interpreted, runtime.profile)
    )
    return apply_gauge_to_intermediate(scanned, gauge)


def bound_material_density(
    target: np.ndarray,
    material: np.ndarray,
    *,
    lower_rgb: np.ndarray,
    upper_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytically scale one RGB density residual into interpretation bounds."""
    base = np.asarray(target, dtype=np.float64)
    candidate = np.asarray(material, dtype=np.float64)
    lower = np.asarray(lower_rgb, dtype=np.float64)
    upper = np.asarray(upper_rgb, dtype=np.float64)
    if (
        base.shape != candidate.shape
        or base.ndim != 3
        or base.shape[-1] != 3
        or lower.shape != (3,)
        or upper.shape != (3,)
        or not np.all(np.isfinite(base))
        or not np.all(np.isfinite(candidate))
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower < 0.0)
        or np.any(upper <= lower)
        or np.any(base < lower - 1e-12)
        or np.any(base > upper + 1e-12)
        or np.any(candidate < 0.0)
    ):
        raise ValueError("invalid material density bound inputs")
    residual = candidate - base
    scale = np.ones(base.shape[:2], dtype=np.float64)
    for channel in range(3):
        delta = residual[..., channel]
        positive = delta > 0.0
        negative = delta < 0.0
        permitted = np.ones(delta.shape, dtype=np.float64)
        permitted[positive] = (
            upper[channel] - base[..., channel][positive]
        ) / delta[positive]
        permitted[negative] = (
            lower[channel] - base[..., channel][negative]
        ) / delta[negative]
        scale = np.minimum(scale, permitted)
    if (
        not np.all(np.isfinite(scale))
        or np.any(scale < -1e-12)
        or np.any(scale > 1.0 + 1e-12)
    ):
        raise PhotographicMaterialStressError(
            "analytical material residual scale left [0,1]"
        )
    scale = np.maximum(0.0, np.minimum(1.0, scale))
    bounded = base + scale[..., None] * residual
    if (
        not np.all(np.isfinite(bounded))
        or np.any(bounded < lower - 1e-12)
        or np.any(bounded > upper + 1e-12)
    ):
        raise PhotographicMaterialStressError(
            "analytically bounded material left interpretation density"
        )
    bounded = np.asarray(bounded, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    bounded.setflags(write=False)
    scale.setflags(write=False)
    return bounded, scale


def _density_diagnostics(
    target: np.ndarray,
    raw_material: np.ndarray,
    bounded_material: np.ndarray,
    scale: np.ndarray,
) -> dict[str, Any]:
    return {
        "target_minimum": float(np.min(target)),
        "target_maximum": float(np.max(target)),
        "raw_material_minimum": float(np.min(raw_material)),
        "raw_material_maximum": float(np.max(raw_material)),
        "bounded_material_minimum": float(np.min(bounded_material)),
        "bounded_material_maximum": float(np.max(bounded_material)),
        "minimum_residual_scale": float(np.min(scale)),
        "compressed_pixel_fraction": float(np.mean(scale < 1.0)),
        "channel_mean_density_error": [
            float(
                abs(
                    np.mean(
                        bounded_material[..., channel], dtype=np.float64
                    )
                    - np.mean(target[..., channel], dtype=np.float64)
                )
            )
            for channel in range(target.shape[2])
        ],
        "material_density_sha256": hashlib.sha256(
            np.ascontiguousarray(bounded_material).tobytes()
        ).hexdigest(),
        "residual_scale_sha256": hashlib.sha256(
            np.ascontiguousarray(scale).tobytes()
        ).hexdigest(),
    }


def render_material_arms(
    source: np.ndarray,
    runtime: Any,
    gauge: Any,
    p4d: dict[str, Any],
    p4t: dict[str, Any],
    *,
    sampling_dpi: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    encoded = np.asarray(source, dtype=np.float64)
    if (
        encoded.ndim != 3
        or encoded.shape[-1] != 3
        or not np.all(np.isfinite(encoded))
        or np.any(encoded < 0.0)
        or np.any(encoded > 1.0)
    ):
        raise ValueError("P4U source must be finite HxWx3 encoded RGB")
    compiled = replace(
        runtime,
        profile=compile_virtual_scan_profile(
            runtime.profile, sampling_dpi=sampling_dpi
        ),
    )
    linear = encoded_srgb_to_linear(encoded)
    exposure = apply_forward_scatter(linear, compiled.profile)
    target_density = compiled.print_operator.sensitometry.apply(exposure)
    maximum = float(p4t["candidate"]["maximum_target_density"])
    if (
        not np.all(np.isfinite(target_density))
        or np.any(target_density < 0.0)
        or np.any(target_density > maximum)
    ):
        raise PhotographicMaterialStressError(
            "developed target density left the frozen material domain"
        )

    p4d_result = render_density_conditioned_structure(
        target_density, _material_profiles(p4d, p4t)
    )
    candidate = p4t["candidate"]
    p4t_density, _ = render_anisotropic_structure(
        target_density,
        grain_optical_density_by_channel=[
            float(value)
            for value in candidate["grain_optical_density_by_rgb_layer"]
        ],
        sigma_yx=tuple(
            float(value) for value in candidate["sigma_yx_pixels"]
        ),
        seeds=[int(value) for value in candidate["layer_seeds"]],
        maximum_target_density=maximum,
        truncate=float(candidate["truncate"]),
    )
    lower = np.asarray(
        compiled.print_operator.interpretation.black_reference_density,
        dtype=np.float64,
    )
    upper = np.asarray(
        compiled.print_operator.interpretation.white_reference_density,
        dtype=np.float64,
    )
    p4d_bounded, p4d_scale = bound_material_density(
        target_density,
        p4d_result.density,
        lower_rgb=lower,
        upper_rgb=upper,
    )
    p4t_bounded, p4t_scale = bound_material_density(
        target_density,
        p4t_density,
        lower_rgb=lower,
        upper_rgb=upper,
    )
    intermediates = {
        "no_material_control": _finish_physical(
            target_density, compiled, gauge
        ),
        "p4d_isotropic_shared_seed": _finish_physical(
            p4d_bounded, compiled, gauge
        ),
        "p4t_anisotropic_shared_seed": _finish_physical(
            p4t_bounded, compiled, gauge
        ),
    }
    apply_colour = compiled.build_source_context_colour(encoded)
    outputs = {"colour_only": apply_colour(encoded)}
    outputs.update(
        {
            arm_id: apply_colour(linear_srgb_to_encoded(values))
            for arm_id, values in intermediates.items()
        }
    )
    for arm_id, values in outputs.items():
        if (
            values.shape != encoded.shape
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise PhotographicMaterialStressError(
                f"{arm_id} left encoded RGB"
            )
    diagnostics = {
        "p4d_isotropic_shared_seed": _density_diagnostics(
            target_density,
            p4d_result.density,
            p4d_bounded,
            p4d_scale,
        ),
        "p4t_anisotropic_shared_seed": _density_diagnostics(
            target_density,
            p4t_density,
            p4t_bounded,
            p4t_scale,
        ),
    }
    return outputs, diagnostics


def _save_rgb(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.rint(np.asarray(values) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(
        path, format="PNG", compress_level=6
    )
    return sha256_file(path)


def _luma(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) @ np.asarray(
        [0.2126, 0.7152, 0.0722], dtype=np.float64
    )


def evaluate_photographic_material_stress(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    runtime, gauge, p7f_report, p4d, p4t = validate_contract(root, config)
    synthetic = np.random.default_rng(2026072904).random((65, 67, 3))
    first, _ = render_material_arms(
        synthetic,
        runtime,
        gauge,
        p4d,
        p4t,
        sampling_dpi=int(config["chain"]["sampling_dpi"]),
    )
    second, _ = render_material_arms(
        synthetic,
        runtime,
        gauge,
        p4d,
        p4t,
        sampling_dpi=int(config["chain"]["sampling_dpi"]),
    )
    repeat_exact = all(
        np.array_equal(first[arm_id], second[arm_id])
        for arm_id in config["chain"]["arms"]
    )
    parent_hashes = {
        row["sample_id"]: row["output_sha256"]
        for row in p7f_report["rows"]
        if row["arm_id"] == "gauged_spatial_4000"
    }
    rows: list[dict[str, Any]] = []
    parent_exact = True
    for sample_id in runtime.eligible_ids:
        source_row = runtime.source_rows[sample_id]
        source_path = root / source_row["decoded_path"]
        if sha256_file(source_path) != source_row["decoded_sha256"]:
            raise PhotographicMaterialStressError(
                f"source hash drift: {sample_id}"
            )
        with Image.open(source_path) as image:
            source = (
                np.asarray(
                    ImageOps.exif_transpose(image).convert("RGB"),
                    dtype=np.float64,
                )
                / 255.0
            )
        outputs, diagnostics = render_material_arms(
            source,
            runtime,
            gauge,
            p4d,
            p4t,
            sampling_dpi=int(config["chain"]["sampling_dpi"]),
        )
        source_sample = _metric_sample(source)
        hashes = {
            arm_id: _save_rgb(
                output_dir / "renders" / arm_id / f"{sample_id}.png",
                values,
            )
            for arm_id, values in outputs.items()
        }
        parent_exact &= (
            hashes["no_material_control"] == parent_hashes[sample_id]
        )
        for arm_id in config["chain"]["arms"][1:]:
            values = outputs[arm_id]
            sample = _metric_sample(values)
            style, non_basic = style_and_basic_residual(
                source_sample, sample
            )
            metrics = {
                "style_delta_e76": style,
                "non_basic_delta_e76": non_basic,
                "to_colour_delta_e76": _median_delta_e76(
                    values, outputs["colour_only"]
                ),
                "to_no_material_delta_e76": _median_delta_e76(
                    values, outputs["no_material_control"]
                ),
                "to_p4d_delta_e76": _median_delta_e76(
                    values, outputs["p4d_isotropic_shared_seed"]
                ),
                "median_absolute_encoded_luma_drift_to_no_material": float(
                    np.median(
                        np.abs(
                            _luma(values)
                            - _luma(outputs["no_material_control"])
                        )
                    )
                ),
                "new_hard_boundary_fraction": new_hard_clipping_fraction(
                    source_sample, sample, 0.5 / 255.0
                ),
            }
            rows.append(
                {
                    "sample_id": sample_id,
                    "make": source_row["make"],
                    "arm_id": arm_id,
                    "source_sha256": source_row["decoded_sha256"],
                    "output_sha256": hashes[arm_id],
                    "metrics": metrics,
                    "density": diagnostics.get(arm_id),
                }
            )
    if len(rows) != 3 * int(config["population"]["expected_images"]):
        raise PhotographicMaterialStressError("P4U result population drift")

    summaries: dict[str, Any] = {}
    for arm_id in config["chain"]["arms"][1:]:
        subset = [row for row in rows if row["arm_id"] == arm_id]
        summaries[arm_id] = {
            "maximum_style_delta_e76": float(
                max(row["metrics"]["style_delta_e76"] for row in subset)
            ),
            "median_to_colour_delta_e76": float(
                np.median(
                    [row["metrics"]["to_colour_delta_e76"] for row in subset]
                )
            ),
            "median_to_no_material_delta_e76": float(
                np.median(
                    [
                        row["metrics"]["to_no_material_delta_e76"]
                        for row in subset
                    ]
                )
            ),
            "median_to_p4d_delta_e76": float(
                np.median(
                    [row["metrics"]["to_p4d_delta_e76"] for row in subset]
                )
            ),
            "maximum_median_absolute_encoded_luma_drift_to_no_material": float(
                max(
                    row["metrics"][
                        "median_absolute_encoded_luma_drift_to_no_material"
                    ]
                    for row in subset
                )
            ),
            "maximum_new_hard_boundary_fraction": float(
                max(
                    row["metrics"]["new_hard_boundary_fraction"]
                    for row in subset
                )
            ),
            "maximum_material_channel_mean_density_error": (
                None
                if arm_id == "no_material_control"
                else float(
                    max(
                        max(row["density"]["channel_mean_density_error"])
                        for row in subset
                    )
                )
            ),
            "maximum_target_density": (
                None
                if arm_id == "no_material_control"
                else float(
                    max(
                        row["density"]["target_maximum"] for row in subset
                    )
                )
            ),
            "minimum_material_residual_scale": (
                None
                if arm_id == "no_material_control"
                else float(
                    min(
                        row["density"]["minimum_residual_scale"]
                        for row in subset
                    )
                )
            ),
        }
    gates = config["automatic_gates"]
    p4d_summary = summaries["p4d_isotropic_shared_seed"]
    p4t_summary = summaries["p4t_anisotropic_shared_seed"]
    checks = {
        "repeat_exact": repeat_exact,
        "no_material_parent_output_exact": parent_exact,
        "target_density_domain": p4t_summary["maximum_target_density"]
        <= float(gates["maximum_target_density"]),
        "material_density_mean": max(
            p4d_summary["maximum_material_channel_mean_density_error"],
            p4t_summary["maximum_material_channel_mean_density_error"],
        )
        <= float(gates["maximum_material_channel_mean_density_error"]),
        "material_residual_scale_domain": min(
            p4d_summary["minimum_material_residual_scale"],
            p4t_summary["minimum_material_residual_scale"],
        )
        >= float(gates["minimum_material_residual_scale"]),
        "p4t_new_boundaries": p4t_summary[
            "maximum_new_hard_boundary_fraction"
        ]
        <= float(gates["maximum_new_hard_boundary_fraction"]),
        "p4t_style_envelope": p4t_summary["maximum_style_delta_e76"]
        <= float(gates["maximum_per_image_style_delta_e76"]),
        "p4t_style_material": p4t_summary["median_to_colour_delta_e76"]
        >= float(gates["minimum_median_to_colour_delta_e76"]),
        "p4t_material_salience": p4t_summary[
            "median_to_no_material_delta_e76"
        ]
        >= float(
            gates["minimum_median_material_to_no_material_delta_e76"]
        ),
        "p4t_distinct_from_p4d": p4t_summary["median_to_p4d_delta_e76"]
        >= float(gates["minimum_median_p4t_to_p4d_delta_e76"]),
        "p4t_tone_drift": p4t_summary[
            "maximum_median_absolute_encoded_luma_drift_to_no_material"
        ]
        <= float(
            gates[
                "maximum_median_absolute_encoded_luma_drift_to_no_material"
            ]
        ),
        "p4t_boundary_not_worse_than_p4d": p4t_summary[
            "maximum_new_hard_boundary_fraction"
        ]
        <= p4d_summary["maximum_new_hard_boundary_fraction"],
    }
    automatic_pass = all(checks.values())
    core = {
        "schema": REPORT_SCHEMA,
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "rows": rows,
        "summaries": summaries,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "branch": config["branch_rule"][
            "automatic_pass" if automatic_pass else "automatic_fail"
        ],
    }
    stable_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "bound_material_density",
    "evaluate_photographic_material_stress",
    "render_material_arms",
    "validate_contract",
    "write_report",
]
