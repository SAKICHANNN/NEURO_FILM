"""U6.P4V fixed stage attribution for the rejected P4T photographic path."""

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
from src.eval.global_frontier import sha256_file
from src.eval.photographic_material_stress import (
    bound_material_density,
    validate_contract as validate_p4u_contract,
)
from src.eval.physical_joint_ablation import (
    _canonicalize_endpoint_roundoff,
)
from src.eval.physical_neutral_gauged_chain import (
    apply_gauge_to_intermediate,
)
from src.eval.physical_virtual_scan_sampling import (
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


SCHEMA = "neuro_film.u6_p4v_material_tone_mechanism_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4v_material_tone_mechanism_report.v1"


class MaterialToneMechanismError(RuntimeError):
    """Raised when P4V evidence or stage identity drifts."""


def _load_exact_json(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = root / binding["path"]
    if sha256_file(path) != binding["sha256"]:
        raise MaterialToneMechanismError(f"hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MaterialToneMechanismError("bound payload must be an object")
    return payload


def validate_contract(
    root: Path, config: dict[str, Any]
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    if (
        config.get("schema") != SCHEMA
        or config["candidate"].get("parameter_changes_allowed")
        or config["candidate"].get("render_changes_allowed")
        or config["execution"].get("new_render_files_allowed")
        or config["execution"].get("post_result_parameter_change_allowed")
    ):
        raise MaterialToneMechanismError("unsupported P4V contract")
    decision = _load_exact_json(root, config["parent_decision"])
    if (
        decision["decision"]
        != config["parent_decision"]["required_decision"]
        or decision["automatic_result"]["passed"]
        or decision["visual_review"]["allowed"]
    ):
        raise MaterialToneMechanismError("P4U closure drift")
    p4u_path = root / decision["contract"]["path"]
    if sha256_file(p4u_path) != decision["contract"]["sha256"]:
        raise MaterialToneMechanismError("P4U contract drift")
    p4u = json.loads(p4u_path.read_text(encoding="utf-8"))
    runtime, gauge, _, _, p4t = validate_p4u_contract(root, p4u)
    if (
        len(runtime.eligible_ids) != int(config["population"]["expected_images"])
        or config["candidate"]["identity"]
        != "exact closed P4U p4t_anisotropic_shared_seed"
    ):
        raise MaterialToneMechanismError("P4V population drift")
    return runtime, gauge, p4u, p4t


def _median_absolute(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.median(
            np.abs(
                np.asarray(left, dtype=np.float64)
                - np.asarray(right, dtype=np.float64)
            )
        )
    )


def _luma(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) @ np.asarray(
        [0.2126, 0.7152, 0.0722], dtype=np.float64
    )


def _trace_one(
    source: np.ndarray,
    runtime: Any,
    gauge: Any,
    p4u: dict[str, Any],
    p4t: dict[str, Any],
) -> dict[str, Any]:
    compiled = replace(
        runtime,
        profile=compile_virtual_scan_profile(
            runtime.profile,
            sampling_dpi=int(p4u["chain"]["sampling_dpi"]),
        ),
    )
    encoded = np.asarray(source, dtype=np.float64)
    linear = encoded_srgb_to_linear(encoded)
    exposure = apply_forward_scatter(linear, compiled.profile)
    target = compiled.print_operator.sensitometry.apply(exposure)
    candidate = p4t["candidate"]
    raw, _ = render_anisotropic_structure(
        target,
        grain_optical_density_by_channel=[
            float(value)
            for value in candidate["grain_optical_density_by_rgb_layer"]
        ],
        sigma_yx=tuple(
            float(value) for value in candidate["sigma_yx_pixels"]
        ),
        seeds=[int(value) for value in candidate["layer_seeds"]],
        maximum_target_density=float(candidate["maximum_target_density"]),
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
    bounded, scale = bound_material_density(
        target, raw, lower_rgb=lower, upper_rgb=upper
    )

    base_adjacency = compiled.apply_adjacency(target, compiled.profile)
    material_adjacency = compiled.apply_adjacency(bounded, compiled.profile)
    base_diffusion = apply_dye_diffusion(
        base_adjacency, compiled.profile
    )
    material_diffusion = apply_dye_diffusion(
        material_adjacency, compiled.profile
    )
    base_interpreted = _canonicalize_endpoint_roundoff(
        compiled.print_operator.interpretation.apply(base_diffusion)
    )
    material_interpreted = _canonicalize_endpoint_roundoff(
        compiled.print_operator.interpretation.apply(material_diffusion)
    )
    base_scanned = _canonicalize_endpoint_roundoff(
        apply_scanner_mtf(base_interpreted, compiled.profile)
    )
    material_scanned = _canonicalize_endpoint_roundoff(
        apply_scanner_mtf(material_interpreted, compiled.profile)
    )
    base_gauged = apply_gauge_to_intermediate(base_scanned, gauge)
    material_gauged = apply_gauge_to_intermediate(material_scanned, gauge)
    apply_colour = compiled.build_source_context_colour(encoded)
    base_final = apply_colour(linear_srgb_to_encoded(base_gauged))
    material_final = apply_colour(linear_srgb_to_encoded(material_gauged))

    density_mean_errors = [
        float(
            abs(
                np.mean(bounded[..., channel], dtype=np.float64)
                - np.mean(target[..., channel], dtype=np.float64)
            )
        )
        for channel in range(3)
    ]
    raw_residual = raw.astype(np.float64) - target
    bounded_residual = bounded.astype(np.float64) - target
    return {
        "maximum_channel_mean_density_error": max(density_mean_errors),
        "median_absolute_raw_density_residual": float(
            np.median(np.abs(raw_residual))
        ),
        "median_absolute_bounded_density_residual": float(
            np.median(np.abs(bounded_residual))
        ),
        "bounded_to_raw_residual_ratio": (
            float(np.median(np.abs(bounded_residual)))
            / max(float(np.median(np.abs(raw_residual))), 1e-15)
        ),
        "compressed_pixel_fraction": float(np.mean(scale < 1.0)),
        "median_absolute_log10_transmittance_drift_raw": _median_absolute(
            np.power(10.0, -raw.astype(np.float64)),
            np.power(10.0, -target),
        ),
        "median_absolute_log10_transmittance_drift_bounded": _median_absolute(
            np.power(10.0, -bounded.astype(np.float64)),
            np.power(10.0, -target),
        ),
        "stage_median_absolute_rgb_or_density_drift": {
            "development_adjacency": _median_absolute(
                material_adjacency, base_adjacency
            ),
            "dye_diffusion": _median_absolute(
                material_diffusion, base_diffusion
            ),
            "interpretation": _median_absolute(
                material_interpreted, base_interpreted
            ),
            "scanner_mtf": _median_absolute(
                material_scanned, base_scanned
            ),
            "neutral_gauge": _median_absolute(
                material_gauged, base_gauged
            ),
        },
        "median_absolute_encoded_luma_drift": float(
            np.median(np.abs(_luma(material_final) - _luma(base_final)))
        ),
        "base_final_sha256": hashlib.sha256(
            np.ascontiguousarray(base_final).tobytes()
        ).hexdigest(),
        "material_final_sha256": hashlib.sha256(
            np.ascontiguousarray(material_final).tobytes()
        ).hexdigest(),
    }


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def evaluate_material_tone_mechanism(
    *, root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    runtime, gauge, p4u, p4t = validate_contract(root, config)
    rows = []
    for sample_id in runtime.eligible_ids:
        source_row = runtime.source_rows[sample_id]
        path = root / source_row["decoded_path"]
        if sha256_file(path) != source_row["decoded_sha256"]:
            raise MaterialToneMechanismError(
                f"source hash drift: {sample_id}"
            )
        with Image.open(path) as image:
            source = (
                np.asarray(
                    ImageOps.exif_transpose(image).convert("RGB"),
                    dtype=np.float64,
                )
                / 255.0
            )
        rows.append(
            {
                "sample_id": sample_id,
                "source_sha256": source_row["decoded_sha256"],
                **_trace_one(source, runtime, gauge, p4u, p4t),
            }
        )
    if len(rows) != int(config["population"]["expected_images"]):
        raise MaterialToneMechanismError("P4V population drift")
    mean_errors = [
        row["maximum_channel_mean_density_error"] for row in rows
    ]
    raw_local = [
        row["median_absolute_raw_density_residual"] for row in rows
    ]
    bounded_local = [
        row["median_absolute_bounded_density_residual"] for row in rows
    ]
    compressed = [row["compressed_pixel_fraction"] for row in rows]
    final = [row["median_absolute_encoded_luma_drift"] for row in rows]
    thresholds = config["diagnostic_thresholds"]
    small_global_mean = max(mean_errors) <= float(
        thresholds["maximum_channel_mean_density_error_considered_small"]
    )
    large_local = float(np.median(bounded_local)) >= float(
        thresholds[
            "minimum_median_absolute_local_density_residual_considered_large"
        ]
    )
    sparse_compression = max(compressed) <= float(
        thresholds["maximum_compressed_pixel_fraction_considered_sparse"]
    )
    tone_failed = max(final) > float(
        thresholds["photographic_tone_drift_failure_threshold"]
    )
    if not small_global_mean:
        attribution = "global_mean_primary"
    elif not sparse_compression:
        attribution = "bound_compression_primary"
    elif large_local and tone_failed:
        attribution = "local_variance_nonlinearity_primary"
    else:
        attribution = "unidentified"
    summary = {
        "maximum_channel_mean_density_error": float(max(mean_errors)),
        "median_of_median_absolute_raw_density_residual": float(
            np.median(raw_local)
        ),
        "median_of_median_absolute_bounded_density_residual": float(
            np.median(bounded_local)
        ),
        "maximum_compressed_pixel_fraction": float(max(compressed)),
        "maximum_final_encoded_luma_drift": float(max(final)),
        "correlation_final_drift_vs_global_mean_error": _correlation(
            final, mean_errors
        ),
        "correlation_final_drift_vs_local_bounded_residual": _correlation(
            final, bounded_local
        ),
        "correlation_final_drift_vs_compressed_fraction": _correlation(
            final, compressed
        ),
    }
    core = {
        "schema": REPORT_SCHEMA,
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "rows": rows,
        "summary": summary,
        "diagnostic_checks": {
            "global_density_mean_small": small_global_mean,
            "local_density_residual_large": large_local,
            "bound_compression_sparse": sparse_compression,
            "photographic_tone_drift_reproduced": tone_failed,
        },
        "attribution": attribution,
        "branch": config["branch_rule"][attribution],
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
    "evaluate_material_tone_mechanism",
    "validate_contract",
    "write_report",
]
