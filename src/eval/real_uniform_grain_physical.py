"""Physical-domain audit for the fixed generic anisotropic grain candidate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

from src.eval.physical_density_conditioned_structure import (
    profiles_from_contract,
)
from src.eval.real_uniform_grain_nps import acf_lag_signature
from src.eval.real_uniform_grain_source import hash_file
from src.film_physics.density_conditioned_structure import (
    counter_poisson_rate_field,
    render_density_conditioned_structure,
)


class AnisotropicGrainPhysicalError(RuntimeError):
    """Raised when the fixed P4T physical audit fails closed."""


def _validate_region(
    target: np.ndarray,
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    grain_optical_density: float,
    sigma_yx: tuple[float, float],
    seed: int,
    maximum_target_density: float,
    truncate: float,
) -> np.ndarray:
    values = np.asarray(target, dtype=np.float64)
    sigma_y, sigma_x = sigma_yx
    if (
        values.ndim != 2
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > maximum_target_density)
        or not math.isfinite(grain_optical_density)
        or grain_optical_density <= 0.0
        or not math.isfinite(sigma_y)
        or not math.isfinite(sigma_x)
        or sigma_y < 0.0
        or sigma_x < 0.0
        or not math.isfinite(maximum_target_density)
        or maximum_target_density <= 0.0
        or not math.isfinite(truncate)
        or truncate <= 0.0
        or not isinstance(seed, int)
        or seed < 0
        or seed >= 2**64
    ):
        raise ValueError("invalid anisotropic density target or profile")
    y0, x0 = origin_yx
    height, width = shape
    if (
        y0 < 0
        or x0 < 0
        or height <= 0
        or width <= 0
        or y0 + height > values.shape[0]
        or x0 + width > values.shape[1]
    ):
        raise ValueError("anisotropic density region leaves target")
    return values


def render_anisotropic_density_region(
    target: np.ndarray,
    *,
    origin_yx: tuple[int, int],
    shape: tuple[int, int],
    grain_optical_density: float,
    sigma_yx: tuple[float, float],
    seed: int,
    maximum_target_density: float,
    truncate: float,
) -> np.ndarray:
    """Render one nonnegative developed-density region."""
    values = _validate_region(
        target,
        origin_yx=origin_yx,
        shape=shape,
        grain_optical_density=grain_optical_density,
        sigma_yx=sigma_yx,
        seed=seed,
        maximum_target_density=maximum_target_density,
        truncate=truncate,
    )
    sigma_y, sigma_x = sigma_yx
    halo_y = int(truncate * sigma_y + 0.5)
    halo_x = int(truncate * sigma_x + 0.5)
    origin_y, origin_x = origin_yx
    height, width = shape
    y0 = max(0, origin_y - halo_y)
    x0 = max(0, origin_x - halo_x)
    y1 = min(values.shape[0], origin_y + height + halo_y)
    x1 = min(values.shape[1], origin_x + width + halo_x)
    rate = values[y0:y1, x0:x1] / grain_optical_density
    counts = counter_poisson_rate_field(
        rate,
        values.shape,
        origin_yx=(y0, x0),
        seed=seed,
        maximum_rate=maximum_target_density / grain_optical_density,
    ).astype(np.float64)
    filtered = gaussian_filter(
        counts,
        sigma=(sigma_y, sigma_x),
        order=0,
        mode="constant",
        cval=0.0,
        truncate=truncate,
    )
    crop_y = origin_y - y0
    crop_x = origin_x - x0
    density = np.asarray(
        grain_optical_density
        * filtered[
            crop_y : crop_y + height,
            crop_x : crop_x + width,
        ],
        dtype=np.float32,
    )
    if np.any(density < 0.0) or not np.all(np.isfinite(density)):
        raise AnisotropicGrainPhysicalError(
            "anisotropic density left its physical domain"
        )
    density.setflags(write=False)
    return density


def render_anisotropic_structure(
    target_density: np.ndarray,
    *,
    grain_optical_density_by_channel: list[float],
    sigma_yx: tuple[float, float],
    seeds: list[int],
    maximum_target_density: float,
    truncate: float,
    origin_yx: tuple[int, int] | None = None,
    shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render HxWxC density and transmittance under one shared kernel."""
    target = np.asarray(target_density, dtype=np.float64)
    if (
        target.ndim != 3
        or target.shape[2] != len(grain_optical_density_by_channel)
        or target.shape[2] != len(seeds)
    ):
        raise ValueError("anisotropic target/profile channel mismatch")
    resolved_origin = (0, 0) if origin_yx is None else origin_yx
    resolved_shape = target.shape[:2] if shape is None else shape
    layers = [
        render_anisotropic_density_region(
            target[..., channel],
            origin_yx=resolved_origin,
            shape=resolved_shape,
            grain_optical_density=float(
                grain_optical_density_by_channel[channel]
            ),
            sigma_yx=sigma_yx,
            seed=int(seeds[channel]),
            maximum_target_density=maximum_target_density,
            truncate=truncate,
        )
        for channel in range(target.shape[2])
    ]
    density = np.stack(layers, axis=-1).astype(np.float32)
    transmittance = np.exp(-density.astype(np.float64)).astype(np.float32)
    if (
        np.any(transmittance <= 0.0)
        or np.any(transmittance > 1.0)
        or not np.all(np.isfinite(transmittance))
    ):
        raise AnisotropicGrainPhysicalError(
            "anisotropic transmittance left its physical domain"
        )
    density.setflags(write=False)
    transmittance.setflags(write=False)
    return density, transmittance


def _load_bound_json(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = root / binding["path"]
    if hash_file(path, "sha256") != binding["sha256"]:
        raise AnisotropicGrainPhysicalError(f"parent hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AnisotropicGrainPhysicalError("parent payload must be an object")
    return payload


def _metrics(
    target: np.ndarray,
    density: np.ndarray,
    transmittance: np.ndarray,
    *,
    border: int,
) -> dict[str, Any]:
    interior = density[border:-border, border:-border]
    target_interior = target[border:-border, border:-border]
    return {
        "channel_mean_density": [
            float(np.mean(interior[..., channel], dtype=np.float64))
            for channel in range(interior.shape[2])
        ],
        "channel_variance_density": [
            float(np.var(interior[..., channel], dtype=np.float64))
            for channel in range(interior.shape[2])
        ],
        "channel_mean_absolute_error": [
            float(
                abs(
                    np.mean(interior[..., channel], dtype=np.float64)
                    - np.mean(
                        target_interior[..., channel],
                        dtype=np.float64,
                    )
                )
            )
            for channel in range(interior.shape[2])
        ],
        "channel_mean_absolute_pixel_deviation": [
            float(
                np.mean(
                    np.abs(
                        interior[..., channel]
                        - target_interior[..., channel]
                    ),
                    dtype=np.float64,
                )
            )
            for channel in range(interior.shape[2])
        ],
        "minimum_density": float(np.min(density)),
        "maximum_transmittance": float(np.max(transmittance)),
        "density_sha256": hashlib.sha256(
            density.tobytes(order="C")
        ).hexdigest(),
        "transmittance_sha256": hashlib.sha256(
            transmittance.tobytes(order="C")
        ).hexdigest(),
    }


def _physical_metrics(
    *,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    candidate = contract["candidate"]
    evaluation = contract["evaluation"]
    height, width = (int(value) for value in evaluation["shape"])
    grain_od = [
        float(value)
        for value in candidate["grain_optical_density_by_rgb_layer"]
    ]
    sigma = tuple(float(value) for value in candidate["sigma_yx_pixels"])
    seeds = [int(value) for value in candidate["layer_seeds"]]
    maximum = float(candidate["maximum_target_density"])
    truncate = float(candidate["truncate"])
    border = int(evaluation["interior_border_pixels"])
    flat_rows = []
    for level in evaluation["flat_density_levels"]:
        target = np.full(
            (height, width, len(grain_od)),
            float(level),
            dtype=np.float64,
        )
        density, transmittance = render_anisotropic_structure(
            target,
            grain_optical_density_by_channel=grain_od,
            sigma_yx=sigma,
            seeds=seeds,
            maximum_target_density=maximum,
            truncate=truncate,
        )
        flat_rows.append(
            {
                "target_density": float(level),
                **_metrics(
                    target,
                    density,
                    transmittance,
                    border=border,
                ),
            }
        )
    ramp = np.linspace(
        0.0,
        float(evaluation["ramp_maximum_density"]),
        width,
        dtype=np.float64,
    )
    ramp_target = np.broadcast_to(
        ramp[None, :, None],
        (height, width, len(grain_od)),
    ).copy()
    ramp_density, ramp_transmittance = render_anisotropic_structure(
        ramp_target,
        grain_optical_density_by_channel=grain_od,
        sigma_yx=sigma,
        seeds=seeds,
        maximum_target_density=maximum,
        truncate=truncate,
    )
    repeat_density, repeat_transmittance = render_anisotropic_structure(
        ramp_target,
        grain_optical_density_by_channel=grain_od,
        sigma_yx=sigma,
        seeds=seeds,
        maximum_target_density=maximum,
        truncate=truncate,
    )
    column_mean = np.mean(
        ramp_density[border:-border],
        axis=0,
        dtype=np.float64,
    )
    correlations = [
        float(np.corrcoef(ramp, column_mean[:, channel])[0, 1])
        for channel in range(len(grain_od))
    ]
    ramp_errors = [
        float(np.mean(np.abs(column_mean[:, channel] - ramp)))
        for channel in range(len(grain_od))
    ]
    partition_exact = True
    for row_height in evaluation["row_partitions"]:
        assembled_density = np.empty_like(ramp_density)
        assembled_transmittance = np.empty_like(ramp_transmittance)
        for y0 in range(0, height, int(row_height)):
            rows = min(int(row_height), height - y0)
            region_density, region_transmittance = (
                render_anisotropic_structure(
                    ramp_target,
                    grain_optical_density_by_channel=grain_od,
                    sigma_yx=sigma,
                    seeds=seeds,
                    maximum_target_density=maximum,
                    truncate=truncate,
                    origin_yx=(y0, 0),
                    shape=(rows, width),
                )
            )
            assembled_density[y0 : y0 + rows] = region_density
            assembled_transmittance[y0 : y0 + rows] = region_transmittance
        partition_exact &= bool(
            np.array_equal(ramp_density, assembled_density)
            and np.array_equal(ramp_transmittance, assembled_transmittance)
        )
    gates = contract["automatic_gates"]
    maximum_flat_mean_error = max(
        error
        for row in flat_rows
        for error in row["channel_mean_absolute_error"]
    )
    levels = [
        float(value) for value in evaluation["flat_density_levels"]
    ]
    low = flat_rows[levels.index(0.3)]
    high = flat_rows[levels.index(1.2)]
    variance_ratios = [
        high["channel_variance_density"][channel]
        / low["channel_variance_density"][channel]
        for channel in range(len(grain_od))
    ]
    gate_results = {
        "flat_mean": maximum_flat_mean_error
        <= float(gates["maximum_flat_mean_absolute_error"]),
        "density_conditioned_variance": min(variance_ratios)
        >= float(gates["minimum_density_variance_high_to_low_ratio"]),
        "ramp_correlation": min(correlations)
        >= float(gates["minimum_ramp_target_output_correlation"]),
        "ramp_mean": max(ramp_errors)
        <= float(gates["maximum_ramp_mean_absolute_error"]),
        "physical_domain": min(
            row["minimum_density"] for row in flat_rows
        )
        >= float(gates["minimum_density"])
        and max(row["maximum_transmittance"] for row in flat_rows)
        <= float(gates["maximum_transmittance"]),
        "zero_identity": all(
            value == 0.0
            for value in flat_rows[0]["channel_mean_density"]
        )
        and flat_rows[0]["maximum_transmittance"] == 1.0,
        "repeat_exact": bool(
            np.array_equal(ramp_density, repeat_density)
            and np.array_equal(ramp_transmittance, repeat_transmittance)
        ),
        "row_partition_exact": partition_exact,
    }
    return (
        {
            "flat_rows": flat_rows,
            "maximum_flat_mean_absolute_error": maximum_flat_mean_error,
            "minimum_high_to_low_variance_ratio": min(variance_ratios),
            "ramp_minimum_correlation": min(correlations),
            "ramp_maximum_mean_absolute_error": max(ramp_errors),
            "ramp_density_sha256": hashlib.sha256(
                ramp_density.tobytes(order="C")
            ).hexdigest(),
            "ramp_transmittance_sha256": hashlib.sha256(
                ramp_transmittance.tobytes(order="C")
            ).hexdigest(),
        },
        gate_results,
    )


def _acf_metrics(
    *,
    p4r_report: dict[str, Any],
    p4r_analysis: dict[str, Any],
    p4s_contract: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    confirmation_ids = contract["evaluation"]["confirmation_source_ids"]
    source_ids = [
        row["source_id"] for row in p4r_report["source_rows"]
    ]
    indexes = [source_ids.index(source_id) for source_id in confirmation_ids]
    channels = ("red", "green", "blue")
    observed = {
        channel: np.median(
            np.asarray(
                p4r_report["median_scan_acf_by_channel"][channel],
                dtype=np.float64,
            )[indexes],
            axis=0,
        )
        for channel in channels
    }
    model = p4s_contract["model"]
    field_shape = tuple(int(value) for value in model["field_shape"])
    crop_y, crop_x, crop_h, crop_w = (
        int(value) for value in model["central_analysis_crop"]
    )
    target_density = float(model["target_density"])
    grain_od = [
        float(value) for value in model["grain_optical_density_by_rgb_layer"]
    ]
    baseline_sigmas = [
        tuple(float(value) for value in row)
        for row in model["baseline_layer_sigmas_yx"]
    ]
    candidate_sigma = tuple(
        float(value) for value in contract["candidate"]["sigma_yx_pixels"]
    )
    truncate = float(contract["candidate"]["truncate"])
    lags = p4r_analysis["pixel_contract"]["acf_lags_pixels_yx"]
    baseline_seeds = [
        int(value) for value in contract["evaluation"]["acf_baseline_seeds"]
    ]
    candidate_seeds = [
        int(value) for value in contract["evaluation"]["acf_candidate_seeds"]
    ]
    if baseline_seeds != candidate_seeds:
        raise AnisotropicGrainPhysicalError(
            "baseline/candidate ACF seeds must match"
        )
    seeds = candidate_seeds

    def centroid(sigmas: list[tuple[float, float]]) -> dict[str, np.ndarray]:
        rows: dict[str, list[np.ndarray]] = {
            channel: [] for channel in channels
        }
        for seed in seeds:
            for channel, name in enumerate(channels):
                target = np.full(
                    field_shape,
                    target_density,
                    dtype=np.float64,
                )
                density = render_anisotropic_density_region(
                    target,
                    origin_yx=(0, 0),
                    shape=field_shape,
                    grain_optical_density=grain_od[channel],
                    sigma_yx=sigmas[channel],
                    seed=seed + 104729 * channel,
                    maximum_target_density=float(
                        contract["candidate"]["maximum_target_density"]
                    ),
                    truncate=truncate,
                )
                rows[name].append(
                    acf_lag_signature(
                        density[
                            crop_y : crop_y + crop_h,
                            crop_x : crop_x + crop_w,
                        ],
                        lags_yx=lags,
                    )
                )
        return {
            channel: np.median(np.asarray(values), axis=0)
            for channel, values in rows.items()
        }

    baseline = centroid(baseline_sigmas)
    candidate = centroid([candidate_sigma] * 3)
    baseline_errors = np.concatenate(
        [np.abs(baseline[name] - observed[name]) for name in channels]
    )
    candidate_errors = np.concatenate(
        [np.abs(candidate[name] - observed[name]) for name in channels]
    )
    baseline_median = float(np.median(baseline_errors))
    candidate_median = float(np.median(candidate_errors))
    baseline_max = float(np.max(baseline_errors))
    candidate_max = float(np.max(candidate_errors))
    improvement = 1.0 - candidate_median / baseline_median
    worst_ratio = candidate_max / baseline_max
    gates = contract["automatic_gates"]
    gate_results = {
        "held_acf_median_improvement": improvement
        >= float(
            gates["minimum_held_acf_median_error_improvement_fraction"]
        ),
        "held_acf_worst_not_worse": worst_ratio
        <= float(gates["maximum_held_acf_worst_error_ratio"]),
    }
    return (
        {
            "confirmation_source_ids": confirmation_ids,
            "baseline_median_absolute_error": baseline_median,
            "candidate_median_absolute_error": candidate_median,
            "median_error_improvement_fraction": improvement,
            "baseline_maximum_absolute_error": baseline_max,
            "candidate_maximum_absolute_error": candidate_max,
            "candidate_to_baseline_worst_error_ratio": worst_ratio,
            "observed_median_acf_by_channel": {
                key: value.tolist() for key, value in observed.items()
            },
            "baseline_acf_by_channel": {
                key: value.tolist() for key, value in baseline.items()
            },
            "candidate_acf_by_channel": {
                key: value.tolist() for key, value in candidate.items()
            },
        },
        gate_results,
    )


def _diagnostic_targets(
    *,
    height: int,
    width: int,
) -> list[tuple[str, np.ndarray]]:
    rows = []
    for level in (0.3, 0.8, 1.2):
        rows.append(
            (
                f"flat-{level}",
                np.full((height, width, 3), level, dtype=np.float64),
            )
        )
    ramp = np.linspace(0.0, 1.8, width, dtype=np.float64)
    rows.append(
        (
            "horizontal-ramp-0-to-1.8",
            np.broadcast_to(ramp[None, :, None], (height, width, 3)).copy(),
        )
    )
    step = np.full((height, width, 3), 0.1, dtype=np.float64)
    step[:, width // 2 :] = 1.2
    rows.append(("hard-step-0.1-to-1.2", step))
    island = np.full((height, width, 3), 0.3, dtype=np.float64)
    y0, y1 = height // 2 - 24, height // 2 + 24
    x0, x1 = width // 2 - 24, width // 2 + 24
    island[y0:y1, x0:x1] = 1.8
    rows.append(("small-highlight-island-1.8-on-0.3", island))
    return rows


def _save_diagnostics(
    *,
    output_dir: Path,
    contract: dict[str, Any],
    p4d: dict[str, Any],
) -> dict[str, Any]:
    evaluation = contract["evaluation"]
    height, width = (int(value) for value in evaluation["shape"])
    candidate = contract["candidate"]
    grain_od = [
        float(value)
        for value in candidate["grain_optical_density_by_rgb_layer"]
    ]
    sigma = tuple(float(value) for value in candidate["sigma_yx_pixels"])
    seeds = [int(value) for value in candidate["layer_seeds"]]
    baseline_profiles = profiles_from_contract(p4d)
    font = ImageFont.load_default()
    panel_width, panel_height = width, height
    rows = _diagnostic_targets(height=height, width=width)
    if [name for name, _ in rows] != evaluation[
        "diagnostic_density_patterns"
    ]:
        raise AnisotropicGrainPhysicalError(
            "diagnostic pattern contract mismatch"
        )
    sheet = Image.new(
        "RGB",
        (panel_width * 2 + 24, len(rows) * (panel_height + 28) + 20),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    artifact_rows = []
    for index, (name, target) in enumerate(rows):
        baseline = render_density_conditioned_structure(
            target,
            baseline_profiles,
        )
        candidate_density, candidate_transmittance = (
            render_anisotropic_structure(
                target,
                grain_optical_density_by_channel=grain_od,
                sigma_yx=sigma,
                seeds=seeds,
                maximum_target_density=float(
                    candidate["maximum_target_density"]
                ),
                truncate=float(candidate["truncate"]),
            )
        )
        baseline_code = np.rint(
            255.0 * (1.0 - baseline.transmittance)
        ).astype(np.uint8)
        candidate_code = np.rint(
            255.0 * (1.0 - candidate_transmittance)
        ).astype(np.uint8)
        y = 20 + index * (panel_height + 28)
        draw.text((4, y - 16), f"{name} / P4D", fill="black", font=font)
        draw.text(
            (panel_width + 20, y - 16),
            f"{name} / P4T",
            fill="black",
            font=font,
        )
        sheet.paste(Image.fromarray(baseline_code, mode="RGB"), (4, y))
        sheet.paste(
            Image.fromarray(candidate_code, mode="RGB"),
            (panel_width + 20, y),
        )
        artifact_rows.append(
            {
                "pattern": name,
                "candidate_density_sha256": hashlib.sha256(
                    candidate_density.tobytes(order="C")
                ).hexdigest(),
                "candidate_transmittance_sha256": hashlib.sha256(
                    candidate_transmittance.tobytes(order="C")
                ).hexdigest(),
                "candidate_minimum_density": float(
                    np.min(candidate_density)
                ),
                "candidate_maximum_density": float(
                    np.max(candidate_density)
                ),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "diagnostic_contact_sheet.png"
    sheet.save(path, "PNG", compress_level=6)
    return {
        "path": str(path),
        "sha256": hash_file(path, "sha256"),
        "rows": artifact_rows,
        "visual_review_required": True,
    }


def run_anisotropic_grain_physical(
    *,
    root: Path,
    contract: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Run fixed held-ACF and explicit physical-domain gates."""
    if (
        contract.get("schema")
        != "neuro_film.u6_p4t_anisotropic_grain_physical_contract.v1"
    ):
        raise AnisotropicGrainPhysicalError("unsupported P4T contract")
    parents = contract["parents"]
    p4s_contract = _load_bound_json(root, parents["p4s_contract"])
    p4s_report = _load_bound_json(root, parents["p4s_report"])
    p4s_decision = _load_bound_json(root, parents["p4s_decision"])
    p4r_report = _load_bound_json(root, parents["p4r_report"])
    p4r_analysis = _load_bound_json(root, parents["p4r_analysis"])
    p4d = _load_bound_json(root, parents["p4d_contract"])
    if (
        p4s_report.get("decision")
        != parents["p4s_report"]["required_decision"]
        or p4s_decision.get("result", {}).get("status")
        != parents["p4s_decision"]["required_status"]
        or p4s_report.get("selected_shared_sigma_yx_pixels")
        != contract["candidate"]["sigma_yx_pixels"]
        or any(
            contract["candidate"].get(key) is not False
            for key in (
                "display_rgb_noise_allowed",
                "hard_clipping_allowed",
                "signed_density_allowed",
            )
        )
    ):
        raise AnisotropicGrainPhysicalError("parent/candidate admission failed")
    physical, physical_gates = _physical_metrics(contract=contract)
    acf, acf_gates = _acf_metrics(
        p4r_report=p4r_report,
        p4r_analysis=p4r_analysis,
        p4s_contract=p4s_contract,
        contract=contract,
    )
    gate_results = {**physical_gates, **acf_gates}
    automatic_pass = all(gate_results.values())
    diagnostics = (
        _save_diagnostics(
            output_dir=output_dir,
            contract=contract,
            p4d=p4d,
        )
        if automatic_pass
        else None
    )
    stable = {
        "schema": "neuro_film.u6_p4t_anisotropic_grain_physical_report.v1",
        "contract_sha256": hash_file(
            root / "configs/u6_p4t_anisotropic_grain_physical_v1.json",
            "sha256",
        ),
        "selected_sigma_yx_pixels": contract["candidate"]["sigma_yx_pixels"],
        "held_acf": acf,
        "physical_metrics": physical,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
        "visual_review_allowed": automatic_pass,
        "diagnostics": diagnostics,
        "decision": (
            "automatic_pass_open_fixed_visual_severe_review"
            if automatic_pass
            else "close_anisotropic_candidate_retain_p4d_p4q"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**stable, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AnisotropicGrainPhysicalError",
    "render_anisotropic_density_region",
    "render_anisotropic_structure",
    "run_anisotropic_grain_physical",
    "write_report",
]
