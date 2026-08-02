"""U6.P4CB typed density-chain comparison for bounded cloud occupancy."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.eval.real_uniform_grain_source import hash_file
from src.eval.typed_thomas_image_formation import (
    _isolated_excursions,
    _lag1_correlations,
    _layer_exposure,
    _mean_density,
    _srgb_code,
)
from src.film_physics.bounded_cloud_image_formation import (
    BoundedCloudImageFormationContext,
    render_typed_bounded_cloud_image_formation,
    render_typed_bounded_cloud_image_formation_region,
)
from src.film_physics.bounded_cloud_occupancy import (
    build_bounded_cloud_dc_receipt,
)
from src.film_physics.contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
    transmittance_to_density,
)
from src.film_physics.density_conditioned_thomas import (
    DensityConditionedThomasProfile,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)
from src.film_physics.thomas_dc_projection import build_thomas_dc_receipt
from src.film_physics.thomas_image_formation import (
    CHANNELS,
    ThomasImageFormationContext,
    render_typed_thomas_image_formation,
)

SCHEMA = "neuro_film.u6_p4cb_bounded_cloud_density_chain_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4cb_bounded_cloud_density_chain_report.v1"


class BoundedCloudDensityChainError(RuntimeError):
    """Raised when frozen P4CB inputs or semantics drift."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(binding["path"])
    if not path.is_file() or hash_file(path, "sha256") != binding["sha256"]:
        raise BoundedCloudDensityChainError(f"P4CB parent mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BoundedCloudDensityChainError("P4CB parent must be an object")
    return payload


def _validate_contract(contract: Mapping[str, Any]) -> None:
    candidate = contract.get("candidate", {})
    evaluation = contract.get("evaluation", {})
    if (
        contract.get("schema") != SCHEMA
        or candidate.get("occupancy_trials_per_cell") != 16
        or candidate.get("occupancy_probability") != 0.5
        or candidate.get("field_shape") != [512, 768]
        or candidate.get("scanner_stages") != ["spectral"]
        or candidate.get("realized_variance_normalization_allowed") is not False
        or candidate.get("pointwise_tail_compression_or_clipping_allowed") is not False
        or evaluation.get("minimum_p99_ratio_vs_p4by_gaussian") != 0.9
        or evaluation.get("maximum_p99_ratio_vs_p4by_gaussian") != 1.1
        or evaluation.get("maximum_maximum_delta_ratio_vs_p4by_gaussian") != 0.9
        or evaluation.get("maximum_isolated_excursion_count") != 0
        or evaluation.get("require_two_byte_identical_reports") is not True
        or evaluation.get("require_visual_severe_artifact_acceptance") is not True
    ):
        raise BoundedCloudDensityChainError("P4CB frozen contract drift")


def _contact_sheet_bytes(
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], gain: float
) -> bytes:
    rendered_rows: list[np.ndarray] = []
    for baseline, gaussian, bounded, difference in rows:
        panels = (
            _srgb_code(baseline),
            _srgb_code(gaussian),
            _srgb_code(bounded),
            np.clip(0.5 + gain * difference, 0.0, 1.0),
        )
        rendered_rows.append(
            np.concatenate(
                [
                    np.mean(
                        panel.reshape(
                            panel.shape[0] // 2,
                            2,
                            panel.shape[1] // 2,
                            2,
                            3,
                        ),
                        axis=(1, 3),
                    )
                    for panel in panels
                ],
                axis=1,
            )
        )
    sheet = np.concatenate(rendered_rows, axis=0)
    image = Image.fromarray(np.rint(sheet * 255.0).astype(np.uint8), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def evaluate_bounded_cloud_density_chain(
    contract: Mapping[str, Any], root: Path, *, diagnostic_path: Path | None = None
) -> dict[str, Any]:
    _validate_contract(contract)
    parents = contract["parents"]
    p4ca = _load_bound(root, parents["p4ca_decision"])
    p4by = _load_bound(root, parents["p4by_decision"])
    profile_payload = _load_bound(root, parents["p4bw_bundle"])
    prior_payload = _load_bound(root, parents["p2q_bundle"])
    if (
        p4ca.get("decision") != parents["p4ca_decision"]["required_decision"]
        or p4by.get("decision") != parents["p4by_decision"]["required_decision"]
        or profile_payload.get("profile_id")
        != parents["p4bw_bundle"]["required_profile_id"]
    ):
        raise BoundedCloudDensityChainError("P4CB parent decision drift")
    profile = DensityConditionedThomasProfile.from_dict(profile_payload)
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    if prior.identity() != parents["p2q_bundle"]["required_prior_id"]:
        raise BoundedCloudDensityChainError("P4CB prior identity drift")

    candidate = contract["candidate"]
    evaluation = contract["evaluation"]
    shape = tuple(int(value) for value in candidate["field_shape"])
    bounded_receipts = tuple(
        build_bounded_cloud_dc_receipt(
            shape,
            profile_id=str(candidate["occupancy_profile_id"]),
            particle_sigma_pixels=profile.particle_sigma_samples,
            cluster_sigma_pixels=profile.cluster_sigma_samples,
            mean_offspring=profile.mean_offspring,
            component_seeds=profile.component_seeds,
            realization_seed=int(seed),
            trials=int(candidate["occupancy_trials_per_cell"]),
            probability=float(candidate["occupancy_probability"]),
            truncate=profile.truncate,
            canonical_row_block_height=int(
                candidate["canonical_receipt_row_block_height"]
            ),
        )
        for seed in candidate["layer_realization_seeds"]
    )
    gaussian_receipts = tuple(
        build_thomas_dc_receipt(
            shape,
            profile_id=profile.spatial_profile_id,
            particle_sigma_pixels=profile.particle_sigma_samples,
            cluster_sigma_pixels=profile.cluster_sigma_samples,
            mean_offspring=profile.mean_offspring,
            component_seeds=profile.component_seeds,
            realization_seed=int(seed),
            truncate=profile.truncate,
            canonical_row_block_height=int(
                candidate["canonical_receipt_row_block_height"]
            ),
        )
        for seed in candidate["layer_realization_seeds"]
    )
    bounded_context = BoundedCloudImageFormationContext(
        density_profile_id=profile.identity(),
        source_spectrum_profile_id=profile.spatial_profile_id,
        occupancy_profile_id=str(candidate["occupancy_profile_id"]),
        prior_id=prior.identity(),
        scanner_profile_id=str(candidate["scanner_profile_id"]),
        full_shape=shape,
        receipt_ids=tuple(receipt.receipt_id for receipt in bounded_receipts),
        layer_realization_seeds=tuple(
            int(value) for value in candidate["layer_realization_seeds"]
        ),
        occupancy_trials=int(candidate["occupancy_trials_per_cell"]),
        occupancy_probability=float(candidate["occupancy_probability"]),
    )
    gaussian_context = ThomasImageFormationContext(
        profile_id=profile.identity(),
        prior_id=prior.identity(),
        scanner_profile_id=str(candidate["scanner_profile_id"]),
        full_shape=shape,
        receipt_ids=tuple(receipt.receipt_id for receipt in gaussian_receipts),
        layer_realization_seeds=tuple(
            int(value) for value in candidate["layer_realization_seeds"]
        ),
    )

    rows: list[dict[str, Any]] = []
    diagnostics: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    bounded_absolute: list[np.ndarray] = []
    gaussian_absolute: list[np.ndarray] = []
    isolated_total = 0
    all_new_boundary = 0
    all_samples = 0
    lag1: list[float] = []
    means: list[float] = []
    roundtrip_errors: list[float] = []
    repeat_exact: list[bool] = []
    partition_exact: list[bool] = []
    immutable: list[bool] = []
    identity_scanner_exact: list[bool] = []
    for fixture in candidate["fixtures"]:
        exposure, relative_log = _layer_exposure(
            fixture,
            shape,
            prior,
            float(candidate["normalized_exposure_fraction_minimum"]),
            float(candidate["normalized_exposure_fraction_maximum"]),
            float(candidate["pixel_pitch_micrometres"]),
        )
        input_sha = hashlib.sha256(exposure.values.tobytes()).hexdigest()
        bounded = render_typed_bounded_cloud_image_formation(
            exposure, profile, prior, bounded_receipts, bounded_context
        )
        repeated = render_typed_bounded_cloud_image_formation(
            exposure, profile, prior, bounded_receipts, bounded_context
        )
        gaussian = render_typed_thomas_image_formation(
            exposure, profile, prior, gaussian_receipts, gaussian_context
        )
        assembled = np.empty_like(bounded.scan_linear.values)
        row_height = int(candidate["row_partition_height"])
        for y0 in range(0, shape[0], row_height):
            height = min(row_height, shape[0] - y0)
            region = render_typed_bounded_cloud_image_formation_region(
                exposure,
                profile,
                prior,
                bounded_receipts,
                bounded_context,
                origin_yx=(y0, 0),
                shape=(height, shape[1]),
            )
            assembled[y0 : y0 + height] = region.scan_linear.values
        baseline_density = _mean_density(prior, relative_log)
        baseline = np.power(10.0, -baseline_density)
        bounded_delta = bounded.scan_linear.values - baseline
        gaussian_delta = gaussian.scan_linear.values - baseline
        bounded_abs = np.abs(bounded_delta)
        gaussian_abs = np.abs(gaussian_delta)
        epsilon = float(evaluation["boundary_epsilon"])
        new_boundary = (
            (bounded.scan_linear.values <= epsilon)
            | (bounded.scan_linear.values >= 1.0 - epsilon)
        ) & ~((baseline <= epsilon) | (baseline >= 1.0 - epsilon))
        row_isolated = _isolated_excursions(
            bounded_delta,
            float(evaluation["isolated_excursion_threshold"]),
            float(evaluation["isolated_neighbour_ratio"]),
        )
        row_lag1 = _lag1_correlations(bounded_delta)
        row_mean = np.mean(bounded_delta, axis=(0, 1), dtype=np.float64)
        roundtrip = transmittance_to_density(bounded.transmittance)
        roundtrip_error = float(
            np.max(np.abs(roundtrip.values - bounded.developed_density.values))
        )
        row_repeat = bool(
            np.array_equal(
                bounded.developed_density.values, repeated.developed_density.values
            )
            and np.array_equal(bounded.scan_linear.values, repeated.scan_linear.values)
        )
        row_partition = bool(np.array_equal(bounded.scan_linear.values, assembled))
        row_immutable = (
            hashlib.sha256(exposure.values.tobytes()).hexdigest() == input_sha
        )
        diagnostics.append(
            (
                baseline,
                gaussian.scan_linear.values,
                bounded.scan_linear.values,
                bounded.scan_linear.values - gaussian.scan_linear.values,
            )
        )
        bounded_absolute.append(bounded_abs.reshape(-1))
        gaussian_absolute.append(gaussian_abs.reshape(-1))
        isolated_total += row_isolated
        all_new_boundary += int(np.count_nonzero(new_boundary))
        all_samples += int(new_boundary.size)
        lag1.extend(row_lag1)
        means.extend(abs(float(value)) for value in row_mean)
        roundtrip_errors.append(roundtrip_error)
        repeat_exact.append(row_repeat)
        partition_exact.append(row_partition)
        immutable.append(row_immutable)
        identity_scanner_exact.append(
            bool(
                np.array_equal(bounded.scan_linear.values, bounded.transmittance.values)
            )
        )
        rows.append(
            {
                "fixture": fixture,
                "layer_exposure_sha256": input_sha,
                "bounded_scan_sha256": hashlib.sha256(
                    bounded.scan_linear.values.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
                "gaussian_scan_sha256": hashlib.sha256(
                    gaussian.scan_linear.values.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
                "bounded_absolute_scan_delta_p99": float(
                    np.percentile(bounded_abs, 99.0)
                ),
                "gaussian_absolute_scan_delta_p99": float(
                    np.percentile(gaussian_abs, 99.0)
                ),
                "bounded_maximum_absolute_scan_delta": float(np.max(bounded_abs)),
                "gaussian_maximum_absolute_scan_delta": float(np.max(gaussian_abs)),
                "isolated_excursion_count": row_isolated,
                "new_boundary_count": int(np.count_nonzero(new_boundary)),
                "minimum_axis_lag1_residual_correlation": min(row_lag1),
                "maximum_absolute_mean_scan_delta": max(
                    abs(float(value)) for value in row_mean
                ),
                "density_transmittance_roundtrip_error": roundtrip_error,
                "repeat_exact": row_repeat,
                "row_partition_exact": row_partition,
                "input_immutable": row_immutable,
            }
        )

    wrong_domain_rejected = False
    try:
        wrong = PhysicalDomainArray(
            exposure.values,
            PhysicalDomain.SCENE_LINEAR,
            PhysicalUnit.RELATIVE_SCENE_EXPOSURE,
            CHANNELS,
            exposure.scale,
        )
        render_typed_bounded_cloud_image_formation(
            wrong, profile, prior, bounded_receipts, bounded_context
        )
    except ValueError:
        wrong_domain_rejected = True
    bounded_combined = np.concatenate(bounded_absolute)
    gaussian_combined = np.concatenate(gaussian_absolute)
    bounded_p99 = float(np.percentile(bounded_combined, 99.0))
    gaussian_p99 = float(np.percentile(gaussian_combined, 99.0))
    p99_ratio = bounded_p99 / gaussian_p99
    bounded_maximum = float(np.max(bounded_combined))
    gaussian_maximum = float(np.max(gaussian_combined))
    maximum_ratio = bounded_maximum / gaussian_maximum
    boundary_fraction = all_new_boundary / all_samples
    sheet_bytes = _contact_sheet_bytes(
        diagnostics, float(candidate["diagnostic_delta_display_gain"])
    )
    sheet_sha = hashlib.sha256(sheet_bytes).hexdigest()
    if diagnostic_path is not None:
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_bytes(sheet_bytes)
    checks = {
        "fixture_count": len(rows) == evaluation["required_fixture_count"],
        "visible_effect": bounded_p99 >= evaluation["minimum_absolute_scan_delta_p99"],
        "bounded_p99": bounded_p99 <= evaluation["maximum_absolute_scan_delta_p99"],
        "p99_strength_parity": evaluation["minimum_p99_ratio_vs_p4by_gaussian"]
        <= p99_ratio
        <= evaluation["maximum_p99_ratio_vs_p4by_gaussian"],
        "bounded_maximum": bounded_maximum <= evaluation["maximum_absolute_scan_delta"],
        "maximum_tail_reduction": maximum_ratio
        <= evaluation["maximum_maximum_delta_ratio_vs_p4by_gaussian"],
        "no_new_boundary": boundary_fraction
        <= evaluation["maximum_new_boundary_fraction"],
        "no_isolated_excursions": isolated_total
        <= evaluation["maximum_isolated_excursion_count"],
        "spatially_correlated_residual": min(lag1)
        >= evaluation["minimum_axis_lag1_residual_correlation"],
        "bounded_mean_drift": max(means)
        <= evaluation["maximum_absolute_mean_scan_delta"],
        "density_transmittance_roundtrip": max(roundtrip_errors)
        <= evaluation["maximum_density_transmittance_roundtrip_error"],
        "identity_scanner_exact": all(identity_scanner_exact),
        "repeat_exact": all(repeat_exact),
        "row_partition_exact": all(partition_exact),
        "input_immutable": all(immutable),
        "wrong_domain_rejected": wrong_domain_rejected,
        "finite_bounded": bool(
            np.all(np.isfinite(bounded_combined))
            and all(
                np.all(scan >= 0.0) and np.all(scan <= 1.0)
                for _, _, scan, _ in diagnostics
            )
        ),
        "no_scanner_double_counting": (
            candidate["scanner_stages"] == ["spectral"]
            and candidate["scanner_convolution_embedded_in_spatial_profile"] is True
            and candidate["additional_scanner_mtf_flare_or_noise_allowed"] is False
        ),
        "no_tail_compression_or_normalization": (
            candidate["realized_variance_normalization_allowed"] is False
            and candidate["pointwise_tail_compression_or_clipping_allowed"] is False
        ),
    }
    automatic_pass = all(checks.values())
    stable = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": hash_file(
            root / "configs/u6_p4cb_bounded_cloud_density_chain_v1.json", "sha256"
        ),
        "density_profile_id": profile.identity(),
        "occupancy_profile_id": bounded_context.occupancy_profile_id,
        "prior_id": prior.identity(),
        "bounded_context_id": bounded_context.context_id,
        "gaussian_context_id": gaussian_context.context_id,
        "bounded_absolute_scan_delta_p99": bounded_p99,
        "gaussian_absolute_scan_delta_p99": gaussian_p99,
        "p99_ratio_vs_p4by_gaussian": p99_ratio,
        "bounded_maximum_absolute_scan_delta": bounded_maximum,
        "gaussian_maximum_absolute_scan_delta": gaussian_maximum,
        "maximum_delta_ratio_vs_p4by_gaussian": maximum_ratio,
        "new_boundary_fraction": boundary_fraction,
        "isolated_excursion_count": isolated_total,
        "minimum_axis_lag1_residual_correlation": min(lag1),
        "maximum_absolute_mean_scan_delta": max(means),
        "maximum_density_transmittance_roundtrip_error": max(roundtrip_errors),
        "diagnostic_contact_sheet_sha256": sheet_sha,
        "gate_results": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_bounded_cloud_density_chain"
            if automatic_pass
            else "close_bounded_cloud_density_chain_without_rescue"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(
            stable, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return {**stable, "stable_evidence_id": stable_id, "rows": rows}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    encoded = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "BoundedCloudDensityChainError",
    "evaluate_bounded_cloud_density_chain",
    "write_report",
]
