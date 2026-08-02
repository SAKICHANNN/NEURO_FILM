"""U6.P4BZ photographic severe-artifact stress for the P4BY chain."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import gaussian_filter

from src.eval.density_witness_frontier import (
    linear_srgb_to_encoded,
)
from src.eval.physical_spatial_photographic_stress import (
    _flat_region_p99,
    _load_source,
)
from src.eval.real_uniform_grain_source import hash_file
from src.film_physics.contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
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
    render_typed_thomas_image_formation_region,
)

SCHEMA = "neuro_film.u6_p4bz_thomas_photographic_stress_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bz_thomas_photographic_stress_report.v1"


class ThomasPhotographicStressError(RuntimeError):
    """Raised when the frozen P4BZ inputs or semantics drift."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _load_bound(root: Path, binding: Mapping[str, Any]) -> Any:
    path = root / str(binding["path"])
    if not path.is_file() or hash_file(path, "sha256") != binding["sha256"]:
        raise ThomasPhotographicStressError(f"P4BZ parent mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_contract(contract: Mapping[str, Any]) -> None:
    adapter = contract.get("adapter", {})
    candidate = contract.get("candidate", {})
    evaluation = contract.get("evaluation", {})
    if (
        contract.get("schema") != SCHEMA
        or contract.get("input", {}).get("expected_rows") != 9
        or contract.get("input", {}).get("expected_camera_makes") != 9
        or adapter.get("minimum_domain_fraction") != 0.15
        or adapter.get("maximum_domain_fraction") != 0.85
        or adapter.get("per_image_normalization_allowed") is not False
        or adapter.get("histogram_or_content_fit_allowed") is not False
        or adapter.get("clipping_allowed") is not False
        or candidate.get("scanner_stages") != ["spectral"]
        or candidate.get("scanner_convolution_embedded_in_spatial_profile") is not True
        or evaluation.get("maximum_population_absolute_scan_delta_p99") != 0.03
        or evaluation.get("maximum_absolute_scan_delta") != 0.08
        or evaluation.get("maximum_new_boundary_fraction") != 0.0
        or evaluation.get("maximum_isolated_excursion_count") != 0
        or evaluation.get("minimum_axis_lag1_residual_correlation") != 0.75
        or evaluation.get("lowpass_structure_sigma_pixels") != 2.0
        or evaluation.get("require_two_byte_identical_reports") is not True
        or evaluation.get("require_visual_severe_artifact_acceptance") is not True
    ):
        raise ThomasPhotographicStressError("P4BZ frozen contract drift")


def _derive_seed(source_id: str, channel: str, base_seed: int) -> int:
    encoded = f"u6.p4bz{source_id}{channel}{base_seed}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


def _adapt_display_linear_to_layer_exposure(
    source: np.ndarray,
    prior: ManufacturerCharacteristicPrior,
    *,
    minimum_fraction: float,
    maximum_fraction: float,
    pixel_pitch_um: float,
) -> tuple[PhysicalDomainArray, np.ndarray]:
    values = np.asarray(source, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape[-1] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ThomasPhotographicStressError(
            "display-linear source must be finite RGB in [0,1]"
        )
    fraction = minimum_fraction + (maximum_fraction - minimum_fraction) * values
    relative_log = np.empty_like(fraction)
    for index, curve in enumerate(prior.curves):
        lower, upper = curve.domain
        relative_log[..., index] = lower + fraction[..., index] * (upper - lower)
    exposure = PhysicalDomainArray(
        np.power(10.0, relative_log),
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        CHANNELS,
        PhysicalScale(pixel_pitch_um),
    )
    return exposure, relative_log


def _mean_density(
    prior: ManufacturerCharacteristicPrior, relative_log: np.ndarray
) -> np.ndarray:
    density = np.empty_like(relative_log)
    for index, curve in enumerate(prior.curves):
        density[..., index] = curve.apply(relative_log[..., index])
    return density


def _lag1_correlations(delta: np.ndarray) -> list[float]:
    correlations: list[float] = []
    for channel in range(3):
        values = delta[..., channel]
        for left, right in (
            (values[:, :-1], values[:, 1:]),
            (values[:-1], values[1:]),
        ):
            if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
                correlations.append(1.0)
            else:
                correlations.append(
                    float(np.corrcoef(left.reshape(-1), right.reshape(-1))[0, 1])
                )
    return correlations


def _isolated_excursions(delta: np.ndarray, threshold: float, ratio: float) -> int:
    absolute = np.abs(delta)
    count = 0
    for channel in range(3):
        values = absolute[..., channel]
        center = values[1:-1, 1:-1]
        neighbours = np.maximum.reduce(
            [
                values[dy : dy + values.shape[0] - 2, dx : dx + values.shape[1] - 2]
                for dy in range(3)
                for dx in range(3)
                if (dy, dx) != (1, 1)
            ]
        )
        count += int(
            np.count_nonzero((center > threshold) & (neighbours < ratio * center))
        )
    return count


def _lowpass_structure_correlation(
    baseline: np.ndarray, candidate: np.ndarray, sigma: float
) -> float:
    correlations: list[float] = []
    for channel in range(3):
        left = gaussian_filter(baseline[..., channel], sigma=sigma, mode="nearest")
        right = gaussian_filter(candidate[..., channel], sigma=sigma, mode="nearest")
        correlations.append(
            float(np.corrcoef(left.reshape(-1), right.reshape(-1))[0, 1])
        )
    return min(correlations)


def _preview(linear: np.ndarray, size: tuple[int, int]) -> Image.Image:
    encoded = linear_srgb_to_encoded(np.clip(linear, 0.0, 1.0))
    image = Image.fromarray(
        np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB"
    )
    return ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)


def _contact_sheet_bytes(rows: list[dict[str, Any]], gain: float) -> bytes:
    tile_width, tile_height, header = 240, 160, 22
    canvas = Image.new(
        "RGB", (4 * tile_width, len(rows) * (tile_height + header)), color=(24, 24, 24)
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(rows):
        y = row_index * (tile_height + header)
        draw.text((4, y + 4), row["id"], fill=(235, 235, 235))
        views = (
            row["source"],
            row["baseline"],
            row["candidate"],
            np.clip(0.5 + gain * row["delta"], 0.0, 1.0),
        )
        for column, values in enumerate(views):
            image = _preview(values, (tile_width, tile_height))
            x = column * tile_width + (tile_width - image.width) // 2
            canvas.paste(image, (x, y + header))
    buffer = BytesIO()
    canvas.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def evaluate_thomas_photographic_stress(
    contract: Mapping[str, Any], root: Path, *, contact_sheet_path: Path | None = None
) -> dict[str, Any]:
    _validate_contract(contract)
    parents = contract["parents"]
    p4by = _load_bound(root, parents["p4by_decision"])
    profile_payload = _load_bound(root, parents["p4bw_bundle"])
    prior_payload = _load_bound(root, parents["p2q_bundle"])
    manifest = _load_bound(root, parents["source_manifest"])
    _load_bound(root, parents["source_preflight"])
    if (
        p4by.get("decision") != parents["p4by_decision"]["required_decision"]
        or profile_payload.get("profile_id")
        != parents["p4bw_bundle"]["required_profile_id"]
    ):
        raise ThomasPhotographicStressError("P4BZ parent decision drift")
    profile = DensityConditionedThomasProfile.from_dict(profile_payload)
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    if prior.identity() != parents["p2q_bundle"]["required_prior_id"]:
        raise ThomasPhotographicStressError("P4BZ prior identity drift")

    input_spec = contract["input"]
    by_id = {row["id"]: row for row in manifest}
    selected = [by_id[source_id] for source_id in input_spec["selected_ids"]]
    if (
        len(selected) != input_spec["expected_rows"]
        or len({row["make"] for row in selected}) != input_spec["expected_camera_makes"]
        or any(
            row["allowed_use"] != input_spec["required_allowed_use"]
            or row["rights_scope"] != input_spec["required_rights_scope"]
            or row["decoded_color_state"] != input_spec["required_decoded_color_state"]
            for row in selected
        )
    ):
        raise ThomasPhotographicStressError("P4BZ source boundary drift")

    adapter = contract["adapter"]
    candidate_spec = contract["candidate"]
    evaluation = contract["evaluation"]
    rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    absolute_values: list[np.ndarray] = []
    receipt_ids: list[str] = []
    for source_row in selected:
        source, input_sha = _load_source(source_row, root)
        shape = source.shape[:2]
        derived_seeds = tuple(
            _derive_seed(source_row["id"], channel, int(base_seed))
            for channel, base_seed in zip(
                CHANNELS,
                candidate_spec["base_layer_realization_seeds"],
                strict=True,
            )
        )
        receipts = tuple(
            build_thomas_dc_receipt(
                shape,
                profile_id=profile.spatial_profile_id,
                particle_sigma_pixels=profile.particle_sigma_samples,
                cluster_sigma_pixels=profile.cluster_sigma_samples,
                mean_offspring=profile.mean_offspring,
                component_seeds=profile.component_seeds,
                realization_seed=seed,
                truncate=profile.truncate,
                canonical_row_block_height=int(
                    candidate_spec["canonical_receipt_row_block_height"]
                ),
            )
            for seed in derived_seeds
        )
        context = ThomasImageFormationContext(
            profile_id=profile.identity(),
            prior_id=prior.identity(),
            scanner_profile_id=str(candidate_spec["scanner_profile_id"]),
            full_shape=shape,
            receipt_ids=tuple(receipt.receipt_id for receipt in receipts),
            layer_realization_seeds=derived_seeds,
        )
        receipt_ids.extend(context.receipt_ids)
        exposure, relative_log = _adapt_display_linear_to_layer_exposure(
            source,
            prior,
            minimum_fraction=float(adapter["minimum_domain_fraction"]),
            maximum_fraction=float(adapter["maximum_domain_fraction"]),
            pixel_pitch_um=float(candidate_spec["pixel_pitch_micrometres"]),
        )
        result = render_typed_thomas_image_formation(
            exposure, profile, prior, receipts, context
        )
        assembled = np.empty_like(result.scan_linear.values)
        row_height = int(candidate_spec["row_partition_height"])
        for y0 in range(0, shape[0], row_height):
            height = min(row_height, shape[0] - y0)
            region = render_typed_thomas_image_formation_region(
                exposure,
                profile,
                prior,
                receipts,
                context,
                origin_yx=(y0, 0),
                shape=(height, shape[1]),
            )
            assembled[y0 : y0 + height] = region.scan_linear.values
        baseline = np.power(10.0, -_mean_density(prior, relative_log))
        delta = result.scan_linear.values - baseline
        absolute = np.abs(delta)
        epsilon = float(evaluation["boundary_epsilon"])
        new_boundary = (
            (result.scan_linear.values <= epsilon)
            | (result.scan_linear.values >= 1.0 - epsilon)
        ) & ~((baseline <= epsilon) | (baseline >= 1.0 - epsilon))
        roundtrip = transmittance_to_density(result.transmittance)
        lag1 = _lag1_correlations(delta)
        isolated = _isolated_excursions(
            delta,
            float(evaluation["isolated_excursion_threshold"]),
            float(evaluation["isolated_neighbour_ratio"]),
        )
        row = {
            "id": source_row["id"],
            "make": source_row["make"],
            "input_sha256": input_sha,
            "shape": list(result.scan_linear.values.shape),
            "layer_realization_seeds": list(derived_seeds),
            "receipt_ids": list(context.receipt_ids),
            "scan_linear_sha256": hashlib.sha256(
                result.scan_linear.values.astype("<f8", copy=False).tobytes()
            ).hexdigest(),
            "absolute_scan_delta_p95": float(np.percentile(absolute, 95.0)),
            "absolute_scan_delta_p99": float(np.percentile(absolute, 99.0)),
            "maximum_absolute_scan_delta": float(np.max(absolute)),
            "flat_region_p99_absolute_scan_delta": _flat_region_p99(source, delta),
            "new_boundary_fraction": float(np.mean(new_boundary)),
            "isolated_excursion_count": isolated,
            "minimum_axis_lag1_residual_correlation": min(lag1),
            "maximum_absolute_mean_scan_delta": float(
                np.max(np.abs(np.mean(delta, axis=(0, 1), dtype=np.float64)))
            ),
            "lowpass_structure_correlation": _lowpass_structure_correlation(
                baseline,
                result.scan_linear.values,
                float(evaluation["lowpass_structure_sigma_pixels"]),
            ),
            "density_transmittance_roundtrip_error": float(
                np.max(np.abs(roundtrip.values - result.developed_density.values))
            ),
            "row_partition_exact": bool(
                np.array_equal(result.scan_linear.values, assembled)
            ),
            "finite_bounded": bool(
                np.all(np.isfinite(result.scan_linear.values))
                and np.all(result.scan_linear.values >= 0.0)
                and np.all(result.scan_linear.values <= 1.0)
            ),
        }
        rows.append(row)
        absolute_values.append(absolute.reshape(-1))
        visual_rows.append(
            {
                "id": source_row["id"],
                "source": source,
                "baseline": baseline,
                "candidate": result.scan_linear.values,
                "delta": delta,
            }
        )

    combined_absolute = np.concatenate(absolute_values)
    population_p95 = float(np.percentile(combined_absolute, 95.0))
    population_p99 = float(np.percentile(combined_absolute, 99.0))
    maximum = max(row["maximum_absolute_scan_delta"] for row in rows)
    sheet_bytes = _contact_sheet_bytes(
        visual_rows, float(candidate_spec["diagnostic_delta_display_gain"])
    )
    sheet_sha = hashlib.sha256(sheet_bytes).hexdigest()
    if contact_sheet_path is not None:
        contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
        contact_sheet_path.write_bytes(sheet_bytes)
    gates = {
        "parent_identity": True,
        "source_count": len(rows) == input_spec["expected_rows"],
        "input_hashes_exact": all(
            row["input_sha256"] == source["decoded_sha256"]
            for row, source in zip(rows, selected, strict=True)
        ),
        "unique_source_receipts": len(receipt_ids) == len(set(receipt_ids)),
        "visible_population_effect": population_p95
        >= evaluation["minimum_population_absolute_scan_delta_p95"],
        "bounded_population_p99": population_p99
        <= evaluation["maximum_population_absolute_scan_delta_p99"],
        "bounded_maximum": maximum <= evaluation["maximum_absolute_scan_delta"],
        "flat_region_bound": max(
            row["flat_region_p99_absolute_scan_delta"] for row in rows
        )
        <= evaluation["maximum_flat_region_p99_absolute_scan_delta"],
        "no_new_boundary": max(row["new_boundary_fraction"] for row in rows)
        <= evaluation["maximum_new_boundary_fraction"],
        "no_isolated_excursions": sum(row["isolated_excursion_count"] for row in rows)
        <= evaluation["maximum_isolated_excursion_count"],
        "spatially_correlated_residual": min(
            row["minimum_axis_lag1_residual_correlation"] for row in rows
        )
        >= evaluation["minimum_axis_lag1_residual_correlation"],
        "bounded_mean_drift": max(
            row["maximum_absolute_mean_scan_delta"] for row in rows
        )
        <= evaluation["maximum_absolute_mean_scan_delta"],
        "lowpass_structure": min(row["lowpass_structure_correlation"] for row in rows)
        >= evaluation["minimum_lowpass_structure_correlation"],
        "density_transmittance_roundtrip": max(
            row["density_transmittance_roundtrip_error"] for row in rows
        )
        <= evaluation["maximum_density_transmittance_roundtrip_error"],
        "row_partition_exact": all(row["row_partition_exact"] for row in rows),
        "finite_bounded": all(row["finite_bounded"] for row in rows),
        "adapter_is_fixed_and_uncalibrated": (
            adapter["per_image_normalization_allowed"] is False
            and adapter["histogram_or_content_fit_allowed"] is False
            and adapter["clipping_allowed"] is False
        ),
        "no_scanner_double_counting": (
            candidate_spec["scanner_stages"] == ["spectral"]
            and candidate_spec["scanner_convolution_embedded_in_spatial_profile"]
            is True
        ),
    }
    automatic_pass = all(gates.values())
    stable = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": hash_file(
            root / "configs/u6_p4bz_thomas_photographic_stress_v1.json", "sha256"
        ),
        "profile_id": profile.identity(),
        "prior_id": prior.identity(),
        "source_count": len(rows),
        "camera_make_count": len({row["make"] for row in rows}),
        "population_absolute_scan_delta_p95": population_p95,
        "population_absolute_scan_delta_p99": population_p99,
        "maximum_absolute_scan_delta": maximum,
        "maximum_flat_region_p99_absolute_scan_delta": max(
            row["flat_region_p99_absolute_scan_delta"] for row in rows
        ),
        "maximum_new_boundary_fraction": max(
            row["new_boundary_fraction"] for row in rows
        ),
        "isolated_excursion_count": sum(
            row["isolated_excursion_count"] for row in rows
        ),
        "minimum_axis_lag1_residual_correlation": min(
            row["minimum_axis_lag1_residual_correlation"] for row in rows
        ),
        "maximum_absolute_mean_scan_delta": max(
            row["maximum_absolute_mean_scan_delta"] for row in rows
        ),
        "minimum_lowpass_structure_correlation": min(
            row["lowpass_structure_correlation"] for row in rows
        ),
        "maximum_density_transmittance_roundtrip_error": max(
            row["density_transmittance_roundtrip_error"] for row in rows
        ),
        "contact_sheet_sha256": sheet_sha,
        "gate_results": gates,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_thomas_photographic_stress_candidate"
            if automatic_pass
            else "close_thomas_photographic_stress_without_rescue"
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
    "ThomasPhotographicStressError",
    "evaluate_thomas_photographic_stress",
    "write_report",
]
