"""U6.P4BY typed synthetic Thomas image-formation evaluation."""

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

SCHEMA = "neuro_film.u6_p4by_typed_thomas_image_formation_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4by_typed_thomas_image_formation_report.v1"


class TypedThomasImageFormationError(RuntimeError):
    """Raised when P4BY frozen evidence or semantics drift."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(binding["path"])
    if not path.is_file() or hash_file(path, "sha256") != binding["sha256"]:
        raise TypedThomasImageFormationError(f"P4BY parent mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypedThomasImageFormationError("P4BY parent must be an object")
    return payload


def _validate_contract(contract: Mapping[str, Any]) -> None:
    candidate = contract.get("candidate", {})
    evaluation = contract.get("evaluation", {})
    if (
        contract.get("schema") != SCHEMA
        or candidate.get("field_shape") != [512, 768]
        or candidate.get("fixtures")
        != [
            "smooth_gradient",
            "high_contrast_edges",
            "chromatic_patches",
            "text_like_bars",
            "highlight_islands",
            "flat_field",
        ]
        or candidate.get("typed_domain_order")
        != [
            "film-layer-linear-exposure",
            "developed-optical-density",
            "film-transmittance",
            "scanner-linear-signal",
        ]
        or candidate.get("scanner_stages") != ["spectral"]
        or candidate.get("scanner_convolution_embedded_in_spatial_profile") is not True
        or candidate.get("additional_scanner_mtf_flare_or_noise_allowed") is not False
        or candidate.get("realized_variance_normalization_allowed") is not False
        or candidate.get("density_or_signal_clipping_allowed") is not False
        or evaluation.get("minimum_absolute_scan_delta_p99") != 0.003
        or evaluation.get("maximum_absolute_scan_delta_p99") != 0.03
        or evaluation.get("maximum_absolute_scan_delta") != 0.08
        or evaluation.get("maximum_new_boundary_fraction") != 0.0
        or evaluation.get("maximum_isolated_excursion_count") != 0
        or evaluation.get("minimum_axis_lag1_residual_correlation") != 0.75
        or evaluation.get("require_two_byte_identical_reports") is not True
        or evaluation.get("require_visual_severe_artifact_acceptance") is not True
    ):
        raise TypedThomasImageFormationError("P4BY frozen contract drift")


def _fixture_fractions(name: str, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float64),
        np.linspace(0.0, 1.0, width, dtype=np.float64),
        indexing="ij",
    )
    if name == "smooth_gradient":
        fractions = np.stack((xx, yy, 0.5 * (xx + yy)), axis=-1)
    elif name == "high_contrast_edges":
        vertical = np.where(xx < 0.5, 0.08, 0.92)
        horizontal = np.where(yy < 0.5, 0.9, 0.1)
        diagonal = np.where(xx + yy < 1.0, 0.12, 0.88)
        fractions = np.stack((vertical, horizontal, diagonal), axis=-1)
    elif name == "chromatic_patches":
        row = np.floor(yy * 6.0).astype(np.int64)
        column = np.floor(xx * 8.0).astype(np.int64)
        code = row * 8 + column
        fractions = np.stack(
            (
                ((code * 17 + 3) % 47) / 46.0,
                ((code * 29 + 11) % 47) / 46.0,
                ((code * 37 + 19) % 47) / 46.0,
            ),
            axis=-1,
        )
    elif name == "text_like_bars":
        fine = ((np.floor(xx * 96) + np.floor(yy * 24)) % 2) == 0
        stems = ((np.floor(xx * 24) % 4) == 0) & ((np.floor(yy * 16) % 3) != 0)
        fractions = np.stack(
            (
                np.where(fine, 0.28, 0.72),
                np.where(stems, 0.8, 0.24),
                np.where(fine ^ stems, 0.68, 0.32),
            ),
            axis=-1,
        )
    elif name == "highlight_islands":
        base = np.full(shape, 0.18, dtype=np.float64)
        islands = np.zeros(shape, dtype=np.float64)
        for cy, cx, radius in (
            (0.27, 0.25, 0.045),
            (0.62, 0.53, 0.08),
            (0.42, 0.82, 0.025),
        ):
            islands = np.maximum(
                islands,
                np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / radius**2),
            )
        fractions = np.stack(
            (base + 0.8 * islands, base + 0.62 * islands, base + 0.42 * islands),
            axis=-1,
        )
    elif name == "flat_field":
        fractions = np.full((*shape, 3), 0.5, dtype=np.float64)
    else:
        raise TypedThomasImageFormationError("unsupported P4BY fixture")
    return np.clip(fractions, 0.0, 1.0)


def _layer_exposure(
    name: str,
    shape: tuple[int, int],
    prior: ManufacturerCharacteristicPrior,
    minimum_fraction: float,
    maximum_fraction: float,
    pixel_pitch_um: float,
) -> tuple[PhysicalDomainArray, np.ndarray]:
    fractions = minimum_fraction + (
        maximum_fraction - minimum_fraction
    ) * _fixture_fractions(name, shape)
    relative_log = np.empty_like(fractions)
    for index, curve in enumerate(prior.curves):
        lower, upper = curve.domain
        relative_log[..., index] = lower + fractions[..., index] * (upper - lower)
    values = np.power(10.0, relative_log)
    return (
        PhysicalDomainArray(
            values,
            PhysicalDomain.LAYER_EXPOSURE,
            PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
            CHANNELS,
            PhysicalScale(pixel_pitch_um),
        ),
        relative_log,
    )


def _mean_density(
    prior: ManufacturerCharacteristicPrior, relative_log: np.ndarray
) -> np.ndarray:
    result = np.empty_like(relative_log)
    for index, curve in enumerate(prior.curves):
        result[..., index] = curve.apply(relative_log[..., index])
    return result


def _lag1_correlations(delta: np.ndarray) -> list[float]:
    correlations: list[float] = []
    for channel in range(3):
        values = delta[..., channel]
        for left, right in ((values[:, :-1], values[:, 1:]), (values[:-1], values[1:])):
            if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
                correlations.append(1.0)
            else:
                correlations.append(
                    float(np.corrcoef(left.ravel(), right.ravel())[0, 1])
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


def _srgb_code(linear: np.ndarray) -> np.ndarray:
    values = np.clip(linear, 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )


def _contact_sheet_bytes(
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]], gain: float
) -> bytes:
    rendered_rows: list[np.ndarray] = []
    for baseline, candidate, delta in rows:
        panels = (
            _srgb_code(baseline),
            _srgb_code(candidate),
            np.clip(0.5 + gain * delta, 0.0, 1.0),
        )
        row = np.concatenate(
            [
                np.mean(
                    panel.reshape(panel.shape[0] // 2, 2, panel.shape[1] // 2, 2, 3),
                    axis=(1, 3),
                )
                for panel in panels
            ],
            axis=1,
        )
        rendered_rows.append(row)
    sheet = np.concatenate(rendered_rows, axis=0)
    image = Image.fromarray(np.rint(sheet * 255.0).astype(np.uint8), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def evaluate_typed_thomas_image_formation(
    contract: Mapping[str, Any], root: Path, *, diagnostic_path: Path | None = None
) -> dict[str, Any]:
    _validate_contract(contract)
    parents = contract["parents"]
    p4bx = _load_bound(root, parents["p4bx_decision"])
    profile_payload = _load_bound(root, parents["p4bw_bundle"])
    prior_payload = _load_bound(root, parents["p2q_bundle"])
    if (
        p4bx.get("decision") != parents["p4bx_decision"]["required_decision"]
        or profile_payload.get("profile_id")
        != parents["p4bw_bundle"]["required_profile_id"]
    ):
        raise TypedThomasImageFormationError("P4BY parent decision drift")
    profile = DensityConditionedThomasProfile.from_dict(profile_payload)
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    if prior.identity() != parents["p2q_bundle"]["required_prior_id"]:
        raise TypedThomasImageFormationError("P4BY prior identity drift")

    candidate = contract["candidate"]
    evaluation = contract["evaluation"]
    shape = tuple(int(value) for value in candidate["field_shape"])
    receipts = tuple(
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
    context = ThomasImageFormationContext(
        profile_id=profile.identity(),
        prior_id=prior.identity(),
        scanner_profile_id=str(candidate["scanner_profile_id"]),
        full_shape=shape,
        receipt_ids=tuple(receipt.receipt_id for receipt in receipts),
        layer_realization_seeds=tuple(
            int(value) for value in candidate["layer_realization_seeds"]
        ),
    )

    rows: list[dict[str, Any]] = []
    diagnostics: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    all_absolute: list[np.ndarray] = []
    all_new_boundary = 0
    all_samples = 0
    isolated_total = 0
    lag1: list[float] = []
    means: list[float] = []
    roundtrip_errors: list[float] = []
    identity_scanner_exact: list[bool] = []
    exact_repeats: list[bool] = []
    exact_partitions: list[bool] = []
    immutable: list[bool] = []

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
        result = render_typed_thomas_image_formation(
            exposure, profile, prior, receipts, context
        )
        repeated = render_typed_thomas_image_formation(
            exposure, profile, prior, receipts, context
        )
        assembled = np.empty_like(result.scan_linear.values)
        row_height = int(candidate["row_partition_height"])
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
        baseline_density = _mean_density(prior, relative_log)
        baseline_scan = np.power(10.0, -baseline_density)
        delta = result.scan_linear.values - baseline_scan
        absolute = np.abs(delta)
        epsilon = float(evaluation["boundary_epsilon"])
        new_boundary = (
            (result.scan_linear.values <= epsilon)
            | (result.scan_linear.values >= 1.0 - epsilon)
        ) & ~((baseline_scan <= epsilon) | (baseline_scan >= 1.0 - epsilon))
        roundtrip = transmittance_to_density(result.transmittance)
        roundtrip_error = float(
            np.max(np.abs(roundtrip.values - result.developed_density.values))
        )
        row_isolated = _isolated_excursions(
            delta,
            float(evaluation["isolated_excursion_threshold"]),
            float(evaluation["isolated_neighbour_ratio"]),
        )
        row_lag1 = _lag1_correlations(delta)
        row_mean = np.mean(delta, axis=(0, 1), dtype=np.float64)
        repeat_exact = bool(
            np.array_equal(
                result.developed_density.values, repeated.developed_density.values
            )
            and np.array_equal(result.scan_linear.values, repeated.scan_linear.values)
        )
        partition_exact = bool(np.array_equal(result.scan_linear.values, assembled))
        input_immutable = (
            hashlib.sha256(exposure.values.tobytes()).hexdigest() == input_sha
        )
        diagnostics.append((baseline_scan, result.scan_linear.values, delta))
        all_absolute.append(absolute.reshape(-1))
        all_new_boundary += int(np.count_nonzero(new_boundary))
        all_samples += int(new_boundary.size)
        isolated_total += row_isolated
        lag1.extend(row_lag1)
        means.extend(abs(float(value)) for value in row_mean)
        roundtrip_errors.append(roundtrip_error)
        identity_scanner_exact.append(
            bool(np.array_equal(result.scan_linear.values, result.transmittance.values))
        )
        exact_repeats.append(repeat_exact)
        exact_partitions.append(partition_exact)
        immutable.append(input_immutable)
        rows.append(
            {
                "fixture": fixture,
                "layer_exposure_sha256": input_sha,
                "developed_density_sha256": hashlib.sha256(
                    result.developed_density.values.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
                "transmittance_sha256": hashlib.sha256(
                    result.transmittance.values.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
                "scan_linear_sha256": hashlib.sha256(
                    result.scan_linear.values.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
                "absolute_scan_delta_p99": float(np.percentile(absolute, 99.0)),
                "maximum_absolute_scan_delta": float(np.max(absolute)),
                "new_boundary_count": int(np.count_nonzero(new_boundary)),
                "isolated_excursion_count": row_isolated,
                "minimum_axis_lag1_residual_correlation": min(row_lag1),
                "maximum_absolute_mean_scan_delta": max(
                    abs(float(value)) for value in row_mean
                ),
                "density_transmittance_roundtrip_error": roundtrip_error,
                "repeat_exact": repeat_exact,
                "row_partition_exact": partition_exact,
                "input_immutable": input_immutable,
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
        render_typed_thomas_image_formation(wrong, profile, prior, receipts, context)
    except ValueError:
        wrong_domain_rejected = True

    combined_absolute = np.concatenate(all_absolute)
    p99 = float(np.percentile(combined_absolute, 99.0))
    maximum = float(np.max(combined_absolute))
    boundary_fraction = all_new_boundary / all_samples
    sheet_bytes = _contact_sheet_bytes(
        diagnostics, float(candidate["diagnostic_delta_display_gain"])
    )
    sheet_sha = hashlib.sha256(sheet_bytes).hexdigest()
    if diagnostic_path is not None:
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_bytes(sheet_bytes)
    gates = {
        "parent_identity": True,
        "fixture_count": len(rows) == evaluation["required_fixture_count"],
        "channel_count": len(CHANNELS) == evaluation["required_channel_count"],
        "visible_stochastic_effect": p99
        >= evaluation["minimum_absolute_scan_delta_p99"],
        "bounded_p99_effect": p99 <= evaluation["maximum_absolute_scan_delta_p99"],
        "bounded_maximum_effect": maximum <= evaluation["maximum_absolute_scan_delta"],
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
        "repeat_exact": all(exact_repeats),
        "row_partition_exact": all(exact_partitions),
        "input_immutable": all(immutable),
        "wrong_domain_rejected": wrong_domain_rejected,
        "finite_bounded": all(
            np.all(np.isfinite(candidate_scan))
            and np.all(candidate_scan >= 0.0)
            and np.all(candidate_scan <= 1.0)
            for _, candidate_scan, _ in diagnostics
        ),
        "no_scanner_double_counting": (
            candidate["scanner_stages"] == ["spectral"]
            and candidate["scanner_convolution_embedded_in_spatial_profile"] is True
            and candidate["additional_scanner_mtf_flare_or_noise_allowed"] is False
        ),
        "no_normalization_or_clipping": (
            candidate["realized_variance_normalization_allowed"] is False
            and candidate["density_or_signal_clipping_allowed"] is False
        ),
    }
    automatic_pass = all(gates.values())
    stable = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": hash_file(
            root / "configs/u6_p4by_typed_thomas_image_formation_v1.json", "sha256"
        ),
        "profile_id": profile.identity(),
        "prior_id": prior.identity(),
        "context_id": context.context_id,
        "scanner_profile_id": context.scanner_profile_id,
        "scanner_stages": list(context.scanner_stages),
        "scanner_convolution_embedded": context.scanner_convolution_embedded,
        "fixture_count": len(rows),
        "absolute_scan_delta_p99": p99,
        "maximum_absolute_scan_delta": maximum,
        "new_boundary_fraction": boundary_fraction,
        "isolated_excursion_count": isolated_total,
        "minimum_axis_lag1_residual_correlation": min(lag1),
        "maximum_absolute_mean_scan_delta": max(means),
        "maximum_density_transmittance_roundtrip_error": max(roundtrip_errors),
        "diagnostic_contact_sheet_sha256": sheet_sha,
        "gate_results": gates,
        "automatic_pass": automatic_pass,
        "decision": "retain_typed_thomas_image_formation"
        if automatic_pass
        else "close_typed_thomas_image_formation_without_rescue",
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
    "TypedThomasImageFormationError",
    "evaluate_typed_thomas_image_formation",
    "write_report",
]
