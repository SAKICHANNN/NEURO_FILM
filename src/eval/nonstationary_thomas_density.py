"""U6.P4BX structured-exposure density/transmittance evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.signal import fftconvolve

from src.eval.real_uniform_grain_source import hash_file
from src.film_physics.density_conditioned_thomas import (
    DensityConditionedThomasProfile,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)
from src.film_physics.thomas_dc_projection import build_thomas_dc_receipt

SCHEMA = "neuro_film.u6_p4bx_nonstationary_thomas_density_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bx_nonstationary_thomas_density_report.v1"
CHANNELS = ("red", "green", "blue")


class NonstationaryThomasDensityError(RuntimeError):
    """Raised when P4BX frozen evidence or semantics drift."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(binding["path"])
    if not path.is_file() or hash_file(path, "sha256") != binding["sha256"]:
        raise NonstationaryThomasDensityError(f"P4BX parent mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise NonstationaryThomasDensityError("P4BX parent must be object")
    return payload


def _validate_contract(contract: Mapping[str, Any]) -> None:
    candidate = contract.get("candidate", {})
    evaluation = contract.get("evaluation", {})
    if (
        contract.get("schema") != SCHEMA
        or candidate.get("field_shape") != [512, 513]
        or candidate.get("patterns")
        != ["horizontal_ramp", "vertical_step", "checker_64", "highlight_island"]
        or candidate.get("layer_realization_seeds")
        != [2608024101, 11400714821016565296, 15111065704576689958]
        or candidate.get("measurement_crop_border_samples") != 13
        or candidate.get("realized_variance_normalization_allowed") is not False
        or candidate.get("clipping_allowed") is not False
        or candidate.get("photographic_render_allowed") is not False
        or evaluation.get("required_local_group_count") != 42
        or evaluation.get(
            "maximum_local_aperture_sigma_median_absolute_relative_error"
        )
        != 0.025
        or evaluation.get("maximum_local_aperture_sigma_p95_absolute_relative_error")
        != 0.08
        or evaluation.get("maximum_local_aperture_sigma_worst_absolute_relative_error")
        != 0.12
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise NonstationaryThomasDensityError("P4BX frozen contract drift")


def _pattern(
    name: str,
    shape: tuple[int, int],
    minimum: float,
    maximum: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    yy, xx = np.indices(shape)
    if name == "horizontal_ramp":
        fraction = minimum + (maximum - minimum) * xx / (width - 1)
        groups = np.minimum(xx * 8 // width, 7)
    elif name == "vertical_step":
        groups = (yy >= height // 2).astype(np.int64)
        fraction = np.where(groups == 0, 0.25, 0.75)
    elif name == "checker_64":
        groups = ((yy // 64 + xx // 64) & 1).astype(np.int64)
        fraction = np.where(groups == 0, 0.2, 0.8)
    elif name == "highlight_island":
        distance = np.square((yy + 0.5) / height - 0.5) + np.square(
            (xx + 0.5) / width - 0.5
        )
        groups = (distance <= 0.22**2).astype(np.int64)
        fraction = np.where(groups == 0, minimum, maximum)
    else:  # pragma: no cover - contract protected
        raise NonstationaryThomasDensityError("unsupported P4BX pattern")
    return np.asarray(fraction, dtype=np.float64), groups


def _exposure_for_curve(curve: Any, fraction: np.ndarray) -> np.ndarray:
    lower, upper = curve.domain
    return lower + fraction * (upper - lower)


def evaluate_nonstationary_thomas_density(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    _validate_contract(contract)
    parents = contract["parents"]
    decision = _load_bound(root, parents["p4bw_decision"])
    bundle = _load_bound(root, parents["p4bw_bundle"])
    prior_payload = _load_bound(root, parents["p2q_bundle"])
    if (
        decision.get("decision")
        != parents["p4bw_decision"]["required_decision"]
        or bundle.get("profile_id")
        != parents["p4bw_bundle"]["required_profile_id"]
    ):
        raise NonstationaryThomasDensityError("P4BX parent decision drift")
    profile = DensityConditionedThomasProfile.from_dict(bundle)
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    if prior.identity() != parents["p2q_bundle"]["required_prior_id"]:
        raise NonstationaryThomasDensityError("P4BX prior identity drift")

    candidate = contract["candidate"]
    evaluation = contract["evaluation"]
    shape = tuple(int(value) for value in candidate["field_shape"])
    receipts = [
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
    ]
    aperture = profile.aperture_kernel()
    aperture_energy = profile.measurement_energy()
    border = int(candidate["measurement_crop_border_samples"])
    erosion = np.ones((2 * border + 1, 2 * border + 1), dtype=bool)
    rows: list[dict[str, Any]] = []
    local_errors: list[float] = []
    density_biases: list[float] = []
    transmittance_mean_errors: list[float] = []
    cross_correlations: list[float] = []
    minimum_density = math.inf
    maximum_transmittance = 0.0
    exact_repeat: list[bool] = []
    exact_partition: list[bool] = []
    finite: list[bool] = []
    for pattern_name in candidate["patterns"]:
        fraction, groups = _pattern(
            str(pattern_name),
            shape,
            float(candidate["normalized_exposure_fraction_minimum"]),
            float(candidate["normalized_exposure_fraction_maximum"]),
        )
        pattern_residuals: list[np.ndarray] = []
        for index, channel in enumerate(CHANNELS):
            curve = prior.curves[index]
            exposure = _exposure_for_curve(curve, fraction)
            density_mean = curve.apply(exposure)
            sigma_d = profile.amplitude_profile.evaluate_channel(
                prior, channel, exposure
            )
            density = profile.render_nonstationary_developed_density_region(
                receipts[index],
                prior,
                channel=channel,
                full_relative_log_exposure=exposure,
                origin_yx=(0, 0),
                shape=shape,
            )
            repeated = profile.render_nonstationary_developed_density_region(
                receipts[index],
                prior,
                channel=channel,
                full_relative_log_exposure=exposure,
                origin_yx=(0, 0),
                shape=shape,
            )
            assembled = np.empty_like(density)
            row_height = int(candidate["row_partition_height"])
            for y0 in range(0, shape[0], row_height):
                height = min(row_height, shape[0] - y0)
                assembled[y0 : y0 + height] = (
                    profile.render_nonstationary_developed_density_region(
                        receipts[index],
                        prior,
                        channel=channel,
                        full_relative_log_exposure=exposure,
                        origin_yx=(y0, 0),
                        shape=(height, shape[1]),
                    )
                )
            transmittance = np.power(10.0, -density)
            residual = density - density_mean
            pattern_residuals.append(residual)
            aperture_residual = fftconvolve(residual, aperture, mode="same")
            local_rows: list[dict[str, Any]] = []
            for group in range(int(np.max(groups)) + 1):
                mask = binary_erosion(groups == group, structure=erosion, border_value=0)
                if np.count_nonzero(mask) == 0:
                    raise NonstationaryThomasDensityError("empty P4BX local group")
                actual = float(np.std(aperture_residual[mask], dtype=np.float64))
                expected = float(
                    np.sqrt(np.mean(np.square(sigma_d[mask]), dtype=np.float64))
                )
                error = abs(actual / expected - 1.0)
                local_errors.append(error)
                local_rows.append(
                    {
                        "group": group,
                        "sample_count": int(np.count_nonzero(mask)),
                        "expected_aperture_sigma_d": expected,
                        "observed_aperture_sigma_d": actual,
                        "absolute_relative_error": error,
                    }
                )
            density_bias = abs(float(np.mean(residual, dtype=np.float64)))
            point_sigma = sigma_d / math.sqrt(aperture_energy)
            expected_transmittance_mean = np.exp(
                -math.log(10.0) * density_mean
                + 0.5 * np.square(math.log(10.0) * point_sigma)
            )
            transmittance_error = abs(
                float(np.mean(transmittance, dtype=np.float64))
                / float(np.mean(expected_transmittance_mean, dtype=np.float64))
                - 1.0
            )
            density_biases.append(density_bias)
            transmittance_mean_errors.append(transmittance_error)
            minimum_density = min(minimum_density, float(np.min(density)))
            maximum_transmittance = max(
                maximum_transmittance, float(np.max(transmittance))
            )
            exact_repeat.append(bool(np.array_equal(density, repeated)))
            exact_partition.append(bool(np.array_equal(density, assembled)))
            finite.append(
                bool(
                    np.all(np.isfinite(density))
                    and np.all(np.isfinite(transmittance))
                )
            )
            rows.append(
                {
                    "pattern": pattern_name,
                    "channel": channel,
                    "realization_seed": int(candidate["layer_realization_seeds"][index]),
                    "density_sha256": hashlib.sha256(
                        np.ascontiguousarray(density, dtype="<f8").tobytes()
                    ).hexdigest(),
                    "transmittance_sha256": hashlib.sha256(
                        np.ascontiguousarray(transmittance, dtype="<f8").tobytes()
                    ).hexdigest(),
                    "global_density_bias_absolute": density_bias,
                    "transmittance_mean_relative_error": transmittance_error,
                    "minimum_developed_density": float(np.min(density)),
                    "maximum_transmittance": float(np.max(transmittance)),
                    "repeat_exact": exact_repeat[-1],
                    "row_partition_exact": exact_partition[-1],
                    "local_groups": local_rows,
                }
            )
        correlation = np.corrcoef(
            [residual.reshape(-1) for residual in pattern_residuals]
        )
        cross_correlations.extend(
            abs(float(correlation[left, right]))
            for left in range(3)
            for right in range(left + 1, 3)
        )
    gates = {
        "parent_identity": True,
        "pattern_count": len(candidate["patterns"])
        == evaluation["required_pattern_count"],
        "channel_count": len(CHANNELS) == evaluation["required_channel_count"],
        "local_group_count": len(local_errors)
        == evaluation["required_local_group_count"],
        "local_aperture_sigma_median": float(np.median(local_errors))
        <= evaluation[
            "maximum_local_aperture_sigma_median_absolute_relative_error"
        ],
        "local_aperture_sigma_p95": float(np.percentile(local_errors, 95.0))
        <= evaluation[
            "maximum_local_aperture_sigma_p95_absolute_relative_error"
        ],
        "local_aperture_sigma_worst": max(local_errors)
        <= evaluation[
            "maximum_local_aperture_sigma_worst_absolute_relative_error"
        ],
        "global_density_bias": max(density_biases)
        <= evaluation["maximum_global_density_bias_absolute"],
        "transmittance_mean": max(transmittance_mean_errors)
        <= evaluation["maximum_transmittance_mean_relative_error"],
        "cross_layer_independence": max(cross_correlations)
        <= evaluation["maximum_cross_layer_residual_correlation"],
        "positive_developed_density": minimum_density
        > evaluation["minimum_developed_density"],
        "unit_interval_transmittance": 0.0 < maximum_transmittance
        <= evaluation["maximum_transmittance"],
        "repeat_exact": all(exact_repeat),
        "row_partition_exact": all(exact_partition),
        "finite": all(finite),
        "no_renormalization_or_clipping": (
            candidate["realized_variance_normalization_allowed"] is False
            and candidate["clipping_allowed"] is False
        ),
    }
    automatic_pass = all(gates.values())
    stable = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": hash_file(
            root / "configs/u6_p4bx_nonstationary_thomas_density_v1.json",
            "sha256",
        ),
        "profile_id": profile.identity(),
        "row_count": len(rows),
        "local_group_count": len(local_errors),
        "local_aperture_sigma_median_absolute_relative_error": float(
            np.median(local_errors)
        ),
        "local_aperture_sigma_p95_absolute_relative_error": float(
            np.percentile(local_errors, 95.0)
        ),
        "local_aperture_sigma_worst_absolute_relative_error": max(local_errors),
        "maximum_global_density_bias_absolute": max(density_biases),
        "maximum_transmittance_mean_relative_error": max(
            transmittance_mean_errors
        ),
        "maximum_cross_layer_residual_correlation": max(cross_correlations),
        "minimum_developed_density": minimum_density,
        "maximum_transmittance": maximum_transmittance,
        "gate_results": gates,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_nonstationary_thomas_density_transmittance_execution"
            if automatic_pass
            else "close_nonstationary_thomas_density_transmittance_execution"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {**stable, "stable_evidence_id": stable_id, "rows": rows}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    encoded = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "NonstationaryThomasDensityError",
    "_validate_contract",
    "evaluate_nonstationary_thomas_density",
    "write_report",
]
