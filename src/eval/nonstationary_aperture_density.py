"""U6.P4BJ nonstationary aperture-amplitude reference evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.aperture_cell_sampler import (
    counter_aperture_scaled_poisson_region,
    sample_nonstationary_aperture_density_region,
)
from src.film_physics.aperture_scaled_granularity import (
    ApertureScaledCompoundPoissonProfile,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)

SCHEMA = "neuro_film.u6_p4bj_nonstationary_aperture_density_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bj_nonstationary_aperture_density_report.v1"


class NonstationaryApertureDensityError(RuntimeError):
    """Raised when frozen P4BJ evidence or semantics drift."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise NonstationaryApertureDensityError("P4BJ paths must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = payload.get("candidate", {})
    evaluation = payload.get("evaluation", {})
    if (
        payload.get("schema") != SCHEMA
        or candidate.get("apertures_micrometres") != [7.25, 48.0, 384.0]
        or candidate.get("field_shape_cells") != [192, 193]
        or candidate.get("patterns")
        != ["horizontal_ramp", "vertical_step", "checker_16", "highlight_island"]
        or candidate.get("normalized_exposure_fraction_minimum") != 0.15
        or candidate.get("normalized_exposure_fraction_maximum") != 0.85
        or candidate.get("ramp_strata") != 8
        or candidate.get("seed_base") != 2608021700
        or candidate.get("seed_stride") != 104729
        or candidate.get("spatial_topology")
        != "independent_measurement_cells_generic_fallback"
        or candidate.get("realized_field_renormalization_allowed") is not False
        or candidate.get("correlation_radius_or_nps_claimed") is not False
        or candidate.get("photographic_render_allowed") is not False
        or evaluation.get("required_row_count") != 36
        or evaluation.get("maximum_global_density_mean_relative_error") != 0.005
        or evaluation.get("maximum_local_density_mean_relative_error") != 0.02
        or evaluation.get("maximum_local_density_rms_relative_error") != 0.08
        or evaluation.get("maximum_local_transmittance_mean_relative_error") != 0.01
        or evaluation.get("maximum_local_transmittance_rms_relative_error") != 0.08
        or evaluation.get("require_constant_field_exact_p4bh_counts") is not True
        or evaluation.get("require_repeat_exact") is not True
        or evaluation.get("require_odd_partition_exact") is not True
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise NonstationaryApertureDensityError("P4BJ frozen contract drift")
    return payload


def _load_parent(parents: Mapping[str, Any], root: Path, stem: str) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise NonstationaryApertureDensityError(
            f"P4BJ parent integrity mismatch: {stem}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _pattern_fraction_and_groups(
    pattern: str, shape: tuple[int, int], *, minimum: float, maximum: float
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    yy, xx = np.indices(shape)
    if pattern == "horizontal_ramp":
        fraction = minimum + (maximum - minimum) * xx / (width - 1)
        groups = np.minimum((xx * 8) // width, 7)
    elif pattern == "vertical_step":
        groups = (yy >= height // 2).astype(np.int64)
        fraction = np.where(groups == 0, 0.25, 0.75)
    elif pattern == "checker_16":
        groups = ((yy // 16 + xx // 16) & 1).astype(np.int64)
        fraction = np.where(groups == 0, 0.2, 0.8)
    elif pattern == "highlight_island":
        distance = np.square((yy + 0.5) / height - 0.5) + np.square(
            (xx + 0.5) / width - 0.5
        )
        groups = (distance <= 0.22**2).astype(np.int64)
        fraction = np.where(groups == 0, minimum, maximum)
    else:  # pragma: no cover - protected by the contract loader
        raise NonstationaryApertureDensityError("unsupported P4BJ pattern")
    return np.asarray(fraction, dtype=np.float64), groups


def _exposure_field(
    prior: ManufacturerCharacteristicPrior, fraction: np.ndarray
) -> np.ndarray:
    channels = []
    for curve in prior.curves:
        lower, upper = curve.domain
        channels.append(lower + fraction * (upper - lower))
    return np.stack(channels, axis=-1)


def _mixture_moments(
    mean: np.ndarray, rms: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    selected_mean = mean[mask]
    selected_rms = rms[mask]
    mixture_mean = float(np.mean(selected_mean, dtype=np.float64))
    second = float(
        np.mean(np.square(selected_rms) + np.square(selected_mean), dtype=np.float64)
    )
    return mixture_mean, float(np.sqrt(max(second - mixture_mean**2, 0.0)))


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual / expected - 1.0)


def _partition_sample(
    rate: np.ndarray, mark: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = rate.shape
    row_cuts = (0, min(73, height - 1), height)
    column_cuts = (0, min(61, width - 1), width)
    count_rows: list[np.ndarray] = []
    density_rows: list[np.ndarray] = []
    for y0, y1 in pairwise(row_cuts):
        count_blocks: list[np.ndarray] = []
        density_blocks: list[np.ndarray] = []
        for x0, x1 in pairwise(column_cuts):
            counts, density = sample_nonstationary_aperture_density_region(
                rate,
                mark,
                origin_yx=(y0, x0),
                shape=(y1 - y0, x1 - x0),
                seed=seed,
            )
            count_blocks.append(counts)
            density_blocks.append(density)
        count_rows.append(np.concatenate(count_blocks, axis=1))
        density_rows.append(np.concatenate(density_blocks, axis=1))
    return np.concatenate(count_rows), np.concatenate(density_rows)


def evaluate_nonstationary_aperture_density(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    parents = contract["parents"]
    prior_payload = _load_parent(parents, root, "p2q_bundle")
    p4bg_decision = _load_parent(parents, root, "p4bg_decision")
    p4bg_bundle = _load_parent(parents, root, "p4bg_bundle")
    p4bg_report = _load_parent(parents, root, "p4bg_report")
    p4bh_decision = _load_parent(parents, root, "p4bh_decision")
    p4bh_report = _load_parent(parents, root, "p4bh_report")
    p4bi_decision = _load_parent(parents, root, "p4bi_decision")
    p4bi_report = _load_parent(parents, root, "p4bi_report")
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    serialized = dict(p4bg_bundle)
    profile_id = str(serialized.pop("profile_id"))
    profile = ApertureScaledCompoundPoissonProfile.from_dict(serialized)
    if (
        prior.identity() != parents["p2q_prior_identity"]
        or profile.identity() != profile_id
        or p4bg_decision.get("automatic_pass") is not True
        or p4bg_report.get("stable_evidence_id") != parents["p4bg_stable_evidence_id"]
        or p4bh_decision.get("automatic_pass") is not True
        or p4bh_report.get("stable_evidence_id") != parents["p4bh_stable_evidence_id"]
        or p4bi_decision.get("automatic_pass") is not False
        or p4bi_report.get("stable_evidence_id") != parents["p4bi_stable_evidence_id"]
    ):
        raise NonstationaryApertureDensityError("P4BJ parent decision mismatch")

    candidate = contract["candidate"]
    shape = tuple(int(value) for value in candidate["field_shape_cells"])
    minimum = float(candidate["normalized_exposure_fraction_minimum"])
    maximum = float(candidate["normalized_exposure_fraction_maximum"])
    seed_base = int(candidate["seed_base"])
    seed_stride = int(candidate["seed_stride"])
    rows: list[dict[str, Any]] = []
    global_density_errors: list[float] = []
    local_density_mean_errors: list[float] = []
    local_density_rms_errors: list[float] = []
    local_transmittance_mean_errors: list[float] = []
    local_transmittance_rms_errors: list[float] = []
    ramp_violations = 0
    contrast_violations = 0
    repeat_exact: list[bool] = []
    partition_exact: list[bool] = []
    domain_valid: list[bool] = []
    row_index = 0
    for pattern in candidate["patterns"]:
        fraction, groups = _pattern_fraction_and_groups(
            str(pattern), shape, minimum=minimum, maximum=maximum
        )
        exposure = _exposure_field(prior, fraction)
        for aperture in (float(value) for value in candidate["apertures_micrometres"]):
            parameters, moments = profile.evaluate(prior, exposure, aperture)
            for channel in range(3):
                seed = seed_base + row_index * seed_stride
                rate = parameters.poisson_rate[..., channel]
                mark = parameters.density_mark[..., channel]
                counts, density = sample_nonstationary_aperture_density_region(
                    rate, mark, origin_yx=(0, 0), shape=shape, seed=seed
                )
                transmittance = np.power(10.0, -density)
                small_rate = rate[:17, :19]
                small_mark = mark[:17, :19]
                small_counts, small_density = (
                    sample_nonstationary_aperture_density_region(
                        small_rate,
                        small_mark,
                        origin_yx=(0, 0),
                        shape=small_rate.shape,
                        seed=seed,
                    )
                )
                repeat_counts, repeat_density = (
                    sample_nonstationary_aperture_density_region(
                        small_rate,
                        small_mark,
                        origin_yx=(0, 0),
                        shape=small_rate.shape,
                        seed=seed,
                    )
                )
                partition_counts, partition_density = _partition_sample(
                    small_rate, small_mark, seed
                )
                repeat_exact.append(
                    np.array_equal(small_counts, repeat_counts)
                    and np.array_equal(small_density, repeat_density)
                )
                partition_exact.append(
                    np.array_equal(small_counts, partition_counts)
                    and np.array_equal(small_density, partition_density)
                )
                domain_valid.append(
                    bool(
                        np.all(density > 0.0)
                        and np.all(np.isfinite(density))
                        and np.all(transmittance > 0.0)
                        and np.all(transmittance <= 1.0)
                    )
                )
                density_mean = moments.density_mean[..., channel]
                density_rms = moments.density_rms[..., channel]
                trans_mean = moments.transmittance_mean[..., channel]
                trans_rms = moments.transmittance_rms[..., channel]
                global_error = _relative_error(
                    float(np.mean(density, dtype=np.float64)),
                    float(np.mean(density_mean, dtype=np.float64)),
                )
                global_density_errors.append(global_error)
                local_rows: list[dict[str, float | int]] = []
                observed_group_means: list[float] = []
                expected_group_means: list[float] = []
                for group in range(int(np.max(groups)) + 1):
                    mask = groups == group
                    expected_density_mean, expected_density_rms = _mixture_moments(
                        density_mean, density_rms, mask
                    )
                    expected_trans_mean, expected_trans_rms = _mixture_moments(
                        trans_mean, trans_rms, mask
                    )
                    observed_density_mean = float(np.mean(density[mask]))
                    observed_density_rms = float(np.std(density[mask]))
                    observed_trans_mean = float(np.mean(transmittance[mask]))
                    observed_trans_rms = float(np.std(transmittance[mask]))
                    density_mean_error = _relative_error(
                        observed_density_mean, expected_density_mean
                    )
                    density_rms_error = _relative_error(
                        observed_density_rms, expected_density_rms
                    )
                    trans_mean_error = _relative_error(
                        observed_trans_mean, expected_trans_mean
                    )
                    trans_rms_error = _relative_error(
                        observed_trans_rms, expected_trans_rms
                    )
                    local_density_mean_errors.append(density_mean_error)
                    local_density_rms_errors.append(density_rms_error)
                    local_transmittance_mean_errors.append(trans_mean_error)
                    local_transmittance_rms_errors.append(trans_rms_error)
                    observed_group_means.append(observed_density_mean)
                    expected_group_means.append(expected_density_mean)
                    local_rows.append(
                        {
                            "group": group,
                            "cell_count": int(np.count_nonzero(mask)),
                            "density_mean_relative_error": density_mean_error,
                            "density_rms_relative_error": density_rms_error,
                            "transmittance_mean_relative_error": trans_mean_error,
                            "transmittance_rms_relative_error": trans_rms_error,
                        }
                    )
                if pattern == "horizontal_ramp":
                    ramp_violations += int(
                        np.count_nonzero(np.diff(observed_group_means) <= 0.0)
                    )
                else:
                    contrast_violations += int(
                        not (
                            expected_group_means[1] > expected_group_means[0]
                            and observed_group_means[1] > observed_group_means[0]
                        )
                    )
                rows.append(
                    {
                        "pattern": pattern,
                        "aperture_micrometres": aperture,
                        "channel": ("red", "green", "blue")[channel],
                        "seed": seed,
                        "count_sha256": hashlib.sha256(
                            np.asarray(counts, dtype="<u4").tobytes(order="C")
                        ).hexdigest(),
                        "density_sha256": hashlib.sha256(
                            np.asarray(density, dtype="<f8").tobytes(order="C")
                        ).hexdigest(),
                        "global_density_mean_relative_error": global_error,
                        "local_groups": local_rows,
                    }
                )
                row_index += 1

    p4bh_by_key = {
        (str(row["channel"]), float(row["domain_fraction"])): row
        for row in p4bh_report["rows"]
        if float(row["aperture_micrometres"]) == 48.0
    }
    reference_exact: list[bool] = []
    for probe_index, ((channel, fraction), parent_row) in enumerate(
        sorted(p4bh_by_key.items())
    ):
        parent = next(
            row
            for row in p4bg_report["rows"]
            if row["channel"] == channel
            and float(row["domain_fraction"]) == fraction
            and float(row["aperture_micrometres"]) == 48.0
        )
        constant_shape = tuple(int(value) for value in p4bh_report["field_shape_cells"])
        seed = 2608021100 + ("red", "green", "blue").index(channel) * 5 * 104729
        seed += [0.15, 0.3, 0.5, 0.7, 0.85].index(fraction) * 104729
        rate = np.full(constant_shape, float(parent["poisson_rate"]), dtype=np.float64)
        mark = np.full(constant_shape, float(parent["density_mark"]), dtype=np.float64)
        counts, _ = sample_nonstationary_aperture_density_region(
            rate, mark, origin_yx=(0, 0), shape=constant_shape, seed=seed
        )
        scalar_counts = counter_aperture_scaled_poisson_region(
            constant_shape,
            origin_yx=(0, 0),
            shape=constant_shape,
            rate=float(parent["poisson_rate"]),
            seed=seed,
        )
        reference_exact.append(
            np.array_equal(counts, scalar_counts)
            and hashlib.sha256(
                np.asarray(counts, dtype="<u4").tobytes(order="C")
            ).hexdigest()
            == parent_row["count_sha256"]
        )

    gates = contract["evaluation"]
    gate_results = {
        "parent_identity": True,
        "pattern_count": len(candidate["patterns"])
        == int(gates["required_pattern_count"]),
        "aperture_count": len(candidate["apertures_micrometres"])
        == int(gates["required_aperture_count"]),
        "channel_count": 3 == int(gates["required_channel_count"]),
        "row_count": len(rows) == int(gates["required_row_count"]),
        "global_density_mean": max(global_density_errors)
        <= float(gates["maximum_global_density_mean_relative_error"]),
        "local_density_mean": max(local_density_mean_errors)
        <= float(gates["maximum_local_density_mean_relative_error"]),
        "local_density_rms": max(local_density_rms_errors)
        <= float(gates["maximum_local_density_rms_relative_error"]),
        "local_transmittance_mean": max(local_transmittance_mean_errors)
        <= float(gates["maximum_local_transmittance_mean_relative_error"]),
        "local_transmittance_rms": max(local_transmittance_rms_errors)
        <= float(gates["maximum_local_transmittance_rms_relative_error"]),
        "ramp_monotonicity": ramp_violations
        <= int(gates["maximum_ramp_monotonicity_violations"]),
        "contrast_sign": contrast_violations
        <= int(gates["maximum_step_or_checker_contrast_sign_violations"]),
        "constant_field_exact_p4bh_counts": all(reference_exact),
        "repeat_exact": all(repeat_exact),
        "odd_partition_exact": all(partition_exact),
        "positive_density": all(domain_valid),
        "unit_interval_transmittance": all(domain_valid),
        "generic_topology_label": candidate["spatial_topology"]
        == "independent_measurement_cells_generic_fallback",
        "no_realized_renormalization": candidate[
            "realized_field_renormalization_allowed"
        ]
        is False,
        "no_correlation_or_nps_claim": candidate["correlation_radius_or_nps_claimed"]
        is False,
        "no_photographic_render": candidate["photographic_render_allowed"] is False,
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "experiment_id": contract["experiment_id"],
        "profile_id": profile_id,
        "row_count": len(rows),
        "maximum_global_density_mean_relative_error": max(global_density_errors),
        "maximum_local_density_mean_relative_error": max(local_density_mean_errors),
        "maximum_local_density_rms_relative_error": max(local_density_rms_errors),
        "maximum_local_transmittance_mean_relative_error": max(
            local_transmittance_mean_errors
        ),
        "maximum_local_transmittance_rms_relative_error": max(
            local_transmittance_rms_errors
        ),
        "ramp_monotonicity_violation_count": ramp_violations,
        "contrast_sign_violation_count": contrast_violations,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "rows": rows,
        "decision": (
            "retain_nonstationary_aperture_amplitude_reference"
            if automatic_pass
            else "retain_flat_field_aperture_amplitude_only"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }


def write_report(report: Mapping[str, Any], path: Path) -> str:
    encoded = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "NonstationaryApertureDensityError",
    "evaluate_nonstationary_aperture_density",
    "load_contract",
    "write_report",
]
