"""U6.P4BW aperture-calibrated density-amplitude composition evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from src.eval.real_uniform_grain_source import hash_file
from src.film_physics.density_conditioned_thomas import (
    DensityConditionedThomasProfile,
)
from src.film_physics.granularity_amplitude import GranularityAmplitudeProfile
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)
from src.film_physics.thomas_dc_projection import (
    build_thomas_dc_receipt,
    render_dc_projected_thomas_region,
)

SCHEMA = "neuro_film.u6_p4bw_density_conditioned_thomas_profile_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bw_density_conditioned_thomas_profile_report.v1"


class DensityConditionedThomasEvaluationError(RuntimeError):
    """Raised when the frozen P4BW inputs or semantics drift."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(binding["path"])
    if not path.is_file() or hash_file(path, "sha256") != binding["sha256"]:
        raise DensityConditionedThomasEvaluationError(
            f"P4BW parent integrity mismatch: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DensityConditionedThomasEvaluationError("P4BW parent must be object")
    return payload


def _validate_contract(contract: Mapping[str, Any]) -> None:
    compiler = contract.get("compiler", {})
    evaluation = contract.get("evaluation", {})
    if (
        contract.get("schema") != SCHEMA
        or compiler.get("source_scanner_resolution_dpi") != 4000.0
        or compiler.get("sample_pitch_micrometres") != 6.35
        or compiler.get("measurement_aperture_diameter_micrometres") != 48.0
        or compiler.get("particle_sigma_samples") != 1.3253699417769098
        or compiler.get("cluster_sigma_samples") != 1.1268619873805532
        or compiler.get("mean_offspring") != 27.765942352935905
        or compiler.get("parameter_refit_allowed") is not False
        or compiler.get("realized_variance_normalization_allowed") is not False
        or evaluation.get("field_shape") != [1024, 1024]
        or evaluation.get("profile_probe_domain_fractions") != [0.2, 0.5, 0.8]
        or evaluation.get("maximum_monte_carlo_median_absolute_relative_error")
        != 0.015
        or evaluation.get("maximum_monte_carlo_p95_absolute_relative_error")
        != 0.05
        or evaluation.get("require_two_byte_identical_bundles_and_reports")
        is not True
    ):
        raise DensityConditionedThomasEvaluationError("P4BW frozen contract drift")


def compile_and_evaluate(
    contract: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_contract(contract)
    parents = contract["parents"]
    p4ax_decision = _load_bound(root, parents["p4ax_decision"])
    p4ax_bundle = _load_bound(root, parents["p4ax_bundle"])
    prior_payload = _load_bound(root, parents["p2q_bundle"])
    p4bs_decision = _load_bound(root, parents["p4bs_decision"])
    p4bv_decision = _load_bound(root, parents["p4bv_decision"])
    if (
        p4ax_decision.get("decision")
        != parents["p4ax_decision"]["required_decision"]
        or p4ax_bundle.get("profile_id")
        != parents["p4ax_bundle"]["required_profile_id"]
        or p4bs_decision.get("decision")
        != parents["p4bs_decision"]["required_decision"]
        or p4bv_decision.get("decision")
        != parents["p4bv_decision"]["required_decision"]
    ):
        raise DensityConditionedThomasEvaluationError("P4BW parent decision drift")
    amplitude = GranularityAmplitudeProfile.from_dict(p4ax_bundle)
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    if prior.identity() != parents["p2q_bundle"]["required_prior_id"]:
        raise DensityConditionedThomasEvaluationError("P4BW prior identity drift")
    compiler = contract["compiler"]
    profile = DensityConditionedThomasProfile(
        amplitude_profile=amplitude,
        spatial_profile_id=str(compiler["spatial_profile_id"]),
        particle_sigma_samples=float(compiler["particle_sigma_samples"]),
        cluster_sigma_samples=float(compiler["cluster_sigma_samples"]),
        mean_offspring=float(compiler["mean_offspring"]),
        component_seeds=tuple(int(value) for value in compiler["component_seeds"]),
        sample_pitch_micrometres=float(compiler["sample_pitch_micrometres"]),
        measurement_aperture_diameter_micrometres=float(
            compiler["measurement_aperture_diameter_micrometres"]
        ),
        truncate=float(compiler["truncate_sigma"]),
    )
    bundle = {**profile.to_dict(), "profile_id": profile.identity()}
    replay = DensityConditionedThomasProfile.from_dict(bundle)
    profile_roundtrip_exact = replay.to_dict() == profile.to_dict()

    probes: list[dict[str, Any]] = []
    analytic_errors: list[float] = []
    for curve in prior.curves:
        lower, upper = curve.domain
        for fraction in contract["evaluation"]["profile_probe_domain_fractions"]:
            exposure = lower + float(fraction) * (upper - lower)
            target = profile.target_sigma_d(prior, curve.layer, exposure)
            analytic = profile.analytic_aperture_sigma(
                prior, curve.layer, exposure
            )
            density_mean = float(
                curve.apply(np.asarray([exposure], dtype=np.float64))[0]
            )
            error = abs(analytic - target)
            analytic_errors.append(error)
            probes.append(
                {
                    "channel": curve.layer,
                    "domain_fraction": float(fraction),
                    "relative_log_exposure": exposure,
                    "density_mean": density_mean,
                    "target_sigma_d_at_48um": target,
                    "point_scale": profile.point_scale(
                        prior, curve.layer, exposure
                    ),
                    "analytic_aperture_sigma_d": analytic,
                    "analytic_absolute_error": error,
                }
            )

    evaluation = contract["evaluation"]
    full_shape = tuple(int(value) for value in evaluation["field_shape"])
    energy = profile.measurement_energy()
    aperture = profile.aperture_kernel()
    ratios: list[float] = []
    full_means: list[float] = []
    minimum_density = math.inf
    row_exact: list[bool] = []
    seed_rows: list[dict[str, Any]] = []
    for seed in evaluation["seeds"]:
        receipt = build_thomas_dc_receipt(
            full_shape,
            profile_id=profile.spatial_profile_id,
            particle_sigma_pixels=profile.particle_sigma_samples,
            cluster_sigma_pixels=profile.cluster_sigma_samples,
            mean_offspring=profile.mean_offspring,
            component_seeds=profile.component_seeds,
            realization_seed=int(seed),
            truncate=profile.truncate,
            canonical_row_block_height=int(
                evaluation["canonical_receipt_row_block_height"]
            ),
        )
        unit = render_dc_projected_thomas_region(
            receipt, origin_yx=(0, 0), shape=full_shape
        )
        normalized = unit / math.sqrt(energy)
        measured = fftconvolve(normalized, aperture, mode="same")
        border = int(evaluation["measurement_crop_border_samples"])
        ratio = float(np.std(measured[border:-border, border:-border]))
        field_mean = abs(float(np.mean(normalized, dtype=np.float64)))
        ratios.append(ratio)
        full_means.append(field_mean)
        for probe in probes:
            minimum_density = min(
                minimum_density,
                float(
                    np.min(
                        float(probe["density_mean"])
                        + float(probe["target_sigma_d_at_48um"]) * normalized
                    )
                ),
            )
        partition_exact = True
        assembled = np.empty_like(unit)
        row_height = int(evaluation["row_partition_height"])
        for y0 in range(0, full_shape[0], row_height):
            height = min(row_height, full_shape[0] - y0)
            assembled[y0 : y0 + height] = render_dc_projected_thomas_region(
                receipt, origin_yx=(y0, 0), shape=(height, full_shape[1])
            )
        partition_exact &= bool(np.array_equal(unit, assembled))
        row_exact.append(partition_exact)
        seed_rows.append(
            {
                "seed": int(seed),
                "receipt_id": receipt.receipt_id,
                "aperture_sigma_ratio": ratio,
                "absolute_relative_error": abs(ratio - 1.0),
                "normalized_full_field_absolute_mean": field_mean,
                "row_partition_exact": partition_exact,
            }
        )
    errors = np.abs(np.asarray(ratios, dtype=np.float64) - 1.0)
    probe_sigmas = [float(probe["target_sigma_d_at_48um"]) for probe in probes]
    gates = {
        "parent_identity": True,
        "analytic_aperture_sigma": max(analytic_errors)
        <= evaluation["maximum_analytic_aperture_sigma_error"],
        "monte_carlo_median": float(np.median(errors))
        <= evaluation["maximum_monte_carlo_median_absolute_relative_error"],
        "monte_carlo_p95": float(np.percentile(errors, 95.0))
        <= evaluation["maximum_monte_carlo_p95_absolute_relative_error"],
        "profile_probe_range": min(probe_sigmas)
        >= evaluation["minimum_profile_probe_sigma_d"]
        and max(probe_sigmas) <= evaluation["maximum_profile_probe_sigma_d"],
        "full_field_mean": max(full_means)
        <= evaluation["maximum_full_field_absolute_mean"],
        "positive_developed_density": minimum_density > 0.0,
        "row_partition_exact": all(row_exact),
        "profile_roundtrip_exact": profile_roundtrip_exact,
        "finite": all(
            math.isfinite(value)
            for value in ratios + full_means + probe_sigmas + [minimum_density]
        ),
        "no_refit_or_realized_normalization": True,
    }
    automatic_pass = all(gates.values())
    stable = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": hash_file(
            root / "configs/u6_p4bw_density_conditioned_thomas_profile_v1.json",
            "sha256",
        ),
        "profile_id": profile.identity(),
        "amplitude_profile_id": amplitude.identity(),
        "spatial_profile_id": profile.spatial_profile_id,
        "measurement_energy": energy,
        "point_scale_for_unit_sigma_d": 1.0 / math.sqrt(energy),
        "profile_probes": probes,
        "seed_rows": seed_rows,
        "monte_carlo_median_absolute_relative_error": float(np.median(errors)),
        "monte_carlo_p95_absolute_relative_error": float(
            np.percentile(errors, 95.0)
        ),
        "minimum_developed_density": minimum_density,
        "gate_results": gates,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_density_conditioned_thomas_hybrid_profile"
            if automatic_pass
            else "close_density_conditioned_thomas_hybrid_profile"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return bundle, {**stable, "stable_evidence_id": stable_id}


def write_json(payload: Mapping[str, Any], path: Path) -> str:
    encoded = _canonical_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "DensityConditionedThomasEvaluationError",
    "_validate_contract",
    "compile_and_evaluate",
    "write_json",
]
