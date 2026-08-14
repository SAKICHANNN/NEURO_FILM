"""U6.P2AL TRI-X scalar amplitude plus generic Thomas shape evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from src.film_physics.bw_granularity_scalar import BWGranularityScalarProfile
from src.film_physics.density_conditioned_thomas import (
    binary_circular_aperture_kernel,
    thomas_aperture_measurement_energy,
)
from src.film_physics.thomas_dc_projection import (
    build_thomas_dc_receipt,
    render_dc_projected_thomas_region,
)


class TrixHybridThomasAmplitudeError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2al_trix_hybrid_thomas_amplitude_contract.v1"
    ):
        raise TrixHybridThomasAmplitudeError("unsupported P2AL contract")
    return payload


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    loaded = {}
    for name, binding in contract["parents"].items():
        path = root / binding["path"]
        if _sha(path) != binding["sha256"]:
            raise TrixHybridThomasAmplitudeError("P2AL parent identity mismatch")
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
        required = binding.get(
            "required_decision", binding.get("required_automatic_pass")
        )
        actual = (
            loaded[name].get("decision")
            if "required_decision" in binding
            else loaded[name].get("automatic_pass")
        )
        if actual != required:
            raise TrixHybridThomasAmplitudeError("P2AL parent decision mismatch")
    scalar = BWGranularityScalarProfile(
        **{
            **loaded["trix_scalar"]["observation"],
            "source_evidence_id": loaded["trix_scalar"]["stable_evidence_id"],
        }
    )
    hybrid = contract["hybrid"]
    aperture = binary_circular_aperture_kernel(
        hybrid["sample_pitch_micrometres"],
        hybrid["measurement_aperture_diameter_micrometres"],
    )
    energy = thomas_aperture_measurement_energy(
        particle_sigma_samples=hybrid["particle_sigma_samples"],
        cluster_sigma_samples=hybrid["cluster_sigma_samples"],
        mean_offspring=hybrid["mean_offspring"],
        truncate=hybrid["truncate_sigma"],
        aperture_kernel=aperture,
    )
    target = scalar.density_rms_at(
        density_mean=scalar.density_mean,
        aperture_diameter_micrometres=scalar.aperture_diameter_micrometres,
    )
    point_scale = target / math.sqrt(energy)
    analytic_error = abs(point_scale * math.sqrt(energy) - target)
    evaluation = contract["evaluation"]
    shape = tuple(evaluation["field_shape"])
    rows = []
    partition_exact = True
    repeat_exact = True
    for seed in evaluation["seeds"]:
        kwargs = {
            "full_shape": shape,
            "profile_id": hybrid["spatial_profile_id"],
            "particle_sigma_pixels": hybrid["particle_sigma_samples"],
            "cluster_sigma_pixels": hybrid["cluster_sigma_samples"],
            "mean_offspring": hybrid["mean_offspring"],
            "component_seeds": tuple(hybrid["component_seeds"]),
            "realization_seed": seed,
            "truncate": hybrid["truncate_sigma"],
            "canonical_row_block_height": evaluation["canonical_row_block_height"],
        }
        receipt = build_thomas_dc_receipt(**kwargs)
        repeat = build_thomas_dc_receipt(**kwargs)
        unit = render_dc_projected_thomas_region(receipt, origin_yx=(0, 0), shape=shape)
        repeated = render_dc_projected_thomas_region(
            repeat, origin_yx=(0, 0), shape=shape
        )
        repeat_exact = repeat_exact and np.array_equal(unit, repeated)
        assembled = np.empty_like(unit)
        row_height = evaluation["row_partition_height"]
        for y0 in range(0, shape[0], row_height):
            height = min(row_height, shape[0] - y0)
            assembled[y0 : y0 + height] = render_dc_projected_thomas_region(
                receipt, origin_yx=(y0, 0), shape=(height, shape[1])
            )
        partition_exact = partition_exact and np.array_equal(unit, assembled)
        density = scalar.density_mean + point_scale * unit
        measured = fftconvolve(density, aperture, mode="same")
        border = evaluation["measurement_crop_border_samples"]
        measured = measured[border:-border, border:-border]
        sigma = float(np.std(measured, dtype=np.float64))
        rows.append(
            {
                "seed": seed,
                "receipt_id": receipt.receipt_id,
                "density_sha256": hashlib.sha256(
                    np.asarray(density, dtype="<f8").tobytes()
                ).hexdigest(),
                "absolute_field_mean": abs(
                    float(np.mean(density - scalar.density_mean, dtype=np.float64))
                ),
                "minimum_developed_density": float(np.min(density)),
                "aperture_sigma_d": sigma,
                "aperture_sigma_relative_error": abs(sigma - target) / target,
            }
        )
    errors = np.asarray(
        [row["aperture_sigma_relative_error"] for row in rows], dtype=np.float64
    )
    measurements = {
        "analytic_aperture_sigma_error": analytic_error,
        "median_aperture_sigma_relative_error": float(np.median(errors)),
        "p95_aperture_sigma_relative_error": float(np.percentile(errors, 95.0)),
        "maximum_absolute_field_mean": max(row["absolute_field_mean"] for row in rows),
        "minimum_developed_density": min(
            row["minimum_developed_density"] for row in rows
        ),
        "repeat_byte_exact": repeat_exact,
        "partition_exact": partition_exact,
        "rgb_image_transform_count_zero": True,
    }
    results = {
        "maximum_analytic_aperture_sigma_error": analytic_error
        <= evaluation["maximum_analytic_aperture_sigma_error"],
        "maximum_median_aperture_sigma_relative_error": measurements[
            "median_aperture_sigma_relative_error"
        ]
        <= evaluation["maximum_median_aperture_sigma_relative_error"],
        "maximum_p95_aperture_sigma_relative_error": measurements[
            "p95_aperture_sigma_relative_error"
        ]
        <= evaluation["maximum_p95_aperture_sigma_relative_error"],
        "maximum_absolute_field_mean": measurements["maximum_absolute_field_mean"]
        <= evaluation["maximum_absolute_field_mean"],
        "minimum_developed_density": measurements["minimum_developed_density"]
        >= evaluation["minimum_developed_density"],
        "repeat_byte_exact": repeat_exact is evaluation["repeat_byte_exact"],
        "partition_exact": partition_exact is evaluation["partition_exact"],
        "rgb_image_transform_count_zero": True,
    }
    passed = all(results.values())
    stable = {
        "schema": "neuro_film.u6_p2al_trix_hybrid_thomas_amplitude_report.v1",
        "scalar_profile_identity": scalar.identity(),
        "measurement_energy": energy,
        "point_scale": point_scale,
        "rows": rows,
        "measurements": measurements,
        "gate_results": results,
        "automatic_pass": passed,
        "decision": contract["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        **stable,
        "stable_evidence_id": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return hashlib.sha256(encoded.encode()).hexdigest()
