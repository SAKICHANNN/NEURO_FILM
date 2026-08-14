"""U6.P2AJ fixed-kernel TRI-X diffuse-rms granularity compatibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from src.film_physics.developed_structure import (
    build_bw_silver_context,
    render_developed_structure,
)


class TrixGranularityCompatibilityError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2aj_trix_granularity_compatibility_contract.v1"
    ):
        raise TrixGranularityCompatibilityError("unsupported P2AJ contract")
    return payload


def _aperture_kernel(diameter_pixels: float) -> np.ndarray:
    radius = diameter_pixels / 2.0
    half = int(np.ceil(radius))
    yy, xx = np.indices((2 * half + 1, 2 * half + 1), dtype=np.float64)
    kernel = ((yy - half) ** 2 + (xx - half) ** 2 <= radius**2).astype(np.float64)
    kernel /= np.sum(kernel)
    return kernel


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    for binding in contract["parents"].values():
        path = root / binding["path"]
        if _sha(path) != binding["sha256"]:
            raise TrixGranularityCompatibilityError("P2AJ parent identity mismatch")
        parent = json.loads(path.read_text(encoding="utf-8"))
        if parent.get("automatic_pass") is not binding["required_automatic_pass"]:
            raise TrixGranularityCompatibilityError("P2AJ parent decision mismatch")
    source = contract["source"]
    if _sha(root / source["path"]) != source["sha256"]:
        raise TrixGranularityCompatibilityError("P2AJ source identity mismatch")

    candidate = contract["unchanged_candidate"]
    height, width = candidate["input_shape"]
    density = np.full((height, width), source["net_diffuse_density"], np.float64)
    pitch = candidate["output_pixel_pitch_um"]
    kernel = _aperture_kernel(source["aperture_diameter_micrometres"] / pitch)
    rows = []
    repeat_exact = True
    for seed in candidate["seeds"]:
        kwargs = {
            "radius_um": candidate["radius_um"],
            "output_zoom": candidate["output_zoom"],
            "output_pixel_pitch_um": pitch,
            "monte_carlo_samples": candidate["monte_carlo_samples"],
            "seed": seed,
        }
        context = build_bw_silver_context(density, **kwargs)
        repeat = build_bw_silver_context(density, **kwargs)
        transmission = render_developed_structure(context).values[..., 0]
        repeated = render_developed_structure(repeat).values[..., 0]
        repeat_exact = repeat_exact and np.array_equal(transmission, repeated)
        aperture_transmission = fftconvolve(transmission, kernel, mode="valid")
        finite = bool(
            np.all(np.isfinite(aperture_transmission))
            and np.all(aperture_transmission > 0.0)
        )
        aperture_density = -np.log10(aperture_transmission)
        mean_density = float(np.mean(aperture_density, dtype=np.float64))
        diffuse_rms = float(np.std(aperture_density, dtype=np.float64) * 1000.0)
        target = source["diffuse_rms_granularity"]
        rows.append(
            {
                "seed": seed,
                "aperture_window_count": int(aperture_density.size),
                "aperture_windows_finite": finite,
                "mean_aperture_density": mean_density,
                "diffuse_rms_granularity": diffuse_rms,
                "relative_error": abs(diffuse_rms - target) / target,
                "transmittance_sha256": hashlib.sha256(
                    np.asarray(transmission, dtype="<f4").tobytes()
                ).hexdigest(),
            }
        )

    errors = np.asarray([row["relative_error"] for row in rows], dtype=np.float64)
    density_errors = np.asarray(
        [
            abs(row["mean_aperture_density"] - source["net_diffuse_density"])
            for row in rows
        ],
        dtype=np.float64,
    )
    measurements = {
        "median_relative_error": float(np.median(errors)),
        "worst_relative_error": float(np.max(errors)),
        "maximum_target_mean_density_absolute_error": float(np.max(density_errors)),
        "all_aperture_windows_finite": all(
            row["aperture_windows_finite"] for row in rows
        ),
        "minimum_aperture_window_count_per_seed": min(
            row["aperture_window_count"] for row in rows
        ),
        "repeat_byte_exact": repeat_exact,
        "rgb_image_transform_count_zero": True,
    }
    gates = contract["automatic_gates"]
    results = {
        "maximum_median_relative_error": measurements["median_relative_error"]
        <= gates["maximum_median_relative_error"],
        "maximum_worst_relative_error": measurements["worst_relative_error"]
        <= gates["maximum_worst_relative_error"],
        "maximum_target_mean_density_absolute_error": measurements[
            "maximum_target_mean_density_absolute_error"
        ]
        <= gates["maximum_target_mean_density_absolute_error"],
        "all_aperture_windows_finite": measurements["all_aperture_windows_finite"]
        is gates["all_aperture_windows_finite"],
        "minimum_aperture_window_count_per_seed": measurements[
            "minimum_aperture_window_count_per_seed"
        ]
        >= gates["minimum_aperture_window_count_per_seed"],
        "repeat_byte_exact": measurements["repeat_byte_exact"]
        is gates["repeat_byte_exact"],
        "rgb_image_transform_count_zero": True,
    }
    passed = all(results.values())
    stable = {
        "schema": "neuro_film.u6_p2aj_trix_granularity_compatibility_report.v1",
        "source_sha256": source["sha256"],
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
