"""U6.P6AH piecewise analytic cloud/aperture evaluator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_callier_source import hash_file
from src.eval.presampling_reference_convergence import (
    build_presampling_convergence_fixture,
)
from src.film_physics.analytic_cloud_aperture import (
    integrate_marked_cloud_aperture_analytic,
    render_analytic_cloud_aperture,
)
from src.film_physics.developed_structure import build_colour_dye_cloud_context
from src.film_physics.direct_cloud_aperture import render_direct_cloud_aperture
from src.film_physics.presampling_reference_convergence import (
    render_fft_presampling_reference,
)

SCHEMA = "neuro_film.u6_p6ah_piecewise_analytic_cloud_aperture_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p6ah_piecewise_analytic_cloud_aperture_report.v1"


class AnalyticCloudApertureAuditError(RuntimeError):
    pass


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise AnalyticCloudApertureAuditError("P6AH paths must be relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload.get("parent", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or parent.get("decision_sha256")
        != "eb845b0e65541dd806fa38af08bd675c39013ae4b0211bba1790909e58a49756"
        or parent.get("decision") != "close_adaptive_quadrature_nonconvergence"
        or gates.get("maximum_analytic_control_error") != 1e-11
        or gates.get("maximum_axis_swap_error") != 1e-10
        or gates.get("maximum_repeat_error") != 0.0
        or gates.get("maximum_partition_error") != 0.0
        or gates.get("minimum_transmittance") != 0.0
        or gates.get("maximum_transmittance") != 1.0
    ):
        raise AnalyticCloudApertureAuditError("P6AH contract drift")
    _relative(parent.get("decision_path", ""))
    return payload


def _sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def _rmse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(first - second), dtype=np.float64)))


def _analytic_control_error() -> float:
    mark = 0.2
    transmission = 10.0**-mark
    common = {
        "mark_optical_density": mark,
        "aperture_center_yx": (0.0, 0.0),
        "aperture_radius": 1.0,
    }
    cases: list[tuple[np.ndarray, np.ndarray | float, float]] = [
        (np.empty((0, 2)), np.empty(0), 1.0),
        (np.asarray([[4.0, 4.0]]), 0.5, 1.0),
        (np.asarray([[0.0, 0.0]]), 2.0, transmission),
        (np.asarray([[0.0, 0.0]]), 0.5, 0.25 * transmission + 0.75),
        (
            np.asarray([[0.0, 0.0], [0.0, 0.0]]),
            np.asarray([0.35, 0.75]),
            0.35**2 * transmission**2
            + (0.75**2 - 0.35**2) * transmission
            + (1.0 - 0.75**2),
        ),
    ]
    errors = []
    for centers, radii, expected in cases:
        horizontal = integrate_marked_cloud_aperture_analytic(
            centers, radii, integration_axis="x", **common
        )
        vertical = integrate_marked_cloud_aperture_analytic(
            centers, radii, integration_axis="y", **common
        )
        errors.extend((abs(horizontal - expected), abs(vertical - expected)))
    return max(errors)


def evaluate_analytic_cloud_aperture(
    config: dict[str, Any], root: Path
) -> dict[str, Any]:
    parent = config["parent"]
    decision_path = root / _relative(parent["decision_path"])
    if hash_file(decision_path) != parent["decision_sha256"]:
        raise AnalyticCloudApertureAuditError("P6AH parent drift")
    if (
        json.loads(decision_path.read_text(encoding="utf-8")).get("decision")
        != parent["decision"]
    ):
        raise AnalyticCloudApertureAuditError("P6AH parent facts drift")

    material = config["material"]
    target = build_presampling_convergence_fixture(tuple(material["input_shape"]))
    context = build_colour_dye_cloud_context(
        target,
        radius_um_cmy=tuple(material["radius_um_cmy"]),
        mark_optical_density_cmy=tuple(material["mark_optical_density_cmy"]),
        output_zoom=1,
        output_pixel_pitch_um=material["target_pixel_pitch_um"],
        monte_carlo_samples=material["monte_carlo_samples"],
        seed=material["seed"],
    )
    common = {"aperture_diameter_um": material["aperture_diameter_um"]}
    candidate = render_analytic_cloud_aperture(
        context, integration_axis="x", **common
    ).values
    axis_swap = render_analytic_cloud_aperture(
        context, integration_axis="y", **common
    ).values
    repeat = render_analytic_cloud_aperture(
        context, integration_axis="x", **common
    ).values
    partition_errors = []
    for partition in config["evaluation"]["row_partitions"]:
        partitioned = render_analytic_cloud_aperture(
            context, integration_axis="x", row_partition=partition, **common
        ).values
        partition_errors.append(float(np.max(np.abs(partitioned - candidate))))
    direct = render_direct_cloud_aperture(
        context,
        aperture_diameter_um=material["aperture_diameter_um"],
        sample_count=256,
        point_chunk_size=4096,
    )
    raster64 = render_fft_presampling_reference(
        target,
        zoom=64,
        target_pixel_pitch_um=material["target_pixel_pitch_um"],
        aperture_diameter_um=material["aperture_diameter_um"],
        aperture_subpixels_per_axis=256,
        radius_um_cmy=tuple(material["radius_um_cmy"]),
        mark_optical_density_cmy=tuple(material["mark_optical_density_cmy"]),
        monte_carlo_samples=material["monte_carlo_samples"],
        seed=material["seed"],
    )
    metrics = {
        "maximum_analytic_control_error": _analytic_control_error(),
        "maximum_axis_swap_error": float(np.max(np.abs(candidate - axis_swap))),
        "maximum_repeat_error": float(np.max(np.abs(candidate - repeat))),
        "maximum_partition_error": max(partition_errors),
        "minimum_transmittance": float(np.min(candidate)),
        "maximum_transmittance": float(np.max(candidate)),
        "rmse_vs_p6ad_256_point": _rmse(candidate, direct),
        "rmse_vs_p6af_64x_raster": _rmse(candidate, raster64),
    }
    gates = config["gates"]
    gate_results = {
        "analytic_controls": metrics["maximum_analytic_control_error"]
        <= gates["maximum_analytic_control_error"],
        "axis_swap": metrics["maximum_axis_swap_error"]
        <= gates["maximum_axis_swap_error"],
        "repeat": metrics["maximum_repeat_error"] <= gates["maximum_repeat_error"],
        "partition": metrics["maximum_partition_error"]
        <= gates["maximum_partition_error"],
        "range": metrics["minimum_transmittance"] > gates["minimum_transmittance"]
        and metrics["maximum_transmittance"] <= gates["maximum_transmittance"],
    }
    passed = all(gate_results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "candidate_sha256": _sha(candidate),
        "axis_swap_sha256": _sha(axis_swap),
        "metrics": metrics,
        "gate_results": gate_results,
        "automatic_pass": passed,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "retain_piecewise_analytic_aperture_reference"
            if passed
            else "close_piecewise_analytic_aperture_reference"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "AnalyticCloudApertureAuditError",
    "evaluate_analytic_cloud_aperture",
    "load_contract",
]
