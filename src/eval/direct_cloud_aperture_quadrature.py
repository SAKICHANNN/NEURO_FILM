"""U6.P6AD direct continuous-cloud aperture quadrature evaluator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_callier_source import hash_file
from src.film_physics.developed_structure import build_colour_dye_cloud_context
from src.film_physics.direct_cloud_aperture import render_direct_cloud_aperture
from src.film_physics.presampling_dye_cloud_scan import (
    render_presampling_dye_cloud_scan,
)

SCHEMA = "neuro_film.u6_p6ad_direct_cloud_aperture_quadrature_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p6ad_direct_cloud_aperture_quadrature_report.v1"


class DirectCloudQuadratureAuditError(RuntimeError):
    """Raised when the P6AD contract or parent reference drifts."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise DirectCloudQuadratureAuditError("P6AD paths must be bounded and relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload.get("parent", {})
    material = payload.get("material", {})
    quadrature = payload.get("quadrature", {})
    evaluation = payload.get("evaluation", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or parent.get("decision_sha256")
        != "fcff603cd8da2978ffa1659bf38c7cffbbd9d72c5927cde6ebf637994bae6c45"
        or parent.get("stable_evidence_id")
        != "f6928274fa5c275878d2b032123092736be9649f6a5d935793e225d052a82f78"
        or parent.get("reference_report_sha256")
        != "203c0676643c58d85bb627c9936bc559af88412682fe52ea99d5ffd4bb30f846"
        or parent.get("target_density_sha256")
        != "6cb2267330726e0915e71a8363f5486d028bb564ed9dbf32f5131cf6c93c02d0"
        or parent.get("reference_output_sha256")
        != "7766720aafe57a3c8c6d10503a8bce59d60169a003700610bfa1a0783fe26420"
        or material.get("input_shape") != [17, 19]
        or material.get("target_pixel_pitch_um") != 6.35
        or material.get("reference_zoom") != 16
        or material.get("reference_aperture_subpixels_per_axis") != 256
        or material.get("aperture_diameter_um") != 12.5
        or material.get("radius_um_cmy") != [1.8, 2.2, 2.6]
        or material.get("mark_optical_density_cmy") != [0.16, 0.20, 0.24]
        or material.get("geometry_seed") != 26_080_263
        or material.get("reference_monte_carlo_samples") != 4
        or quadrature.get("candidate_samples") != 256
        or quadrature.get("low_sample_control") != 64
        or quadrature.get("point_chunk_size") != 4096
        or quadrature.get("output_dtype") != "float64"
        or quadrature.get("intermediate_raster_allowed") is not False
        or evaluation.get("interior_crop_target_pixels") != 2
        or evaluation.get("row_partitions") != [3, 7, 11]
        or gates.get("maximum_candidate_rmse") != 0.005
        or gates.get("maximum_candidate_p95_absolute_error") != 0.0125
        or gates.get("minimum_rmse_improvement_vs_8x_raster") != 0.20
        or gates.get("minimum_rmse_improvement_vs_low_sample") != 0.10
        or gates.get("minimum_transmittance") != 0.0
        or gates.get("maximum_transmittance") != 1.0
        or gates.get("maximum_repeat_error") != 0.0
        or gates.get("maximum_partition_error") != 0.0
    ):
        raise DirectCloudQuadratureAuditError("P6AD frozen contract drift")
    _relative_path(parent.get("decision_path", ""))
    _relative_path(parent.get("reference_report_path", ""))
    return payload


def _verify_parent(config: dict[str, Any], root: Path) -> None:
    parent = config["parent"]
    decision_path = root / _relative_path(parent["decision_path"])
    report_path = root / _relative_path(parent["reference_report_path"])
    if (
        not decision_path.is_file()
        or hash_file(decision_path) != parent["decision_sha256"]
        or not report_path.is_file()
        or hash_file(report_path) != parent["reference_report_sha256"]
    ):
        raise DirectCloudQuadratureAuditError("P6AD parent evidence drift")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        decision.get("stable_evidence_id") != parent["stable_evidence_id"]
        or decision.get("decision")
        != "retain_16x_presampling_reference_close_8x_approximation"
        or report.get("stable_evidence_id") != parent["stable_evidence_id"]
        or report.get("target_density_sha256") != parent["target_density_sha256"]
        or report.get("reference_sha256") != parent["reference_output_sha256"]
    ):
        raise DirectCloudQuadratureAuditError("P6AD parent facts drift")


def _density_fixture(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y, x = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float64),
        np.linspace(0.0, 1.0, width, dtype=np.float64),
        indexing="ij",
    )
    checker = ((np.indices(shape).sum(axis=0) // 3) % 2).astype(np.float64)
    sparse = ((x > 0.62) & (y > 0.47)).astype(np.float64)
    return np.stack(
        (
            0.08 + 0.44 * x + 0.08 * checker,
            0.10 + 0.36 * y + 0.10 * sparse,
            0.06 + 0.24 * (x + y) + 0.06 * checker * sparse,
        ),
        axis=-1,
    )


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes(order="C")).hexdigest()


def _rmse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(first - second), dtype=np.float64)))


def evaluate_direct_cloud_aperture_quadrature(
    config: dict[str, Any], root: Path
) -> dict[str, Any]:
    _verify_parent(config, root)
    material = config["material"]
    quadrature = config["quadrature"]
    evaluation = config["evaluation"]
    target = _density_fixture(tuple(material["input_shape"]))
    if _array_sha256(target) != config["parent"]["target_density_sha256"]:
        raise DirectCloudQuadratureAuditError("P6AD target fixture drift")
    raster = render_presampling_dye_cloud_scan(
        target,
        target_pixel_pitch_um=material["target_pixel_pitch_um"],
        candidate_zoom=8,
        reference_zoom=material["reference_zoom"],
        aperture_diameter_um=material["aperture_diameter_um"],
        aperture_subpixels_per_axis=material["reference_aperture_subpixels_per_axis"],
        radius_um_cmy=tuple(material["radius_um_cmy"]),
        mark_optical_density_cmy=tuple(material["mark_optical_density_cmy"]),
        monte_carlo_samples=material["reference_monte_carlo_samples"],
        seed=material["geometry_seed"],
    )
    if _array_sha256(raster.reference) != config["parent"]["reference_output_sha256"]:
        raise DirectCloudQuadratureAuditError("P6AD reconstructed reference drift")
    context = build_colour_dye_cloud_context(
        target,
        radius_um_cmy=tuple(material["radius_um_cmy"]),
        mark_optical_density_cmy=tuple(material["mark_optical_density_cmy"]),
        output_zoom=1,
        output_pixel_pitch_um=material["target_pixel_pitch_um"],
        monte_carlo_samples=material["reference_monte_carlo_samples"],
        seed=material["geometry_seed"],
    )
    direct_kwargs = {
        "aperture_diameter_um": material["aperture_diameter_um"],
        "point_chunk_size": quadrature["point_chunk_size"],
    }
    candidate = render_direct_cloud_aperture(
        context,
        **direct_kwargs,
        sample_count=quadrature["candidate_samples"],
    )
    repeat = render_direct_cloud_aperture(
        context,
        **direct_kwargs,
        sample_count=quadrature["candidate_samples"],
    )
    low_sample = render_direct_cloud_aperture(
        context,
        **direct_kwargs,
        sample_count=quadrature["low_sample_control"],
    )
    partition_error = 0.0
    for rows in evaluation["row_partitions"]:
        partitioned = render_direct_cloud_aperture(
            context,
            **direct_kwargs,
            sample_count=quadrature["candidate_samples"],
            row_partition=rows,
        )
        partition_error = max(
            partition_error, float(np.max(np.abs(partitioned - candidate)))
        )
    crop = evaluation["interior_crop_target_pixels"]
    interior = np.s_[crop:-crop, crop:-crop, :]
    reference_interior = raster.reference[interior]
    candidate_interior = candidate[interior]
    candidate_error = np.abs(candidate_interior - reference_interior)
    candidate_rmse = _rmse(candidate_interior, reference_interior)
    raster_rmse = _rmse(raster.candidate[interior], reference_interior)
    low_sample_rmse = _rmse(low_sample[interior], reference_interior)
    all_outputs = np.stack((candidate, low_sample))
    metrics = {
        "candidate_rmse": candidate_rmse,
        "candidate_p95_absolute_error": float(np.quantile(candidate_error, 0.95)),
        "p6ac_8x_raster_rmse": raster_rmse,
        "low_sample_rmse": low_sample_rmse,
        "rmse_improvement_vs_8x_raster": (raster_rmse - candidate_rmse) / raster_rmse,
        "rmse_improvement_vs_low_sample": (low_sample_rmse - candidate_rmse)
        / low_sample_rmse,
        "minimum_transmittance": float(np.min(all_outputs)),
        "maximum_transmittance": float(np.max(all_outputs)),
        "repeat_error": float(np.max(np.abs(repeat - candidate))),
        "partition_error": partition_error,
        "cloud_counts_cmy": [len(layer) for layer in context.centers_by_layer],
    }
    gates = config["gates"]
    gate_results = {
        "candidate_rmse": metrics["candidate_rmse"] <= gates["maximum_candidate_rmse"],
        "candidate_p95": metrics["candidate_p95_absolute_error"]
        <= gates["maximum_candidate_p95_absolute_error"],
        "raster_control": metrics["rmse_improvement_vs_8x_raster"]
        >= gates["minimum_rmse_improvement_vs_8x_raster"],
        "low_sample_control": metrics["rmse_improvement_vs_low_sample"]
        >= gates["minimum_rmse_improvement_vs_low_sample"],
        "range": metrics["minimum_transmittance"] > gates["minimum_transmittance"]
        and metrics["maximum_transmittance"] <= gates["maximum_transmittance"],
        "repeat": metrics["repeat_error"] <= gates["maximum_repeat_error"],
        "partitions": metrics["partition_error"] <= gates["maximum_partition_error"],
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "parent_stable_evidence_id": config["parent"]["stable_evidence_id"],
        "reference_sha256": _array_sha256(raster.reference),
        "candidate_sha256": _array_sha256(candidate),
        "low_sample_sha256": _array_sha256(low_sample),
        "metrics": metrics,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "retain_direct_cloud_aperture_quadrature"
            if automatic_pass
            else "close_direct_cloud_aperture_quadrature"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "DirectCloudQuadratureAuditError",
    "evaluate_direct_cloud_aperture_quadrature",
    "load_contract",
]
