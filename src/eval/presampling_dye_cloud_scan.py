"""U6.P6AC presampling dye-cloud scanner reference evaluator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_callier_source import hash_file
from src.film_physics.circular_scanner_aperture import (
    apply_circular_aperture,
    compile_circular_aperture_kernel,
)
from src.film_physics.presampling_dye_cloud_scan import (
    render_presampling_dye_cloud_scan,
)

SCHEMA = "neuro_film.u6_p6ac_presampling_dye_cloud_scan_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p6ac_presampling_dye_cloud_scan_report.v1"


class PresamplingDyeCloudAuditError(RuntimeError):
    """Raised when the P6AC contract or parent evidence drifts."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise PresamplingDyeCloudAuditError("P6AC paths must be bounded and relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parents = payload.get("parents", {})
    material = payload.get("material", {})
    evaluation = payload.get("evaluation", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or parents.get("developed_structure_decision_sha256")
        != "a7a8a6402e5de35922723b0a46b8afbfb2f4b95741bac32a7c42a4314bfcacef"
        or parents.get("developed_structure_stable_evidence_id")
        != "099aeea91cec96b91a31b2c01423a673c66c608617880666b2558c8370894cf8"
        or parents.get("target_compiler_decision_sha256")
        != "9267de83bab912caceb89b4022f1a18dd5255da63ad4d84107ba36816bfba7a1"
        or parents.get("target_compiler_stable_evidence_id")
        != "9ca2cb6cf50f0b4aca1390c723dd21f6fb8fd177ffbebba2f715a8609dea77b4"
        or material.get("input_shape") != [17, 19]
        or material.get("target_pixel_pitch_um") != 6.35
        or material.get("candidate_zoom") != 8
        or material.get("reference_zoom") != 16
        or material.get("aperture_diameter_um") != 12.5
        or material.get("aperture_subpixels_per_axis") != 256
        or material.get("radius_um_cmy") != [1.8, 2.2, 2.6]
        or material.get("mark_optical_density_cmy") != [0.16, 0.20, 0.24]
        or material.get("monte_carlo_samples") != 4
        or material.get("seed") != 26_080_263
        or evaluation.get("interior_crop_target_pixels") != 2
        or evaluation.get("row_partitions") != [3, 7, 11]
        or evaluation.get("constant_transmittance") != 0.37
        or gates.get("maximum_candidate_rmse") != 0.005
        or gates.get("maximum_candidate_p95_absolute_error") != 0.0125
        or gates.get("minimum_rmse_improvement_vs_best_post_raster") != 0.25
        or gates.get("minimum_rmse_improvement_vs_wrong_order") != 0.25
        or gates.get("maximum_constant_error") != 1e-12
        or gates.get("minimum_transmittance") != 0.0
        or gates.get("maximum_transmittance") != 1.0
        or gates.get("maximum_repeat_error") != 0.0
        or gates.get("maximum_partition_error") != 1e-15
    ):
        raise PresamplingDyeCloudAuditError("P6AC frozen contract drift")
    _relative_path(parents.get("developed_structure_decision_path", ""))
    _relative_path(parents.get("target_compiler_decision_path", ""))
    return payload


def _verify_parents(config: dict[str, Any], root: Path) -> None:
    parents = config["parents"]
    developed_path = root / _relative_path(parents["developed_structure_decision_path"])
    target_path = root / _relative_path(parents["target_compiler_decision_path"])
    if (
        not developed_path.is_file()
        or hash_file(developed_path) != parents["developed_structure_decision_sha256"]
        or not target_path.is_file()
        or hash_file(target_path) != parents["target_compiler_decision_sha256"]
    ):
        raise PresamplingDyeCloudAuditError("P6AC parent evidence drift")
    developed = json.loads(developed_path.read_text(encoding="utf-8"))
    target = json.loads(target_path.read_text(encoding="utf-8"))
    if (
        developed.get("stable_evidence_id")
        != parents["developed_structure_stable_evidence_id"]
        or developed.get("status") != "automatic-pass-reference-only"
        or target.get("stable_evidence_id")
        != parents["target_compiler_stable_evidence_id"]
        or target.get("decision") != "close_direct_4000dpi_circular_aperture_compiler"
    ):
        raise PresamplingDyeCloudAuditError("P6AC parent facts drift")


def _density_fixture(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y, x = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float64),
        np.linspace(0.0, 1.0, width, dtype=np.float64),
        indexing="ij",
    )
    checker = ((np.indices(shape).sum(axis=0) // 3) % 2).astype(np.float64)
    sparse = ((x > 0.62) & (y > 0.47)).astype(np.float64)
    density = np.stack(
        (
            0.08 + 0.44 * x + 0.08 * checker,
            0.10 + 0.36 * y + 0.10 * sparse,
            0.06 + 0.24 * (x + y) + 0.06 * checker * sparse,
        ),
        axis=-1,
    )
    return np.asarray(density, dtype=np.float64)


def _rmse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(first - second), dtype=np.float64)))


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes(order="C")).hexdigest()


def evaluate_presampling_dye_cloud_scan(
    config: dict[str, Any], root: Path
) -> dict[str, Any]:
    _verify_parents(config, root)
    material = config["material"]
    evaluation = config["evaluation"]
    target = _density_fixture(tuple(material["input_shape"]))
    kwargs = {
        "target_pixel_pitch_um": material["target_pixel_pitch_um"],
        "candidate_zoom": material["candidate_zoom"],
        "reference_zoom": material["reference_zoom"],
        "aperture_diameter_um": material["aperture_diameter_um"],
        "aperture_subpixels_per_axis": material["aperture_subpixels_per_axis"],
        "radius_um_cmy": tuple(material["radius_um_cmy"]),
        "mark_optical_density_cmy": tuple(material["mark_optical_density_cmy"]),
        "monte_carlo_samples": material["monte_carlo_samples"],
        "seed": material["seed"],
    }
    result = render_presampling_dye_cloud_scan(target, **kwargs)
    crop = evaluation["interior_crop_target_pixels"]
    interior = np.s_[crop:-crop, crop:-crop, :]
    reference = result.reference[interior]
    candidate = result.candidate[interior]
    wrong = result.wrong_order[interior]
    post_circular = result.post_raster_circular[interior]
    post_gaussian = result.post_raster_gaussian[interior]
    candidate_error = np.abs(candidate - reference)
    candidate_rmse = _rmse(candidate, reference)
    wrong_rmse = _rmse(wrong, reference)
    circular_rmse = _rmse(post_circular, reference)
    gaussian_rmse = _rmse(post_gaussian, reference)
    best_post_rmse = min(circular_rmse, gaussian_rmse)

    partition_error = 0.0
    repeat_error = 0.0
    for row_partition in evaluation["row_partitions"]:
        replay = render_presampling_dye_cloud_scan(
            target, **kwargs, row_partition=row_partition
        )
        partition_error = max(
            partition_error,
            float(np.max(np.abs(replay.candidate - result.candidate))),
            float(
                np.max(
                    np.abs(replay.post_raster_circular - result.post_raster_circular)
                )
            ),
        )
        repeat_error = max(
            repeat_error,
            float(np.max(np.abs(replay.reference - result.reference))),
        )

    constant = np.full((31, 29, 3), evaluation["constant_transmittance"])
    constant_kernel = compile_circular_aperture_kernel(
        aperture_diameter_um=material["aperture_diameter_um"],
        pixel_pitch_um=material["target_pixel_pitch_um"] / material["candidate_zoom"],
        subpixels_per_axis=material["aperture_subpixels_per_axis"],
    )
    constant_error = float(
        np.max(np.abs(apply_circular_aperture(constant, constant_kernel) - constant))
    )
    all_outputs = np.stack(
        (
            result.reference,
            result.candidate,
            result.wrong_order,
            result.post_raster_circular,
            result.post_raster_gaussian,
        )
    )
    metrics = {
        "candidate_rmse": candidate_rmse,
        "candidate_p95_absolute_error": float(np.quantile(candidate_error, 0.95)),
        "wrong_order_rmse": wrong_rmse,
        "post_raster_circular_rmse": circular_rmse,
        "post_raster_gaussian_rmse": gaussian_rmse,
        "best_post_raster_rmse": best_post_rmse,
        "rmse_improvement_vs_best_post_raster": (best_post_rmse - candidate_rmse)
        / best_post_rmse,
        "rmse_improvement_vs_wrong_order": (wrong_rmse - candidate_rmse) / wrong_rmse,
        "minimum_transmittance": float(np.min(all_outputs)),
        "maximum_transmittance": float(np.max(all_outputs)),
        "constant_error": constant_error,
        "repeat_error": repeat_error,
        "partition_error": partition_error,
        "candidate_kernel_shape": list(result.candidate_kernel.shape),
        "reference_kernel_shape": list(result.reference_kernel.shape),
    }
    gates = config["gates"]
    gate_results = {
        "candidate_rmse": metrics["candidate_rmse"] <= gates["maximum_candidate_rmse"],
        "candidate_p95": metrics["candidate_p95_absolute_error"]
        <= gates["maximum_candidate_p95_absolute_error"],
        "post_raster_control": metrics["rmse_improvement_vs_best_post_raster"]
        >= gates["minimum_rmse_improvement_vs_best_post_raster"],
        "wrong_order_control": metrics["rmse_improvement_vs_wrong_order"]
        >= gates["minimum_rmse_improvement_vs_wrong_order"],
        "constant": metrics["constant_error"] <= gates["maximum_constant_error"],
        "range": metrics["minimum_transmittance"] > gates["minimum_transmittance"]
        and metrics["maximum_transmittance"] <= gates["maximum_transmittance"],
        "repeat": metrics["repeat_error"] <= gates["maximum_repeat_error"],
        "partitions": metrics["partition_error"] <= gates["maximum_partition_error"],
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "parent_evidence_ids": [
            config["parents"]["developed_structure_stable_evidence_id"],
            config["parents"]["target_compiler_stable_evidence_id"],
        ],
        "target_density_sha256": _array_sha256(target),
        "reference_sha256": _array_sha256(result.reference),
        "candidate_sha256": _array_sha256(result.candidate),
        "wrong_order_sha256": _array_sha256(result.wrong_order),
        "post_raster_circular_sha256": _array_sha256(result.post_raster_circular),
        "post_raster_gaussian_sha256": _array_sha256(result.post_raster_gaussian),
        "metrics": metrics,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "retain_presampling_dye_cloud_scan_reference"
            if automatic_pass
            else "close_8x_presampling_dye_cloud_scan_approximation"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "PresamplingDyeCloudAuditError",
    "evaluate_presampling_dye_cloud_scan",
    "load_contract",
]
