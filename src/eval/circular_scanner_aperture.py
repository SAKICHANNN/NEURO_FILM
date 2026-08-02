"""U6.P6AA evaluator for a positive circular scanner-aperture kernel."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_callier_source import hash_file
from src.film_physics.circular_scanner_aperture import (
    analytic_circular_aperture_mtf,
    apply_circular_aperture,
    compile_circular_aperture_kernel,
    sampled_kernel_mtf,
)

SCHEMA = "neuro_film.u6_p6aa_circular_scanner_aperture_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p6aa_circular_scanner_aperture_report.v1"


class CircularScannerAuditError(RuntimeError):
    """Raised when the P6AA contract or source binding drifts."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise CircularScannerAuditError("P6AA paths must be bounded and relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("source", {})
    compiler = payload.get("compiler", {})
    evaluation = payload.get("evaluation", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or source.get("record_id") != "19730022682"
        or source.get("source_sha256")
        != "8773bb57946c38573d2feb42a08ddf3b0d5001745bc46ac23d825d6cdad90814"
        or source.get("source_report_sha256")
        != "53748b3eeb92fa3a9b103db50ea32f1bb2edba4278ae72e2a9b74f8aa2fd447b"
        or compiler.get("aperture_diameter_um") != 12.5
        or compiler.get("reference_pixel_pitch_um") != 1.0
        or compiler.get("subpixels_per_axis") != 512
        or compiler.get("dtype") != "float64"
        or compiler.get("production_import_allowed") is not False
        or evaluation.get("frequencies_cycles_per_mm")
        != [0, 10, 20, 30, 40, 50, 60, 70, 80]
        or evaluation.get("equal_second_moment_gaussian_sigma_um") != 3.125
        or evaluation.get("test_shape") != [61, 47, 3]
        or evaluation.get("row_partitions") != [7, 17, 31]
        or evaluation.get("seed") != 26_080_261
        or gates.get("maximum_kernel_mtf_absolute_error") != 0.02
        or gates.get("minimum_rmse_improvement_vs_gaussian") != 0.75
        or gates.get("minimum_rmse_improvement_vs_point") != 0.75
        or gates.get("maximum_weight_sum_error") != 1e-12
        or gates.get("minimum_weight") != 0.0
        or gates.get("maximum_rotational_symmetry_error") != 1e-15
        or gates.get("maximum_constant_error") != 1e-12
        or gates.get("maximum_partition_error") != 1e-15
        or gates.get("maximum_repeat_error") != 0.0
        or not gates.get("impulse_must_equal_kernel")
    ):
        raise CircularScannerAuditError("P6AA frozen contract drift")
    _relative_path(source.get("source_report_path", ""))
    return payload


def _verify_source(config: dict[str, Any], root: Path) -> None:
    source = config["source"]
    path = root / _relative_path(source["source_report_path"])
    if not path.is_file() or hash_file(path) != source["source_report_sha256"]:
        raise CircularScannerAuditError("P6AA source report drift")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("source_sha256") != source["source_sha256"]
        or not report.get("source_pass")
        or not report.get("factual_statements", {}).get("composite_system_mtf")
    ):
        raise CircularScannerAuditError("P6AA source mechanism facts drift")


def _rmse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(first - second))))


def evaluate_circular_scanner_aperture(
    config: dict[str, Any], root: Path
) -> dict[str, Any]:
    _verify_source(config, root)
    compiler = config["compiler"]
    evaluation = config["evaluation"]
    kernel = compile_circular_aperture_kernel(
        aperture_diameter_um=compiler["aperture_diameter_um"],
        pixel_pitch_um=compiler["reference_pixel_pitch_um"],
        subpixels_per_axis=compiler["subpixels_per_axis"],
    )
    repeat = compile_circular_aperture_kernel(
        aperture_diameter_um=compiler["aperture_diameter_um"],
        pixel_pitch_um=compiler["reference_pixel_pitch_um"],
        subpixels_per_axis=compiler["subpixels_per_axis"],
    )
    frequencies = np.asarray(evaluation["frequencies_cycles_per_mm"], dtype=np.float64)
    target = analytic_circular_aperture_mtf(
        frequencies, compiler["aperture_diameter_um"]
    )
    candidate = sampled_kernel_mtf(
        kernel, frequencies, compiler["reference_pixel_pitch_um"]
    )
    sigma_mm = evaluation["equal_second_moment_gaussian_sigma_um"] / 1000.0
    gaussian = np.exp(-2.0 * math.pi**2 * sigma_mm**2 * frequencies**2)
    point = np.ones_like(frequencies)
    candidate_rmse = _rmse(candidate, target)
    gaussian_rmse = _rmse(gaussian, target)
    point_rmse = _rmse(point, target)

    shape = tuple(evaluation["test_shape"])
    constant = np.ones(shape, dtype=np.float64)
    constant_output = apply_circular_aperture(constant, kernel)
    rng = np.random.default_rng(evaluation["seed"])
    random_image = rng.random(shape, dtype=np.float64)
    full = apply_circular_aperture(random_image, kernel)
    partition_error = 0.0
    for rows in evaluation["row_partitions"]:
        partitioned = apply_circular_aperture(random_image, kernel, row_partition=rows)
        partition_error = max(
            partition_error, float(np.max(np.abs(partitioned - full)))
        )
    impulse_shape = (kernel.shape[0] + 8, kernel.shape[1] + 8, 3)
    impulse = np.zeros(impulse_shape, dtype=np.float64)
    centre = (impulse_shape[0] // 2, impulse_shape[1] // 2)
    impulse[centre[0], centre[1], :] = 1.0
    impulse_output = apply_circular_aperture(impulse, kernel)
    radius = kernel.shape[0] // 2
    impulse_crop = impulse_output[
        centre[0] - radius : centre[0] + radius + 1,
        centre[1] - radius : centre[1] + radius + 1,
        :,
    ]
    impulse_error = float(np.max(np.abs(impulse_crop - kernel[:, :, None])))
    metrics = {
        "kernel_shape": list(kernel.shape),
        "positive_weight_count": int(np.count_nonzero(kernel > 0.0)),
        "weight_sum_error": abs(float(np.sum(kernel)) - 1.0),
        "minimum_weight": float(np.min(kernel)),
        "rotational_symmetry_error": float(
            max(np.max(np.abs(kernel - np.rot90(kernel, turns))) for turns in (1, 2, 3))
        ),
        "maximum_kernel_mtf_absolute_error": float(np.max(np.abs(candidate - target))),
        "kernel_mtf_rmse": candidate_rmse,
        "gaussian_mtf_rmse": gaussian_rmse,
        "point_mtf_rmse": point_rmse,
        "rmse_improvement_vs_gaussian": (gaussian_rmse - candidate_rmse)
        / gaussian_rmse,
        "rmse_improvement_vs_point": (point_rmse - candidate_rmse) / point_rmse,
        "constant_error": float(np.max(np.abs(constant_output - constant))),
        "partition_error": partition_error,
        "repeat_error": float(np.max(np.abs(repeat - kernel))),
        "impulse_error": impulse_error,
    }
    gates = config["gates"]
    gate_results = {
        "kernel_mtf_error": metrics["maximum_kernel_mtf_absolute_error"]
        <= gates["maximum_kernel_mtf_absolute_error"],
        "gaussian_control": metrics["rmse_improvement_vs_gaussian"]
        >= gates["minimum_rmse_improvement_vs_gaussian"],
        "point_control": metrics["rmse_improvement_vs_point"]
        >= gates["minimum_rmse_improvement_vs_point"],
        "normalization": metrics["weight_sum_error"]
        <= gates["maximum_weight_sum_error"],
        "positivity": metrics["minimum_weight"] >= gates["minimum_weight"],
        "rotational_symmetry": metrics["rotational_symmetry_error"]
        <= gates["maximum_rotational_symmetry_error"],
        "constant": metrics["constant_error"] <= gates["maximum_constant_error"],
        "partitions": metrics["partition_error"] <= gates["maximum_partition_error"],
        "repeat": metrics["repeat_error"] <= gates["maximum_repeat_error"],
        "impulse": metrics["impulse_error"] == 0.0,
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "source_report_sha256": config["source"]["source_report_sha256"],
        "kernel_sha256": hashlib.sha256(kernel.tobytes(order="C")).hexdigest(),
        "frequencies_cycles_per_mm": frequencies.tolist(),
        "analytic_mtf": target.tolist(),
        "compiled_kernel_mtf": candidate.tolist(),
        "gaussian_control_mtf": gaussian.tolist(),
        "metrics": metrics,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "retain_generic_circular_scanner_aperture_reference"
            if automatic_pass
            else "close_discrete_circular_scanner_aperture_compiler"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "CircularScannerAuditError",
    "evaluate_circular_scanner_aperture",
    "load_contract",
]
