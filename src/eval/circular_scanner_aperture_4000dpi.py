"""U6.P6AB direct 4000-dpi circular scanner-aperture evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.eval.circular_scanner_aperture import evaluate_circular_scanner_aperture
from src.eval.physical_callier_source import hash_file

SCHEMA = "neuro_film.u6_p6ab_circular_scanner_aperture_4000dpi_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p6ab_circular_scanner_aperture_4000dpi_report.v1"


class CircularScanner4000DpiError(RuntimeError):
    """Raised when the P6AB target compiler contract or lineage drifts."""


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise CircularScanner4000DpiError("P6AB paths must be bounded and relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload.get("parent", {})
    source = payload.get("source", {})
    compiler = payload.get("compiler", {})
    evaluation = payload.get("evaluation", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or parent.get("decision_sha256")
        != "f3f6aac78ce8b69e784fe17b4c01406484b99ce0c76b243ce85c2a9415643048"
        or parent.get("stable_evidence_id")
        != "c2e6e1d9002577d17e25f749b4c9570b2cb9c2751ffb4bdcb2514e141e5f9305"
        or source.get("record_id") != "19730022682"
        or source.get("source_sha256")
        != "8773bb57946c38573d2feb42a08ddf3b0d5001745bc46ac23d825d6cdad90814"
        or source.get("source_report_sha256")
        != "53748b3eeb92fa3a9b103db50ea32f1bb2edba4278ae72e2a9b74f8aa2fd447b"
        or compiler.get("aperture_diameter_um") != 12.5
        or compiler.get("reference_pixel_pitch_um") != 6.35
        or compiler.get("subpixels_per_axis") != 1024
        or compiler.get("dtype") != "float64"
        or compiler.get("production_import_allowed") is not False
        or evaluation.get("frequencies_cycles_per_mm")
        != [0, 10, 20, 30, 40, 50, 60, 70, 78.74015748031496]
        or evaluation.get("equal_second_moment_gaussian_sigma_um") != 3.125
        or evaluation.get("test_shape") != [61, 47, 3]
        or evaluation.get("row_partitions") != [7, 17, 31]
        or evaluation.get("seed") != 26_080_262
        or gates.get("maximum_kernel_mtf_absolute_error") != 0.05
        or gates.get("minimum_rmse_improvement_vs_gaussian") != 0.5
        or gates.get("minimum_rmse_improvement_vs_point") != 0.8
        or gates.get("maximum_weight_sum_error") != 1e-12
        or gates.get("minimum_weight") != 0.0
        or gates.get("maximum_rotational_symmetry_error") != 1e-15
        or gates.get("maximum_constant_error") != 1e-12
        or gates.get("maximum_partition_error") != 1e-15
        or gates.get("maximum_repeat_error") != 0.0
        or not gates.get("impulse_must_equal_kernel")
    ):
        raise CircularScanner4000DpiError("P6AB frozen contract drift")
    _relative_path(parent.get("decision_path", ""))
    _relative_path(source.get("source_report_path", ""))
    return payload


def _verify_parent(config: dict[str, Any], root: Path) -> None:
    parent = config["parent"]
    path = root / _relative_path(parent["decision_path"])
    if not path.is_file() or hash_file(path) != parent["decision_sha256"]:
        raise CircularScanner4000DpiError("P6AB parent decision drift")
    decision = json.loads(path.read_text(encoding="utf-8"))
    if (
        decision.get("decision") != "retain_generic_circular_scanner_aperture_reference"
        or decision.get("stable_evidence_id") != parent["stable_evidence_id"]
    ):
        raise CircularScanner4000DpiError("P6AB parent facts drift")


def evaluate_circular_scanner_aperture_4000dpi(
    config: dict[str, Any], root: Path
) -> dict[str, Any]:
    _verify_parent(config, root)
    report = evaluate_circular_scanner_aperture(config, root)
    report["schema"] = REPORT_SCHEMA
    report["decision"] = (
        "retain_direct_4000dpi_circular_aperture_compiler"
        if report["automatic_pass"]
        else "close_direct_4000dpi_circular_aperture_compiler"
    )
    return report


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "CircularScanner4000DpiError",
    "evaluate_circular_scanner_aperture_4000dpi",
    "load_contract",
]
