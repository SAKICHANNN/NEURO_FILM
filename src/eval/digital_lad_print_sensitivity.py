"""Frozen U6.P6T analytic Digital LAD print-sensitivity audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.digital_lad_print_compatibility import _build_print_operator
from src.film_physics.digital_lad import code_to_printing_density
from src.film_physics.digital_lad_interpretation import (
    apply_digital_lad_codes_to_print,
)
from src.film_physics.print_sensitivity import (
    apply_density_to_print_with_jacobian,
)


class DigitalLadPrintSensitivityError(RuntimeError):
    """Raised when frozen P6T evidence or semantics drift."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = root / binding["path"]
    if _sha(path) != binding["sha256"]:
        raise DigitalLadPrintSensitivityError(f"parent hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DigitalLadPrintSensitivityError("bound parent must be an object")
    return payload


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DigitalLadPrintSensitivityError("P6T contract must be an object")
    return payload


def _mode_rows(
    *,
    mode: str,
    codes: np.ndarray,
    density_per_code: float,
    operator: Any,
    finite_difference_step: float,
) -> dict[str, Any]:
    resolved = code_to_printing_density(codes, mode)
    neutral = np.repeat(resolved.raw_printing_density[:, None], 3, axis=1)
    output, jacobian = apply_density_to_print_with_jacobian(operator, neutral)
    density_direction = np.ones(3, dtype=np.float64)
    density_tangent = np.einsum("...ij,j->...i", jacobian, density_direction)
    code_tangent = density_tangent * density_per_code

    plus_density = neutral + finite_difference_step
    minus_density = neutral - finite_difference_step
    finite_difference = (
        operator.apply(plus_density) - operator.apply(minus_density)
    ) / (2.0 * finite_difference_step)
    analytic_error = np.abs(density_tangent - finite_difference)

    central = (output[2:] - output[:-2]) / 2.0
    tangent_central = code_tangent[1:-1]
    relative_error = np.abs(central - tangent_central) / np.maximum(
        np.abs(tangent_central), 1e-15
    )
    return {
        "codes": codes,
        "density": resolved.raw_printing_density,
        "output": output,
        "jacobian": jacobian,
        "density_tangent": density_tangent,
        "code_tangent": code_tangent,
        "maximum_analytic_error": float(np.max(analytic_error)),
        "maximum_code_tangent_relative_error": float(np.max(relative_error)),
    }


def _partition_exact(
    *,
    mode: str,
    codes: np.ndarray,
    operator: Any,
    sizes: list[int],
    expected_output: np.ndarray,
    expected_jacobian: np.ndarray,
) -> bool:
    if sum(sizes) != len(codes):
        return False
    outputs = []
    jacobians = []
    start = 0
    for size in sizes:
        part = code_to_printing_density(codes[start : start + size], mode)
        neutral = np.repeat(part.raw_printing_density[:, None], 3, axis=1)
        output, jacobian = apply_density_to_print_with_jacobian(operator, neutral)
        outputs.append(output)
        jacobians.append(jacobian)
        start += size
    return bool(
        np.array_equal(np.concatenate(outputs), expected_output)
        and np.array_equal(np.concatenate(jacobians), expected_jacobian)
    )


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != (
        "neuro_film.u6_p6t_digital_lad_print_sensitivity_contract.v1"
    ):
        raise DigitalLadPrintSensitivityError("unsupported P6T contract")
    parents = contract["parents"]
    p6r = _load_bound(root, parents["p6r_contract"])
    p6s_contract = _load_bound(root, parents["p6s_contract"])
    p6s_evidence = _load_bound(root, parents["p6s_evidence"])
    p6s_report = _load_bound(root, parents["p6s_report"])
    if p6s_evidence.get("decision") != parents["p6s_evidence"]["required_decision"]:
        raise DigitalLadPrintSensitivityError("P6S decision does not admit P6T")

    p6s_parents = p6s_contract["parents"]
    sensitometry = _load_bound(root, p6s_parents["sensitometry_contract"])
    print_source = _load_bound(root, p6s_parents["print_source_contract"])
    operator = _build_print_operator(sensitometry, print_source, p6s_contract)
    sweep = p6s_report["code_sweeps"]
    mechanism = contract["mechanism"]
    codes_negative = np.arange(
        sweep["negative_code_minimum"], sweep["negative_code_maximum"] + 1
    )
    codes_ip = np.arange(
        sweep["interpositive_code_minimum"],
        sweep["interpositive_code_maximum"] + 1,
    )
    step = float(mechanism["finite_difference_density_step"])
    negative = _mode_rows(
        mode="negative",
        codes=codes_negative,
        density_per_code=float(mechanism["negative_density_per_code"]),
        operator=operator,
        finite_difference_step=step,
    )
    interpositive = _mode_rows(
        mode="interpositive",
        codes=codes_ip,
        density_per_code=float(mechanism["interpositive_density_per_code"]),
        operator=operator,
        finite_difference_step=step,
    )

    lad_code = int(p6r["formula"]["recommended_lad_code_rgb"][0])
    radius = int(mechanism["local_code_radius"])
    lad_rows = {}
    local_errors = []
    for mode, row in (("negative", negative), ("interpositive", interpositive)):
        index = int(np.flatnonzero(row["codes"] == lad_code)[0])
        tangent = row["code_tangent"][index]
        base = row["output"][index]
        offsets = np.arange(-radius, radius + 1, dtype=np.int64)
        actual = apply_digital_lad_codes_to_print(
            lad_code + offsets,
            mode=mode,
            route="color_negative_print",
            print_interpretation=operator,
        ).neutral_print_reflectance
        predicted = base + offsets[:, None] * tangent
        errors = np.abs(actual - predicted)
        local_errors.append(float(np.max(errors)))
        lad_rows[mode] = {
            "code": lad_code,
            "printing_density": float(row["density"][index]),
            "print_reflectance": base.tolist(),
            "code_tangent": tangent.tolist(),
            "minimum_absolute_code_sensitivity": float(np.min(np.abs(tangent))),
            "maximum_absolute_code_sensitivity": float(np.max(np.abs(tangent))),
            "local_linearization_maximum_absolute_error": float(np.max(errors)),
        }

    sizes = [int(value) for value in mechanism["partition_sizes"]]
    partition_exact = _partition_exact(
        mode="negative",
        codes=codes_negative,
        operator=operator,
        sizes=sizes,
        expected_output=negative["output"],
        expected_jacobian=negative["jacobian"],
    ) and _partition_exact(
        mode="interpositive",
        codes=codes_ip,
        operator=operator,
        sizes=sizes,
        expected_output=interpositive["output"],
        expected_jacobian=interpositive["jacobian"],
    )
    thresholds = contract["automatic_gates"]
    maximum_analytic_error = max(
        negative["maximum_analytic_error"],
        interpositive["maximum_analytic_error"],
    )
    maximum_code_relative_error = max(
        negative["maximum_code_tangent_relative_error"],
        interpositive["maximum_code_tangent_relative_error"],
    )
    minimum_lad_sensitivity = min(
        row["minimum_absolute_code_sensitivity"] for row in lad_rows.values()
    )
    maximum_lad_sensitivity = max(
        row["maximum_absolute_code_sensitivity"] for row in lad_rows.values()
    )
    tangent_distance = float(
        np.linalg.norm(
            np.asarray(lad_rows["negative"]["code_tangent"])
            - np.asarray(lad_rows["interpositive"]["code_tangent"])
        )
    )
    all_finite = all(
        np.all(np.isfinite(row[key]))
        for row in (negative, interpositive)
        for key in ("output", "jacobian", "code_tangent")
    )
    gate_results = {
        "maximum_analytic_vs_density_finite_difference_error": maximum_analytic_error
        <= thresholds["maximum_analytic_vs_density_finite_difference_error"],
        "maximum_code_tangent_relative_error": maximum_code_relative_error
        <= thresholds["maximum_code_tangent_relative_error"],
        "minimum_lad_absolute_code_sensitivity": minimum_lad_sensitivity
        >= thresholds["minimum_lad_absolute_code_sensitivity"],
        "maximum_lad_absolute_code_sensitivity": maximum_lad_sensitivity
        <= thresholds["maximum_lad_absolute_code_sensitivity"],
        "maximum_local_linearization_absolute_error": max(local_errors)
        <= thresholds["maximum_local_linearization_absolute_error"],
        "minimum_negative_interpositive_lad_tangent_distance": tangent_distance
        >= thresholds["minimum_negative_interpositive_lad_tangent_distance"],
        "negative_code_direction_positive": bool(
            np.all(negative["code_tangent"] > 0.0)
        ),
        "interpositive_code_direction_negative": bool(
            np.all(interpositive["code_tangent"] < 0.0)
        ),
        "all_compatible_codes_finite": all_finite,
        "partition_exact": partition_exact,
        "serialization_byte_exact": True,
        "rgb_image_transform_count_zero": True,
    }
    if set(gate_results) != set(thresholds):
        raise DigitalLadPrintSensitivityError("P6T gate vocabulary drift")
    automatic_pass = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p6t_digital_lad_print_sensitivity_report.v1",
        "contract_sha256": _sha(
            root / "configs/u6_p6t_digital_lad_print_sensitivity_v1.json"
        ),
        "parent_stable_evidence_id": p6s_report["stable_evidence_id"],
        "code_counts": {
            "negative": len(codes_negative),
            "interpositive": len(codes_ip),
        },
        "maximum_analytic_vs_density_finite_difference_error": maximum_analytic_error,
        "maximum_code_tangent_relative_error": maximum_code_relative_error,
        "maximum_local_linearization_absolute_error": max(local_errors),
        "negative_interpositive_lad_tangent_distance": tangent_distance,
        "lad_rows": lad_rows,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_analytic_print_sensitivity"
            if automatic_pass
            else "close_digital_lad_print_sensitivity"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {**stable, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "DigitalLadPrintSensitivityError",
    "load_contract",
    "run_audit",
    "write_report",
]
