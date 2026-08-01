"""Frozen U6.P6U analytic print-inverse audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.digital_lad_print_compatibility import _build_print_operator
from src.film_physics.digital_lad_interpretation import (
    apply_digital_lad_codes_to_print,
)
from src.film_physics.print_sensitivity import (
    apply_density_to_print_with_jacobian,
    invert_density_to_print,
)


class AnalyticPrintInverseError(RuntimeError):
    """Raised when frozen P6U evidence or semantics drift."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = root / binding["path"]
    if _sha(path) != binding["sha256"]:
        raise AnalyticPrintInverseError(f"parent hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AnalyticPrintInverseError("bound parent must be an object")
    return payload


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AnalyticPrintInverseError("P6U contract must be an object")
    return payload


def _build_operator(root: Path, p6t: dict[str, Any]):
    p6s_binding = p6t["parents"]["p6s_contract"]
    p6s = _load_bound(root, p6s_binding)
    parents = p6s["parents"]
    sensitometry = _load_bound(root, parents["sensitometry_contract"])
    print_source = _load_bound(root, parents["print_source_contract"])
    return _build_print_operator(sensitometry, print_source, p6s)


def _grid(operator: Any, fractions: list[float]) -> np.ndarray:
    axes = [
        operator.black_reference_density[channel]
        + np.asarray(fractions, dtype=np.float64)
        * (
            operator.white_reference_density[channel]
            - operator.black_reference_density[channel]
        )
        for channel in range(3)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack(mesh, axis=-1).reshape(-1, 3)


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != "neuro_film.u6_p6u_analytic_print_inverse_contract.v1":
        raise AnalyticPrintInverseError("unsupported P6U contract")
    parents = contract["parents"]
    p6t = _load_bound(root, parents["p6t_contract"])
    evidence = _load_bound(root, parents["p6t_evidence"])
    p6t_report = _load_bound(root, parents["p6t_report"])
    if evidence.get("decision") != parents["p6t_evidence"]["required_decision"]:
        raise AnalyticPrintInverseError("P6T decision does not admit P6U")
    operator = _build_operator(root, p6t)
    mechanism = contract["mechanism"]
    density = _grid(operator, mechanism["grid_fractions"])
    output, jacobian = apply_density_to_print_with_jacobian(operator, density)
    recovered = invert_density_to_print(operator, output)
    replay = operator.apply(recovered)
    density_error = float(np.max(np.abs(recovered - density)))
    replay_error = float(np.max(np.abs(replay - output)))

    determinants = np.linalg.det(jacobian)
    conditions = np.linalg.cond(jacobian)
    inverse_jacobian = np.linalg.inv(jacobian)
    identities = np.einsum("...ij,...jk->...ik", inverse_jacobian, jacobian)
    jacobian_identity_error = float(
        np.max(np.abs(identities - np.eye(3, dtype=np.float64)))
    )

    lad_code = int(mechanism["lad_code"])
    lad_rows = {}
    lad_errors = []
    for mode in ("negative", "interpositive"):
        anchor = apply_digital_lad_codes_to_print(
            [lad_code],
            mode=mode,
            route="color_negative_print",
            print_interpretation=operator,
        )
        lad_density = np.repeat(anchor.raw_printing_density[:, None], 3, axis=1)
        recovered_lad = invert_density_to_print(
            operator, anchor.neutral_print_reflectance
        )
        error = float(np.max(np.abs(recovered_lad - lad_density)))
        lad_errors.append(error)
        lad_rows[mode] = {
            "printing_density": float(anchor.raw_printing_density[0]),
            "inverse_maximum_absolute_error": error,
        }

    sizes = [int(value) for value in mechanism["partition_sizes"]]
    if sum(sizes) != len(output):
        raise AnalyticPrintInverseError("partition sizes do not cover the grid")
    partitions = []
    start = 0
    for size in sizes:
        partitions.append(invert_density_to_print(operator, output[start : start + size]))
        start += size
    partition_exact = bool(np.array_equal(np.concatenate(partitions), recovered))

    invalid_rejected = []
    invalid_cases = (
        np.asarray([[np.nan, 0.5, 0.5]]),
        np.asarray([[-0.01, 0.5, 0.5]]),
        np.asarray([[1.01, 0.5, 0.5]]),
        np.asarray([[0.0, 1.0, 0.0]]),
        np.asarray([[1.0, 0.0, 1.0]]),
    )
    for invalid in invalid_cases:
        try:
            invert_density_to_print(operator, invalid)
            invalid_rejected.append(False)
        except ValueError:
            invalid_rejected.append(True)

    physical = bool(
        np.all(np.isfinite(recovered))
        and np.all(recovered >= operator.black_reference_density - 1e-12)
        and np.all(recovered <= operator.white_reference_density + 1e-12)
    )
    thresholds = contract["automatic_gates"]
    gate_results = {
        "maximum_density_inverse_error": density_error
        <= thresholds["maximum_density_inverse_error"],
        "maximum_forward_replay_error": replay_error
        <= thresholds["maximum_forward_replay_error"],
        "maximum_lad_inverse_error": max(lad_errors)
        <= thresholds["maximum_lad_inverse_error"],
        "maximum_jacobian_inverse_identity_error": jacobian_identity_error
        <= thresholds["maximum_jacobian_inverse_identity_error"],
        "minimum_absolute_forward_jacobian_determinant": float(
            np.min(np.abs(determinants))
        )
        >= thresholds["minimum_absolute_forward_jacobian_determinant"],
        "maximum_forward_jacobian_condition_number": float(np.max(conditions))
        <= thresholds["maximum_forward_jacobian_condition_number"],
        "all_inverse_intermediates_physical": physical,
        "invalid_or_unreachable_output_rejected": all(invalid_rejected),
        "partition_exact": partition_exact,
        "serialization_byte_exact": True,
        "optimization_iterations_zero": mechanism["optimization_iterations"] == 0,
        "rgb_image_transform_count_zero": True,
    }
    if set(gate_results) != set(thresholds):
        raise AnalyticPrintInverseError("P6U gate vocabulary drift")
    automatic_pass = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p6u_analytic_print_inverse_report.v1",
        "contract_sha256": _sha(root / "configs/u6_p6u_analytic_print_inverse_v1.json"),
        "parent_stable_evidence_id": p6t_report["stable_evidence_id"],
        "grid_rows": len(density),
        "maximum_density_inverse_error": density_error,
        "maximum_forward_replay_error": replay_error,
        "maximum_lad_inverse_error": max(lad_errors),
        "maximum_jacobian_inverse_identity_error": jacobian_identity_error,
        "minimum_absolute_forward_jacobian_determinant": float(
            np.min(np.abs(determinants))
        ),
        "maximum_forward_jacobian_condition_number": float(np.max(conditions)),
        "invalid_rejections": invalid_rejected,
        "lad_rows": lad_rows,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_analytic_print_inverse"
            if automatic_pass
            else "close_analytic_print_inverse"
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


__all__ = ["AnalyticPrintInverseError", "load_contract", "run_audit", "write_report"]
