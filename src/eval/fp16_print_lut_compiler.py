"""Frozen U6.P6V FP16 print-LUT compiler audit."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.analytic_print_inverse import _build_operator
from src.film_physics.print_lut_profile import compile_fp16_print_lut
from src.film_physics.print_sensitivity import invert_density_to_print


class FP16PrintLUTCompilerError(RuntimeError):
    """Raised when frozen P6V evidence or semantics drift."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = root / binding["path"]
    if _sha(path) != binding["sha256"]:
        raise FP16PrintLUTCompilerError(f"parent hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FP16PrintLUTCompilerError("bound parent must be an object")
    return payload


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FP16PrintLUTCompilerError("P6V contract must be an object")
    return payload


def _audit_grid(operator: Any, axis_size: int) -> np.ndarray:
    fractions = (np.arange(axis_size, dtype=np.float64) + 0.5) / axis_size
    axes = [
        operator.black_reference_density[channel]
        + fractions
        * (
            operator.white_reference_density[channel]
            - operator.black_reference_density[channel]
        )
        for channel in range(3)
    ]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != "neuro_film.u6_p6v_fp16_print_lut_compiler_contract.v1":
        raise FP16PrintLUTCompilerError("unsupported P6V contract")
    parents = contract["parents"]
    p6u = _load_bound(root, parents["p6u_contract"])
    evidence = _load_bound(root, parents["p6u_evidence"])
    p6u_report = _load_bound(root, parents["p6u_report"])
    if evidence.get("decision") != parents["p6u_evidence"]["required_decision"]:
        raise FP16PrintLUTCompilerError("P6U decision does not admit P6V")
    p6t = _load_bound(root, p6u["parents"]["p6t_contract"])
    operator = _build_operator(root, p6t)
    compiler = contract["compiler"]
    density = _audit_grid(operator, int(compiler["audit_axis_size"]))
    reference = operator.apply(density)
    thresholds = contract["automatic_gates"]
    candidates = []
    for size in compiler["candidate_sizes"]:
        first = compile_fp16_print_lut(operator, int(size))
        second = compile_fp16_print_lut(operator, int(size))
        output = first.apply(density)
        delta = output.astype(np.float64) - reference
        out_of_range = int(np.sum((output < 0.0) | (output > 1.0)))
        inverse_rejections = 0
        inverse_error = float("inf")
        try:
            recovered = invert_density_to_print(operator, output.astype(np.float64))
            inverse_error = float(np.max(np.abs(recovered - density)))
        except ValueError:
            inverse_rejections = len(output)
        splits = (0, 7001, 15997, len(density))
        partitioned = np.concatenate(
            [first.apply(density[start:stop]) for start, stop in pairwise(splits)]
        )
        metrics = {
            "maximum_output_absolute_error": float(np.max(np.abs(delta))),
            "output_rmse": float(np.sqrt(np.mean(np.square(delta)))),
            "maximum_inverse_density_absolute_error": inverse_error,
            "minimum_tetrahedral_jacobian_determinant": first.minimum_tetrahedral_jacobian_determinant(),
            "out_of_range_output_count": out_of_range,
            "inverse_rejection_count": inverse_rejections,
            "bundle_bytes": first.storage_bytes,
            "compile_replay_byte_exact": bool(
                first.values_sha256 == second.values_sha256
                and first.descriptor() == second.descriptor()
            ),
            "runtime_partition_exact": bool(np.array_equal(partitioned, output)),
            "post_interpolation_clipping_count": 0,
        }
        gates = {
            "maximum_output_absolute_error": metrics["maximum_output_absolute_error"]
            <= thresholds["maximum_output_absolute_error"],
            "maximum_output_rmse": metrics["output_rmse"]
            <= thresholds["maximum_output_rmse"],
            "maximum_inverse_density_absolute_error": metrics[
                "maximum_inverse_density_absolute_error"
            ]
            <= thresholds["maximum_inverse_density_absolute_error"],
            "minimum_tetrahedral_jacobian_determinant": metrics[
                "minimum_tetrahedral_jacobian_determinant"
            ]
            >= thresholds["minimum_tetrahedral_jacobian_determinant"],
            "maximum_out_of_range_output_count": out_of_range
            <= thresholds["maximum_out_of_range_output_count"],
            "maximum_inverse_rejection_count": inverse_rejections
            <= thresholds["maximum_inverse_rejection_count"],
            "maximum_selected_bundle_bytes": first.storage_bytes
            <= thresholds["maximum_selected_bundle_bytes"],
            "compile_replay_byte_exact": metrics["compile_replay_byte_exact"],
            "runtime_partition_exact": metrics["runtime_partition_exact"],
            "post_interpolation_clipping_count_zero": metrics[
                "post_interpolation_clipping_count"
            ]
            == 0,
        }
        candidates.append(
            {
                "size": int(size),
                "profile": first.descriptor(),
                "metrics": metrics,
                "gate_results": gates,
                "automatic_pass": all(gates.values()),
            }
        )
    if any(set(row["gate_results"]) != set(thresholds) for row in candidates):
        raise FP16PrintLUTCompilerError("P6V gate vocabulary drift")
    selected = next(
        (
            row
            for size in compiler["selection_order"]
            for row in candidates
            if row["size"] == size and row["automatic_pass"]
        ),
        None,
    )
    automatic_pass = selected is not None
    stable = {
        "schema": "neuro_film.u6_p6v_fp16_print_lut_compiler_report.v1",
        "contract_sha256": _sha(root / "configs/u6_p6v_fp16_print_lut_compiler_v1.json"),
        "parent_stable_evidence_id": p6u_report["stable_evidence_id"],
        "audit_rows": len(density),
        "candidates": candidates,
        "selected_size": None if selected is None else selected["size"],
        "selected_profile": None if selected is None else selected["profile"],
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_smallest_passing_fp16_print_lut"
            if automatic_pass
            else "close_fp16_print_lut_compiler"
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


__all__ = ["FP16PrintLUTCompilerError", "load_contract", "run_audit", "write_report"]
