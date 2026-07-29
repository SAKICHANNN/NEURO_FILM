"""Frozen U6.P3J functional evaluation of row-streamed FFT cores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    apply_fft_backing_return,
    backing_return_profile_from_contract,
    compile_backing_return_profile,
    iter_fft_backing_return_row_cores,
)


P3D_SCHEMA = "neuro_film.u6_p3d_backing_return_reference_contract.v1"
P3J_SCHEMA = "neuro_film.u6_p3j_streamed_fft_backing_return_contract.v1"


def load_json(path: Path, schema: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != schema:
        raise ValueError(f"unsupported contract: expected {schema}")
    return payload


def _array(values: np.ndarray, pitch: float) -> PhysicalDomainArray:
    return PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red-sensitive", "green-sensitive", "blue-sensitive"),
        PhysicalScale(pitch),
    )


def _collect(
    source: PhysicalDomainArray,
    profile: Any,
    *,
    tile_rows: int,
    order: str,
) -> tuple[np.ndarray, list[list[int]], int]:
    output = np.empty_like(source.values)
    coverage = np.zeros(source.values.shape[0], dtype=np.uint8)
    ranges = []
    maximum_core_rows = 0
    for y0, y1, core in iter_fft_backing_return_row_cores(
        source, profile, tile_rows=tile_rows, order=order
    ):
        output[y0:y1] = core
        coverage[y0:y1] += 1
        ranges.append([y0, y1])
        maximum_core_rows = max(maximum_core_rows, core.shape[0])
    if not np.all(coverage == 1):
        raise RuntimeError("streamed FFT row coverage is not exactly once")
    return output, ranges, maximum_core_rows


def evaluate_streamed_fft_backing_return(
    parent_contract: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    if parent_contract.get("schema") != P3D_SCHEMA:
        raise ValueError("invalid P3D parent contract")
    if contract.get("schema") != P3J_SCHEMA:
        raise ValueError("invalid P3J contract")
    reference = backing_return_profile_from_contract(parent_contract)
    compiled = compile_backing_return_profile(reference)
    if compiled.parent_profile_sha256 != contract["parents"]["p3d_profile_sha256"]:
        raise ValueError("P3D profile identity drift")
    witnesses = contract["witnesses"]
    gates = contract["automatic_gates"]
    shape = tuple(witnesses["shape"])
    values = np.random.default_rng(int(witnesses["random_seed"])).random(
        shape, dtype=np.float32
    )
    source = _array(values, compiled.pixel_pitch_um)
    full = apply_fft_backing_return(source, compiled).values

    partition_metrics: dict[str, Any] = {}
    maximum_error = 0.0
    maximum_mean_error = 0.0
    maximum_core_rows = 0
    all_order_exact = True
    all_repeat_exact = True
    all_coverage = True
    minimum_output = float("inf")
    minimum_direct_increment = float("inf")
    for tile_rows in witnesses["partition_rows"]:
        forward, forward_ranges, forward_core_rows = _collect(
            source, compiled, tile_rows=int(tile_rows), order="forward"
        )
        reverse, reverse_ranges, reverse_core_rows = _collect(
            source, compiled, tile_rows=int(tile_rows), order="reverse"
        )
        repeat, repeat_ranges, repeat_core_rows = _collect(
            source, compiled, tile_rows=int(tile_rows), order="forward"
        )
        difference = forward.astype(np.float64) - full.astype(np.float64)
        max_error = float(np.max(np.abs(difference)))
        mean_error = float(np.mean(np.abs(difference)))
        order_exact = np.array_equal(forward, reverse)
        repeat_exact = np.array_equal(forward, repeat)
        expected_ranges = [
            [y0, min(shape[0], y0 + int(tile_rows))]
            for y0 in range(0, shape[0], int(tile_rows))
        ]
        coverage_exact = (
            forward_ranges == expected_ranges
            and reverse_ranges == list(reversed(expected_ranges))
            and repeat_ranges == expected_ranges
        )
        maximum_core = max(
            forward_core_rows, reverse_core_rows, repeat_core_rows
        )
        partition_metrics[str(tile_rows)] = {
            "full_frame_max_abs_error": max_error,
            "full_frame_mean_abs_error": mean_error,
            "forward_reverse_exact": order_exact,
            "repeat_exact": repeat_exact,
            "coverage_exactly_once": coverage_exact,
            "maximum_live_output_core_rows": maximum_core,
        }
        maximum_error = max(maximum_error, max_error)
        maximum_mean_error = max(maximum_mean_error, mean_error)
        maximum_core_rows = max(maximum_core_rows, maximum_core)
        all_order_exact = all_order_exact and order_exact
        all_repeat_exact = all_repeat_exact and repeat_exact
        all_coverage = all_coverage and coverage_exact
        minimum_output = min(minimum_output, float(np.min(forward)))
        minimum_direct_increment = min(
            minimum_direct_increment, float(np.min(forward - values))
        )
    metrics = {
        "partition_metrics": partition_metrics,
        "full_frame_max_abs_error": maximum_error,
        "full_frame_mean_abs_error": maximum_mean_error,
        "forward_reverse_exact_within_partition": all_order_exact,
        "repeat_exact_within_partition_order": all_repeat_exact,
        "coverage_exactly_once": all_coverage,
        "maximum_live_output_core_rows": maximum_core_rows,
        "minimum_output": minimum_output,
        "minimum_direct_increment": minimum_direct_increment,
        "full_output_allocation_in_iterator": False,
    }
    decisions = {
        "max_error": metrics["full_frame_max_abs_error"]
        <= float(gates["full_frame_max_abs_error"]),
        "mean_error": metrics["full_frame_mean_abs_error"]
        <= float(gates["full_frame_mean_abs_error"]),
        "order": bool(metrics["forward_reverse_exact_within_partition"]),
        "repeat": bool(metrics["repeat_exact_within_partition_order"]),
        "coverage": bool(metrics["coverage_exactly_once"]),
        "core_bound": metrics["maximum_live_output_core_rows"]
        <= int(gates["maximum_live_output_core_rows"]),
        "nonnegative": metrics["minimum_output"] >= float(gates["minimum_output"]),
        "direct_retention": metrics["minimum_direct_increment"]
        >= float(gates["minimum_direct_increment"]),
        "no_full_output": metrics["full_output_allocation_in_iterator"] is False,
    }
    passed = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p3j_streamed_fft_backing_return_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "parent_profile_sha256": compiled.parent_profile_sha256,
        "metrics": metrics,
        "decisions": decisions,
        "automatic_pass": passed,
        "branch": contract["branch_rule"]["pass" if passed else "fail"],
    }
    stable_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
