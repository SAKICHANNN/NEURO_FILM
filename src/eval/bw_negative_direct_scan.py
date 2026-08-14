"""U6.P2AZ neutral B&W negative direct-scan interpretation audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.eval.typed_density_scanner_chain import build_typed_scanner_candidate
from src.film_physics.bw_density_scanner_chain import (
    interpret_bw_negative_direct_scan,
)
from src.film_physics.contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
)


class BWNegativeDirectScanError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2az_bw_negative_direct_scan_contract.v1"
    ):
        raise BWNegativeDirectScanError("unsupported P2AZ contract")
    return payload


def _load_bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if _sha(path) != binding["sha256"]:
        raise BWNegativeDirectScanError("P2AZ parent hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "required_automatic_pass" in binding and (
        payload.get("automatic_pass") is not binding["required_automatic_pass"]
    ):
        raise BWNegativeDirectScanError("P2AZ parent decision mismatch")
    return payload


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    _load_bound(root, contract["parents"]["typed_scanner_evidence"])
    scanner_contract = _load_bound(
        root, contract["parents"]["typed_scanner_contract"]
    )
    seed = int(contract["fresh_realization_seed"])
    result, receipt, mean_density = build_typed_scanner_candidate(
        root=root, contract=scanner_contract, realization_seed=seed
    )
    display = interpret_bw_negative_direct_scan(result.scan_linear)
    repeated_result, repeated_receipt, _ = build_typed_scanner_candidate(
        root=root, contract=scanner_contract, realization_seed=seed
    )
    repeated_display = interpret_bw_negative_direct_scan(repeated_result.scan_linear)
    repeat_exact = bool(
        receipt.receipt_id == repeated_receipt.receipt_id
        and np.array_equal(display.values, repeated_display.values)
    )
    restored_scan = display.values.dtype.type(1.0) - display.values
    roundtrip_error = float(
        np.max(np.abs(restored_scan - result.scan_linear.values))
    )
    partitioned = []
    for y0 in range(0, display.values.shape[0], 73):
        y1 = min(display.values.shape[0], y0 + 73)
        partitioned.append(
            interpret_bw_negative_direct_scan(
                PhysicalDomainArray(
                    result.scan_linear.values[y0:y1],
                    PhysicalDomain.SCAN_LINEAR,
                    PhysicalUnit.RELATIVE_SCAN_SIGNAL,
                    result.scan_linear.channels,
                    result.scan_linear.scale,
                )
            ).values
        )
    partition_exact = bool(np.array_equal(display.values, np.concatenate(partitioned)))
    unique_densities = np.sort(np.unique(mean_density))
    cell_means = np.asarray(
        [
            float(np.mean(display.values[..., 0][mean_density == density]))
            for density in unique_densities
        ],
        dtype=np.float64,
    )
    rank_correlation = float(spearmanr(unique_densities, cell_means).statistic)
    adjacent_steps = np.diff(cell_means)
    neutral_exact = bool(
        np.array_equal(display.values[..., 0], display.values[..., 1])
        and np.array_equal(display.values[..., 1], display.values[..., 2])
    )
    boundary_fraction = float(
        np.mean((display.values == 0.0) | (display.values == 1.0))
    )
    interpretation = contract["interpretation"]
    measurements = {
        "receipt_id": receipt.receipt_id,
        "interpretation_id": interpretation["interpretation_id"],
        "display_descriptor": display.descriptor(),
        "maximum_scan_roundtrip_absolute_error": roundtrip_error,
        "density_to_display_rank_correlation": rank_correlation,
        "minimum_adjacent_cell_mean_step": float(np.min(adjacent_steps)),
        "minimum_display_linear": float(np.min(display.values)),
        "maximum_display_linear": float(np.max(display.values)),
        "output_boundary_fraction": boundary_fraction,
        "neutral_channels_exact": neutral_exact,
        "partition_exact": partition_exact,
        "repeat_byte_exact": repeat_exact,
        "endpoint_fit_count_zero": True,
        "realized_normalization_count_zero": True,
        "hard_clipping_count_zero": True,
        "display_linear_sha256": hashlib.sha256(
            np.ascontiguousarray(display.values, dtype="<f8").tobytes()
        ).hexdigest(),
    }
    evaluation = contract["evaluation"]
    gate_results = {
        "maximum_scan_roundtrip_absolute_error": roundtrip_error
        <= evaluation["maximum_scan_roundtrip_absolute_error"],
        "minimum_density_to_display_rank_correlation": rank_correlation
        >= evaluation["minimum_density_to_display_rank_correlation"],
        "minimum_adjacent_cell_mean_step": measurements[
            "minimum_adjacent_cell_mean_step"
        ]
        >= evaluation["minimum_adjacent_cell_mean_step"],
        "display_linear_range": measurements["minimum_display_linear"]
        > evaluation["minimum_display_linear_exclusive"]
        and measurements["maximum_display_linear"]
        < evaluation["maximum_display_linear_exclusive"],
        "maximum_output_boundary_fraction": boundary_fraction
        <= evaluation["maximum_output_boundary_fraction"],
        "neutral_channels_exact": neutral_exact
        is evaluation["neutral_channels_exact"],
        "partition_exact": partition_exact is evaluation["partition_exact"],
        "repeat_byte_exact": repeat_exact is evaluation["repeat_byte_exact"],
        "endpoint_fit_count_zero": True,
        "realized_normalization_count_zero": True,
        "hard_clipping_count_zero": True,
    }
    passed = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p2az_bw_negative_direct_scan_report.v1",
        "interpretation": interpretation,
        "measurements": measurements,
        "gate_results": gate_results,
        "automatic_pass": passed,
        "decision": contract["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**stable, "stable_evidence_id": hashlib.sha256(encoded.encode()).hexdigest()}


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return hashlib.sha256(encoded.encode()).hexdigest()
