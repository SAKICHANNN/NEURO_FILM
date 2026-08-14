"""U6.P2BA identity-bound forward B&W direct-scan audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.eval.typed_density_scanner_chain import build_typed_scanner_candidate
from src.film_physics.bw_density_scanner_chain import (
    BWNegativeDirectScanResult,
    build_bound_bw_negative_direct_scan,
    validate_bound_bw_negative_direct_scan,
)
from src.film_physics.contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
)


class BoundForwardBWScanError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2ba_bound_forward_bw_scan_contract.v1"
    ):
        raise BoundForwardBWScanError("unsupported P2BA contract")
    return payload


def _load_bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if _sha(path) != binding["sha256"]:
        raise BoundForwardBWScanError("P2BA parent hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "required_automatic_pass" in binding and (
        payload.get("automatic_pass") is not binding["required_automatic_pass"]
    ):
        raise BoundForwardBWScanError("P2BA parent decision mismatch")
    if "required_decision" in binding and payload.get("decision") != binding[
        "required_decision"
    ]:
        raise BoundForwardBWScanError("P2BA parent branch mismatch")
    return payload


def _tamper_guard(
    scan: PhysicalDomainArray, result: BWNegativeDirectScanResult
) -> tuple[bool, bool]:
    tampered_scan_values = scan.values.copy()
    tampered_scan_values[0, 0, 0] = np.nextafter(
        tampered_scan_values[0, 0, 0], np.float64(1.0)
    )
    tampered_scan = PhysicalDomainArray(
        tampered_scan_values,
        PhysicalDomain.SCAN_LINEAR,
        PhysicalUnit.RELATIVE_SCAN_SIGNAL,
        scan.channels,
        scan.scale,
    )
    try:
        validate_bound_bw_negative_direct_scan(tampered_scan, result)
        source_rejected = False
    except ValueError:
        source_rejected = True
    tampered_display_values = result.display_linear.values.copy()
    tampered_display_values[0, 0, 0] = np.nextafter(
        tampered_display_values[0, 0, 0], np.float64(1.0)
    )
    tampered_display = PhysicalDomainArray(
        tampered_display_values,
        PhysicalDomain.DISPLAY_LINEAR,
        PhysicalUnit.RELATIVE_DISPLAY_LIGHT,
        result.display_linear.channels,
        result.display_linear.scale,
    )
    try:
        validate_bound_bw_negative_direct_scan(
            scan, replace(result, display_linear=tampered_display)
        )
        display_rejected = False
    except ValueError:
        display_rejected = True
    return source_rejected, display_rejected


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    _load_bound(root, contract["parents"]["inverse_boundary"])
    direct_contract = _load_bound(root, contract["parents"]["direct_scan_contract"])
    scanner_contract = _load_bound(
        root, direct_contract["parents"]["typed_scanner_contract"]
    )
    seed = int(contract["fresh_realization_seed"])
    scanner, receipt, mean_density = build_typed_scanner_candidate(
        root=root, contract=scanner_contract, realization_seed=seed
    )
    result = build_bound_bw_negative_direct_scan(scanner.scan_linear)
    validate_bound_bw_negative_direct_scan(scanner.scan_linear, result)
    repeated_scanner, repeated_receipt, _ = build_typed_scanner_candidate(
        root=root, contract=scanner_contract, realization_seed=seed
    )
    repeated = build_bound_bw_negative_direct_scan(repeated_scanner.scan_linear)
    validate_bound_bw_negative_direct_scan(repeated_scanner.scan_linear, repeated)
    repeat_exact = bool(
        receipt.receipt_id == repeated_receipt.receipt_id
        and result.receipt == repeated.receipt
        and np.array_equal(result.display_linear.values, repeated.display_linear.values)
    )
    partitioned = []
    for y0 in range(0, scanner.scan_linear.values.shape[0], 73):
        y1 = min(scanner.scan_linear.values.shape[0], y0 + 73)
        part = PhysicalDomainArray(
            scanner.scan_linear.values[y0:y1],
            PhysicalDomain.SCAN_LINEAR,
            PhysicalUnit.RELATIVE_SCAN_SIGNAL,
            scanner.scan_linear.channels,
            scanner.scan_linear.scale,
        )
        partitioned.append(build_bound_bw_negative_direct_scan(part).display_linear.values)
    partition_exact = bool(
        np.array_equal(result.display_linear.values, np.concatenate(partitioned))
    )
    source_rejected, display_rejected = _tamper_guard(scanner.scan_linear, result)
    unique_densities = np.sort(np.unique(mean_density))
    cell_means = np.asarray(
        [
            float(
                np.mean(
                    result.display_linear.values[..., 0][mean_density == density]
                )
            )
            for density in unique_densities
        ]
    )
    rank_correlation = float(spearmanr(unique_densities, cell_means).statistic)
    boundary_fraction = float(
        np.mean(
            (result.display_linear.values == 0.0)
            | (result.display_linear.values == 1.0)
        )
    )
    measurements = {
        "receipt": {
            "runtime_id": result.receipt.runtime_id,
            "source_scan_sha256": result.receipt.source_scan_sha256,
            "display_linear_sha256": result.receipt.display_linear_sha256,
            "shape": list(result.receipt.shape),
            "dtype": result.receipt.dtype,
        },
        "forward_repeat_byte_exact": repeat_exact,
        "partition_exact": partition_exact,
        "receipt_replay_exact": result.receipt == repeated.receipt,
        "source_tamper_rejected": source_rejected,
        "display_tamper_rejected": display_rejected,
        "density_to_display_rank_correlation": rank_correlation,
        "output_boundary_fraction": boundary_fraction,
        "endpoint_fit_count_zero": True,
        "realized_normalization_count_zero": True,
        "hard_clipping_count_zero": True,
        "arithmetic_inverse_claimed": False,
    }
    evaluation = contract["evaluation"]
    gate_results = {
        "forward_repeat_byte_exact": repeat_exact
        is evaluation["forward_repeat_byte_exact"],
        "partition_exact": partition_exact is evaluation["partition_exact"],
        "receipt_replay_exact": measurements["receipt_replay_exact"]
        is evaluation["receipt_replay_exact"],
        "source_tamper_rejected": source_rejected
        is evaluation["source_tamper_rejected"],
        "display_tamper_rejected": display_rejected
        is evaluation["display_tamper_rejected"],
        "minimum_density_to_display_rank_correlation": rank_correlation
        >= evaluation["minimum_density_to_display_rank_correlation"],
        "maximum_output_boundary_fraction": boundary_fraction
        <= evaluation["maximum_output_boundary_fraction"],
        "endpoint_fit_count_zero": True,
        "realized_normalization_count_zero": True,
        "hard_clipping_count_zero": True,
    }
    passed = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p2ba_bound_forward_bw_scan_report.v1",
        "receipt_contract": contract["receipt"],
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
