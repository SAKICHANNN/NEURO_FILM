"""U6.P4IG characteristic-density to scanner-positive analytical chain."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_characteristic_prior import compile_prior
from src.film_physics.compact_log_scanner_compiler import (
    CompactLogScannerCompiler,
    apply_compact_log_scanner,
    invert_compact_log_scanner,
)

SCHEMA = "neuro-film.u6-p4ig-characteristic-scanner-chain-d0-contract.v1"
REPORT_SCHEMA = "neuro-film.u6-p4ig-characteristic-scanner-chain-d0-result.v1"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported P4IG contract")
    return payload


def _load_bound_json(root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(record["path"])
    if not path.is_file() or _hash_file(path) != record["sha256"]:
        raise ValueError("P4IG parent integrity mismatch")
    return json.loads(path.read_text(encoding="utf-8"))


def _compiler(row: Mapping[str, Any]) -> CompactLogScannerCompiler:
    return CompactLogScannerCompiler(
        compiler_id=str(row["compiler_id"]),
        matrix_density_to_log10_rgb=tuple(
            tuple(map(float, values)) for values in row["matrix_density_to_log10_rgb"]
        ),
        bias_log10_rgb=tuple(map(float, row["bias_log10_rgb"])),
    )


def evaluate(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parent_binding = config["parents"]["p4if_evidence"]
    parent = _load_bound_json(root, parent_binding)
    if (
        parent.get("decision") != parent_binding["required_decision"]
        or parent.get("formal_runs", {}).get("stable_evidence_id")
        != parent_binding["required_stable_evidence_id"]
    ):
        raise ValueError("P4IG parent decision drift")
    trace_binding = config["parents"]["characteristic_trace"]
    trace = _load_bound_json(root, trace_binding)
    prior, _ = compile_prior(trace, source_evidence_id=str(trace_binding["sha256"]))
    compiler = _compiler(config["compiler"])

    levels = np.linspace(0.0, 1.0, int(config["population"]["levels"]), dtype=np.float64)
    source = np.asarray(
        [(red, green, blue) for red in levels for green in levels for blue in levels],
        dtype=np.float64,
    )
    exposure = np.empty_like(source)
    for index, curve in enumerate(prior.curves):
        lower, upper = curve.domain
        exposure[:, index] = lower + source[:, index] * (upper - lower)
    density = prior.apply(exposure)
    normalized_density = np.empty_like(density)
    direct_transmittance = np.power(10.0, -density)
    endpoint_control = np.empty_like(density)
    for index, curve in enumerate(prior.curves):
        lower, upper = curve.density_bounds
        normalized_density[:, index] = (density[:, index] - lower) / (upper - lower)
        clear = 10.0**-lower
        maximum = 10.0**-upper
        endpoint_control[:, index] = (clear - direct_transmittance[:, index]) / (clear - maximum)

    scanner_density = np.ascontiguousarray(normalized_density[:, [2, 1, 0]], dtype=np.float32)
    scan = apply_compact_log_scanner(scanner_density, compiler)
    recovered = invert_compact_log_scanner(scan, compiler)
    positive = np.ascontiguousarray(recovered[:, [2, 1, 0]])
    truth = np.ascontiguousarray(normalized_density, dtype=np.float32)
    chain_error = np.abs(positive - truth)
    control_error = np.abs(endpoint_control - truth)
    materiality = np.abs(truth - source.astype(np.float32))

    monotonic_positive_steps: list[float] = []
    for channel, curve in enumerate(prior.curves):
        axis_source = np.zeros((len(levels), 3), dtype=np.float64)
        lower, upper = curve.domain
        axis_source[:, channel] = lower + levels * (upper - lower)
        for other, other_curve in enumerate(prior.curves):
            if other != channel:
                axis_source[:, other] = other_curve.domain[0]
        axis_density = prior.apply(axis_source)
        axis_amount = np.empty_like(axis_density)
        for index, item in enumerate(prior.curves):
            dlo, dhi = item.density_bounds
            axis_amount[:, index] = (axis_density[:, index] - dlo) / (dhi - dlo)
        axis_scan = apply_compact_log_scanner(
            np.ascontiguousarray(axis_amount[:, [2, 1, 0]], dtype=np.float32), compiler
        )
        axis_positive = invert_compact_log_scanner(axis_scan, compiler)[:, [2, 1, 0]]
        monotonic_positive_steps.append(float(np.max(-np.diff(axis_positive[:, channel]))))

    metrics = {
        "sample_count": len(source),
        "chain_max_abs_error": float(np.max(chain_error)),
        "monotonic_max_positive_violation": max(monotonic_positive_steps),
        "output_minimum": float(np.min(positive)),
        "output_maximum": float(np.max(positive)),
        "direct_transmittance_control_median_abs_error": float(np.median(control_error)),
        "direct_transmittance_control_p95_abs_error": float(np.percentile(control_error, 95)),
        "direct_transmittance_control_max_abs_error": float(np.max(control_error)),
        "characteristic_materiality_median_abs": float(np.median(materiality)),
        "characteristic_materiality_p95_abs": float(np.percentile(materiality, 95)),
        "characteristic_materiality_max_abs": float(np.max(materiality)),
        "scanner_raw_sha256": hashlib.sha256(scan.tobytes()).hexdigest(),
        "positive_rgb_sha256": hashlib.sha256(positive.tobytes()).hexdigest(),
    }
    gates = config["gates"]
    checks = {
        "chain_exact": metrics["chain_max_abs_error"] <= gates["maximum_chain_absolute_error"],
        "monotonic": metrics["monotonic_max_positive_violation"] <= gates["maximum_monotonic_positive_step"],
        "bounded": metrics["output_minimum"] >= -gates["maximum_output_boundary_error"]
        and metrics["output_maximum"] <= 1.0 + gates["maximum_output_boundary_error"],
        "direct_transmittance_control_rejected": metrics["direct_transmittance_control_p95_abs_error"]
        >= gates["minimum_direct_transmittance_control_p95_error"],
        "characteristic_material": metrics["characteristic_materiality_p95_abs"]
        >= gates["minimum_characteristic_materiality_p95"],
    }
    passed = all(checks.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(config)).hexdigest(),
        "prior_identity": prior.identity(),
        "compiler_id": compiler.compiler_id,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": passed,
        "decision": config["decision_if_pass"] if passed else config["decision_if_fail"],
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = ["evaluate", "load_contract", "write_report"]
