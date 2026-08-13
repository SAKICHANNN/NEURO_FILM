"""U6.P4IF analytical negative scanner inverse D0."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.compact_log_scanner_compiler import (
    CompactLogScannerCompiler,
    apply_compact_log_scanner,
    interpret_negative_scan_relative,
    invert_compact_log_scanner,
)

SCHEMA = "neuro-film.u6-p4if-negative-scanner-inverse-d0-contract.v2"
REPORT_SCHEMA = "neuro-film.u6-p4if-negative-scanner-inverse-d0-result.v2"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported P4IF contract")
    return payload


def _compiler(row: Mapping[str, Any]) -> CompactLogScannerCompiler:
    return CompactLogScannerCompiler(
        compiler_id=str(row["compiler_id"]),
        matrix_density_to_log10_rgb=tuple(tuple(map(float, values)) for values in row["matrix_density_to_log10_rgb"]),
        bias_log10_rgb=tuple(map(float, row["bias_log10_rgb"])),
    )


def evaluate(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    binding = config["parent"]["evidence"]
    path = root / str(binding["path"])
    if hash_file(path) != binding["sha256"]:
        raise ValueError("P4IF parent integrity mismatch")
    parent = json.loads(path.read_text(encoding="utf-8"))
    if (
        parent.get("decision") != binding["required_decision"]
        or parent.get("formal_runs", {}).get("stable_evidence_id")
        != binding["required_stable_evidence_id"]
    ):
        raise ValueError("P4IF parent decision drift")
    compiler = _compiler(config["compiler"])
    population = config["population"]
    levels = np.linspace(
        0.0,
        float(population["maximum_density"]),
        int(population["levels"]),
        dtype=np.float32,
    )
    density = np.asarray(
        [(a, b, c) for a in levels for b in levels for c in levels],
        dtype=np.float32,
    )
    scan = apply_compact_log_scanner(density, compiler)
    recovered = invert_compact_log_scanner(scan, compiler)
    positive_truth = np.ascontiguousarray(density[:, [2, 1, 0]])
    positive = np.ascontiguousarray(recovered[:, [2, 1, 0]])

    clear = apply_compact_log_scanner(np.zeros((1, 3), dtype=np.float32), compiler)[0]
    maximum = apply_compact_log_scanner(np.ones((1, 3), dtype=np.float32), compiler)[0]
    endpoint_control = interpret_negative_scan_relative(
        scan, clear_scan_rgb=clear, maximum_density_scan_rgb=maximum
    )
    inverse_error = np.abs(recovered - density)
    positive_error = np.abs(positive - positive_truth)
    control_error = np.abs(endpoint_control - positive_truth)

    neutral = np.asarray([(level, level, level) for level in levels], dtype=np.float32)
    neutral_positive = invert_compact_log_scanner(
        apply_compact_log_scanner(neutral, compiler), compiler
    )[:, [2, 1, 0]]
    primaries = np.eye(3, dtype=np.float32)
    primary_positive = invert_compact_log_scanner(
        apply_compact_log_scanner(primaries[:, [2, 1, 0]], compiler), compiler
    )[:, [2, 1, 0]]
    expected_primary = primaries
    # A unit cyan/magenta/yellow dye amount records red/green/blue exposure.
    # Inverting a negative therefore maps C/M/Y amounts to positive R/G/B.
    primary_error = float(np.max(np.abs(primary_positive - expected_primary)))

    gates = config["gates"]
    metrics = {
        "sample_count": len(density),
        "inverse_density_max_abs": float(np.max(inverse_error)),
        "positive_rgb_max_abs": float(np.max(positive_error)),
        "endpoint_control_median_abs": float(np.median(control_error)),
        "endpoint_control_p95_abs": float(np.percentile(control_error, 95)),
        "endpoint_control_max_abs": float(np.max(control_error)),
        "neutral_rgb_spread_max": float(np.max(np.ptp(neutral_positive, axis=1))),
        "primary_channel_order_max_abs": primary_error,
        "positive_rgb_minimum": float(np.min(positive)),
        "positive_rgb_maximum": float(np.max(positive)),
        "scan_sha256": hashlib.sha256(scan.tobytes()).hexdigest(),
        "recovered_density_sha256": hashlib.sha256(recovered.tobytes()).hexdigest(),
        "positive_rgb_sha256": hashlib.sha256(positive.tobytes()).hexdigest(),
    }
    decisions = {
        "inverse_density": metrics["inverse_density_max_abs"] <= gates["maximum_inverse_density_absolute_error"],
        "positive_rgb": metrics["positive_rgb_max_abs"] <= gates["maximum_positive_rgb_absolute_error"],
        "endpoint_control_is_insufficient": metrics["endpoint_control_p95_abs"] >= gates["minimum_endpoint_control_p95_absolute_error"],
        "neutral": metrics["neutral_rgb_spread_max"] <= gates["maximum_neutral_rgb_spread"],
        "channel_order": metrics["primary_channel_order_max_abs"] <= gates["maximum_primary_off_channel_absolute_error"],
        "boundary": metrics["positive_rgb_minimum"] >= -gates["maximum_positive_rgb_boundary_error"]
        and metrics["positive_rgb_maximum"] <= 1.0 + gates["maximum_positive_rgb_boundary_error"],
    }
    passed = all(decisions.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(config)).hexdigest(),
        "compiler_id": compiler.compiler_id,
        "metrics": metrics,
        "decisions": decisions,
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
