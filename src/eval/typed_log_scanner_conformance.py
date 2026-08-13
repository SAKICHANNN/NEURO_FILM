"""U6.P4IE typed compact log-scanner execution conformance."""

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
)

SCHEMA = "neuro-film.u6-p4ie-typed-log-scanner-conformance-contract.v1"
REPORT_SCHEMA = "neuro-film.u6-p4ie-typed-log-scanner-conformance-result.v1"


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
        raise ValueError("unsupported P4IE contract")
    return payload


def _compiler(row: Mapping[str, Any]) -> CompactLogScannerCompiler:
    return CompactLogScannerCompiler(
        compiler_id=str(row["compiler_id"]),
        matrix_density_to_log10_rgb=tuple(
            tuple(float(value) for value in values)
            for values in row["matrix_density_to_log10_rgb"]
        ),
        bias_log10_rgb=tuple(float(value) for value in row["bias_log10_rgb"]),
    )


def _cube(level_count: int, maximum: float) -> np.ndarray:
    levels = np.linspace(0.0, maximum, level_count, dtype=np.float32)
    return np.asarray(
        [(a, b, c) for a in levels for b in levels for c in levels],
        dtype=np.float32,
    )


def _oracle(values: np.ndarray, compiler: CompactLogScannerCompiler) -> np.ndarray:
    matrix = np.asarray(compiler.matrix_density_to_log10_rgb, dtype=np.float64)
    bias = np.asarray(compiler.bias_log10_rgb, dtype=np.float64)
    return np.power(10.0, values.astype(np.float64) @ matrix + bias)


def evaluate(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parent = config["parent"]
    evidence_path = root / str(parent["evidence"]["path"])
    implementation_path = root / str(parent["implementation"]["path"])
    if hash_file(evidence_path) != parent["evidence"]["sha256"]:
        raise ValueError("P4IE evidence integrity mismatch")
    if hash_file(implementation_path) != parent["implementation"]["sha256"]:
        raise ValueError("P4IE implementation integrity mismatch")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("decision") != parent["evidence"]["required_decision"]
        or evidence.get("formal_runs", {}).get("stable_evidence_id")
        != parent["evidence"]["required_stable_evidence_id"]
    ):
        raise ValueError("P4IE parent decision drift")

    compiler = _compiler(config["compiler"])
    population = config["population"]
    values = _cube(
        int(population["cube_levels"]), float(population["maximum_density"])
    )
    chunk_rows = int(population["chunk_rows"])
    output = np.empty_like(values)
    repeat = np.empty_like(values)
    maximum_oracle_error = 0.0
    digest = hashlib.sha256()
    repeat_digest = hashlib.sha256()
    for start in range(0, len(values), chunk_rows):
        stop = min(start + chunk_rows, len(values))
        block = apply_compact_log_scanner(values[start:stop], compiler)
        replay = apply_compact_log_scanner(values[start:stop], compiler)
        output[start:stop] = block
        repeat[start:stop] = replay
        maximum_oracle_error = max(
            maximum_oracle_error,
            float(np.max(np.abs(block.astype(np.float64) - _oracle(values[start:stop], compiler)))),
        )
        digest.update(np.ascontiguousarray(block).tobytes())
        repeat_digest.update(np.ascontiguousarray(replay).tobytes())

    axis_levels = np.linspace(
        0.0,
        float(population["maximum_density"]),
        int(population["axis_levels"]),
        dtype=np.float32,
    )
    maximum_positive_step = -np.inf
    axis_hashes: list[str] = []
    for channel in range(3):
        axis = np.zeros((len(axis_levels), 3), dtype=np.float32)
        axis[:, channel] = axis_levels
        axis_output = apply_compact_log_scanner(axis, compiler)
        maximum_positive_step = max(
            maximum_positive_step, float(np.max(np.diff(axis_output, axis=0)))
        )
        axis_hashes.append(hashlib.sha256(axis_output.tobytes()).hexdigest())

    invalid_rejected: dict[str, bool] = {}
    invalids = {
        "negative": np.full((2, 3), -0.01, dtype=np.float32),
        "nan": np.full((2, 3), np.nan, dtype=np.float32),
        "infinity": np.full((2, 3), np.inf, dtype=np.float32),
        "wrong_dtype": np.zeros((2, 3), dtype=np.float64),
        "wrong_shape": np.zeros((2, 2), dtype=np.float32),
    }
    for name, invalid in invalids.items():
        try:
            apply_compact_log_scanner(invalid, compiler)
        except (TypeError, ValueError):
            invalid_rejected[name] = True
        else:
            invalid_rejected[name] = False

    gates = config["gates"]
    repeat_error = float(np.max(np.abs(output - repeat)))
    minimum = float(np.min(output))
    maximum = float(np.max(output))
    decisions = {
        "float32_oracle": maximum_oracle_error
        <= float(gates["maximum_float32_vs_float64_absolute_error"]),
        "repeat_exact": repeat_error <= float(gates["maximum_repeat_absolute_error"])
        and digest.hexdigest() == repeat_digest.hexdigest(),
        "monotone": maximum_positive_step
        <= float(gates["maximum_monotonicity_positive_step"]),
        "finite": bool(np.all(np.isfinite(output)))
        is bool(gates["require_all_values_finite"]),
        "range": minimum > float(gates["minimum_output"])
        and maximum <= float(gates["maximum_output"]),
        "invalid_rejection": all(invalid_rejected.values())
        is bool(gates["require_all_invalid_inputs_rejected"]),
    }
    passed = all(decisions.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(config)).hexdigest(),
        "compiler_id": compiler.compiler_id,
        "cube_sample_count": len(values),
        "axis_sample_count_per_dye": len(axis_levels),
        "output_sha256": digest.hexdigest(),
        "repeat_output_sha256": repeat_digest.hexdigest(),
        "axis_output_sha256": axis_hashes,
        "maximum_float32_vs_float64_absolute_error": maximum_oracle_error,
        "maximum_repeat_absolute_error": repeat_error,
        "maximum_monotonicity_positive_step": maximum_positive_step,
        "output_minimum": minimum,
        "output_maximum": maximum,
        "invalid_inputs_rejected": invalid_rejected,
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
