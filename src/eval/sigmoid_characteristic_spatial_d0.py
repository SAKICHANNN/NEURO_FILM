"""U6.P4IJ explicit sigmoid characteristic spatial D0."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.characteristic_scanner_spatial_smoke import (
    _source,
    _structured_transmittance,
)
from src.eval.physical_characteristic_prior import compile_prior
from src.film_physics.compact_log_scanner_compiler import CompactLogScannerCompiler
from src.film_physics.sigmoid_characteristic import (
    fit_sigmoid_characteristic,
    render_sigmoid_scanner_positive,
)

SCHEMA = "neuro-film.u6-p4ij-sigmoid-characteristic-spatial-d0-contract.v1"
REPORT_SCHEMA = "neuro-film.u6-p4ij-sigmoid-characteristic-spatial-d0-result.v1"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported P4IJ contract")
    return payload


def _runtime(root: Path, contract: Mapping[str, Any]):
    trace = json.loads(
        (root / "configs/data/kodak_250d_characteristic_curve_pixels_v1.json").read_text()
    )
    prior, _ = compile_prior(trace, source_evidence_id="0" * 64)
    mechanism = contract["mechanism"]
    curves, fit_rmse = fit_sigmoid_characteristic(
        prior,
        samples=int(mechanism["fit_samples_per_layer"]),
        initial_slope=float(mechanism["initial_slope"]),
        maximum_iterations=int(mechanism["maximum_iterations"]),
    )
    row = json.loads(
        (root / "configs/u6_p4if_negative_scanner_inverse_d0_v2.json").read_text()
    )["compiler"]
    compiler = CompactLogScannerCompiler(
        row["compiler_id"],
        tuple(tuple(values) for values in row["matrix_density_to_log10_rgb"]),
        tuple(row["bias_log10_rgb"]),
    )
    return curves, fit_rmse, compiler


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    curves, fit_rmse, compiler = _runtime(root, contract)
    population = contract["population"]
    rows = []
    for index in range(int(population["images"])):
        source = _source(index, int(population["height"]), int(population["width"]))
        density = np.stack(
            [curve.apply_normalized(source[..., channel]) for channel, curve in enumerate(curves)],
            axis=-1,
        )
        base_transmittance = np.ascontiguousarray(np.power(10.0, -density), dtype=np.float32)
        baseline, _ = render_sigmoid_scanner_positive(
            source, base_transmittance, curves=curves, compiler=compiler
        )
        structured = _structured_transmittance(
            base_transmittance,
            index,
            float(contract["mechanism"]["structure_amplitude_unchanged_from_p4ii"]),
        )
        candidate, receipt = render_sigmoid_scanner_positive(
            source, structured, curves=curves, compiler=compiler
        )
        replay, _ = render_sigmoid_scanner_positive(
            source, structured, curves=curves, compiler=compiler
        )
        partition_error = 0.0
        for block in population["row_partitions"]:
            pieces = []
            for start in range(0, source.shape[0], int(block)):
                piece, _ = render_sigmoid_scanner_positive(
                    source[start : start + int(block)],
                    structured[start : start + int(block)],
                    curves=curves,
                    compiler=compiler,
                )
                pieces.append(piece)
            partition_error = max(
                partition_error,
                float(np.max(np.abs(np.concatenate(pieces, axis=0) - candidate))),
            )
        absolute = np.abs(candidate - baseline)
        epsilon = 1.0 / 65535.0
        source_boundary = (baseline <= epsilon) | (baseline >= 1.0 - epsilon)
        output_boundary = (candidate <= epsilon) | (candidate >= 1.0 - epsilon)
        rows.append(
            {
                "id": f"procedural-{index:02d}",
                "p95_abs_difference": float(np.percentile(absolute, 95)),
                "p99_abs_difference": float(np.percentile(absolute, 99)),
                "new_boundary_fraction": float(np.mean(output_boundary & ~source_boundary)),
                "partition_max_abs_error": partition_error,
                "repeat_max_abs_error": float(np.max(np.abs(candidate - replay))),
                "minimum_shared_scale": receipt["minimum_shared_scale"],
            }
        )
    metrics = {
        "curve_fit_rmse": list(fit_rmse),
        "minimum_input_003_headroom": min(
            float((curve.apply_normalized(np.asarray(0.03)) - curve.lower) / (curve.upper - curve.lower))
            for curve in curves
        ),
        "population_p95_abs_difference": float(np.percentile([r["p95_abs_difference"] for r in rows], 95)),
        "population_p99_abs_difference": float(np.percentile([r["p99_abs_difference"] for r in rows], 99)),
        "maximum_new_boundary_fraction": max(r["new_boundary_fraction"] for r in rows),
        "maximum_partition_error": max(r["partition_max_abs_error"] for r in rows),
        "maximum_repeat_error": max(r["repeat_max_abs_error"] for r in rows),
        "minimum_shared_scale": min(r["minimum_shared_scale"] for r in rows),
    }
    gates = contract["gates"]
    checks = {
        "fit": max(fit_rmse) <= gates["maximum_curve_fit_rmse"],
        "headroom": metrics["minimum_input_003_headroom"] >= gates["minimum_input_003_headroom"],
        "material": metrics["population_p95_abs_difference"] >= gates["minimum_population_p95_abs_difference"],
        "tail": metrics["population_p99_abs_difference"] <= gates["maximum_population_p99_abs_difference"],
        "boundary": metrics["maximum_new_boundary_fraction"] <= gates["maximum_new_boundary_fraction"],
        "partition": metrics["maximum_partition_error"] <= gates["maximum_partition_error"],
        "repeat": metrics["maximum_repeat_error"] <= gates["maximum_repeat_error"],
        "retained_direction": metrics["minimum_shared_scale"] >= gates["minimum_shared_scale"],
    }
    passed = all(checks.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "curve_parameters": [curve.__dict__ for curve in curves],
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": passed,
        "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = ["evaluate", "load_contract", "write_report"]
