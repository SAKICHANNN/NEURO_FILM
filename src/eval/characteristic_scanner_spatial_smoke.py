"""U6.P4II procedural spatial smoke of the complete scanner chain."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_characteristic_prior import compile_prior
from src.film_physics.characteristic_scanner_chain import (
    render_characteristic_scanner_positive,
)
from src.film_physics.compact_log_scanner_compiler import CompactLogScannerCompiler
from src.film_physics.relative_display_characteristic_ingress import (
    relative_display_to_finite_density_transmittance,
)

SCHEMA = "neuro-film.u6-p4ii-characteristic-scanner-spatial-smoke-contract.v1"
REPORT_SCHEMA = "neuro-film.u6-p4ii-characteristic-scanner-spatial-smoke-result.v1"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported P4II contract")
    return payload


def _runtime(root: Path):
    trace_path = root / "configs/data/kodak_250d_characteristic_curve_pixels_v1.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    prior, _ = compile_prior(trace, source_evidence_id=trace_sha)
    config = json.loads(
        (root / "configs/u6_p4if_negative_scanner_inverse_d0_v2.json").read_text(
            encoding="utf-8"
        )
    )
    row = config["compiler"]
    compiler = CompactLogScannerCompiler(
        row["compiler_id"],
        tuple(tuple(values) for values in row["matrix_density_to_log10_rgb"]),
        tuple(row["bias_log10_rgb"]),
    )
    return prior, compiler


def _source(index: int, height: int, width: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    xf = x / max(width - 1, 1)
    yf = y / max(height - 1, 1)
    phase = 0.31 * index
    red = 0.08 + 0.78 * xf + 0.05 * np.sin(7.0 * yf + phase)
    green = 0.1 + 0.72 * yf + 0.06 * np.cos(5.0 * xf - phase)
    blue = 0.12 + 0.62 * (0.55 * xf + 0.45 * yf)
    source = np.stack((red, green, blue), axis=-1)
    source[y > height * (0.58 + 0.01 * (index % 3)), 1] += 0.08
    return np.ascontiguousarray(np.clip(source, 0.03, 0.97), dtype=np.float32)


def _structured_transmittance(
    base: np.ndarray, index: int, amplitude: float
) -> np.ndarray:
    height, width = base.shape[:2]
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    field = np.sin((x + 11 * index) / 23.0) * np.cos((y - 7 * index) / 19.0)
    direction = np.asarray((0.8, -0.55, 0.35), dtype=np.float64)
    residual = float(amplitude) * field[..., None] * direction
    return np.ascontiguousarray(
        base.astype(np.float64) * np.power(10.0, -residual), dtype=np.float32
    )


def evaluate(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    prior, compiler = _runtime(root)
    population = config["population"]
    rows = []
    for index in range(int(population["images"])):
        source = _source(index, int(population["height"]), int(population["width"]))
        _, base_transmittance, _ = relative_display_to_finite_density_transmittance(
            source, prior
        )
        baseline, _ = render_characteristic_scanner_positive(
            source, base_transmittance, prior=prior, compiler=compiler
        )
        structured = _structured_transmittance(
            base_transmittance, index, float(population["density_residual_amplitude"])
        )
        candidate, receipt = render_characteristic_scanner_positive(
            source, structured, prior=prior, compiler=compiler
        )
        replay, _ = render_characteristic_scanner_positive(
            source, structured, prior=prior, compiler=compiler
        )
        partition_error = 0.0
        for block in population["row_partitions"]:
            pieces = []
            for start in range(0, source.shape[0], int(block)):
                piece, _ = render_characteristic_scanner_positive(
                    source[start : start + int(block)],
                    structured[start : start + int(block)],
                    prior=prior,
                    compiler=compiler,
                )
                pieces.append(piece)
            joined = np.concatenate(pieces, axis=0)
            partition_error = max(partition_error, float(np.max(np.abs(joined - candidate))))
        absolute = np.abs(candidate - baseline)
        epsilon = 1.0 / 65535.0
        base_boundary = (baseline <= epsilon) | (baseline >= 1.0 - epsilon)
        candidate_boundary = (candidate <= epsilon) | (candidate >= 1.0 - epsilon)
        rows.append(
            {
                "id": f"procedural-{index:02d}",
                "source_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
                "candidate_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
                "p95_abs_difference": float(np.percentile(absolute, 95)),
                "p99_abs_difference": float(np.percentile(absolute, 99)),
                "new_boundary_fraction": float(np.mean(candidate_boundary & ~base_boundary)),
                "partition_max_abs_error": partition_error,
                "repeat_max_abs_error": float(np.max(np.abs(candidate - replay))),
                "minimum_shared_scale": receipt["minimum_shared_scale"],
                "limited_fraction": receipt["limited_fraction"],
            }
        )
    metrics = {
        "row_count": len(rows),
        "population_p95_abs_difference": float(
            np.percentile([row["p95_abs_difference"] for row in rows], 95)
        ),
        "population_p99_abs_difference": float(
            np.percentile([row["p99_abs_difference"] for row in rows], 99)
        ),
        "maximum_new_boundary_fraction": max(row["new_boundary_fraction"] for row in rows),
        "maximum_partition_error": max(row["partition_max_abs_error"] for row in rows),
        "maximum_repeat_error": max(row["repeat_max_abs_error"] for row in rows),
        "minimum_shared_scale": min(row["minimum_shared_scale"] for row in rows),
    }
    gates = config["gates"]
    checks = {
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
        "experiment_id": config["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(config)).hexdigest(),
        "rows": rows,
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
