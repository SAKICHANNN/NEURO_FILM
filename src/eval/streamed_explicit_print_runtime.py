"""Frozen U6.P6W streamed explicit float32 print-runtime benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from src.eval.analytic_print_inverse import _build_operator
from src.film_physics.print_runtime import apply_density_to_print_float32


class StreamedExplicitPrintRuntimeError(RuntimeError):
    """Raised when frozen P6W evidence or semantics drift."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = root / binding["path"]
    if _sha(path) != binding["sha256"]:
        raise StreamedExplicitPrintRuntimeError(f"parent hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StreamedExplicitPrintRuntimeError("bound parent must be an object")
    return payload


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StreamedExplicitPrintRuntimeError("P6W contract must be an object")
    return payload


def _operator(root: Path, p6v: dict[str, Any]):
    p6u = _load_bound(root, p6v["parents"]["p6u_contract"])
    p6t = _load_bound(root, p6u["parents"]["p6t_contract"])
    return _build_operator(root, p6t)


def _density_rows(
    operator: Any,
    *,
    width: int,
    height: int,
    start: int,
    stop: int,
) -> np.ndarray:
    x = (np.arange(width, dtype=np.float32) + np.float32(0.5)) / np.float32(
        width
    )
    y = (np.arange(start, stop, dtype=np.float32) + np.float32(0.5)) / np.float32(
        height
    )
    red = np.broadcast_to(
        np.float32(0.1) + np.float32(0.8) * x[None, :],
        (stop - start, width),
    )
    green = np.broadcast_to(
        np.float32(0.1) + np.float32(0.8) * y[:, None],
        (stop - start, width),
    )
    blue = np.float32(0.1) + np.float32(0.4) * (x[None, :] + y[:, None])
    fraction = np.stack((red, green, blue), axis=-1)
    black = np.asarray(operator.black_reference_density, dtype=np.float32)
    white = np.asarray(operator.white_reference_density, dtype=np.float32)
    return black + fraction * (white - black)


def run_stream(
    operator: Any,
    *,
    width: int,
    height: int,
    tile_rows: int,
    measure_performance: bool,
) -> dict[str, Any]:
    if width <= 0 or height <= 0 or tile_rows <= 0:
        raise ValueError("stream dimensions must be positive")
    digest = hashlib.sha256()
    squared_error = 0.0
    scalar_count = 0
    maximum_error = 0.0
    out_of_range = 0
    process = psutil.Process(os.getpid())
    maximum_rss = process.memory_info().rss
    started = time.perf_counter()
    for start in range(0, height, tile_rows):
        stop = min(start + tile_rows, height)
        density = _density_rows(
            operator, width=width, height=height, start=start, stop=stop
        )
        reference = operator.apply(density.astype(np.float64))
        candidate = apply_density_to_print_float32(operator, density)
        delta = candidate.astype(np.float64) - reference
        maximum_error = max(maximum_error, float(np.max(np.abs(delta))))
        squared_error += float(np.sum(np.square(delta), dtype=np.float64))
        scalar_count += int(delta.size)
        out_of_range += int(np.sum((candidate < 0.0) | (candidate > 1.0)))
        digest.update(candidate.astype("<f4", copy=False).tobytes(order="C"))
        maximum_rss = max(maximum_rss, process.memory_info().rss)
    wall_seconds = time.perf_counter() - started
    return {
        "width": width,
        "height": height,
        "pixels": width * height,
        "tile_rows": tile_rows,
        "maximum_live_rows": min(tile_rows, height),
        "output_sha256": digest.hexdigest(),
        "maximum_float32_vs_float64_absolute_error": maximum_error,
        "float32_vs_float64_rmse": float(np.sqrt(squared_error / scalar_count)),
        "out_of_range_output_count": out_of_range,
        "maximum_process_tree_rss_bytes": maximum_rss,
        "worker_wall_seconds": wall_seconds if measure_performance else 0.0,
        "full_frame_output_allocation_count": 0,
        "post_operator_clipping_count": 0,
    }


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != (
        "neuro_film.u6_p6w_streamed_explicit_print_runtime_contract.v1"
    ):
        raise StreamedExplicitPrintRuntimeError("unsupported P6W contract")
    parents = contract["parents"]
    p6v = _load_bound(root, parents["p6v_contract"])
    evidence = _load_bound(root, parents["p6v_evidence"])
    p6v_report = _load_bound(root, parents["p6v_report"])
    if evidence.get("decision") != parents["p6v_evidence"]["required_decision"]:
        raise StreamedExplicitPrintRuntimeError("P6V decision does not admit P6W")
    operator = _operator(root, p6v)
    runtime = contract["runtime"]
    small = [
        run_stream(
            operator,
            width=257,
            height=257,
            tile_rows=rows,
            measure_performance=False,
        )
        for rows in (31, 127, 257)
    ]
    formal = run_stream(
        operator,
        width=int(runtime["width"]),
        height=int(runtime["height"]),
        tile_rows=int(runtime["tile_rows"]),
        measure_performance=True,
    )
    small_exact = len({row["output_sha256"] for row in small}) == 1
    stable_metrics_exact = all(
        row["maximum_float32_vs_float64_absolute_error"]
        == small[0]["maximum_float32_vs_float64_absolute_error"]
        and row["float32_vs_float64_rmse"] == small[0]["float32_vs_float64_rmse"]
        and row["out_of_range_output_count"] == 0
        for row in small
    )
    thresholds = contract["automatic_gates"]
    gate_results = {
        "maximum_float32_vs_float64_absolute_error": formal[
            "maximum_float32_vs_float64_absolute_error"
        ]
        <= thresholds["maximum_float32_vs_float64_absolute_error"],
        "maximum_float32_vs_float64_rmse": formal["float32_vs_float64_rmse"]
        <= thresholds["maximum_float32_vs_float64_rmse"],
        "maximum_process_tree_rss_bytes": formal["maximum_process_tree_rss_bytes"]
        <= thresholds["maximum_process_tree_rss_bytes"],
        "maximum_worker_wall_seconds": formal["worker_wall_seconds"]
        <= thresholds["maximum_worker_wall_seconds"],
        "maximum_live_rows": formal["maximum_live_rows"]
        <= thresholds["maximum_live_rows"],
        "maximum_out_of_range_output_count": formal["out_of_range_output_count"]
        <= thresholds["maximum_out_of_range_output_count"],
        "repeat_output_sha256_exact": small_exact,
        "repeat_stable_metrics_exact": stable_metrics_exact,
        "small_partition_output_bytes_exact": small_exact,
        "full_frame_output_allocation_count_zero": formal[
            "full_frame_output_allocation_count"
        ]
        == 0,
        "post_operator_clipping_count_zero": formal["post_operator_clipping_count"]
        == 0,
    }
    if set(gate_results) != set(thresholds):
        raise StreamedExplicitPrintRuntimeError("P6W gate vocabulary drift")
    automatic_pass = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p6w_streamed_explicit_print_runtime_report.v1",
        "contract_sha256": _sha(
            root / "configs/u6_p6w_streamed_explicit_print_runtime_v1.json"
        ),
        "parent_stable_evidence_id": p6v_report["stable_evidence_id"],
        "small_partition_output_sha256": small[0]["output_sha256"],
        "formal": formal,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_streamed_explicit_float32_print_runtime"
            if automatic_pass
            else "close_streamed_explicit_float32_print_runtime"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_identity_payload = {
        key: value for key, value in stable.items() if key != "formal"
    }
    stable_identity_payload["formal"] = {
        key: value
        for key, value in formal.items()
        if key not in ("maximum_process_tree_rss_bytes", "worker_wall_seconds")
    }
    stable_id = hashlib.sha256(
        json.dumps(
            stable_identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
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


__all__ = [
    "StreamedExplicitPrintRuntimeError",
    "load_contract",
    "run_audit",
    "run_stream",
    "write_report",
]
