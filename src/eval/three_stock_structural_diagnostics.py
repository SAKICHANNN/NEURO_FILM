"""Deterministic structural review queues for existing RF3.D0 renders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.filmcase.diagnostics import (
    chroma_speckle_diagnostics,
    structural_render_diagnostics,
)


class ThreeStockStructuralDiagnosticError(ValueError):
    """Raised when the frozen diagnostic inputs do not match."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_exact_json(root: Path, spec: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    path = root / spec["path"]
    data = path.read_bytes()
    actual = _sha256_bytes(data)
    if actual != spec["sha256"]:
        raise ThreeStockStructuralDiagnosticError(
            f"input hash mismatch: {spec['path']}"
        )
    return json.loads(data), {
        "path": spec["path"],
        "bytes": len(data),
        "sha256": actual,
    }


def _read_rgb8(path: Path, expected_sha256: str) -> tuple[np.ndarray, dict[str, Any]]:
    data = path.read_bytes()
    actual = _sha256_bytes(data)
    if actual != expected_sha256:
        raise ThreeStockStructuralDiagnosticError(f"PNG hash mismatch: {path}")
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB":
            raise ThreeStockStructuralDiagnosticError(f"expected RGB8 PNG: {path}")
        array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ThreeStockStructuralDiagnosticError(f"invalid RGB geometry: {path}")
    return array, {
        "bytes": len(data),
        "sha256": actual,
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
    }


def _queue(
    rows: list[dict[str, Any]], metric: str, count: int, *, reverse: bool
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["metrics"][metric])
            if reverse
            else float(row["metrics"][metric]),
            row["source_id"],
            row["arm_id"],
        ),
    )
    return [
        {
            "source_id": row["source_id"],
            "arm_id": row["arm_id"],
            "metric": metric,
            "value": row["metrics"][metric],
        }
        for row in ordered[:count]
    ]


def run_diagnostics(
    config: dict[str, Any], root: Path, *, reverse: bool = False
) -> dict[str, Any]:
    inputs = config["inputs"]
    _contract, contract_fact = _read_exact_json(root, inputs["rf3_d0_contract"])
    report, report_fact = _read_exact_json(root, inputs["rf3_d0_report"])
    manifest, manifest_fact = _read_exact_json(root, inputs["source_manifest"])
    source_rows = {row["id"]: row for row in manifest}
    render_rows = list(report["rows"])
    if reverse:
        render_rows.reverse()
    required_sources = int(inputs["required_sources"])
    required_arms = sorted(inputs["required_arms"])
    render_source_ids = [row["source_id"] for row in render_rows]
    if len(render_rows) != required_sources or len(set(render_source_ids)) != required_sources:
        raise ThreeStockStructuralDiagnosticError("source count mismatch")
    if any(source_id not in source_rows for source_id in render_source_ids):
        raise ThreeStockStructuralDiagnosticError("source identities mismatch")

    diagnostic_config = config["diagnostics"]
    rows: list[dict[str, Any]] = []
    render_root = root / inputs["render_root"]
    for render_row in render_rows:
        source_id = render_row["source_id"]
        source_record = source_rows[source_id]
        source, source_fact = _read_rgb8(
            root / source_record["decoded_path"], source_record["decoded_sha256"]
        )
        if source.shape[:2] != (render_row["height"], render_row["width"]):
            raise ThreeStockStructuralDiagnosticError("source geometry mismatch")
        arm_items = list(render_row["outputs"].items())
        if reverse:
            arm_items.reverse()
        if sorted(arm for arm, _ in arm_items) != required_arms:
            raise ThreeStockStructuralDiagnosticError("render arm mismatch")
        for arm_id, output_record in arm_items:
            output, output_fact = _read_rgb8(
                render_root / output_record["relative_path"],
                output_record["png_sha256"],
            )
            structural = structural_render_diagnostics(
                source,
                output,
                source_edge_percentile=float(
                    diagnostic_config["sobel_source_edge_percentile"]
                ),
                texture_sigma=float(diagnostic_config["texture_sigma"]),
                texture_source_percentile=float(
                    diagnostic_config["texture_source_percentile"]
                ),
                flat_source_percentile=float(
                    diagnostic_config["flat_source_percentile"]
                ),
                minimum_denominator=float(diagnostic_config["minimum_denominator"]),
            )
            speckle = chroma_speckle_diagnostics(source, output)
            metrics = {
                key: value
                for key, value in structural.items()
                if key not in {"diagnostic_only", "review_instruction"}
            }
            metrics.update(
                {
                    "speckle_candidate_percent": speckle["speckle_candidate_percent"],
                    "speckle_candidate_pixel_count": speckle[
                        "speckle_candidate_pixel_count"
                    ],
                    "speckle_small_island_count": speckle["small_island_count"],
                    "speckle_largest_island_pixels": speckle["largest_island_pixels"],
                }
            )
            rows.append(
                {
                    "source_id": source_id,
                    "arm_id": arm_id,
                    "source": source_fact,
                    "output": {
                        **output_fact,
                        "relative_path": output_record["relative_path"],
                    },
                    "metrics": metrics,
                }
            )

    rows.sort(key=lambda row: (row["source_id"], row["arm_id"]))
    queue_count = int(diagnostic_config["review_queue_rows_per_metric"])
    review_queues = {
        "lowest_edge_direction_cosine_p05": _queue(
            rows, "edge_direction_cosine_p05", queue_count, reverse=False
        ),
        "lowest_edge_gain_p05": _queue(
            rows, "edge_gain_p05", queue_count, reverse=False
        ),
        "highest_edge_gain_p95": _queue(
            rows, "edge_gain_p95", queue_count, reverse=True
        ),
        "lowest_texture_gain_median": _queue(
            rows, "texture_gain_median", queue_count, reverse=False
        ),
        "highest_texture_gain_p95": _queue(
            rows, "texture_gain_p95", queue_count, reverse=True
        ),
        "highest_flat_new_high_frequency_p99": _queue(
            rows, "flat_new_high_frequency_p99", queue_count, reverse=True
        ),
        "highest_speckle_candidate_percent": _queue(
            rows, "speckle_candidate_percent", queue_count, reverse=True
        ),
    }
    numeric_metrics = [
        value
        for row in rows
        for value in row["metrics"].values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    gates = {
        "exact_input_hashes": True,
        "all_sources_present": len({row["source_id"] for row in rows})
        == required_sources,
        "all_arms_present": all(
            {row["arm_id"] for row in rows if row["source_id"] == source_id}
            == set(required_arms)
            for source_id in render_source_ids
        ),
        "finite_metrics": bool(numeric_metrics)
        and bool(np.isfinite(np.asarray(numeric_metrics, dtype=np.float64)).all()),
        "render_calls": 0 <= int(config["gates"]["render_calls_max"]),
        "network_reads": 0 <= int(config["gates"]["network_reads_max"]),
    }
    scientific_payload = {
        "inputs": [contract_fact, report_fact, manifest_fact],
        "rows": rows,
        "review_queues": review_queues,
        "gates": gates,
        "aggregate_scalar_score_present": False,
        "automatic_veto_present": False,
    }
    passed = all(gates.values())
    return {
        "schema": "neuro-film.u4-3a-three-stock-structural-diagnostics-result.v1",
        "experiment_id": config["experiment_id"],
        "status": "PASS_DIAGNOSTIC_REVIEW_QUEUES" if passed else "FAIL_CLOSED",
        "scientific_payload": scientific_payload,
        "scientific_identity": _sha256_bytes(_canonical_bytes(scientific_payload)),
        "render_calls": 0,
        "network_reads": 0,
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = ["ThreeStockStructuralDiagnosticError", "run_diagnostics"]
