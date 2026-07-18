"""Reproduce the U1.6G4B batch-sensitive G1-v1 area-reduction audit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmfx import (  # noqa: E402
    StagedGlobalField,
    build_shape_stable_global_stage,
    plan_shape_stable_global_resample,
    reconstruct_shape_stable_full,
)
from src.filmfx.fast_blur import gaussian_filter_direct  # noqa: E402
from src.filmfx.global_resample import _area_downsample_axis  # noqa: E402


def _candidate_coarse(
    field: np.ndarray,
    sigma: float | tuple[float, float],
    coarse_row_chunk: int,
) -> tuple[np.ndarray, dict[str, int]]:
    plan = plan_shape_stable_global_resample(field.shape[:2], sigma)
    coarse_height, coarse_width = plan.coarse_shape
    source_height = field.shape[0]
    trailing = field.shape[2:]
    coarse = np.empty((coarse_height, coarse_width) + trailing, dtype=np.float32)
    scale = source_height / coarse_height
    calls = 0
    max_rows = 0
    for coarse_y0 in range(0, coarse_height, coarse_row_chunk):
        coarse_y1 = min(coarse_height, coarse_y0 + coarse_row_chunk)
        source_y0 = int(math.floor(coarse_y0 * scale))
        source_y1 = int(math.ceil(coarse_y1 * scale))
        rows = field[source_y0:source_y1]
        horizontal = _area_downsample_axis(rows, coarse_width, axis=1)
        for coarse_y in range(coarse_y0, coarse_y1):
            start = coarse_y * scale
            end = (coarse_y + 1) * scale
            first = int(math.floor(start))
            stop = int(math.ceil(end))
            cells = np.arange(first, stop, dtype=np.float64)
            weights = np.minimum(end, cells + 1.0) - np.maximum(start, cells)
            weights = (weights / (end - start)).astype(np.float32)
            coarse[coarse_y] = np.tensordot(
                weights,
                horizontal[first - source_y0 : stop - source_y0],
                axes=(0, 0),
            )
        calls += 1
        max_rows = max(max_rows, source_y1 - source_y0)
    resolved_sigma = plan.coarse_sigma + ((0.0,) if field.ndim == 3 else ())
    blurred = np.ascontiguousarray(
        gaussian_filter_direct(coarse, resolved_sigma, truncate=plan.truncate),
        dtype=np.float32,
    )
    blurred.setflags(write=False)
    return blurred, {"reader_calls": calls, "max_source_rows": max_rows}


def run(config: dict) -> dict:
    rows = []
    for case in config["cases"]:
        shape = tuple(int(value) for value in case["shape"])
        sigma_value = case["sigma"]
        sigma = (
            tuple(float(value) for value in sigma_value)
            if isinstance(sigma_value, list)
            else float(sigma_value)
        )
        field = np.random.default_rng(int(case["seed"])).normal(size=shape).astype(np.float32)
        plan = plan_shape_stable_global_resample(field.shape[:2], sigma)
        full_stage = build_shape_stable_global_stage(field, plan)
        full_output = reconstruct_shape_stable_full(full_stage)
        for chunk in case["coarse_row_chunks"]:
            candidate, metadata = _candidate_coarse(field, sigma, int(chunk))
            candidate_stage = StagedGlobalField(plan=plan, coarse=candidate)
            candidate_output = reconstruct_shape_stable_full(candidate_stage)
            coarse_difference = np.abs(candidate - full_stage.coarse)
            output_difference = np.abs(candidate_output - full_output)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "shape": shape,
                    "coarse_shape": plan.coarse_shape,
                    "coarse_row_chunk": int(chunk),
                    "reader_calls": metadata["reader_calls"],
                    "max_source_rows": metadata["max_source_rows"],
                    "reader_never_full_height": metadata["max_source_rows"] < shape[0],
                    "coarse_byte_identical": candidate.tobytes() == full_stage.coarse.tobytes(),
                    "coarse_max_error": float(coarse_difference.max()),
                    "coarse_different_values": int(
                        np.count_nonzero(candidate.view(np.uint32) != full_stage.coarse.view(np.uint32))
                    ),
                    "output_byte_identical": candidate_output.tobytes() == full_output.tobytes(),
                    "output_max_error": float(output_difference.max()),
                    "output_different_values": int(
                        np.count_nonzero(candidate_output.view(np.uint32) != full_output.view(np.uint32))
                    ),
                }
            )
    all_coarse = all(row["coarse_byte_identical"] for row in rows)
    all_output = all(row["output_byte_identical"] for row in rows)
    bounded = all(row["reader_never_full_height"] for row in rows)
    return {
        "schema_version": 1,
        "node": config["node"],
        "candidate": config["candidate"],
        "rows": rows,
        "gate_result": {
            "all_coarse_bytes_equal": all_coarse,
            "all_reconstructed_bytes_equal": all_output,
            "reader_call_never_full_height": bounded,
            "passed": all_coarse and all_output and bounded,
        },
        "decision": "pass" if all_coarse and all_output and bounded else config["failure_branch"],
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["gate_result"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
