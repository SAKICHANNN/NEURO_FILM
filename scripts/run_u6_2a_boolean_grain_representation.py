#!/usr/bin/env python
"""Run the frozen U6.2A Boolean/Poisson grain representation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmfx.boolean_grain import (  # noqa: E402
    BooleanGrainContext,
    build_boolean_grain_context,
    render_boolean_grain,
    render_boolean_grain_region,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _array_sha256(values: np.ndarray) -> str:
    return _sha256(np.ascontiguousarray(values).tobytes())


def _lag1_autocorrelation(values: np.ndarray) -> float:
    pairs = (
        (values[:, :-1].reshape(-1), values[:, 1:].reshape(-1)),
        (values[:-1, :].reshape(-1), values[1:, :].reshape(-1)),
    )
    correlations = []
    for left, right in pairs:
        left_centered = left - np.mean(left)
        right_centered = right - np.mean(right)
        denominator = float(
            np.sqrt(
                np.sum(left_centered * left_centered)
                * np.sum(right_centered * right_centered)
            )
        )
        correlations.append(
            float(np.sum(left_centered * right_centered) / denominator)
            if denominator > 0.0
            else 0.0
        )
    return float(np.mean(correlations))


def _build_context(
    config: dict[str, Any],
    *,
    radius: float,
    level: float,
) -> BooleanGrainContext:
    candidate = config["candidate"]
    shape = tuple(int(value) for value in config["audit"]["flat_input_shape"])
    return build_boolean_grain_context(
        np.full(shape, level, dtype=np.float64),
        radius_input_pixels=radius,
        monte_carlo_samples=int(candidate["monte_carlo_samples"]),
        gaussian_filter_sigma_output_pixels=float(
            candidate["gaussian_filter_sigma_output_pixels"]
        ),
        maximum_input_intensity=float(candidate["maximum_input_intensity"]),
        epsilon=float(candidate["epsilon"]),
        seed=int(candidate["seed"]),
    )


def run_audit(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    candidate = config["candidate"]
    audit = config["audit"]
    gates = config["gates"]
    zoom = int(candidate["output_zoom"])
    margin = int(audit["metric_interior_margin_input_pixels"]) * zoom
    radius_policies = {
        "small": float(candidate["small_radius_input_pixels"]),
        "large": float(candidate["large_radius_input_pixels"]),
    }
    levels = [float(value) for value in audit["flat_levels"]]
    reports: dict[str, dict[str, Any]] = {}
    contexts: dict[tuple[str, float], BooleanGrainContext] = {}
    outputs: dict[tuple[str, float], np.ndarray] = {}

    for radius_name, radius in radius_policies.items():
        reports[radius_name] = {}
        for level in levels:
            context = _build_context(config, radius=radius, level=level)
            output = render_boolean_grain(context, output_zoom=zoom)
            interior = output[margin:-margin, margin:-margin]
            if not interior.size:
                raise RuntimeError("metric interior is empty")
            contexts[(radius_name, level)] = context
            outputs[(radius_name, level)] = output
            reports[radius_name][f"{level:.1f}"] = {
                "grain_count": context.grain_count,
                "context_fingerprint": context.fingerprint(),
                "output_sha256": _array_sha256(output),
                "output_minimum": float(np.min(output)),
                "output_maximum": float(np.max(output)),
                "interior_mean": float(np.mean(interior, dtype=np.float64)),
                "interior_mean_absolute_error": abs(
                    float(np.mean(interior, dtype=np.float64)) - level
                ),
                "interior_variance": float(np.var(interior, dtype=np.float64)),
                "interior_lag1_autocorrelation": _lag1_autocorrelation(interior),
            }

    mid_key = min(levels, key=lambda value: abs(value - 0.5))
    primary_context = contexts[("small", mid_key)]
    primary_output = outputs[("small", mid_key)]
    repeat_output = render_boolean_grain(primary_context, output_zoom=zoom)
    height, width = primary_output.shape
    cuts = [0] + [int(value) for value in audit["partition_rows"]] + [height]
    if cuts != sorted(set(cuts)) or cuts[0] != 0 or cuts[-1] != height:
        raise ValueError("partition rows must be unique increasing interior cuts")
    partitioned = np.concatenate(
        [
            render_boolean_grain_region(
                primary_context,
                output_zoom=zoom,
                output_origin_yx=(start, 0),
                output_shape=(end - start, width),
            )
            for start, end in zip(cuts[:-1], cuts[1:], strict=True)
        ]
    )
    replay_context = BooleanGrainContext.from_dict(
        json.loads(json.dumps(primary_context.to_dict(), sort_keys=True))
    )
    replay_output = render_boolean_grain(replay_context, output_zoom=zoom)

    mean_errors = [
        report["interior_mean_absolute_error"]
        for radius_report in reports.values()
        for report in radius_report.values()
    ]
    variance_ratios = {}
    for radius_name, radius_report in reports.items():
        endpoint_maximum = max(
            radius_report[f"{levels[0]:.1f}"]["interior_variance"],
            radius_report[f"{levels[-1]:.1f}"]["interior_variance"],
        )
        variance_ratios[radius_name] = (
            radius_report[f"{mid_key:.1f}"]["interior_variance"]
            / endpoint_maximum
        )
    autocorrelation_delta = (
        reports["large"][f"{mid_key:.1f}"]["interior_lag1_autocorrelation"]
        - reports["small"][f"{mid_key:.1f}"]["interior_lag1_autocorrelation"]
    )
    grain_counts = [
        report["grain_count"]
        for radius_report in reports.values()
        for report in radius_report.values()
    ]
    repeat_error = float(np.max(np.abs(repeat_output - primary_output)))
    partition_error = float(np.max(np.abs(partitioned - primary_output)))
    replay_error = float(np.max(np.abs(replay_output - primary_output)))
    checks = {
        "bounded": all(
            report["output_minimum"] >= gates["output_minimum"]
            and report["output_maximum"] <= gates["output_maximum"]
            for radius_report in reports.values()
            for report in radius_report.values()
        ),
        "flat_mean": max(mean_errors)
        <= gates["maximum_flat_mean_absolute_error"],
        "midtone_variance": min(variance_ratios.values())
        >= gates["minimum_midtone_variance_ratio_over_endpoint_maximum"],
        "radius_autocorrelation": autocorrelation_delta
        >= gates["minimum_large_minus_small_radius_lag1_autocorrelation"],
        "grain_count": min(grain_counts) >= gates["minimum_grain_count"]
        and max(grain_counts) <= gates["maximum_total_grain_count"],
        "repeat_exact": repeat_error <= gates["repeat_maximum_absolute_error"],
        "partition_exact": partition_error
        <= gates["partition_maximum_absolute_error"],
        "serialization_replay": replay_error
        <= gates["serialization_replay_maximum_absolute_error"],
    }
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": _git_commit(),
        "config_sha256": config_sha256,
        "radius_level_reports": reports,
        "maximum_flat_mean_absolute_error": max(mean_errors),
        "midtone_variance_ratios": variance_ratios,
        "large_minus_small_radius_lag1_autocorrelation": autocorrelation_delta,
        "minimum_grain_count": min(grain_counts),
        "maximum_grain_count": max(grain_counts),
        "repeat_maximum_absolute_error": repeat_error,
        "partition_maximum_absolute_error": partition_error,
        "serialization_replay_maximum_absolute_error": replay_error,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_2a_boolean_grain_representation_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/u6_2a_boolean_grain_representation_v1/report.json",
    )
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    report = run_audit(json.loads(config_bytes), _sha256(config_bytes))
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(encoded),
                "all_checks_passed": report["all_checks_passed"],
                "checks": report["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
