"""U6.P2U gauge-free shared manufacturer characteristic-shape audit."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_characteristic_diversity import (
    CHANNELS,
    STOCKS,
    _normalized_shape,
    _value_from_pixel,
    load_trace,
)

SCHEMA = "neuro_film.u6_p2u_shared_characteristic_shape_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p2u_shared_characteristic_shape_report.v1"
BUNDLE_SCHEMA = "neuro_film.gauge_free_characteristic_shape_bundle.v1"


class SharedCharacteristicShapeError(RuntimeError):
    """Raised when the frozen P2U evidence or protocol drifts."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise SharedCharacteristicShapeError("P2U paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload.get("parent", {})
    protocol = payload.get("protocol", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != "U6.P2U"
        or parent.get("decision_sha256")
        != "9cd529029ad2495382750c6b61d3c0c167cdbb559ddc0cdecb0671411a14e6db"
        or parent.get("stable_evidence_id")
        != "cdff0ee6c2178e4ee7088840c1b27695bc3345a0d689f312b6bdbcba3d9ed496"
        or parent.get("trace_sha256")
        != "e7d9d45cd56066d1cc770c3a506117fcddc4ae52417c6d69d4088e8d0da61dfb"
        or tuple(protocol.get("stocks", ())) != STOCKS
        or tuple(protocol.get("channels", ())) != CHANNELS
        or protocol.get("grid_min") != -0.05
        or protocol.get("grid_max") != 1.05
        or protocol.get("grid_samples") != 257
        or protocol.get("baseline")
        != "canonical straight 10-to-90 line clip(0.1 + 0.8*x, 0, 1)"
        or gates
        != {
            "required_loso_rows": 9,
            "minimum_candidate_win_rows": 7,
            "minimum_candidate_wins_per_channel": 2,
            "minimum_median_rmse_improvement_fraction": 0.2,
            "minimum_p95_rmse_improvement_fraction": 0.0,
            "maximum_loso_rmse": 0.025,
            "maximum_final_uncertainty_width": 0.05,
            "maximum_anchor_error": 1e-12,
            "minimum_template_step": 0.0,
            "two_byte_identical_audits": True,
        }
    ):
        raise SharedCharacteristicShapeError("P2U frozen contract drift")
    for key in ("decision_path", "trace_path"):
        _relative_path(str(parent.get(key, "")))
    return payload


def _frozen_grid(protocol: Mapping[str, Any]) -> np.ndarray:
    grid = np.linspace(
        float(protocol["grid_min"]),
        float(protocol["grid_max"]),
        int(protocol["grid_samples"]),
        dtype=np.float64,
    )
    grid[int(np.argmin(np.abs(grid)))] = 0.0
    grid[int(np.argmin(np.abs(grid - 1.0)))] = 1.0
    if not np.all(np.diff(grid) > 0.0):
        raise SharedCharacteristicShapeError("P2U invalid frozen grid")
    return grid


def _load_normalized_shapes(
    trace: Mapping[str, Any], grid: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    shapes: dict[str, dict[str, np.ndarray]] = {}
    for stock in STOCKS:
        row = trace["stocks"][stock]
        axes = row["graph_axes"]
        shapes[stock] = {}
        for channel in CHANNELS:
            coordinates = np.asarray(row["curves"][channel], dtype=np.float64)
            exposure = np.asarray(
                [_value_from_pixel(float(x), axes["x_value_pixels"]) for x in coordinates[:, 0]],
                dtype=np.float64,
            )
            density = np.asarray(
                [_value_from_pixel(float(y), axes["y_value_pixels"]) for y in coordinates[:, 1]],
                dtype=np.float64,
            )
            x_norm, y_norm, _, _, _ = _normalized_shape(exposure, density)
            if grid[0] < x_norm[0] or grid[-1] > x_norm[-1]:
                raise SharedCharacteristicShapeError(f"P2U grid outside trace: {stock}/{channel}")
            shapes[stock][channel] = np.interp(grid, x_norm, y_norm)
    return shapes


def evaluate_shared_shape(
    config: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent = config["parent"]
    decision_path = root / _relative_path(str(parent["decision_path"]))
    trace_path = root / _relative_path(str(parent["trace_path"]))
    if (
        not decision_path.is_file()
        or _hash_file(decision_path) != parent["decision_sha256"]
        or not trace_path.is_file()
        or _hash_file(trace_path) != parent["trace_sha256"]
    ):
        raise SharedCharacteristicShapeError("P2U parent evidence integrity mismatch")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        decision.get("formal_evidence", {}).get("stable_evidence_id")
        != parent["stable_evidence_id"]
        or decision.get("decision")
        != "retain_one_shared_manufacturer_shape_prior_and_close_stock_specific_shape_bank"
    ):
        raise SharedCharacteristicShapeError("P2U parent decision mismatch")
    trace = load_trace(trace_path)
    protocol = config["protocol"]
    gates = config["gates"]
    grid = _frozen_grid(protocol)
    shapes = _load_normalized_shapes(trace, grid)
    baseline = np.clip(0.1 + 0.8 * grid, 0.0, 1.0)

    rows: list[dict[str, Any]] = []
    candidate_errors: list[float] = []
    baseline_errors: list[float] = []
    wins_by_channel = {channel: 0 for channel in CHANNELS}
    for held_stock in STOCKS:
        development = tuple(stock for stock in STOCKS if stock != held_stock)
        for channel in CHANNELS:
            candidate = np.median(
                np.asarray([shapes[stock][channel] for stock in development]), axis=0
            )
            target = shapes[held_stock][channel]
            candidate_rmse = float(np.sqrt(np.mean(np.square(candidate - target))))
            baseline_rmse = float(np.sqrt(np.mean(np.square(baseline - target))))
            won = candidate_rmse < baseline_rmse
            wins_by_channel[channel] += int(won)
            candidate_errors.append(candidate_rmse)
            baseline_errors.append(baseline_rmse)
            lower = np.minimum(shapes[development[0]][channel], shapes[development[1]][channel])
            upper = np.maximum(shapes[development[0]][channel], shapes[development[1]][channel])
            rows.append(
                {
                    "held_stock": held_stock,
                    "channel": channel,
                    "development_stocks": list(development),
                    "candidate_rmse": candidate_rmse,
                    "baseline_rmse": baseline_rmse,
                    "candidate_wins": won,
                    "held_inside_development_envelope_fraction": float(
                        np.mean((target >= lower) & (target <= upper))
                    ),
                    "maximum_absolute_error": float(np.max(np.abs(candidate - target))),
                }
            )

    candidate_array = np.asarray(candidate_errors, dtype=np.float64)
    baseline_array = np.asarray(baseline_errors, dtype=np.float64)
    candidate_median = float(np.median(candidate_array))
    baseline_median = float(np.median(baseline_array))
    candidate_p95 = float(np.quantile(candidate_array, 0.95))
    baseline_p95 = float(np.quantile(baseline_array, 0.95))
    median_improvement = 1.0 - candidate_median / baseline_median
    p95_improvement = 1.0 - candidate_p95 / baseline_p95

    final_channels: dict[str, Any] = {}
    maximum_uncertainty_width = 0.0
    maximum_anchor_error = 0.0
    minimum_step = math.inf
    for channel in CHANNELS:
        population = np.asarray([shapes[stock][channel] for stock in STOCKS])
        template = np.median(population, axis=0)
        lower = np.min(population, axis=0)
        upper = np.max(population, axis=0)
        uncertainty = upper - lower
        anchor_error = max(
            abs(float(template[np.flatnonzero(grid == 0.0)[0]]) - 0.1),
            abs(float(template[np.flatnonzero(grid == 1.0)[0]]) - 0.9),
        )
        maximum_uncertainty_width = max(maximum_uncertainty_width, float(np.max(uncertainty)))
        maximum_anchor_error = max(maximum_anchor_error, anchor_error)
        minimum_step = min(minimum_step, float(np.min(np.diff(template))))
        final_channels[channel] = {
            "template": template.tolist(),
            "uncertainty_min": lower.tolist(),
            "uncertainty_max": upper.tolist(),
            "maximum_uncertainty_width": float(np.max(uncertainty)),
            "median_uncertainty_width": float(np.median(uncertainty)),
            "anchor_error": anchor_error,
            "minimum_step": float(np.min(np.diff(template))),
        }

    bundle_core = {
        "schema": BUNDLE_SCHEMA,
        "source_decision_sha256": parent["decision_sha256"],
        "source_trace_sha256": parent["trace_sha256"],
        "domain": {
            "x": "gauge-free normalized log-relative-exposure shape coordinate",
            "y": "normalized source-document density shape coordinate",
            "grid": grid.tolist(),
            "anchors": [[0.0, 0.1], [1.0, 0.9]],
        },
        "stocks": list(STOCKS),
        "channels": final_channels,
        "claim_ceiling": config["claim_ceiling"],
    }
    bundle = {
        **bundle_core,
        "bundle_id": hashlib.sha256(_canonical_json(bundle_core)).hexdigest(),
    }
    bundle_sha256 = hashlib.sha256(_canonical_json(bundle)).hexdigest()

    gate_results = {
        "row_count": len(rows) == int(gates["required_loso_rows"]),
        "candidate_win_count": sum(row["candidate_wins"] for row in rows)
        >= int(gates["minimum_candidate_win_rows"]),
        "every_channel_win_count": all(
            wins >= int(gates["minimum_candidate_wins_per_channel"])
            for wins in wins_by_channel.values()
        ),
        "median_rmse_improvement": median_improvement
        >= float(gates["minimum_median_rmse_improvement_fraction"]),
        "p95_rmse_improvement": p95_improvement
        >= float(gates["minimum_p95_rmse_improvement_fraction"]),
        "maximum_loso_rmse": float(np.max(candidate_array))
        <= float(gates["maximum_loso_rmse"]),
        "final_uncertainty_width": maximum_uncertainty_width
        <= float(gates["maximum_final_uncertainty_width"]),
        "anchor_identity": maximum_anchor_error <= float(gates["maximum_anchor_error"]),
        "template_monotonicity": minimum_step >= float(gates["minimum_template_step"]),
        "no_parameter_fit": True,
    }
    passed = all(gate_results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "parent_decision_sha256": parent["decision_sha256"],
        "trace_sha256": parent["trace_sha256"],
        "grid": grid.tolist(),
        "loso_rows": rows,
        "candidate_win_count": sum(row["candidate_wins"] for row in rows),
        "wins_by_channel": wins_by_channel,
        "candidate_median_rmse": candidate_median,
        "baseline_median_rmse": baseline_median,
        "candidate_p95_rmse": candidate_p95,
        "baseline_p95_rmse": baseline_p95,
        "median_rmse_improvement_fraction": median_improvement,
        "p95_rmse_improvement_fraction": p95_improvement,
        "maximum_loso_rmse": float(np.max(candidate_array)),
        "maximum_final_uncertainty_width": maximum_uncertainty_width,
        "maximum_anchor_error": maximum_anchor_error,
        "minimum_template_step": minimum_step,
        "bundle_sha256": bundle_sha256,
        "gate_results": gate_results,
    }
    report = {
        "schema": REPORT_SCHEMA,
        **stable,
        "shared_template_pass": passed,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            config["decision_branches"]["pass"]
            if passed
            else config["decision_branches"]["fail"]
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    return report, bundle


__all__ = [
    "BUNDLE_SCHEMA",
    "REPORT_SCHEMA",
    "SCHEMA",
    "SharedCharacteristicShapeError",
    "evaluate_shared_shape",
    "load_contract",
]
