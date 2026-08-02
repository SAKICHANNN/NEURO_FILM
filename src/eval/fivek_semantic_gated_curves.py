"""Held-block capacity test for fixed semantic-gated explicit colour curves."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.eval.fivek_bilateral_gain_capacity import _held_mask


class FiveKSemanticGatedCurvesError(ValueError):
    """Raised when the frozen BV0 curve or evaluation contract is invalid."""


def _rgb(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 3
        or array.shape[-1] != 3
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
        or np.any(array > 1.0)
    ):
        raise FiveKSemanticGatedCurvesError("invalid bounded encoded RGB")
    return array


def _zsigmoid(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise FiveKSemanticGatedCurvesError("invalid gate source")
    scale = float(np.std(array))
    if scale <= 1.0e-12:
        return np.full(array.shape, 0.5, dtype=np.float64)
    z = (array - float(np.mean(array))) / scale
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def _curve_features(source: np.ndarray, knots: Sequence[float]) -> np.ndarray:
    rgb = _rgb(source)
    axis = np.asarray(knots, dtype=np.float64)
    if (
        axis.ndim != 1
        or len(axis) < 3
        or axis[0] != 0.0
        or axis[-1] != 1.0
        or not np.all(np.diff(axis) > 0.0)
        or not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=0.0, atol=1e-12)
    ):
        raise FiveKSemanticGatedCurvesError("invalid uniform curve knots")
    spacing = axis[1] - axis[0]
    flat = rgb.reshape(-1, 3)
    channel_features = []
    for channel in range(3):
        distance = np.abs(flat[:, channel, None] - axis[None, 1:-1])
        channel_features.append(np.maximum(1.0 - distance / spacing, 0.0))
    return np.concatenate(channel_features, axis=1)


def _fit_curves(
    source: np.ndarray,
    target: np.ndarray,
    gate: np.ndarray,
    fit_mask: np.ndarray,
    spec: Mapping[str, Any],
) -> np.ndarray:
    source_rgb = _rgb(source)
    target_rgb = _rgb(target)
    if source_rgb.shape != target_rgb.shape or gate.shape != source_rgb.shape[:2]:
        raise FiveKSemanticGatedCurvesError("curve fit shape mismatch")
    features = _curve_features(source_rgb, spec["knot_positions"])
    selected = np.asarray(fit_mask, dtype=bool).reshape(-1)
    design = features[selected] * gate.reshape(-1)[selected, None]
    target_delta = (target_rgb - source_rgb).reshape(-1, 3)[selected]
    normal = design.T @ design
    normal += float(spec["ridge"]) * np.eye(normal.shape[0])
    coefficients = np.linalg.solve(normal, design.T @ target_delta)
    bound = float(spec["coefficient_bound"])
    return np.clip(coefficients, -bound, bound)


def _apply_curves(
    source: np.ndarray,
    gate: np.ndarray,
    coefficients: np.ndarray,
    spec: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    source_rgb = _rgb(source)
    if gate.shape != source_rgb.shape[:2]:
        raise FiveKSemanticGatedCurvesError("curve gate shape mismatch")
    features = _curve_features(source_rgb, spec["knot_positions"])
    residual = gate.reshape(-1, 1) * (features @ np.asarray(coefficients))
    flat = source_rgb.reshape(-1, 3)
    epsilon = float(spec["boundary_epsilon"])
    dose = np.ones(len(flat), dtype=np.float64)
    source_boundary = np.any((flat <= epsilon) | (flat >= 1.0 - epsilon), axis=1)
    dose[source_boundary] = 0.0
    positive = residual > 0.0
    negative = residual < 0.0
    limits = np.full_like(residual, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        limits[positive] = (1.0 - epsilon - flat[positive]) / residual[positive]
        limits[negative] = (epsilon - flat[negative]) / residual[negative]
    dose = np.minimum(dose, np.min(limits, axis=1))
    dose = np.clip(dose, 0.0, 1.0)
    output = flat + dose[:, None] * residual
    if (
        not np.all(np.isfinite(output))
        or np.any(output < -1e-12)
        or np.any(output > 1.0 + 1e-12)
    ):
        raise FiveKSemanticGatedCurvesError("analytical curve dose failed")
    return output.reshape(source_rgb.shape), dose.reshape(source_rgb.shape[:2])


def _rmse(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    delta = np.asarray(first)[mask] - np.asarray(second)[mask]
    return float(np.sqrt(np.mean(delta * delta)))


def _new_boundary_fraction(
    source: np.ndarray, output: np.ndarray, mask: np.ndarray, epsilon: float
) -> float:
    source_boundary = np.any((source <= epsilon) | (source >= 1.0 - epsilon), axis=-1)
    output_boundary = np.any((output <= epsilon) | (output >= 1.0 - epsilon), axis=-1)
    return float(np.mean((output_boundary & ~source_boundary)[mask]))


def evaluate_semantic_gated_curves(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    expected = int(config["population"]["source_count_exact"])
    ordered = sorted(rows, key=lambda row: str(row["pair_id"]))
    if (
        len(ordered) != expected
        or len({str(row["group"]) for row in ordered}) != expected
    ):
        raise FiveKSemanticGatedCurvesError("population identity drift")
    spec = config["explicit_operator"]
    held_spec = config["held_block_protocol"]
    epsilon = float(spec["boundary_epsilon"])
    output_rows: list[dict[str, Any]] = []
    gate_hashes: dict[str, str] = {}
    for row in ordered:
        source = _rgb(row["source"])
        target = _rgb(row["target"])
        semantic = np.asarray(row["semantic_gate"], dtype=np.float64)
        if source.shape != target.shape or semantic.shape != source.shape[:2]:
            raise FiveKSemanticGatedCurvesError("paired or semantic shape mismatch")
        if (
            not np.all(np.isfinite(semantic))
            or np.any(semantic <= 0.0)
            or np.any(semantic >= 1.0)
        ):
            raise FiveKSemanticGatedCurvesError(
                "semantic gate outside strict unit interval"
            )
        luma = np.tensordot(
            source, np.asarray([0.2126, 0.7152, 0.0722]), axes=([-1], [0])
        )
        gates = {
            "semantic": semantic,
            "global": np.ones(source.shape[:2], dtype=np.float64),
            "luminance": _zsigmoid(luma),
            "shifted": np.roll(
                semantic,
                shift=(semantic.shape[0] // 2, semantic.shape[1] // 2),
                axis=(0, 1),
            ),
        }
        gate_hashes[str(row["pair_id"])] = hashlib.sha256(
            np.ascontiguousarray(semantic, dtype="<f4").tobytes()
        ).hexdigest()
        for fold in map(int, held_spec["folds"]):
            held = _held_mask(source.shape[:2], held_spec, fold)
            fit = ~held
            outputs: dict[str, np.ndarray] = {}
            doses: dict[str, np.ndarray] = {}
            for name, gate in gates.items():
                coefficients = _fit_curves(source, target, gate, fit, spec)
                outputs[name], doses[name] = _apply_curves(
                    source, gate, coefficients, spec
                )
            identity_rmse = _rmse(source, target, held)
            output_rows.append(
                {
                    "pair_id": str(row["pair_id"]),
                    "group": str(row["group"]),
                    "target_variant": str(row["target_variant"]),
                    "fold": fold,
                    "fit_pixels": int(np.sum(fit)),
                    "held_pixels": int(np.sum(held)),
                    "semantic_gate_mean": float(np.mean(semantic)),
                    "semantic_gate_std": float(np.std(semantic)),
                    "identity_rmse": identity_rmse,
                    "semantic_fit_rmse": _rmse(outputs["semantic"], target, fit),
                    "semantic_rmse": _rmse(outputs["semantic"], target, held),
                    "global_rmse": _rmse(outputs["global"], target, held),
                    "luminance_rmse": _rmse(outputs["luminance"], target, held),
                    "shifted_rmse": _rmse(outputs["shifted"], target, held),
                    "semantic_style_retention": _rmse(outputs["semantic"], source, held)
                    / max(identity_rmse, 1e-12),
                    "minimum_semantic_dose": float(np.min(doses["semantic"][held])),
                    "new_boundary_fraction": _new_boundary_fraction(
                        source, outputs["semantic"], held, epsilon
                    ),
                    "out_of_cube_fraction": float(
                        np.mean(
                            (outputs["semantic"] < 0.0) | (outputs["semantic"] > 1.0)
                        )
                    ),
                }
            )
    semantic_error = np.asarray([row["semantic_rmse"] for row in output_rows])
    global_error = np.asarray([row["global_rmse"] for row in output_rows])
    luma_error = np.asarray([row["luminance_rmse"] for row in output_rows])
    shifted_error = np.asarray([row["shifted_rmse"] for row in output_rows])
    fit_error = np.asarray([row["semantic_fit_rmse"] for row in output_rows])
    metrics = {
        "rows": len(output_rows),
        "minimum_semantic_gate_spatial_std": float(
            min(row["semantic_gate_std"] for row in output_rows)
        ),
        "mean_improvement_over_global_gate": float(
            (global_error.mean() - semantic_error.mean()) / global_error.mean()
        ),
        "win_fraction_over_global_gate": float(np.mean(semantic_error < global_error)),
        "p95_error_ratio_to_global_gate": float(
            np.quantile(semantic_error, 0.95) / np.quantile(global_error, 0.95)
        ),
        "worst_error_ratio_to_global_gate": float(
            np.max(semantic_error) / np.max(global_error)
        ),
        "mean_improvement_over_luminance_gate": float(
            (luma_error.mean() - semantic_error.mean()) / luma_error.mean()
        ),
        "mean_improvement_over_shifted_semantic_gate": float(
            (shifted_error.mean() - semantic_error.mean()) / shifted_error.mean()
        ),
        "median_style_retention": float(
            np.median([row["semantic_style_retention"] for row in output_rows])
        ),
        "p95_held_to_fit_error_ratio": float(
            np.quantile(semantic_error / np.maximum(fit_error, 1e-12), 0.95)
        ),
        "maximum_new_boundary_fraction": float(
            max(row["new_boundary_fraction"] for row in output_rows)
        ),
        "maximum_out_of_cube_fraction": float(
            max(row["out_of_cube_fraction"] for row in output_rows)
        ),
        "minimum_semantic_dose": float(
            min(row["minimum_semantic_dose"] for row in output_rows)
        ),
        "semantic_error_sha256": hashlib.sha256(
            np.ascontiguousarray(semantic_error, dtype="<f8").tobytes()
        ).hexdigest(),
    }
    gates = config["evaluation"]["automatic_gates"]
    checks = {
        "source_count": len(ordered) == int(gates["source_count_exact"]),
        "folds": len({row["fold"] for row in output_rows})
        == int(gates["fold_count_exact"]),
        "semantic_variation": metrics["minimum_semantic_gate_spatial_std"]
        >= float(gates["minimum_semantic_gate_spatial_std"]),
        "global_mean": metrics["mean_improvement_over_global_gate"]
        >= float(gates["minimum_mean_improvement_over_global_gate"]),
        "global_wins": metrics["win_fraction_over_global_gate"]
        >= float(gates["minimum_win_fraction_over_global_gate"]),
        "global_p95": metrics["p95_error_ratio_to_global_gate"]
        <= float(gates["maximum_p95_error_ratio_to_global_gate"]),
        "global_worst": metrics["worst_error_ratio_to_global_gate"]
        <= float(gates["maximum_worst_error_ratio_to_global_gate"]),
        "luminance": metrics["mean_improvement_over_luminance_gate"]
        >= float(gates["minimum_mean_improvement_over_luminance_gate"]),
        "shifted": metrics["mean_improvement_over_shifted_semantic_gate"]
        >= float(gates["minimum_mean_improvement_over_shifted_semantic_gate"]),
        "style": metrics["median_style_retention"]
        >= float(gates["minimum_median_style_retention"]),
        "held_fit": metrics["p95_held_to_fit_error_ratio"]
        <= float(gates["maximum_p95_held_to_fit_error_ratio"]),
        "boundary": metrics["maximum_new_boundary_fraction"]
        <= float(gates["maximum_new_boundary_fraction"]),
        "cube": metrics["maximum_out_of_cube_fraction"]
        <= float(gates["maximum_out_of_cube_fraction"]),
    }
    return {
        "schema": "neuro_film.u5_r2bv0_fivek_semantic_gated_curves_report.v1",
        "metrics": metrics,
        "gates": checks,
        "automatic_pass": all(checks.values()),
        "gate_sha256": gate_hashes,
        "rows": output_rows,
    }


__all__ = [
    "FiveKSemanticGatedCurvesError",
    "_apply_curves",
    "_curve_features",
    "_fit_curves",
    "_zsigmoid",
    "evaluate_semantic_gated_curves",
]
