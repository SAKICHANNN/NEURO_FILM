"""Paired FilmSet recipe explainability with bounded explicit colour flows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .baselines import fit_joint_basic_adjustment
from .cube_diffeomorphic_flow import (
    CubeDiffeomorphicColourFlow,
    finite_difference_jacobians,
)
from .hierarchical_colour_coupling import fit_paired_cube_diffeomorphic_flow


@dataclass(frozen=True)
class SpatialPixelSample:
    """Disjoint deterministic samples from every spatial grid cell."""

    fit_pixels: np.ndarray
    evaluation_pixels: np.ndarray
    fit_cell_ids: np.ndarray
    evaluation_cell_ids: np.ndarray
    fit_flat_indices: np.ndarray
    evaluation_flat_indices: np.ndarray


def sample_aligned_spatial_pixels(
    image: np.ndarray,
    *,
    identity: str,
    grid_rows: int,
    grid_columns: int,
    pixels_per_cell_fit: int,
    pixels_per_cell_evaluation: int,
    fit_seed: int,
    evaluation_seed: int,
) -> SpatialPixelSample:
    """Sample one image; reuse returned indices for every aligned domain."""

    values = np.asarray(image, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape[2] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("image must be finite [0,1] HxWx3 linear RGB")
    if (
        not identity
        or grid_rows < 1
        or grid_columns < 1
        or pixels_per_cell_fit < 1
        or pixels_per_cell_evaluation < 1
        or fit_seed == evaluation_seed
    ):
        raise ValueError("invalid spatial sampling contract")
    height, width = values.shape[:2]
    fit_indices: list[np.ndarray] = []
    evaluation_indices: list[np.ndarray] = []
    fit_cells: list[np.ndarray] = []
    evaluation_cells: list[np.ndarray] = []
    for row in range(grid_rows):
        y0 = row * height // grid_rows
        y1 = (row + 1) * height // grid_rows
        for column in range(grid_columns):
            x0 = column * width // grid_columns
            x1 = (column + 1) * width // grid_columns
            yy, xx = np.meshgrid(
                np.arange(y0, y1, dtype=np.int64),
                np.arange(x0, x1, dtype=np.int64),
                indexing="ij",
            )
            available = (yy * width + xx).reshape(-1)
            required = pixels_per_cell_fit + pixels_per_cell_evaluation
            if len(available) < required:
                raise ValueError(
                    f"spatial cell {row},{column} has {len(available)} pixels, "
                    f"requires {required}"
                )
            cell_id = row * grid_columns + column
            fit_material = (
                f"{fit_seed}:{identity}:{row}:{column}:fit"
            ).encode("utf-8")
            fit_rng = np.random.default_rng(
                int.from_bytes(
                    hashlib.sha256(fit_material).digest()[:8], "little"
                )
            )
            fit = np.sort(
                fit_rng.choice(
                    available, size=pixels_per_cell_fit, replace=False
                )
            )
            remaining = np.setdiff1d(available, fit, assume_unique=True)
            evaluation_material = (
                f"{evaluation_seed}:{identity}:{row}:{column}:evaluation"
            ).encode("utf-8")
            evaluation_rng = np.random.default_rng(
                int.from_bytes(
                    hashlib.sha256(evaluation_material).digest()[:8], "little"
                )
            )
            evaluation = np.sort(
                evaluation_rng.choice(
                    remaining,
                    size=pixels_per_cell_evaluation,
                    replace=False,
                )
            )
            fit_indices.append(fit)
            evaluation_indices.append(evaluation)
            fit_cells.append(np.full(len(fit), cell_id, dtype=np.int64))
            evaluation_cells.append(
                np.full(len(evaluation), cell_id, dtype=np.int64)
            )
    flat = values.reshape(-1, 3)
    fit_flat = np.concatenate(fit_indices)
    evaluation_flat = np.concatenate(evaluation_indices)
    return SpatialPixelSample(
        fit_pixels=flat[fit_flat],
        evaluation_pixels=flat[evaluation_flat],
        fit_cell_ids=np.concatenate(fit_cells),
        evaluation_cell_ids=np.concatenate(evaluation_cells),
        fit_flat_indices=fit_flat,
        evaluation_flat_indices=evaluation_flat,
    )


def apply_spatial_indices(
    image: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    selected = np.asarray(indices, dtype=np.int64)
    if (
        values.ndim != 3
        or values.shape[2] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or selected.ndim != 1
        or len(selected) == 0
        or np.any(selected < 0)
        or np.any(selected >= values.shape[0] * values.shape[1])
    ):
        raise ValueError("invalid aligned image or spatial indices")
    return values.reshape(-1, 3)[selected]


def _validate_image_samples(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 3
        or array.shape[2] != 3
        or array.shape[1] < 16
        or not len(array)
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
        or np.any(array > 1.0)
    ):
        raise ValueError(f"{name} must be finite [0,1] image x pixel x RGB")
    return array


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _per_image_rmse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((prediction - target) ** 2, axis=(1, 2)))


def spatial_residual_between_cell_fraction(
    residual: np.ndarray, cell_ids: np.ndarray
) -> np.ndarray:
    """Fraction of residual energy explained by spatial-cell mean shifts."""

    values = np.asarray(residual, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape[2] != 3
        or values.shape[1] < 16
        or not len(values)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("residual must be finite image x pixel x RGB")
    cells = np.asarray(cell_ids, dtype=np.int64)
    if cells.shape != (values.shape[1],) or len(np.unique(cells)) < 2:
        raise ValueError("cell IDs must cover every sampled pixel and multiple cells")
    fractions = []
    for image in values:
        overall = np.mean(image, axis=0)
        total = float(np.mean(np.sum((image - overall) ** 2, axis=1)))
        weighted_between = 0.0
        for cell in np.unique(cells):
            mask = cells == cell
            delta = np.mean(image[mask], axis=0) - overall
            weighted_between += float(np.mean(mask)) * float(np.sum(delta**2))
        fractions.append(weighted_between / max(total, 1e-30))
    return np.clip(np.asarray(fractions, dtype=np.float64), 0.0, 1.0)


def evaluate_flow_structure(
    operators: list[CubeDiffeomorphicColourFlow],
    *,
    coefficient_cap: float,
) -> dict[str, float | int]:
    if not operators:
        raise ValueError("at least one flow is required")
    axis = np.linspace(0.08, 0.92, 5, dtype=np.float64)
    points = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    replay_axis = np.linspace(0.0, 1.0, 7, dtype=np.float64)
    replay_points = np.stack(
        np.meshgrid(replay_axis, replay_axis, replay_axis, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    minimum_output = np.inf
    maximum_output = -np.inf
    minimum_determinant = np.inf
    maximum_norm = 0.0
    maximum_inverse = 0.0
    maximum_replay = 0.0
    maximum_coefficient = 0.0
    for operator in operators:
        output = operator.apply(points)
        jacobians = finite_difference_jacobians(operator, points, step=1e-6)
        minimum_output = min(minimum_output, float(np.min(output)))
        maximum_output = max(maximum_output, float(np.max(output)))
        minimum_determinant = min(
            minimum_determinant, float(np.min(np.linalg.det(jacobians)))
        )
        maximum_norm = max(
            maximum_norm,
            float(np.max(np.linalg.svd(jacobians, compute_uv=False)[:, 0])),
        )
        maximum_inverse = max(
            maximum_inverse,
            float(np.max(np.abs(operator.inverse(output) - points))),
        )
        replay = CubeDiffeomorphicColourFlow.from_dict(
            json.loads(json.dumps(operator.to_dict(), sort_keys=True))
        )
        maximum_replay = max(
            maximum_replay,
            float(
                np.max(
                    np.abs(
                        replay.apply(replay_points)
                        - operator.apply(replay_points)
                    )
                )
            ),
        )
        maximum_coefficient = max(
            maximum_coefficient,
            float(np.max(np.linalg.norm(operator.velocity_grid, axis=-1))),
        )
    return {
        "sampled_operator_count": len(operators),
        "minimum_output": minimum_output,
        "maximum_output": maximum_output,
        "minimum_jacobian_determinant": minimum_determinant,
        "maximum_jacobian_spectral_norm": maximum_norm,
        "maximum_inverse_error": maximum_inverse,
        "maximum_replay_error": maximum_replay,
        "maximum_coefficient_vector_norm": maximum_coefficient,
        "coefficient_cap": float(coefficient_cap),
    }


def _fit_flow(
    source: np.ndarray,
    target: np.ndarray,
    *,
    settings: dict[str, Any],
    seed: int,
    device: str,
) -> tuple[CubeDiffeomorphicColourFlow, dict[str, float]]:
    return fit_paired_cube_diffeomorphic_flow(
        source,
        target,
        axis_size=int(settings["axis_size"]),
        integration_steps=int(settings["integration_steps"]),
        coefficient_vector_norm_cap=float(
            settings["coefficient_vector_norm_cap"]
        ),
        steps=int(settings["steps"]),
        learning_rate=float(settings["learning_rate"]),
        coefficient_l2=float(settings["coefficient_l2"]),
        velocity_smoothness_l2=float(settings["velocity_smoothness_l2"]),
        gradient_clip_norm=float(settings["gradient_clip_norm"]),
        seed=seed,
        device=device,
        deterministic_algorithms=bool(settings["deterministic_algorithms"]),
        optimization_dtype=str(settings["optimization_dtype"]),
    )


def _gate_results(
    metrics: dict[str, Any], gates: dict[str, Any]
) -> dict[str, bool]:
    structure = metrics["structure"]
    result = {
        "shared_beats_identity": metrics[
            "shared_improvement_over_identity_fraction"
        ]
        >= float(gates["minimum_shared_improvement_over_identity_fraction"]),
        "shared_beats_basic": metrics[
            "shared_improvement_over_basic_fraction"
        ]
        >= float(gates["minimum_shared_improvement_over_basic_fraction"]),
        "oracle_regret": metrics["shared_normalized_regret_to_per_pair_oracle"][
            "median"
        ]
        <= float(gates["maximum_shared_normalized_regret_to_per_pair_oracle"]),
        "grid_dispersion": metrics["per_pair_grid_rmse_to_shared"]["median"]
        <= float(gates["maximum_median_per_pair_grid_rmse_to_shared"]),
        "spatial_residual": metrics[
            "spatial_residual_between_cell_fraction"
        ]["median"]
        <= float(gates["maximum_spatial_residual_between_cell_fraction"]),
        "basic_beats_identity": metrics[
            "basic_improvement_over_identity_fraction"
        ]
        >= float(gates["minimum_basic_improvement_over_identity_fraction"]),
        "shared_incremental_is_basic_only": metrics[
            "shared_improvement_over_basic_fraction"
        ]
        <= float(
            gates[
                "maximum_shared_incremental_improvement_over_basic_for_basic_only"
            ]
        ),
        "per_pair_beats_identity": metrics[
            "per_pair_improvement_over_identity_fraction"
        ]
        >= float(gates["minimum_per_pair_improvement_over_identity_fraction"]),
        "range": structure["minimum_output"] >= float(gates["minimum_output"])
        and structure["maximum_output"] <= float(gates["maximum_output"]),
        "jacobian": structure["minimum_jacobian_determinant"]
        > float(gates["minimum_jacobian_determinant_exclusive"]),
        "norm": structure["maximum_jacobian_spectral_norm"]
        <= float(gates["maximum_jacobian_spectral_norm"]),
        "inverse": structure["maximum_inverse_error"]
        <= float(gates["maximum_inverse_error"]),
        "replay": structure["maximum_replay_error"]
        <= float(gates["maximum_replay_error"]),
        "coefficient": structure["maximum_coefficient_vector_norm"]
        <= float(gates["maximum_coefficient_vector_norm"]),
    }
    result["structure_all"] = all(
        result[key]
        for key in ("range", "jacobian", "norm", "inverse", "replay", "coefficient")
    )
    return result


def decide_recipe_branch(gates: dict[str, bool]) -> str:
    if not gates["structure_all"]:
        return "structure_or_repeat_fails"
    if all(
        gates[key]
        for key in (
            "shared_beats_identity",
            "shared_beats_basic",
            "oracle_regret",
            "grid_dispersion",
            "spatial_residual",
        )
    ):
        return "global_operator_coherent"
    if gates["basic_beats_identity"] and gates["shared_incremental_is_basic_only"]:
        return "basic_only"
    if gates["per_pair_beats_identity"] and not all(
        gates[key]
        for key in ("oracle_regret", "grid_dispersion", "spatial_residual")
    ):
        return "adaptive_or_spatial_recipe"
    return "operator_unidentified"


def evaluate_recipe_global_explainability(
    development_input_fit: np.ndarray,
    development_target_fit: np.ndarray,
    confirmatory_input_fit: np.ndarray,
    confirmatory_target_fit: np.ndarray,
    confirmatory_input_evaluation: np.ndarray,
    confirmatory_target_evaluation: np.ndarray,
    evaluation_cell_ids: np.ndarray,
    *,
    operator_settings: dict[str, Any],
    gates: dict[str, Any],
    shared_seed: int,
    per_pair_seed_base: int,
    device: str,
) -> dict[str, Any]:
    dev_source = _validate_image_samples(
        development_input_fit, name="development input"
    )
    dev_target = _validate_image_samples(
        development_target_fit, name="development target"
    )
    con_source_fit = _validate_image_samples(
        confirmatory_input_fit, name="confirmatory fit input"
    )
    con_target_fit = _validate_image_samples(
        confirmatory_target_fit, name="confirmatory fit target"
    )
    con_source_eval = _validate_image_samples(
        confirmatory_input_evaluation, name="confirmatory evaluation input"
    )
    con_target_eval = _validate_image_samples(
        confirmatory_target_evaluation, name="confirmatory evaluation target"
    )
    if (
        dev_source.shape != dev_target.shape
        or con_source_fit.shape != con_target_fit.shape
        or con_source_eval.shape != con_target_eval.shape
        or len(con_source_fit) != len(con_source_eval)
    ):
        raise ValueError("paired FilmSet sample shapes do not align")

    basic = fit_joint_basic_adjustment(
        dev_source.reshape(-1, 3), dev_target.reshape(-1, 3)
    )
    shared, shared_trace = _fit_flow(
        dev_source.reshape(-1, 3),
        dev_target.reshape(-1, 3),
        settings=operator_settings,
        seed=shared_seed,
        device=device,
    )
    per_pair: list[CubeDiffeomorphicColourFlow] = []
    per_pair_traces: list[dict[str, float]] = []
    for index in range(len(con_source_fit)):
        operator, trace = _fit_flow(
            con_source_fit[index],
            con_target_fit[index],
            settings=operator_settings,
            seed=per_pair_seed_base + index,
            device=device,
        )
        per_pair.append(operator)
        per_pair_traces.append(trace)

    identity_output = con_source_eval
    basic_raw = basic.apply(con_source_eval.reshape(-1, 3)).reshape(
        con_source_eval.shape
    )
    basic_output = np.clip(basic_raw, 0.0, 1.0)
    shared_output = shared.apply(con_source_eval.reshape(-1, 3)).reshape(
        con_source_eval.shape
    )
    per_pair_output = np.stack(
        [
            operator.apply(con_source_eval[index])
            for index, operator in enumerate(per_pair)
        ]
    )
    identity_rmse = _per_image_rmse(identity_output, con_target_eval)
    basic_rmse = _per_image_rmse(basic_output, con_target_eval)
    shared_rmse = _per_image_rmse(shared_output, con_target_eval)
    per_pair_rmse = _per_image_rmse(per_pair_output, con_target_eval)
    identity_median = float(np.median(identity_rmse))
    basic_median = float(np.median(basic_rmse))
    shared_median = float(np.median(shared_rmse))
    per_pair_median = float(np.median(per_pair_rmse))
    regret = (shared_rmse - per_pair_rmse) / np.maximum(identity_rmse, 1e-30)
    grid_dispersion = np.asarray(
        [
            np.sqrt(
                np.mean((operator.velocity_grid - shared.velocity_grid) ** 2)
            )
            for operator in per_pair
        ]
    )
    spatial = spatial_residual_between_cell_fraction(
        con_target_eval - shared_output, evaluation_cell_ids
    )
    metrics: dict[str, Any] = {
        "linear_rgb_rmse": {
            "identity": _summary(identity_rmse),
            "shared_joint_basic_clipped": _summary(basic_rmse),
            "shared_o0": _summary(shared_rmse),
            "per_pair_o0_evaluator_oracle": _summary(per_pair_rmse),
        },
        "shared_improvement_over_identity_fraction": 1.0
        - shared_median / max(identity_median, 1e-30),
        "shared_improvement_over_basic_fraction": 1.0
        - shared_median / max(basic_median, 1e-30),
        "basic_improvement_over_identity_fraction": 1.0
        - basic_median / max(identity_median, 1e-30),
        "per_pair_improvement_over_identity_fraction": 1.0
        - per_pair_median / max(identity_median, 1e-30),
        "shared_normalized_regret_to_per_pair_oracle": _summary(regret),
        "per_pair_grid_rmse_to_shared": _summary(grid_dispersion),
        "spatial_residual_between_cell_fraction": _summary(spatial),
        "basic_raw_out_of_range_fraction": float(
            np.mean((basic_raw < 0.0) | (basic_raw > 1.0))
        ),
        "shared_optimization_trace": shared_trace,
        "per_pair_optimization_trace": {
            key: _summary(
                np.asarray([trace[key] for trace in per_pair_traces])
            )
            for key in per_pair_traces[0]
        },
        "shared_operator": shared.to_dict(),
        "shared_basic_operator": basic.to_dict(),
        "structure": evaluate_flow_structure(
            [shared, *per_pair],
            coefficient_cap=float(
                operator_settings["coefficient_vector_norm_cap"]
            ),
        ),
    }
    gate_results = _gate_results(metrics, gates)
    return {
        "metrics": metrics,
        "gate_results": gate_results,
        "decision_branch_before_repeat": decide_recipe_branch(gate_results),
        "development_images": len(dev_source),
        "confirmatory_images": len(con_source_eval),
        "device": device,
    }


__all__ = [
    "SpatialPixelSample",
    "apply_spatial_indices",
    "decide_recipe_branch",
    "evaluate_flow_structure",
    "evaluate_recipe_global_explainability",
    "sample_aligned_spatial_pixels",
    "spatial_residual_between_cell_fraction",
]
