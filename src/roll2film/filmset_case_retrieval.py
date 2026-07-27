"""Within-recipe explicit-operator case-bank and retrieval controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .cube_diffeomorphic_flow import CubeDiffeomorphicColourFlow
from .filmset_recipe_explainability import evaluate_flow_structure
from .hierarchical_colour_coupling import fit_paired_cube_diffeomorphic_flow


@dataclass(frozen=True)
class CaseBankEvaluation:
    """Public metrics plus private operators needed for cross-domain controls."""

    report: dict[str, Any]
    eligible_operators: tuple[CubeDiffeomorphicColourFlow, ...]
    query_input_evaluation: np.ndarray
    query_target_evaluation: np.ndarray
    oracle_rmse: np.ndarray


def _validate_samples(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 3
        or array.shape[2] != 3
        or array.shape[1] < 16
        or len(array) < 2
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
        or np.any(array > 1.0)
    ):
        raise ValueError(f"{name} must be finite [0,1] image x pixel x RGB")
    return array


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary values must be a non-empty finite vector")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _rmse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((prediction - target) ** 2, axis=(-2, -1)))


def content_descriptor(
    samples: np.ndarray,
    cell_ids: np.ndarray,
    *,
    include_spatial_cells: bool,
) -> np.ndarray:
    """Build deterministic photometric descriptors from input pixels only."""

    values = _validate_samples(samples, name="descriptor samples")
    cells = np.asarray(cell_ids, dtype=np.int64)
    if cells.shape != (values.shape[1],) or len(np.unique(cells)) < 2:
        raise ValueError("cell IDs must cover sampled pixels and multiple cells")
    luma = (
        0.2126 * values[..., 0]
        + 0.7152 * values[..., 1]
        + 0.0722 * values[..., 2]
    )
    chroma = np.max(values, axis=-1) - np.min(values, axis=-1)
    quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
    features: list[np.ndarray] = [
        np.mean(values, axis=1),
        np.std(values, axis=1),
    ]
    for quantile in quantiles:
        features.append(np.quantile(values, quantile, axis=1))
    for scalar in (luma, chroma):
        features.extend(
            [
                np.mean(scalar, axis=1, keepdims=True),
                np.std(scalar, axis=1, keepdims=True),
            ]
        )
        for quantile in quantiles:
            features.append(
                np.quantile(scalar, quantile, axis=1, keepdims=True)
            )
    if include_spatial_cells:
        for cell in np.unique(cells):
            selected = cells == cell
            cell_rgb = values[:, selected]
            cell_luma = luma[:, selected]
            features.extend(
                [
                    np.mean(cell_rgb, axis=1),
                    np.mean(cell_luma, axis=1, keepdims=True),
                    np.std(cell_luma, axis=1, keepdims=True),
                ]
            )
    result = np.concatenate(features, axis=1)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("content descriptor became non-finite")
    return result


def standardized_nearest_indices(
    bank_features: np.ndarray,
    query_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest development case after development-only z-scoring."""

    bank = np.asarray(bank_features, dtype=np.float64)
    query = np.asarray(query_features, dtype=np.float64)
    if (
        bank.ndim != 2
        or query.ndim != 2
        or bank.shape[1] != query.shape[1]
        or len(bank) < 2
        or len(query) < 1
        or not np.all(np.isfinite(bank))
        or not np.all(np.isfinite(query))
    ):
        raise ValueError("invalid bank/query feature matrices")
    mean = np.mean(bank, axis=0)
    scale = np.std(bank, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    bank_z = (bank - mean) / scale
    query_z = (query - mean) / scale
    distances = np.sqrt(
        np.sum((query_z[:, None, :] - bank_z[None, :, :]) ** 2, axis=-1)
    )
    indices = np.argmin(distances, axis=1)
    margins = np.partition(distances, 1, axis=1)[:, 1] - distances[
        np.arange(len(query)), indices
    ]
    return indices.astype(np.int64), margins


def oracle_gap_closure(
    shared_rmse: np.ndarray,
    selected_rmse: np.ndarray,
    oracle_rmse: np.ndarray,
) -> float:
    shared = float(np.mean(shared_rmse))
    selected = float(np.mean(selected_rmse))
    oracle = float(np.mean(oracle_rmse))
    return (shared - selected) / max(shared - oracle, 1e-30)


def bootstrap_mean_improvement_ci(
    baseline: np.ndarray,
    challenger: np.ndarray,
    *,
    seed: int,
    samples: int,
    confidence: float,
) -> dict[str, float]:
    left = np.asarray(baseline, dtype=np.float64)
    right = np.asarray(challenger, dtype=np.float64)
    if (
        left.ndim != 1
        or right.shape != left.shape
        or len(left) < 2
        or samples < 100
        or not 0.5 < confidence < 1.0
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        raise ValueError("invalid paired bootstrap contract")
    improvement = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(left), size=(samples, len(left)))
    distribution = np.mean(improvement[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(improvement)),
        "lower": float(np.quantile(distribution, alpha)),
        "upper": float(np.quantile(distribution, 1.0 - alpha)),
        "confidence": float(confidence),
        "bootstrap_samples": int(samples),
    }


def decide_case_retrieval_branch(
    gates: dict[str, bool],
) -> str:
    if not gates["structure_all"] or not gates["minimum_eligible_bank"]:
        return "case_bank_invalid"
    if not (
        gates["oracle_improvement"]
        and gates["oracle_win_fraction"]
        and gates["oracle_bootstrap"]
    ):
        return "case_bank_no_oracle_value"
    if all(
        gates[key]
        for key in (
            "retrieval_gap_closure",
            "retrieval_beats_random",
            "retrieval_beats_shuffled",
        )
    ):
        return "case_bank_oracle_and_retrieval_pass"
    return "case_bank_oracle_only"


def _fit_flow(
    source: np.ndarray,
    target: np.ndarray,
    *,
    settings: dict[str, Any],
    seed: int,
    device: str,
) -> CubeDiffeomorphicColourFlow:
    operator, _trace = fit_paired_cube_diffeomorphic_flow(
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
        velocity_smoothness_l2=float(
            settings["velocity_smoothness_l2"]
        ),
        gradient_clip_norm=float(settings["gradient_clip_norm"]),
        seed=seed,
        device=device,
        deterministic_algorithms=bool(settings["deterministic_algorithms"]),
        optimization_dtype=str(settings["optimization_dtype"]),
    )
    return operator


def _apply_operator_bank(
    operators: tuple[CubeDiffeomorphicColourFlow, ...],
    query_input: np.ndarray,
    query_target: np.ndarray,
) -> np.ndarray:
    losses = []
    for operator in operators:
        prediction = np.stack(
            [operator.apply(image) for image in query_input]
        )
        losses.append(_rmse(prediction, query_target))
    return np.stack(losses, axis=1)


def evaluate_within_recipe_case_bank(
    development_input_fit: np.ndarray,
    development_target_fit: np.ndarray,
    development_input_evaluation: np.ndarray,
    development_target_evaluation: np.ndarray,
    query_input_evaluation: np.ndarray,
    query_target_evaluation: np.ndarray,
    cell_ids: np.ndarray,
    *,
    operator_settings: dict[str, Any],
    gates: dict[str, Any],
    shared_seed: int,
    case_seed_base: int,
    shuffle_seed: int,
    bootstrap_seed: int,
    device: str,
) -> CaseBankEvaluation:
    """Fit development cases and evaluate input-only retrieval on queries."""

    dev_fit = _validate_samples(
        development_input_fit, name="development fit input"
    )
    dev_target_fit = _validate_samples(
        development_target_fit, name="development fit target"
    )
    dev_eval = _validate_samples(
        development_input_evaluation, name="development evaluation input"
    )
    dev_target_eval = _validate_samples(
        development_target_evaluation,
        name="development evaluation target",
    )
    query_eval = _validate_samples(
        query_input_evaluation, name="query evaluation input"
    )
    query_target = _validate_samples(
        query_target_evaluation, name="query evaluation target"
    )
    if (
        dev_fit.shape != dev_target_fit.shape
        or dev_eval.shape != dev_target_eval.shape
        or query_eval.shape != query_target.shape
        or len(dev_fit) != len(dev_eval)
    ):
        raise ValueError("case-bank aligned sample shapes differ")

    shared = _fit_flow(
        dev_fit.reshape(-1, 3),
        dev_target_fit.reshape(-1, 3),
        settings=operator_settings,
        seed=shared_seed,
        device=device,
    )
    all_cases = tuple(
        _fit_flow(
            dev_fit[index],
            dev_target_fit[index],
            settings=operator_settings,
            seed=case_seed_base + index,
            device=device,
        )
        for index in range(len(dev_fit))
    )
    self_identity_rmse = _rmse(dev_eval, dev_target_eval)
    self_case_prediction = np.stack(
        [
            operator.apply(dev_eval[index])
            for index, operator in enumerate(all_cases)
        ]
    )
    self_case_rmse = _rmse(self_case_prediction, dev_target_eval)
    self_improvement = 1.0 - self_case_rmse / np.maximum(
        self_identity_rmse, 1e-30
    )
    eligible_mask = self_improvement >= float(
        gates["minimum_case_self_improvement_fraction"]
    )
    eligible_indices = np.flatnonzero(eligible_mask)
    eligible = tuple(all_cases[index] for index in eligible_indices)
    if len(eligible) < 2:
        structure = {
            "sampled_operator_count": 1,
            "minimum_output": 0.0,
            "maximum_output": 1.0,
            "minimum_jacobian_determinant": 0.0,
            "maximum_jacobian_spectral_norm": 0.0,
            "maximum_inverse_error": 0.0,
            "maximum_replay_error": 0.0,
            "maximum_coefficient_vector_norm": 0.0,
            "coefficient_cap": float(
                operator_settings["coefficient_vector_norm_cap"]
            ),
        }
    else:
        structure = evaluate_flow_structure(
            [shared, *eligible],
            coefficient_cap=float(
                operator_settings["coefficient_vector_norm_cap"]
            ),
        )

    identity_rmse = _rmse(query_eval, query_target)
    shared_prediction = np.stack(
        [shared.apply(image) for image in query_eval]
    )
    shared_rmse = _rmse(shared_prediction, query_target)
    if len(eligible) >= 2:
        loss_matrix = _apply_operator_bank(
            eligible, query_eval, query_target
        )
        oracle_indices = np.argmin(loss_matrix, axis=1)
        oracle_rmse = loss_matrix[np.arange(len(query_eval)), oracle_indices]
        random_expected_rmse = np.mean(loss_matrix, axis=1)

        bank_input = dev_eval[eligible_indices]
        bank_global = content_descriptor(
            bank_input, cell_ids, include_spatial_cells=False
        )
        query_global = content_descriptor(
            query_eval, cell_ids, include_spatial_cells=False
        )
        global_indices, global_margins = standardized_nearest_indices(
            bank_global, query_global
        )
        global_rmse = loss_matrix[
            np.arange(len(query_eval)), global_indices
        ]

        bank_spatial = content_descriptor(
            bank_input, cell_ids, include_spatial_cells=True
        )
        query_spatial = content_descriptor(
            query_eval, cell_ids, include_spatial_cells=True
        )
        spatial_indices, spatial_margins = standardized_nearest_indices(
            bank_spatial, query_spatial
        )
        spatial_rmse = loss_matrix[
            np.arange(len(query_eval)), spatial_indices
        ]
        permutation = np.random.default_rng(shuffle_seed).permutation(
            len(eligible)
        )
        shuffled_indices = permutation[spatial_indices]
        shuffled_rmse = loss_matrix[
            np.arange(len(query_eval)), shuffled_indices
        ]
    else:
        loss_matrix = shared_rmse[:, None].copy()
        oracle_indices = np.zeros(len(query_eval), dtype=np.int64)
        global_indices = np.zeros(len(query_eval), dtype=np.int64)
        spatial_indices = np.zeros(len(query_eval), dtype=np.int64)
        shuffled_indices = np.zeros(len(query_eval), dtype=np.int64)
        oracle_rmse = shared_rmse.copy()
        random_expected_rmse = shared_rmse.copy()
        global_rmse = shared_rmse.copy()
        spatial_rmse = shared_rmse.copy()
        shuffled_rmse = shared_rmse.copy()
        global_margins = np.zeros(len(query_eval))
        spatial_margins = np.zeros(len(query_eval))

    oracle_improvement = 1.0 - float(np.mean(oracle_rmse)) / max(
        float(np.mean(shared_rmse)), 1e-30
    )
    oracle_win_fraction = float(np.mean(oracle_rmse < shared_rmse))
    spatial_gap_closure = oracle_gap_closure(
        shared_rmse, spatial_rmse, oracle_rmse
    )
    bootstrap = bootstrap_mean_improvement_ci(
        shared_rmse,
        oracle_rmse,
        seed=bootstrap_seed,
        samples=int(gates["bootstrap_samples"]),
        confidence=float(gates["bootstrap_confidence"]),
    )
    structure_all = (
        structure["minimum_output"] >= float(gates["minimum_output"])
        and structure["maximum_output"] <= float(gates["maximum_output"])
        and structure["minimum_jacobian_determinant"]
        > float(gates["minimum_jacobian_determinant_exclusive"])
        and structure["maximum_jacobian_spectral_norm"]
        <= float(gates["maximum_jacobian_spectral_norm"])
        and structure["maximum_inverse_error"]
        <= float(gates["maximum_inverse_error"])
        and structure["maximum_replay_error"]
        <= float(gates["maximum_replay_error"])
        and structure["maximum_coefficient_vector_norm"]
        <= float(gates["maximum_coefficient_vector_norm"])
    )
    gate_results = {
        "minimum_eligible_bank": len(eligible)
        >= int(gates["minimum_eligible_case_count"]),
        "structure_all": bool(structure_all),
        "oracle_improvement": oracle_improvement
        >= float(gates["minimum_oracle_improvement_over_shared_fraction"]),
        "oracle_win_fraction": oracle_win_fraction
        >= float(gates["minimum_oracle_win_fraction"]),
        "oracle_bootstrap": bootstrap["lower"]
        > float(gates["minimum_oracle_bootstrap_improvement_lower"]),
        "retrieval_gap_closure": spatial_gap_closure
        >= float(gates["minimum_retrieval_oracle_gap_closure"]),
        "retrieval_beats_random": float(np.mean(spatial_rmse))
        <= float(np.mean(random_expected_rmse))
        * (1.0 - float(gates["minimum_retrieval_gain_over_random_fraction"])),
        "retrieval_beats_shuffled": float(np.mean(spatial_rmse))
        <= float(np.mean(shuffled_rmse))
        * (
            1.0
            - float(gates["minimum_retrieval_gain_over_shuffled_fraction"])
        ),
    }
    report = {
        "development_case_count": len(all_cases),
        "eligible_case_count": len(eligible),
        "eligible_development_indices": eligible_indices.tolist(),
        "case_self_improvement_fraction": _summary(self_improvement),
        "linear_rgb_rmse": {
            "identity": _summary(identity_rmse),
            "shared_o0": _summary(shared_rmse),
            "case_bank_oracle": _summary(oracle_rmse),
            "global_photometric_nn": _summary(global_rmse),
            "spatial_photometric_nn": _summary(spatial_rmse),
            "random_case_expected": _summary(random_expected_rmse),
            "shuffled_spatial_assignment": _summary(shuffled_rmse),
        },
        "oracle_improvement_over_shared_fraction": oracle_improvement,
        "oracle_win_fraction": oracle_win_fraction,
        "oracle_bootstrap_mean_improvement": bootstrap,
        "oracle_gap_closure": {
            "global_photometric_nn": oracle_gap_closure(
                shared_rmse, global_rmse, oracle_rmse
            ),
            "spatial_photometric_nn": spatial_gap_closure,
            "random_case_expected": oracle_gap_closure(
                shared_rmse, random_expected_rmse, oracle_rmse
            ),
            "shuffled_spatial_assignment": oracle_gap_closure(
                shared_rmse, shuffled_rmse, oracle_rmse
            ),
        },
        "selection": {
            "oracle_indices": oracle_indices.tolist(),
            "global_photometric_nn_indices": global_indices.tolist(),
            "spatial_photometric_nn_indices": spatial_indices.tolist(),
            "shuffled_spatial_indices": shuffled_indices.tolist(),
            "global_nn_margin": _summary(global_margins),
            "spatial_nn_margin": _summary(spatial_margins),
        },
        "structure": structure,
        "gate_results": gate_results,
        "decision_branch_before_repeat": decide_case_retrieval_branch(
            gate_results
        ),
    }
    return CaseBankEvaluation(
        report=report,
        eligible_operators=eligible,
        query_input_evaluation=query_eval,
        query_target_evaluation=query_target,
        oracle_rmse=oracle_rmse,
    )


def evaluate_wrong_recipe_oracle(
    evaluation: CaseBankEvaluation,
    wrong_operators: tuple[CubeDiffeomorphicColourFlow, ...],
) -> dict[str, Any]:
    """Best post-hoc operator from other known-recipe banks."""

    if len(wrong_operators) < 2:
        return {
            "status": "unavailable_insufficient_wrong_recipe_bank",
            "wrong_operator_count": len(wrong_operators),
            "best_wrong_recipe_rmse": None,
            "correct_recipe_oracle_improvement_over_wrong_fraction": None,
            "correct_recipe_oracle_win_fraction": None,
            "note": (
                "post-hoc negative control unavailable; never an "
                "inference-time selector"
            ),
        }
    loss_matrix = _apply_operator_bank(
        wrong_operators,
        evaluation.query_input_evaluation,
        evaluation.query_target_evaluation,
    )
    best = np.min(loss_matrix, axis=1)
    correct = evaluation.oracle_rmse
    return {
        "status": "available",
        "wrong_operator_count": len(wrong_operators),
        "best_wrong_recipe_rmse": _summary(best),
        "correct_recipe_oracle_improvement_over_wrong_fraction": 1.0
        - float(np.mean(correct)) / max(float(np.mean(best)), 1e-30),
        "correct_recipe_oracle_win_fraction": float(np.mean(correct < best)),
        "note": "post-hoc negative control; never an inference-time selector",
    }


__all__ = [
    "CaseBankEvaluation",
    "bootstrap_mean_improvement_ci",
    "content_descriptor",
    "decide_case_retrieval_branch",
    "evaluate_within_recipe_case_bank",
    "evaluate_wrong_recipe_oracle",
    "oracle_gap_closure",
    "standardized_nearest_indices",
]
