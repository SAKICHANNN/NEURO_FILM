"""Development-only asymmetric query/operator applicability controls.

The module predicts a scalar loss for each member of a fixed explicit operator
bank.  It never predicts pixels or operator parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .cube_diffeomorphic_flow import CubeDiffeomorphicColourFlow
from .filmset_case_retrieval import (
    bootstrap_mean_improvement_ci,
    oracle_gap_closure,
    standardized_nearest_indices,
)


def _matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 2
        or min(array.shape) < 1
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite non-empty matrix")
    return array


def _vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite non-empty vector")
    return array


def _summary(values: np.ndarray) -> dict[str, float]:
    array = _vector(values, name="summary values")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def _fractional_improvement(
    baseline: np.ndarray, challenger: np.ndarray
) -> float:
    base = float(np.mean(_vector(baseline, name="baseline")))
    candidate = float(np.mean(_vector(challenger, name="challenger")))
    return 1.0 - candidate / max(base, 1e-30)


@dataclass(frozen=True)
class StandardizedPCA:
    """A deterministic train-only z-score plus PCA transform."""

    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = _matrix(values, name="PCA values")
        if matrix.shape[1] != len(self.mean):
            raise ValueError("PCA feature dimension differs")
        return ((matrix - self.mean) / self.scale) @ self.components.T


def fit_standardized_pca(
    values: np.ndarray, *, rank: int
) -> StandardizedPCA:
    """Fit deterministic PCA and orient component signs reproducibly."""

    matrix = _matrix(values, name="PCA training values")
    if rank < 1 or rank > min(matrix.shape):
        raise ValueError("PCA rank exceeds training matrix support")
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - mean) / scale
    _u, _singular, vt = np.linalg.svd(standardized, full_matrices=False)
    components = vt[:rank].copy()
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return StandardizedPCA(
        mean=mean.copy(),
        scale=scale.copy(),
        components=components,
    )


@dataclass(frozen=True)
class RidgeLossModel:
    """Z-scored scalar ridge regressor with an unpenalised intercept."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    interaction: bool

    def predict(
        self, query_features: np.ndarray, operator_features: np.ndarray
    ) -> np.ndarray:
        raw = pair_feature_matrix(
            query_features,
            operator_features,
            interaction=self.interaction,
        )
        normalized = (raw - self.feature_mean) / self.feature_scale
        design = np.column_stack([np.ones(len(normalized)), normalized])
        return (design @ self.coefficients).reshape(
            len(query_features), len(operator_features)
        )


def pair_feature_matrix(
    query_features: np.ndarray,
    operator_features: np.ndarray,
    *,
    interaction: bool,
) -> np.ndarray:
    """Return ordered query/operator pair features."""

    query = _matrix(query_features, name="query features")
    operator = _matrix(operator_features, name="operator features")
    query_pairs = np.repeat(query, len(operator), axis=0)
    operator_pairs = np.tile(operator, (len(query), 1))
    parts = [query_pairs, operator_pairs]
    if interaction:
        outer = np.einsum(
            "ni,nj->nij", query_pairs, operator_pairs
        ).reshape(len(query_pairs), -1)
        parts.append(outer)
    return np.concatenate(parts, axis=1)


def fit_ridge_loss_model(
    query_features: np.ndarray,
    operator_features: np.ndarray,
    loss_matrix: np.ndarray,
    *,
    query_source_indices: np.ndarray,
    operator_source_indices: np.ndarray,
    interaction: bool,
    ridge_penalty: float,
) -> RidgeLossModel:
    """Fit off-diagonal pair loss without same-case pairs."""

    query = _matrix(query_features, name="query features")
    operator = _matrix(operator_features, name="operator features")
    losses = _matrix(loss_matrix, name="loss matrix")
    query_sources = np.asarray(query_source_indices, dtype=np.int64)
    operator_sources = np.asarray(operator_source_indices, dtype=np.int64)
    if (
        losses.shape != (len(query), len(operator))
        or query_sources.shape != (len(query),)
        or operator_sources.shape != (len(operator),)
        or ridge_penalty < 0.0
    ):
        raise ValueError("ridge pair shapes or penalty are invalid")
    raw = pair_feature_matrix(query, operator, interaction=interaction)
    keep = (
        query_sources[:, None] != operator_sources[None, :]
    ).reshape(-1)
    if int(np.sum(keep)) <= raw.shape[1] + 1:
        raise ValueError("insufficient off-diagonal pairs for ridge fit")
    features = raw[keep]
    targets = losses.reshape(-1)[keep]
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (features - mean) / scale
    design = np.column_stack([np.ones(len(normalized)), normalized])
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_penalty
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ targets,
    )
    return RidgeLossModel(
        feature_mean=mean,
        feature_scale=scale,
        coefficients=coefficients,
        interaction=interaction,
    )


def operator_signature(
    operator: CubeDiffeomorphicColourFlow,
    shared_operator: CubeDiffeomorphicColourFlow,
) -> np.ndarray:
    """Return the frozen shared-centred explicit operator signature."""

    residual = np.asarray(
        operator.velocity_grid - shared_operator.velocity_grid,
        dtype=np.float64,
    )
    if residual.shape != (4, 4, 4, 3) or not np.all(np.isfinite(residual)):
        raise ValueError("Z1 requires finite 4x4x4x3 O0 velocity grids")
    channel_means = np.mean(residual, axis=(0, 1, 2))
    channel_rms = np.sqrt(np.mean(residual**2, axis=(0, 1, 2)))
    global_rms = np.array([np.sqrt(np.mean(residual**2))])
    maximum_node_norm = np.array(
        [np.max(np.linalg.norm(residual, axis=-1))]
    )
    return np.concatenate(
        [
            residual.reshape(-1),
            channel_means,
            channel_rms,
            global_rms,
            maximum_node_norm,
        ]
    )


def operator_signature_matrix(
    operators: tuple[CubeDiffeomorphicColourFlow, ...],
    shared_operator: CubeDiffeomorphicColourFlow,
) -> np.ndarray:
    if not operators:
        raise ValueError("operator bank is empty")
    return np.stack(
        [operator_signature(operator, shared_operator) for operator in operators]
    )


def _standardized_support_distance(
    training: np.ndarray, query: np.ndarray
) -> float:
    train = _matrix(training, name="support training")
    item = np.asarray(query, dtype=np.float64)
    if item.shape != (train.shape[1],):
        raise ValueError("support query dimension differs")
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    train_z = (train - mean) / scale
    query_z = (item - mean) / scale
    distance = np.linalg.norm(train_z - query_z, axis=1)
    return float(np.min(distance) / np.sqrt(train.shape[1]))


def development_support_distances(query_raw: np.ndarray) -> np.ndarray:
    query = _matrix(query_raw, name="development query descriptors")
    if len(query) < 3:
        raise ValueError("at least three development queries are required")
    distances = []
    for index in range(len(query)):
        keep = np.arange(len(query)) != index
        distances.append(
            _standardized_support_distance(query[keep], query[index])
        )
    return np.asarray(distances, dtype=np.float64)


@dataclass(frozen=True)
class DevelopmentSelection:
    rank: int
    selected_indices: np.ndarray
    selected_loss: np.ndarray
    raw_selected_loss: np.ndarray
    score_margins: np.ndarray
    support_distances: np.ndarray
    ood_threshold: float
    fallback_mask: np.ndarray


def leave_one_case_selection(
    query_raw: np.ndarray,
    operator_raw: np.ndarray,
    loss_matrix: np.ndarray,
    shared_loss: np.ndarray,
    operator_source_indices: np.ndarray,
    *,
    rank: int,
    ridge_penalty: float,
    ood_quantile: float,
    interaction: bool = True,
    signature_permutation_seed: int | None = None,
) -> DevelopmentSelection:
    """Run leakage-safe leave-one-query-and-operator-out selection."""

    query = _matrix(query_raw, name="development query descriptors")
    operator = _matrix(operator_raw, name="operator signatures")
    losses = _matrix(loss_matrix, name="development loss matrix")
    shared = _vector(shared_loss, name="development shared loss")
    operator_sources = np.asarray(operator_source_indices, dtype=np.int64)
    if (
        losses.shape != (len(query), len(operator))
        or len(shared) != len(query)
        or operator_sources.shape != (len(operator),)
        or not (0.0 <= ood_quantile <= 1.0)
    ):
        raise ValueError("development selection shapes are invalid")
    query_sources = np.arange(len(query), dtype=np.int64)
    support = development_support_distances(query)
    threshold = float(np.quantile(support, ood_quantile))
    selected_indices: list[int] = []
    selected_losses: list[float] = []
    raw_losses: list[float] = []
    margins: list[float] = []
    fallback: list[bool] = []
    for held_index in range(len(query)):
        query_keep = query_sources != held_index
        operator_keep = operator_sources != held_index
        if int(np.sum(query_keep)) <= rank or int(np.sum(operator_keep)) <= rank:
            raise ValueError("development fold does not support PCA rank")
        query_pca = fit_standardized_pca(query[query_keep], rank=rank)
        fold_operator = operator[operator_keep]
        if signature_permutation_seed is not None:
            permutation = np.random.default_rng(
                signature_permutation_seed + held_index
            ).permutation(len(fold_operator))
            fold_operator = fold_operator[permutation]
        operator_pca = fit_standardized_pca(
            fold_operator, rank=rank
        )
        train_query = query_pca.transform(query[query_keep])
        train_operator = operator_pca.transform(fold_operator)
        model = fit_ridge_loss_model(
            train_query,
            train_operator,
            losses[np.ix_(query_keep, operator_keep)],
            query_source_indices=query_sources[query_keep],
            operator_source_indices=operator_sources[operator_keep],
            interaction=interaction,
            ridge_penalty=ridge_penalty,
        )
        scores = model.predict(
            query_pca.transform(query[held_index : held_index + 1]),
            train_operator,
        )[0]
        local_index = int(np.argmin(scores))
        candidate_indices = np.flatnonzero(operator_keep)
        selected_index = int(candidate_indices[local_index])
        ordered = np.sort(scores)
        margin = float(ordered[1] - ordered[0]) if len(ordered) > 1 else 0.0
        raw_loss = float(losses[held_index, selected_index])
        is_ood = bool(support[held_index] > threshold)
        selected_indices.append(-1 if is_ood else selected_index)
        raw_losses.append(raw_loss)
        selected_losses.append(float(shared[held_index]) if is_ood else raw_loss)
        margins.append(margin)
        fallback.append(is_ood)
    return DevelopmentSelection(
        rank=rank,
        selected_indices=np.asarray(selected_indices, dtype=np.int64),
        selected_loss=np.asarray(selected_losses, dtype=np.float64),
        raw_selected_loss=np.asarray(raw_losses, dtype=np.float64),
        score_margins=np.asarray(margins, dtype=np.float64),
        support_distances=support,
        ood_threshold=threshold,
        fallback_mask=np.asarray(fallback, dtype=bool),
    )


def leave_one_case_oracle(
    loss_matrix: np.ndarray, operator_source_indices: np.ndarray
) -> np.ndarray:
    losses = _matrix(loss_matrix, name="development loss matrix")
    operator_sources = np.asarray(operator_source_indices, dtype=np.int64)
    if operator_sources.shape != (losses.shape[1],):
        raise ValueError("operator source indices differ")
    result = []
    for query_index in range(losses.shape[0]):
        candidates = operator_sources != query_index
        if not np.any(candidates):
            raise ValueError("no off-diagonal oracle candidate")
        result.append(float(np.min(losses[query_index, candidates])))
    return np.asarray(result, dtype=np.float64)


def leave_one_case_random(
    loss_matrix: np.ndarray, operator_source_indices: np.ndarray
) -> np.ndarray:
    losses = _matrix(loss_matrix, name="development loss matrix")
    operator_sources = np.asarray(operator_source_indices, dtype=np.int64)
    return np.asarray(
        [
            np.mean(losses[index, operator_sources != index])
            for index in range(len(losses))
        ],
        dtype=np.float64,
    )


def leave_one_case_medoid(
    loss_matrix: np.ndarray, operator_source_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    losses = _matrix(loss_matrix, name="development loss matrix")
    operator_sources = np.asarray(operator_source_indices, dtype=np.int64)
    selected = []
    values = []
    query_indices = np.arange(len(losses), dtype=np.int64)
    for held_index in query_indices:
        query_keep = query_indices != held_index
        operator_keep = operator_sources != held_index
        candidate_columns = np.flatnonzero(operator_keep)
        candidate_means = []
        for column in candidate_columns:
            valid_queries = query_keep & (
                query_indices != operator_sources[column]
            )
            candidate_means.append(np.mean(losses[valid_queries, column]))
        chosen = int(candidate_columns[int(np.argmin(candidate_means))])
        selected.append(chosen)
        values.append(float(losses[held_index, chosen]))
    return (
        np.asarray(selected, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
    )


def leave_one_case_photometric(
    bank_features: np.ndarray,
    query_features: np.ndarray,
    loss_matrix: np.ndarray,
    operator_source_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bank = _matrix(bank_features, name="photometric bank")
    query = _matrix(query_features, name="photometric query")
    losses = _matrix(loss_matrix, name="photometric loss matrix")
    operator_sources = np.asarray(operator_source_indices, dtype=np.int64)
    if (
        bank.shape[0] != losses.shape[1]
        or query.shape[0] != losses.shape[0]
        or bank.shape[1] != query.shape[1]
    ):
        raise ValueError("photometric control shapes differ")
    selected = []
    values = []
    for query_index in range(len(query)):
        keep = operator_sources != query_index
        candidates = np.flatnonzero(keep)
        local, _margin = standardized_nearest_indices(
            bank[keep], query[query_index : query_index + 1]
        )
        chosen = int(candidates[int(local[0])])
        selected.append(chosen)
        values.append(float(losses[query_index, chosen]))
    return (
        np.asarray(selected, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
    )


@dataclass(frozen=True)
class SealedApplicabilityPolicy:
    """Development-fitted hard selector plus immutable OOD fallback state."""

    rank: int
    query_pca: StandardizedPCA
    operator_pca: StandardizedPCA
    loss_model: RidgeLossModel
    development_query_raw: np.ndarray
    operator_raw: np.ndarray
    ood_threshold: float

    def select(
        self, query_raw: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query = _matrix(query_raw, name="sealed-policy query")
        scores = self.loss_model.predict(
            self.query_pca.transform(query),
            self.operator_pca.transform(self.operator_raw),
        )
        indices = np.argmin(scores, axis=1).astype(np.int64)
        margins = np.sort(scores, axis=1)[:, 1] - np.sort(scores, axis=1)[:, 0]
        support = np.asarray(
            [
                _standardized_support_distance(
                    self.development_query_raw, item
                )
                for item in query
            ],
            dtype=np.float64,
        )
        indices[support > self.ood_threshold] = -1
        return indices, margins, support


def fit_sealed_policy(
    query_raw: np.ndarray,
    operator_raw: np.ndarray,
    loss_matrix: np.ndarray,
    operator_source_indices: np.ndarray,
    *,
    rank: int,
    ridge_penalty: float,
    ood_threshold: float,
    interaction: bool = True,
) -> SealedApplicabilityPolicy:
    query = _matrix(query_raw, name="sealed development query")
    operator = _matrix(operator_raw, name="sealed operator signature")
    losses = _matrix(loss_matrix, name="sealed development loss")
    sources = np.asarray(operator_source_indices, dtype=np.int64)
    query_pca = fit_standardized_pca(query, rank=rank)
    operator_pca = fit_standardized_pca(operator, rank=rank)
    model = fit_ridge_loss_model(
        query_pca.transform(query),
        operator_pca.transform(operator),
        losses,
        query_source_indices=np.arange(len(query), dtype=np.int64),
        operator_source_indices=sources,
        interaction=interaction,
        ridge_penalty=ridge_penalty,
    )
    return SealedApplicabilityPolicy(
        rank=rank,
        query_pca=query_pca,
        operator_pca=operator_pca,
        loss_model=model,
        development_query_raw=query.copy(),
        operator_raw=operator.copy(),
        ood_threshold=float(ood_threshold),
    )


def selected_policy_loss(
    indices: np.ndarray,
    loss_matrix: np.ndarray,
    shared_loss: np.ndarray,
) -> np.ndarray:
    chosen = np.asarray(indices, dtype=np.int64)
    losses = _matrix(loss_matrix, name="policy loss matrix")
    shared = _vector(shared_loss, name="policy shared loss")
    if chosen.shape != (len(losses),) or len(shared) != len(losses):
        raise ValueError("selected policy shapes differ")
    if np.any(chosen < -1) or np.any(chosen >= losses.shape[1]):
        raise ValueError("selected policy index out of range")
    return np.asarray(
        [
            shared[index]
            if column == -1
            else losses[index, int(column)]
            for index, column in enumerate(chosen)
        ],
        dtype=np.float64,
    )


def applicability_metrics(
    *,
    shared_loss: np.ndarray,
    oracle_loss: np.ndarray,
    policy_loss: np.ndarray,
    medoid_loss: np.ndarray,
    spatial_loss: np.ndarray,
    random_loss: np.ndarray,
    shuffled_loss: np.ndarray,
    fallback_mask: np.ndarray,
    bootstrap_seed: int,
    bootstrap_samples: int,
    bootstrap_confidence: float,
) -> dict[str, Any]:
    shared = _vector(shared_loss, name="shared loss")
    oracle = _vector(oracle_loss, name="oracle loss")
    policy = _vector(policy_loss, name="policy loss")
    controls = {
        "medoid": _vector(medoid_loss, name="medoid loss"),
        "spatial_photometric": _vector(
            spatial_loss, name="spatial loss"
        ),
        "random": _vector(random_loss, name="random loss"),
        "shuffled": _vector(shuffled_loss, name="shuffled loss"),
    }
    fallback = np.asarray(fallback_mask, dtype=bool)
    if (
        any(len(value) != len(shared) for value in [oracle, policy, *controls.values()])
        or fallback.shape != (len(shared),)
    ):
        raise ValueError("applicability metric vectors differ")
    bootstrap = bootstrap_mean_improvement_ci(
        shared,
        policy,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
        confidence=bootstrap_confidence,
    )
    return {
        "linear_rgb_rmse": {
            "shared_o0": _summary(shared),
            "case_bank_oracle": _summary(oracle),
            "hard_asymmetric_policy": _summary(policy),
            **{name: _summary(value) for name, value in controls.items()},
        },
        "oracle_improvement_over_shared_fraction": _fractional_improvement(
            shared, oracle
        ),
        "oracle_win_fraction": float(np.mean(oracle < shared)),
        "policy_oracle_gap_closure": oracle_gap_closure(
            shared, policy, oracle
        ),
        "policy_improvement_fraction": {
            "shared": _fractional_improvement(shared, policy),
            **{
                name: _fractional_improvement(value, policy)
                for name, value in controls.items()
            },
        },
        "policy_win_fraction_over_shared": float(np.mean(policy < shared)),
        "policy_bootstrap_mean_improvement": bootstrap,
        "ood_fallback_fraction": float(np.mean(fallback)),
    }


def oracle_metrics(
    shared_loss: np.ndarray,
    oracle_loss: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
    bootstrap_confidence: float,
) -> dict[str, Any]:
    shared = _vector(shared_loss, name="oracle shared loss")
    oracle = _vector(oracle_loss, name="oracle candidate loss")
    bootstrap = bootstrap_mean_improvement_ci(
        shared,
        oracle,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
        confidence=bootstrap_confidence,
    )
    return {
        "improvement_over_shared_fraction": _fractional_improvement(
            shared, oracle
        ),
        "win_fraction": float(np.mean(oracle < shared)),
        "bootstrap_mean_improvement": bootstrap,
    }


def applicability_gate_results(
    metrics: dict[str, Any],
    oracle: dict[str, Any],
    *,
    eligible_case_count: int,
    structure_all: bool,
    gates: dict[str, Any],
) -> dict[str, bool]:
    improvements = metrics["policy_improvement_fraction"]
    return {
        "minimum_eligible_bank": eligible_case_count
        >= int(gates["minimum_eligible_case_count"]),
        "structure_all": bool(structure_all),
        "oracle_improvement": oracle["improvement_over_shared_fraction"]
        >= float(gates["minimum_oracle_improvement_over_shared_fraction"]),
        "oracle_win_fraction": oracle["win_fraction"]
        >= float(gates["minimum_oracle_win_fraction"]),
        "oracle_bootstrap": oracle["bootstrap_mean_improvement"]["lower"]
        > float(gates["minimum_oracle_bootstrap_improvement_lower"]),
        "policy_oracle_gap_closure": metrics["policy_oracle_gap_closure"]
        >= float(gates["minimum_policy_oracle_gap_closure"]),
        "policy_beats_shared": improvements["shared"]
        >= float(gates["minimum_policy_improvement_over_shared_fraction"]),
        "policy_beats_medoid": improvements["medoid"]
        >= float(gates["minimum_policy_improvement_over_medoid_fraction"]),
        "policy_beats_spatial": improvements["spatial_photometric"]
        >= float(
            gates[
                "minimum_policy_improvement_over_spatial_photometric_fraction"
            ]
        ),
        "policy_beats_random": improvements["random"]
        >= float(gates["minimum_policy_improvement_over_random_fraction"]),
        "policy_beats_shuffled": improvements["shuffled"]
        >= float(gates["minimum_policy_improvement_over_shuffled_fraction"]),
        "policy_win_fraction": metrics["policy_win_fraction_over_shared"]
        >= float(gates["minimum_policy_win_fraction_over_shared"]),
        "policy_bootstrap": metrics["policy_bootstrap_mean_improvement"][
            "lower"
        ]
        > float(gates["minimum_policy_bootstrap_improvement_lower"]),
        "ood_coverage": metrics["ood_fallback_fraction"]
        <= float(gates["maximum_ood_fallback_fraction"]),
    }


def all_gates_pass(gate_results: dict[str, bool]) -> bool:
    return bool(gate_results) and all(
        isinstance(value, (bool, np.bool_)) and bool(value)
        for value in gate_results.values()
    )


__all__ = [
    "DevelopmentSelection",
    "RidgeLossModel",
    "SealedApplicabilityPolicy",
    "StandardizedPCA",
    "all_gates_pass",
    "applicability_gate_results",
    "applicability_metrics",
    "development_support_distances",
    "fit_ridge_loss_model",
    "fit_sealed_policy",
    "fit_standardized_pca",
    "leave_one_case_medoid",
    "leave_one_case_oracle",
    "leave_one_case_photometric",
    "leave_one_case_random",
    "leave_one_case_selection",
    "operator_signature",
    "operator_signature_matrix",
    "oracle_metrics",
    "pair_feature_matrix",
    "selected_policy_loss",
]
