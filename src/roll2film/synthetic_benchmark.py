"""CPU benchmark for recovering generated bounded colour operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2lab

from .lab_statistics import extract_lab_statistics
from .synthetic_recovery import (
    SyntheticBoundedOperator,
    audit_synthetic_operator,
    operator_from_manifest,
    palette_cloud,
    palette_pairs,
    project_operator_parameters,
    uniform_probe_grid,
)


@dataclass(frozen=True)
class RecoveryObservation:
    operator_id: str
    family: str
    split: str
    query_palette: str
    reference_palette: str
    parameters: np.ndarray
    features: dict[str, np.ndarray]


class PCAPredictor:
    """Development-fit-only scaler/PCA plus one deterministic regressor."""

    def __init__(
        self,
        *,
        kind: str,
        hyperparameter: float | int,
        components: int,
        seed: int,
    ) -> None:
        if kind not in {"ridge", "knn"}:
            raise ValueError(f"unsupported predictor kind: {kind!r}")
        self.kind = kind
        self.hyperparameter = hyperparameter
        self.components = int(components)
        self.seed = int(seed)
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.model: Ridge | KNeighborsRegressor | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "PCAPredictor":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or y.shape[1] != 9:
            raise ValueError("features/targets must be aligned 2D arrays with nine targets")
        standardized = self.scaler.fit_transform(x)
        component_count = min(self.components, len(x) - 1, x.shape[1])
        if component_count < 1:
            raise ValueError("not enough observations for PCA")
        self.pca = PCA(
            n_components=component_count,
            svd_solver="randomized",
            random_state=self.seed,
        )
        reduced = self.pca.fit_transform(standardized)
        if self.kind == "ridge":
            self.model = Ridge(alpha=float(self.hyperparameter), solver="lsqr")
        else:
            self.model = KNeighborsRegressor(
                n_neighbors=int(self.hyperparameter),
                weights="distance",
                p=2,
            )
        self.model.fit(reduced, y)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.pca is None or self.model is None:
            raise RuntimeError("predictor must be fit before prediction")
        x = np.asarray(features, dtype=np.float64)
        return np.asarray(
            self.model.predict(self.pca.transform(self.scaler.transform(x))),
            dtype=np.float64,
        )


def build_observations(
    config: dict[str, Any],
    manifest: list[dict[str, Any]],
) -> list[RecoveryObservation]:
    palette_spec = config["palettes"]
    pixel_count = int(palette_spec["pixels_per_cloud"])
    seed = int(config["seed"])
    names_by_split = {
        "fit": list(palette_spec["development"]),
        "validation": list(palette_spec["development"]),
        "confirmation": list(palette_spec["confirmation"]),
        "stress": list(palette_spec["stress"]),
    }
    all_names = sorted({name for names in names_by_split.values() for name in names})
    clouds = {name: palette_cloud(name, pixel_count, seed) for name in all_names}
    raw_descriptors = {
        name: extract_lab_statistics(cloud).vector() for name, cloud in clouds.items()
    }

    observations: list[RecoveryObservation] = []
    for row in manifest:
        operator = operator_from_manifest(row)
        split = str(row["split"])
        pairs = palette_pairs(
            names_by_split[split],
            operator_index=int(row["family_index"]),
            observations=2,
        )
        for query_name, reference_name in pairs:
            query = clouds[query_name]
            reference = clouds[reference_name]
            transformed_query = operator.apply(query)
            transformed_reference = operator.apply(reference)
            query_descriptor = raw_descriptors[query_name]
            reference_descriptor = raw_descriptors[reference_name]
            transformed_query_descriptor = extract_lab_statistics(
                transformed_query
            ).vector()
            transformed_reference_descriptor = extract_lab_statistics(
                transformed_reference
            ).vector()
            features = {
                "source_content_only_negative": query_descriptor,
                "style_target_only": transformed_reference_descriptor,
                "interaction_source_target": np.concatenate(
                    (query_descriptor, transformed_reference_descriptor)
                ),
                "canonical_reference_delta_oracle": (
                    transformed_reference_descriptor - reference_descriptor
                ),
                "matched_query_delta_oracle": (
                    transformed_query_descriptor - query_descriptor
                ),
                "shuffled_operator_negative": np.concatenate(
                    (query_descriptor, transformed_reference_descriptor)
                ),
            }
            observations.append(
                RecoveryObservation(
                    operator_id=operator.operator_id,
                    family=operator.family,
                    split=split,
                    query_palette=query_name,
                    reference_palette=reference_name,
                    parameters=operator.parameters,
                    features=features,
                )
            )
    return observations


def rows_for_split(
    observations: list[RecoveryObservation],
    split: str,
) -> list[RecoveryObservation]:
    rows = [row for row in observations if row.split == split]
    if not rows:
        raise ValueError(f"no observations for split {split!r}")
    return rows


def feature_matrix(
    observations: list[RecoveryObservation],
    representation: str,
) -> np.ndarray:
    try:
        return np.stack([row.features[representation] for row in observations])
    except KeyError as error:
        raise ValueError(f"unknown representation: {representation!r}") from error


def target_matrix(observations: list[RecoveryObservation]) -> np.ndarray:
    return np.stack([row.parameters for row in observations])


def shuffled_targets(
    observations: list[RecoveryObservation],
    *,
    seed: int,
) -> np.ndarray:
    operator_ids = sorted({row.operator_id for row in observations})
    rng = np.random.default_rng(seed)
    shuffled_ids = list(np.asarray(operator_ids)[rng.permutation(len(operator_ids))])
    source_by_id: dict[str, np.ndarray] = {}
    for row in observations:
        source_by_id.setdefault(row.operator_id, row.parameters)
    mapping = {
        source: source_by_id[target] for source, target in zip(operator_ids, shuffled_ids)
    }
    return np.stack([mapping[row.operator_id] for row in observations])


def _project_rows(
    raw_predictions: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    operator_spec = config["operator"]
    projected = np.stack(
        [
            project_operator_parameters(
                row,
                maximum_off_diagonal_sum=float(
                    operator_spec["matrix_maximum_total_off_diagonal_per_row"]
                ),
                minimum_tone_increment=float(
                    operator_spec["tone_minimum_knot_increment"]
                ),
            )
            for row in raw_predictions
        ]
    )
    movement = np.sqrt(np.mean((projected - raw_predictions) ** 2, axis=1))
    return projected, movement


def evaluate_predictions(
    observations: list[RecoveryObservation],
    raw_predictions: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    if len(observations) != len(raw_predictions):
        raise ValueError("one prediction is required per observation")
    projected, movement = _project_rows(raw_predictions, config)
    probes = uniform_probe_grid(int(config["evaluation"]["uniform_probe_grid_size"]))
    identity = probes
    per_observation: list[dict[str, Any]] = []
    for row, predicted_parameters, projection_movement in zip(
        observations, projected, movement
    ):
        true_operator = SyntheticBoundedOperator(
            row.parameters,
            row.family,
            row.operator_id,
        )
        predicted_operator = SyntheticBoundedOperator(
            predicted_parameters,
            "combined",
            f"predicted-{row.operator_id}",
        )
        true_rgb = true_operator.apply(probes)
        predicted_rgb = predicted_operator.apply(probes)
        difference = predicted_rgb - true_rgb
        per_pixel_l2 = np.linalg.norm(difference, axis=-1)
        predicted_lab = rgb2lab(predicted_rgb)
        true_lab = rgb2lab(true_rgb)
        delta_e = np.linalg.norm(predicted_lab - true_lab, axis=-1)
        rgb_rmse = float(np.sqrt(np.mean(difference**2)))
        identity_rmse = float(np.sqrt(np.mean((identity - true_rgb) ** 2)))
        validity = audit_synthetic_operator(
            predicted_operator,
            minimum_determinant=float(config["operator"]["matrix_minimum_determinant"]),
            maximum_off_diagonal_sum=float(
                config["operator"]["matrix_maximum_total_off_diagonal_per_row"]
            ),
            minimum_tone_increment=float(
                config["operator"]["tone_minimum_knot_increment"]
            ),
            neutral_axis_tolerance=float(
                config["operator"]["neutral_axis_maximum_error"]
            ),
        )
        per_observation.append(
            {
                "operator_id": row.operator_id,
                "family": row.family,
                "split": row.split,
                "query_palette": row.query_palette,
                "reference_palette": row.reference_palette,
                "parameter_rmse": float(
                    np.sqrt(np.mean((predicted_parameters - row.parameters) ** 2))
                ),
                "uniform_rgb_rmse": rgb_rmse,
                "uniform_rgb_p95_l2": float(np.percentile(per_pixel_l2, 95)),
                "uniform_rgb_maximum_l2": float(np.max(per_pixel_l2)),
                "delta_e76_median": float(np.median(delta_e)),
                "delta_e76_p95": float(np.percentile(delta_e, 95)),
                "identity_rgb_rmse": identity_rmse,
                "captured_style_fraction": float(
                    1.0 - rgb_rmse / max(identity_rmse, 1e-15)
                ),
                "projection_parameter_rmse": float(projection_movement),
                "valid": bool(validity["valid"]),
                "neutral_axis_error": float(validity["neutral_axis_error"]),
                "matrix_determinant": float(validity["matrix_determinant"]),
            }
        )
    return {
        "per_observation": per_observation,
        "summary": summarize_by_operator(per_observation),
    }


def summarize_by_operator(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("records must be non-empty")
    operator_ids = sorted({str(row["operator_id"]) for row in records})
    grouped = []
    numeric_keys = (
        "parameter_rmse",
        "uniform_rgb_rmse",
        "uniform_rgb_p95_l2",
        "uniform_rgb_maximum_l2",
        "delta_e76_median",
        "delta_e76_p95",
        "identity_rgb_rmse",
        "captured_style_fraction",
        "projection_parameter_rmse",
        "neutral_axis_error",
        "matrix_determinant",
    )
    for operator_id in operator_ids:
        rows = [row for row in records if row["operator_id"] == operator_id]
        grouped.append(
            {
                "operator_id": operator_id,
                "family": rows[0]["family"],
                **{
                    key: float(np.mean([float(row[key]) for row in rows]))
                    for key in numeric_keys
                },
                "valid": all(bool(row["valid"]) for row in rows),
            }
        )

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "operators": len(rows),
            "median_uniform_rgb_rmse": float(
                np.median([row["uniform_rgb_rmse"] for row in rows])
            ),
            "p95_uniform_rgb_rmse": float(
                np.percentile([row["uniform_rgb_rmse"] for row in rows], 95)
            ),
            "median_parameter_rmse": float(
                np.median([row["parameter_rmse"] for row in rows])
            ),
            "median_delta_e76": float(
                np.median([row["delta_e76_median"] for row in rows])
            ),
            "median_identity_rgb_rmse": float(
                np.median([row["identity_rgb_rmse"] for row in rows])
            ),
            "median_captured_style_fraction": float(
                np.median([row["captured_style_fraction"] for row in rows])
            ),
            "valid_fraction": float(np.mean([row["valid"] for row in rows])),
            "maximum_neutral_axis_error": float(
                np.max([row["neutral_axis_error"] for row in rows])
            ),
            "maximum_projection_parameter_rmse": float(
                np.max([row["projection_parameter_rmse"] for row in rows])
            ),
        }

    families = sorted({str(row["family"]) for row in grouped})
    return {
        "overall": aggregate(grouped),
        "by_family": {
            family: aggregate([row for row in grouped if row["family"] == family])
            for family in families
        },
        "operator_metrics": grouped,
    }


def bootstrap_median_interval(
    operator_metrics: list[dict[str, Any]],
    *,
    key: str,
    replicates: int,
    seed: int,
) -> list[float]:
    values = np.asarray([float(row[key]) for row in operator_metrics])
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        estimates[index] = np.median(rng.choice(values, size=len(values), replace=True))
    return [
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    ]


def fit_predictor_candidates(
    representation: str,
    fit_rows: list[RecoveryObservation],
    validation_rows: list[RecoveryObservation],
    config: dict[str, Any],
    *,
    shuffled: bool = False,
) -> tuple[PCAPredictor, dict[str, Any]]:
    x_fit = feature_matrix(fit_rows, representation)
    x_validation = feature_matrix(validation_rows, representation)
    y_fit = (
        shuffled_targets(fit_rows, seed=int(config["seed"]) + 91)
        if shuffled
        else target_matrix(fit_rows)
    )
    candidates: list[tuple[float, str, float | int, PCAPredictor, dict[str, Any]]] = []
    predictor_spec = config["predictors"]
    for kind, values in (
        ("ridge", predictor_spec["ridge_alphas"]),
        ("knn", predictor_spec["knn_neighbors"]),
    ):
        for value in values:
            predictor = PCAPredictor(
                kind=kind,
                hyperparameter=value,
                components=int(predictor_spec["pca_components"]),
                seed=int(config["seed"]),
            ).fit(x_fit, y_fit)
            evaluation = evaluate_predictions(
                validation_rows,
                predictor.predict(x_validation),
                config,
            )
            score = float(evaluation["summary"]["overall"]["median_uniform_rgb_rmse"])
            candidates.append((score, kind, value, predictor, evaluation))
    candidates.sort(key=lambda item: (item[0], item[1], float(item[2])))
    score, kind, value, predictor, evaluation = candidates[0]
    selection = {
        "representation": representation,
        "predictor": kind,
        "hyperparameter": value,
        "validation_median_uniform_rgb_rmse": score,
        "validation_summary": evaluation["summary"],
        "all_candidates": [
            {
                "predictor": candidate_kind,
                "hyperparameter": candidate_value,
                "validation_median_uniform_rgb_rmse": candidate_score,
            }
            for candidate_score, candidate_kind, candidate_value, _, _ in candidates
        ],
    }
    return predictor, selection
