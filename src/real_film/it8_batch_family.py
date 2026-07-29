"""Batch-held-out target-family measurement analysis primitives."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.real_film.it8_reference import (
    find_charge_table_member,
    find_unique_suffix_member,
    parse_cgats_spectral,
    parse_it8,
)


@dataclass(frozen=True)
class ChargeMeasurement:
    name: str
    family: str
    year: int
    sample_ids: tuple[str, ...]
    lab: np.ndarray
    spectra_pct: np.ndarray
    mean_de_median: float
    mean_de_p90: float


class CappedVariancePCA(TransformerMixin, BaseEstimator):
    """Development-fitted full SVD retaining variance within a hard rank cap."""

    def __init__(
        self, variance_fraction: float = 0.95, maximum_components: int = 32
    ) -> None:
        self.variance_fraction = variance_fraction
        self.maximum_components = maximum_components

    def fit(self, x: np.ndarray, y: np.ndarray | None = None):
        del y
        fitted = PCA(svd_solver="full").fit(x)
        cumulative = np.cumsum(fitted.explained_variance_ratio_)
        required = int(np.searchsorted(cumulative, self.variance_fraction) + 1)
        self.n_components_ = min(required, self.maximum_components)
        self.mean_ = fitted.mean_
        self.components_ = fitted.components_[: self.n_components_]
        self.retained_variance_ratio_ = float(
            np.sum(fitted.explained_variance_ratio_[: self.n_components_])
        )
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) @ self.components_.T


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_charge(path: Path, family: str) -> ChargeMeasurement:
    with ZipFile(path) as archive:
        it8 = parse_it8(
            archive.read(find_charge_table_member(archive, path.stem))
        )
        spectral = parse_cgats_spectral(
            archive.read(find_unique_suffix_member(archive, ".cgt"))
        )
    if it8.sample_ids != spectral.sample_ids:
        raise ValueError(f"sample order mismatch: {path}")
    mean_de = [float(row["MEAN_DE"]) for row in it8.rows]
    return ChargeMeasurement(
        name=path.name,
        family=family,
        year=2000 + int(path.name[1:3]),
        sample_ids=it8.sample_ids,
        lab=np.asarray(spectral.lab, dtype=np.float64),
        spectra_pct=np.asarray(spectral.spectra_pct, dtype=np.float64),
        mean_de_median=float(np.median(mean_de)),
        mean_de_p90=percentile(mean_de, 0.9),
    )


def residual_features(
    charges: list[ChargeMeasurement],
    development_indices: np.ndarray,
    *,
    view: str,
) -> np.ndarray:
    if view == "spectral":
        values = np.stack([charge.spectra_pct for charge in charges])
    elif view == "lab":
        values = np.stack([charge.lab for charge in charges])
    else:
        raise ValueError(f"unsupported view: {view}")
    pooled_aim = np.mean(values[development_indices], axis=0, dtype=np.float64)
    residual = values - pooled_aim[None, ...]
    if view == "spectral":
        residual = np.sign(residual) * np.log1p(np.abs(residual))
    return residual.reshape(len(charges), -1)


def make_family_pipeline(c_value: float, maximum_iterations: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "pca",
                CappedVariancePCA(
                    variance_fraction=0.95, maximum_components=32
                ),
            ),
            (
                "logistic",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=maximum_iterations,
                    random_state=0,
                ),
            ),
        ]
    )


def select_c_leave_one_year_out(
    model_factory,
    x: np.ndarray,
    y: np.ndarray,
    years: np.ndarray,
    c_grid: list[float],
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for c_value in c_grid:
        fold_scores = []
        for year in sorted(set(int(value) for value in years)):
            test = years == year
            train = ~test
            if len(set(y[train])) < 2 or len(set(y[test])) < 2:
                continue
            model = model_factory(c_value)
            model.fit(x[train], y[train])
            fold_scores.append(
                balanced_accuracy_score(y[test], model.predict(x[test]))
            )
        if not fold_scores:
            raise ValueError("no valid leave-one-year-out folds")
        scores[str(c_value)] = float(np.mean(fold_scores))
    chosen = min(
        c_grid,
        key=lambda value: (-scores[str(value)], value),
    )
    return float(chosen), scores


def evaluate_fixed_split(
    model,
    x: np.ndarray,
    y: np.ndarray,
    development_indices: np.ndarray,
    confirmatory_indices: np.ndarray,
) -> dict[str, object]:
    fitted = clone(model).fit(x[development_indices], y[development_indices])
    scores = fitted.predict_proba(x[confirmatory_indices])[:, 1]
    predictions = (scores >= 0.5).astype(np.int64)
    truth = y[confirmatory_indices]
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(truth, predictions)
        ),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "scores": scores.tolist(),
        "predictions": predictions.tolist(),
        "truth": truth.tolist(),
    }


def exact_stratified_fixed_score_p(
    scores: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, int]:
    positive_count = int(np.sum(truth))
    observed = float(roc_auc_score(truth, scores))
    exceed = 0
    total = 0
    indices = range(len(truth))
    for positives in combinations(indices, positive_count):
        permuted = np.zeros(len(truth), dtype=np.int64)
        permuted[list(positives)] = 1
        candidate = float(roc_auc_score(permuted, scores))
        exceed += candidate >= observed - 1e-12
        total += 1
    return exceed / total, total
