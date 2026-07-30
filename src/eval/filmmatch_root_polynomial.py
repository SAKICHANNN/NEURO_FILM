"""Grouped root-polynomial directionality audit for the FilmMatch chart source."""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.filmmatch_chart_pairs import slog3_to_linear_reflection
from src.eval.filmmatch_code_domain_capacity import prediction_metrics
from src.eval.filmmatch_paired_source import canonical_sha256, sha256_file


MODEL_NAMES = (
    "identity",
    "homogeneous_affine",
    "full_affine",
    "ordinary_polynomial_degree2",
    "ordinary_polynomial_degree3",
    "signed_root_polynomial_degree2",
    "signed_root_polynomial_degree3",
)


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype="<f8").tobytes()
    ).hexdigest()


def _degree_exponents(degree: int) -> list[tuple[int, int, int]]:
    return [
        (red, green, degree - red - green)
        for red in range(degree, -1, -1)
        for green in range(degree - red, -1, -1)
    ]


def ordinary_exponents(maximum_degree: int) -> tuple[tuple[int, int, int], ...]:
    if maximum_degree < 1:
        raise ValueError("maximum degree must be positive")
    return tuple(
        exponent
        for degree in range(1, maximum_degree + 1)
        for exponent in _degree_exponents(degree)
    )


def root_exponents(maximum_degree: int) -> tuple[tuple[int, int, int], ...]:
    """Return one representative for each normalized exponent ray."""

    if maximum_degree < 1:
        raise ValueError("maximum degree must be positive")
    output: list[tuple[int, int, int]] = []
    seen: set[tuple[Fraction, Fraction, Fraction]] = set()
    for degree in range(1, maximum_degree + 1):
        for exponent in _degree_exponents(degree):
            key = tuple(Fraction(value, degree) for value in exponent)
            if key not in seen:
                output.append(exponent)
                seen.add(key)
    return tuple(output)


def polynomial_features(
    values: np.ndarray,
    exponents: Sequence[tuple[int, int, int]],
    *,
    root: bool,
) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64)
    if (
        samples.ndim != 2
        or samples.shape[1] != 3
        or not np.all(np.isfinite(samples))
    ):
        raise ValueError("features require finite Nx3 values")
    columns = []
    for exponent in exponents:
        degree = sum(exponent)
        if degree < 1:
            raise ValueError("feature exponents must have positive degree")
        monomial = np.prod(
            np.power(samples, np.asarray(exponent, dtype=np.int64)),
            axis=1,
        )
        if root and degree > 1:
            monomial = np.sign(monomial) * np.power(
                np.abs(monomial), 1.0 / degree
            )
        columns.append(monomial)
    return np.stack(columns, axis=1)


@dataclass(frozen=True)
class ExplicitPolynomialOperator:
    name: str
    exponents: tuple[tuple[int, int, int], ...]
    coefficients: np.ndarray
    bias: np.ndarray
    root: bool

    def apply(self, values: np.ndarray) -> np.ndarray:
        features = polynomial_features(values, self.exponents, root=self.root)
        return features @ self.coefficients + self.bias

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "exponents": [list(value) for value in self.exponents],
            "coefficients": np.asarray(
                self.coefficients, dtype=np.float64
            ).tolist(),
            "bias": np.asarray(self.bias, dtype=np.float64).tolist(),
            "root": bool(self.root),
        }


def _fit_linear(
    features: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float,
    minimum_scale: float,
    include_bias: bool,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.asarray(features, dtype=np.float64)
    reference = np.asarray(target, dtype=np.float64)
    if (
        design.ndim != 2
        or reference.shape != (len(design), 3)
        or not np.all(np.isfinite(design))
        or not np.all(np.isfinite(reference))
    ):
        raise ValueError("fit requires matching finite feature and RGB arrays")
    if include_bias:
        augmented = np.column_stack([design, np.ones(len(design))])
        solution, *_ = np.linalg.lstsq(augmented, reference, rcond=None)
        return solution[:-1], solution[-1]
    scale = np.sqrt(np.mean(np.square(design), axis=0))
    scale = np.maximum(scale, minimum_scale)
    standardized = design / scale
    gram = standardized.T @ standardized / len(standardized)
    right = standardized.T @ reference / len(standardized)
    regularized = gram + float(ridge) * np.eye(gram.shape[0])
    standardized_coefficients = np.linalg.solve(regularized, right)
    return standardized_coefficients / scale[:, None], np.zeros(3)


def fit_models(
    source: np.ndarray, target: np.ndarray, config: Mapping[str, Any]
) -> dict[str, ExplicitPolynomialOperator]:
    ridge = config["models"]["ridge"]
    lambda_value = float(ridge["lambda"])
    minimum_scale = float(ridge["minimum_feature_scale"])
    output: dict[str, ExplicitPolynomialOperator] = {}
    specifications = {
        "homogeneous_affine": (ordinary_exponents(1), False, False),
        "full_affine": (ordinary_exponents(1), False, True),
        "ordinary_polynomial_degree2": (ordinary_exponents(2), False, False),
        "ordinary_polynomial_degree3": (ordinary_exponents(3), False, False),
        "signed_root_polynomial_degree2": (root_exponents(2), True, False),
        "signed_root_polynomial_degree3": (root_exponents(3), True, False),
    }
    for name, (exponents, root, include_bias) in specifications.items():
        features = polynomial_features(source, exponents, root=root)
        coefficients, bias = _fit_linear(
            features,
            target,
            ridge=0.0 if name == "full_affine" else lambda_value,
            minimum_scale=minimum_scale,
            include_bias=include_bias,
        )
        output[name] = ExplicitPolynomialOperator(
            name=name,
            exponents=exponents,
            coefficients=coefficients,
            bias=bias,
            root=root,
        )
    return output


def apply_models(
    source: np.ndarray, fits: Mapping[str, ExplicitPolynomialOperator]
) -> dict[str, np.ndarray]:
    return {
        "identity": np.asarray(source, dtype=np.float64),
        **{name: fits[name].apply(source) for name in MODEL_NAMES[1:]},
    }


def _metric_payload(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    row = prediction_metrics(prediction, target)
    target_rms = float(np.sqrt(np.mean(np.square(target))))
    row["relative_rgb_rmse"] = float(
        row["rgb_rmse"] / max(target_rms, np.finfo(np.float64).tiny)
    )
    return row


def _fold(
    source: np.ndarray,
    target: np.ndarray,
    held: np.ndarray,
    *,
    fold_id: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    fits = fit_models(source[~held], target[~held], config)
    predictions = apply_models(source[held], fits)
    return {
        "fold_id": fold_id,
        "development_samples": int(np.sum(~held)),
        "held_samples": int(np.sum(held)),
        "metrics": {
            name: _metric_payload(prediction, target[held])
            for name, prediction in predictions.items()
        },
    }


def _aggregate(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in MODEL_NAMES:
        rmse = np.asarray(
            [fold["metrics"][name]["rgb_rmse"] for fold in folds]
        )
        oog = np.asarray(
            [
                fold["metrics"][name]["out_of_cube_sample_fraction"]
                for fold in folds
            ]
        )
        output[name] = {
            "folds": len(folds),
            "mean_rgb_rmse": float(np.mean(rmse)),
            "median_rgb_rmse": float(np.median(rmse)),
            "worst_rgb_rmse": float(np.max(rmse)),
            "maximum_held_sample_out_of_cube_fraction": float(np.max(oog)),
        }
    for name in MODEL_NAMES:
        current = np.asarray(
            [fold["metrics"][name]["rgb_rmse"] for fold in folds]
        )
        for baseline in ("homogeneous_affine", "full_affine"):
            base = np.asarray(
                [fold["metrics"][baseline]["rgb_rmse"] for fold in folds]
            )
            improvements = 1.0 - current / base
            output[name][f"median_improvement_over_{baseline}"] = float(
                1.0
                - output[name]["median_rgb_rmse"]
                / output[baseline]["median_rgb_rmse"]
            )
            output[name][f"worst_fold_improvement_over_{baseline}"] = float(
                np.min(improvements)
            )
    return output


def _protocols(
    source: np.ndarray,
    target: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    illuminants = np.asarray([row["illuminant"] for row in records])
    exposures = np.asarray([int(row["exposure_ev"]) for row in records])
    illuminant_folds = [
        _fold(
            source,
            target,
            illuminants == label,
            fold_id=f"held-illuminant-{label}",
            config=config,
        )
        for label in sorted(set(illuminants))
    ]
    exposure_folds = [
        _fold(
            source,
            target,
            exposures == exposure,
            fold_id=f"held-exposure-{exposure:+d}EV",
            config=config,
        )
        for exposure in sorted(set(exposures))
    ]
    return {
        "held_illuminant_folds": illuminant_folds,
        "held_illuminant_aggregate": _aggregate(illuminant_folds),
        "held_exposure_folds": exposure_folds,
        "held_exposure_aggregate": _aggregate(exposure_folds),
    }


def _structural_cube(config: Mapping[str, Any]) -> np.ndarray:
    controls = config["development"]["structural_audit"]
    slog_axis = np.linspace(
        float(controls["slog3_code_axis_minimum"]),
        float(controls["slog3_code_axis_maximum"]),
        int(controls["cube_size"]),
    )
    linear_axis = slog3_to_linear_reflection(slog_axis)
    return np.stack(
        np.meshgrid(linear_axis, linear_axis, linear_axis, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)


def _jacobian_determinants(
    operator: ExplicitPolynomialOperator, points: np.ndarray
) -> np.ndarray:
    jacobian = np.empty((len(points), 3, 3), dtype=np.float64)
    for channel in range(3):
        step = 1e-6 * np.maximum(1.0, np.abs(points[:, channel]))
        lower = points.copy()
        upper = points.copy()
        lower[:, channel] -= step
        upper[:, channel] += step
        jacobian[:, :, channel] = (
            operator.apply(upper) - operator.apply(lower)
        ) / (2.0 * step[:, None])
    return np.linalg.det(jacobian)


def _structural_audit(
    fits: Mapping[str, ExplicitPolynomialOperator],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    points = _structural_cube(config)
    controls = config["development"]["structural_audit"]
    scales = config["development"]["positive_scale_equivariance_scales"]
    output: dict[str, Any] = {}
    for name, prediction in apply_models(points, fits).items():
        if name == "identity":
            determinants = np.ones(len(points))
            equivariance = 0.0
        else:
            operator = fits[name]
            determinants = _jacobian_determinants(operator, points)
            base = operator.apply(points)
            equivariance = max(
                float(
                    np.max(
                        np.abs(
                            operator.apply(points * float(scale))
                            - base * float(scale)
                        )
                    )
                )
                for scale in scales
            )
        row = {
            "finite": bool(
                np.all(np.isfinite(prediction))
                and np.all(np.isfinite(determinants))
            ),
            "minimum_output": float(np.min(prediction)),
            "maximum_output": float(np.max(prediction)),
            "out_of_target_cube_sample_fraction": float(
                np.mean(np.any((prediction < 0.0) | (prediction > 1.0), axis=1))
            ),
            "minimum_jacobian_determinant": float(np.min(determinants)),
            "maximum_jacobian_determinant": float(np.max(determinants)),
            "positive_scale_equivariance_maximum_absolute_error": equivariance,
        }
        row["render_candidate_safe"] = bool(
            row["finite"]
            and row["out_of_target_cube_sample_fraction"]
            <= float(
                controls["maximum_out_of_target_cube_fraction_for_render_candidate"]
            )
            and row["minimum_jacobian_determinant"]
            >= float(controls["minimum_jacobian_determinant_for_render_candidate"])
        )
        output[name] = row
    return output


def _select_root_candidate(
    protocols: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    candidates = (
        "signed_root_polynomial_degree2",
        "signed_root_polynomial_degree3",
    )
    illuminant = protocols["held_illuminant_aggregate"]
    exposure = protocols["held_exposure_aggregate"]
    scores = {
        name: float(
            illuminant[name]["mean_rgb_rmse"]
            + exposure[name]["mean_rgb_rmse"]
        )
        for name in candidates
    }
    best = min(scores.values())
    tolerance = float(
        config["development"]["selection"]["near_best_relative_tolerance"]
    )
    near_best = [name for name in candidates if scores[name] <= best * (1 + tolerance)]
    selected = min(near_best, key=lambda value: int(value.rsplit("degree", 1)[1]))
    return {
        "selected": selected,
        "scores": scores,
        "near_best": near_best,
        "reason": "lowest two-protocol mean RMSE with frozen lower-degree tie break",
    }


def _capacity_decision(
    protocols: Mapping[str, Any],
    structural: Mapping[str, Any],
    selection: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    name = str(selection["selected"])
    gates = config["development"]["capacity_gates"]
    rows = [
        protocols["held_illuminant_aggregate"][name],
        protocols["held_exposure_aggregate"][name],
    ]
    checks = {
        "median_improvement_over_homogeneous_affine_each_protocol": all(
            row["median_improvement_over_homogeneous_affine"]
            >= float(
                gates[
                    "minimum_median_improvement_over_homogeneous_affine_each_protocol"
                ]
            )
            for row in rows
        ),
        "median_improvement_over_full_affine_each_protocol": all(
            row["median_improvement_over_full_affine"]
            >= float(
                gates["minimum_median_improvement_over_full_affine_each_protocol"]
            )
            for row in rows
        ),
        "worst_fold_improvement_over_homogeneous_affine_each_protocol": all(
            row["worst_fold_improvement_over_homogeneous_affine"]
            >= float(
                gates[
                    "minimum_worst_fold_improvement_over_homogeneous_affine_each_protocol"
                ]
            )
            for row in rows
        ),
        "held_sample_out_of_cube_fraction_each_protocol": all(
            row["maximum_held_sample_out_of_cube_fraction"]
            <= float(
                gates[
                    "maximum_held_sample_out_of_cube_fraction_each_protocol"
                ]
            )
            for row in rows
        ),
        "positive_scale_equivariance": structural[name][
            "positive_scale_equivariance_maximum_absolute_error"
        ]
        <= float(
            gates["maximum_positive_scale_equivariance_absolute_error"]
        ),
    }
    return {
        "selected": name,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "render_candidate_safe": bool(structural[name]["render_candidate_safe"]),
    }


def validate_parent(
    root: Path,
    config: Mapping[str, Any],
    datasets: Mapping[str, Any],
) -> None:
    parent = config["parent"]
    config_path = root / str(parent["config"])
    report_path = root / str(parent["report"])
    if (
        sha256_file(config_path) != parent["config_sha256"]
        or sha256_file(report_path) != parent["report_sha256"]
    ):
        raise ValueError("BF0 parent file identity drift")
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("stable_evidence_id") != parent["stable_evidence_id"]:
        raise ValueError("BF0 parent stable evidence drift")
    for name, expected in parent["input_array_sha256"].items():
        if _array_sha256(np.asarray(datasets[name])) != expected:
            raise ValueError(f"BF0 input array drift: {name}")


def evaluate_root_polynomial(
    datasets: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    validate_parent(root, config, datasets)
    digital = slog3_to_linear_reflection(
        np.asarray(datasets["reflective_source"], dtype=np.float64)
    )
    film = np.asarray(datasets["reflective_target"], dtype=np.float64)
    records = datasets["reflective_records"]
    forward = _protocols(digital, film, records, config)
    inverse = _protocols(film, digital, records, config)
    forward_fits = fit_models(digital, film, config)
    inverse_fits = fit_models(film, digital, config)
    forward_structural = _structural_audit(forward_fits, config)
    forward_selection = _select_root_candidate(forward, config)
    inverse_selection = _select_root_candidate(inverse, config)
    forward_decision = _capacity_decision(
        forward, forward_structural, forward_selection, config
    )
    report = {
        "schema_version": "u5-r2bf0-filmmatch-root-polynomial-report-v1",
        "experiment_id": config["experiment_id"],
        "input_array_sha256": {
            "digital_scene_linear": _array_sha256(digital),
            "film_scan_code": _array_sha256(film),
        },
        "feature_contract": {
            name: [list(value) for value in fit.exponents]
            for name, fit in forward_fits.items()
        },
        "forward": forward,
        "inverse_diagnostic": inverse,
        "forward_final_fits": {
            name: fit.to_dict() for name, fit in forward_fits.items()
        },
        "inverse_final_fits": {
            name: fit.to_dict() for name, fit in inverse_fits.items()
        },
        "forward_structural_audit": forward_structural,
        "forward_selection": forward_selection,
        "inverse_selection": inverse_selection,
        "forward_capacity_decision": forward_decision,
        "inverse_is_diagnostic_only": True,
        "rendering_opened": bool(
            forward_decision["passed"]
            and forward_decision["render_candidate_safe"]
        ),
        "validation_opened": False,
        "promotion_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report


__all__ = [
    "ExplicitPolynomialOperator",
    "MODEL_NAMES",
    "apply_models",
    "evaluate_root_polynomial",
    "fit_models",
    "ordinary_exponents",
    "polynomial_features",
    "root_exponents",
    "validate_parent",
]
