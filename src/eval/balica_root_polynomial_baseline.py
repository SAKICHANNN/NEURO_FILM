"""Cross-domain root-polynomial baseline on the Balica Velvia proxies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.film2paint_curve_matrix_capacity import load_ao4m_pairs
from src.eval.filmmatch_root_polynomial import (
    ExplicitPolynomialOperator,
    _fit_linear,
    ordinary_exponents,
    polynomial_features,
    root_exponents,
)
from src.real_film.velvia_chart_explainability import _fit_affine, _metrics
from src.roll2film.positive_film_fitting import (
    PositiveFilmFitResult,
    fit_positive_film_response_operator,
)


MODEL_NAMES = (
    "identity",
    "full_affine_linear_srgb",
    "bounded_one_matrix_linear_srgb",
    "ordinary_polynomial_degree3_gamma_prophoto",
    "root_polynomial_degree3_gamma_prophoto",
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(np.ascontiguousarray(value, dtype="<f8").tobytes())


def _matrix(config: dict[str, Any], name: str) -> np.ndarray:
    value = np.asarray(config["models"]["color_space"][name], dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError(f"AO4R invalid colour matrix: {name}")
    return value


def _romm_encode(linear: np.ndarray) -> np.ndarray:
    value = np.asarray(linear, dtype=np.float64)
    magnitude = np.abs(value)
    encoded = np.where(
        magnitude < (1.0 / 512.0),
        16.0 * magnitude,
        np.power(magnitude, 1.0 / 1.8),
    )
    return np.copysign(encoded, value)


def _romm_decode(encoded: np.ndarray) -> np.ndarray:
    value = np.asarray(encoded, dtype=np.float64)
    magnitude = np.abs(value)
    linear = np.where(
        magnitude < (16.0 / 512.0),
        magnitude / 16.0,
        np.power(magnitude, 1.8),
    )
    return np.copysign(linear, value)


def linear_srgb_to_gamma_prophoto(
    values: np.ndarray, config: dict[str, Any]
) -> np.ndarray:
    rgb = np.asarray(values, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3 or not np.all(np.isfinite(rgb)):
        raise ValueError("AO4R colour conversion requires finite Nx3 values")
    srgb_to_xyz = _matrix(config, "linear_srgb_d65_to_xyz_d65")
    d65_to_d50 = _matrix(config, "bradford_d65_to_d50")
    prophoto_to_xyz = _matrix(config, "prophoto_linear_to_xyz_d50")
    xyz_d65 = rgb @ srgb_to_xyz.T
    xyz_d50 = xyz_d65 @ d65_to_d50.T
    prophoto = xyz_d50 @ np.linalg.inv(prophoto_to_xyz).T
    return _romm_encode(prophoto)


def gamma_prophoto_to_linear_srgb(
    values: np.ndarray, config: dict[str, Any]
) -> np.ndarray:
    encoded = np.asarray(values, dtype=np.float64)
    if (
        encoded.ndim != 2
        or encoded.shape[1] != 3
        or not np.all(np.isfinite(encoded))
    ):
        raise ValueError("AO4R inverse colour conversion requires finite Nx3 values")
    srgb_to_xyz = _matrix(config, "linear_srgb_d65_to_xyz_d65")
    d65_to_d50 = _matrix(config, "bradford_d65_to_d50")
    prophoto_to_xyz = _matrix(config, "prophoto_linear_to_xyz_d50")
    xyz_d50 = _romm_decode(encoded) @ prophoto_to_xyz.T
    xyz_d65 = xyz_d50 @ np.linalg.inv(d65_to_d50).T
    return xyz_d65 @ np.linalg.inv(srgb_to_xyz).T


@dataclass(frozen=True)
class GammaProPhotoPolynomialOperator:
    polynomial: ExplicitPolynomialOperator
    config: dict[str, Any]

    def apply(self, values: np.ndarray) -> np.ndarray:
        encoded = linear_srgb_to_gamma_prophoto(values, self.config)
        prediction = self.polynomial.apply(encoded)
        return gamma_prophoto_to_linear_srgb(prediction, self.config)

    def to_dict(self) -> dict[str, Any]:
        return {
            "working_space": "gamma_encoded_prophoto_rgb_d50",
            "polynomial": self.polynomial.to_dict(),
        }


def _fit_polynomial(
    source: np.ndarray,
    target: np.ndarray,
    config: dict[str, Any],
    *,
    root: bool,
) -> GammaProPhotoPolynomialOperator:
    encoded_source = linear_srgb_to_gamma_prophoto(source, config)
    encoded_target = linear_srgb_to_gamma_prophoto(target, config)
    exponents = root_exponents(3) if root else ordinary_exponents(3)
    features = polynomial_features(encoded_source, exponents, root=root)
    coefficients, bias = _fit_linear(
        features,
        encoded_target,
        ridge=float(config["models"]["ridge_lambda"]),
        minimum_scale=float(config["models"]["minimum_feature_scale"]),
        include_bias=bool(config["models"]["include_intercept"]),
    )
    name = (
        "root_polynomial_degree3_gamma_prophoto"
        if root
        else "ordinary_polynomial_degree3_gamma_prophoto"
    )
    return GammaProPhotoPolynomialOperator(
        polynomial=ExplicitPolynomialOperator(
            name=name,
            exponents=exponents,
            coefficients=coefficients,
            bias=bias,
            root=root,
        ),
        config=config,
    )


def _fit_all(
    source: np.ndarray, target: np.ndarray, config: dict[str, Any]
) -> dict[str, Any]:
    _, affine = _fit_affine(source, target, per_channel=False)
    return {
        "full_affine_linear_srgb": affine,
        "bounded_one_matrix_linear_srgb": fit_positive_film_response_operator(
            source,
            target,
            model="one_matrix",
            identity_mixture=0.25,
            restart_count=3,
            maximum_function_evaluations=2500,
            function_tolerance=1e-11,
            parameter_tolerance=1e-11,
            gradient_tolerance=1e-11,
            loss="linear",
            loss_scale=0.01,
            seed=int(config["execution"]["seed"]),
        ),
        "ordinary_polynomial_degree3_gamma_prophoto": _fit_polynomial(
            source, target, config, root=False
        ),
        "root_polynomial_degree3_gamma_prophoto": _fit_polynomial(
            source, target, config, root=True
        ),
    }


def _predict_all(source: np.ndarray, fits: dict[str, Any]) -> dict[str, np.ndarray]:
    affine = fits["full_affine_linear_srgb"]
    bounded: PositiveFilmFitResult = fits["bounded_one_matrix_linear_srgb"]
    return {
        "identity": source,
        "full_affine_linear_srgb": (
            source @ np.asarray(affine["matrix"], dtype=np.float64).T
            + np.asarray(affine["bias"], dtype=np.float64)
        ),
        "bounded_one_matrix_linear_srgb": bounded.operator.apply(source),
        "ordinary_polynomial_degree3_gamma_prophoto": fits[
            "ordinary_polynomial_degree3_gamma_prophoto"
        ].apply(source),
        "root_polynomial_degree3_gamma_prophoto": fits[
            "root_polynomial_degree3_gamma_prophoto"
        ].apply(source),
    }


def _fit_record(fits: dict[str, Any]) -> dict[str, Any]:
    bounded: PositiveFilmFitResult = fits["bounded_one_matrix_linear_srgb"]
    return {
        "full_affine_linear_srgb": fits["full_affine_linear_srgb"],
        "bounded_one_matrix_linear_srgb": {
            "converged": bounded.converged,
            "development_rgb_rmse": bounded.development_rgb_rmse,
            "operator": bounded.operator.to_dict(),
        },
        "ordinary_polynomial_degree3_gamma_prophoto": fits[
            "ordinary_polynomial_degree3_gamma_prophoto"
        ].to_dict(),
        "root_polynomial_degree3_gamma_prophoto": fits[
            "root_polynomial_degree3_gamma_prophoto"
        ].to_dict(),
    }


def _cube(size: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )


def _structure(
    operator: GammaProPhotoPolynomialOperator,
    config: dict[str, Any],
) -> dict[str, Any]:
    points = _cube(int(config["evaluation"]["structural_cube_size"]))
    output = operator.apply(points)
    step = float(config["evaluation"]["jacobian_step"])
    jacobian = np.empty((len(points), 3, 3), dtype=np.float64)
    for channel in range(3):
        lower = points.copy()
        upper = points.copy()
        lower[:, channel] -= step
        upper[:, channel] += step
        jacobian[:, :, channel] = (
            operator.apply(upper) - operator.apply(lower)
        ) / (2.0 * step)
    determinants = np.linalg.det(jacobian)
    scale_errors = []
    interior = points[(np.min(points, axis=1) > 0.0) & (np.max(points, axis=1) < 1.0)]
    for scale in (0.5, 0.75):
        scale_errors.append(
            float(
                np.max(
                    np.abs(
                        operator.apply(interior * scale)
                        - operator.apply(interior) * scale
                    )
                )
            )
        )
    return {
        "finite": bool(np.all(np.isfinite(output)) and np.all(np.isfinite(determinants))),
        "minimum_output": float(np.min(output)),
        "maximum_output": float(np.max(output)),
        "out_of_cube_sample_fraction": float(
            np.mean(np.any((output < 0.0) | (output > 1.0), axis=1))
        ),
        "minimum_jacobian_determinant": float(np.min(determinants)),
        "maximum_jacobian_determinant": float(np.max(determinants)),
        "positive_scale_equivariance_maximum_absolute_error": max(scale_errors),
    }


def _gain(candidate: dict[str, Any], control: dict[str, Any], key: str) -> float:
    return float(1.0 - candidate[key] / control[key])


def evaluate_balica_root_polynomial(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
) -> dict[str, Any]:
    counts = {"velvia_chart": 24, "velvia_palette": 47}
    if set(datasets) != set(counts):
        raise ValueError("AO4R requires exact chart and palette domains")
    for name, count in counts.items():
        source, target = datasets[name]
        if source.shape != (count, 3) or target.shape != source.shape:
            raise ValueError(f"AO4R {name} shape mismatch")
        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
            raise ValueError(f"AO4R {name} contains non-finite values")
    originals = {
        name: (source.copy(), target.copy())
        for name, (source, target) in datasets.items()
    }
    fold_count = int(config["evaluation"]["cross_validation"]["fold_count"])
    predictions = {
        model: {
            domain: np.full_like(datasets[domain][1], np.nan, dtype=np.float64)
            for domain in counts
        }
        for model in MODEL_NAMES
    }
    folds: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    development_rmse: list[float] = []
    fold_oog: list[float] = []
    structures: list[dict[str, Any]] = []
    candidate = "root_polynomial_degree3_gamma_prophoto"
    for fold in range(fold_count):
        development_source = []
        development_target = []
        held: dict[str, np.ndarray] = {}
        for domain in counts:
            indices = np.arange(counts[domain])
            held[domain] = indices[indices % fold_count == fold]
            keep = indices[indices % fold_count != fold]
            development_source.append(datasets[domain][0][keep])
            development_target.append(datasets[domain][1][keep])
        try:
            fits = _fit_all(
                np.concatenate(development_source),
                np.concatenate(development_target),
                config,
            )
            train_prediction = fits[candidate].apply(np.concatenate(development_source))
            development_rmse.append(
                float(
                    np.sqrt(
                        np.mean(
                            np.square(train_prediction - np.concatenate(development_target))
                        )
                    )
                )
            )
            structure = _structure(fits[candidate], config)
            structures.append(structure)
            metrics: dict[str, Any] = {}
            for domain in counts:
                source, target = datasets[domain]
                indices = held[domain]
                outputs = _predict_all(source[indices], fits)
                metrics[domain] = {}
                for model, output in outputs.items():
                    predictions[model][domain][indices] = output
                    metrics[domain][model] = _metrics(output, target[indices])
                fold_oog.append(metrics[domain][candidate]["raw_out_of_cube_fraction"])
            folds.append(
                {
                    "fold": fold,
                    "status": "complete",
                    "held_indices": {name: value.tolist() for name, value in held.items()},
                    "fit": _fit_record(fits),
                    "candidate_structure": structure,
                    "confirmation": metrics,
                }
            )
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failures.append({"scope": "cross_validation", "fold": fold, "error": str(exc)})
            folds.append({"fold": fold, "status": "fit_failure", "error": str(exc)})

    aggregate: dict[str, Any] = {}
    for model in MODEL_NAMES:
        domain_metrics = {
            domain: _metrics(predictions[model][domain], datasets[domain][1])
            if np.all(np.isfinite(predictions[model][domain]))
            else None
            for domain in counts
        }
        complete = all(value is not None for value in domain_metrics.values())
        aggregate[model] = {
            "domains": domain_metrics,
            "combined": _metrics(
                np.concatenate([predictions[model][domain] for domain in counts]),
                np.concatenate([datasets[domain][1] for domain in counts]),
            )
            if complete
            else None,
        }

    leave_domain_out: list[dict[str, Any]] = []
    for direction in config["evaluation"]["leave_domain_out_directions"]:
        fit_domain = str(direction["fit"])
        held_domain = str(direction["confirm"])
        try:
            fits = _fit_all(*datasets[fit_domain], config)
            outputs = _predict_all(datasets[held_domain][0], fits)
            metrics = {
                model: _metrics(output, datasets[held_domain][1])
                for model, output in outputs.items()
            }
            structure = _structure(fits[candidate], config)
            structures.append(structure)
            leave_domain_out.append(
                {
                    "fit": fit_domain,
                    "confirm": held_domain,
                    "status": "complete",
                    "fit_record": _fit_record(fits),
                    "candidate_structure": structure,
                    "confirmation": metrics,
                    "candidate_rgb_rmse_gain_over_bounded_one_matrix": _gain(
                        metrics[candidate], metrics["bounded_one_matrix_linear_srgb"], "rgb_rmse"
                    ),
                }
            )
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failures.append(
                {"scope": "leave_domain_out", "fit": fit_domain, "confirm": held_domain, "error": str(exc)}
            )
            leave_domain_out.append(
                {"fit": fit_domain, "confirm": held_domain, "status": "fit_failure", "error": str(exc)}
            )

    candidate_metrics = aggregate[candidate]["combined"]
    control_metrics = aggregate["bounded_one_matrix_linear_srgb"]["combined"]
    rgb_gain = _gain(candidate_metrics, control_metrics, "rgb_rmse")
    delta_gain = _gain(candidate_metrics, control_metrics, "mean_delta_e76")
    per_domain_gain = {
        domain: _gain(
            aggregate[candidate]["domains"][domain],
            aggregate["bounded_one_matrix_linear_srgb"]["domains"][domain],
            "rgb_rmse",
        )
        for domain in counts
    }
    ldo_gains = [
        row["candidate_rgb_rmse_gain_over_bounded_one_matrix"]
        for row in leave_domain_out
        if row["status"] == "complete"
    ]
    train_test_gap = candidate_metrics["rgb_rmse"] - float(np.mean(development_rmse))
    source_nonmutation = all(
        np.array_equal(datasets[name][0], originals[name][0])
        and np.array_equal(datasets[name][1], originals[name][1])
        for name in counts
    )
    gates = config["gates"]
    checks = [
        {"name": "all_fits_complete", "passed": not failures and len(folds) == fold_count and len(ldo_gains) == 2},
        {"name": "candidate_combined_rgb_rmse_gain", "passed": rgb_gain >= float(gates["minimum_candidate_combined_rgb_rmse_gain_over_bounded_one_matrix"])},
        {"name": "candidate_combined_delta_e76_gain", "passed": delta_gain >= float(gates["minimum_candidate_combined_delta_e76_gain_over_bounded_one_matrix"])},
        {"name": "candidate_rgb_rmse_gain_each_domain", "passed": min(per_domain_gain.values()) >= float(gates["minimum_candidate_rgb_rmse_gain_each_domain"])},
        {"name": "candidate_leave_domain_out_gain_each_direction", "passed": len(ldo_gains) == 2 and min(ldo_gains) >= float(gates["minimum_candidate_leave_domain_out_rgb_rmse_gain_each_direction"])},
        {"name": "candidate_train_test_gap", "passed": train_test_gap <= float(gates["maximum_mean_train_test_rgb_rmse_gap"])},
        {"name": "candidate_mean_raw_out_of_cube_fraction", "passed": candidate_metrics["raw_out_of_cube_fraction"] <= float(gates["maximum_mean_raw_out_of_cube_fraction"])},
        {"name": "candidate_maximum_fold_raw_out_of_cube_fraction", "passed": bool(fold_oog) and max(fold_oog) <= float(gates["maximum_fold_raw_out_of_cube_fraction"])},
        {"name": "candidate_minimum_jacobian_determinant", "passed": bool(structures) and min(row["minimum_jacobian_determinant"] for row in structures) >= float(gates["minimum_jacobian_determinant"])},
        {"name": "source_nonmutation", "passed": source_nonmutation},
    ]
    capacity_checks = checks[1:6]
    structure_checks = checks[6:9]
    capacity_passed = bool(all(row["passed"] for row in capacity_checks))
    structure_passed = bool(all(row["passed"] for row in structure_checks))
    automatic_pass = bool(all(row["passed"] for row in checks))
    decision = (
        config["decision_if_pass"]
        if automatic_pass
        else config["decision_if_capacity_passes_but_structure_fails"]
        if capacity_passed and not structure_passed
        else config["decision_if_fail"]
    )
    return {
        "input_array_sha256": {
            f"{domain}_{side}": _array_sha256(datasets[domain][index])
            for domain in counts
            for index, side in enumerate(("source", "target"))
        },
        "folds": folds,
        "aggregate": aggregate,
        "leave_domain_out": leave_domain_out,
        "fit_failures": failures,
        "candidate_summary": {
            "rgb_rmse_gain_over_bounded_one_matrix": rgb_gain,
            "mean_delta_e76_gain_over_bounded_one_matrix": delta_gain,
            "rgb_rmse_gain_by_domain": per_domain_gain,
            "leave_domain_out_rgb_rmse_gains": ldo_gains,
            "mean_train_test_rgb_rmse_gap": train_test_gap,
            "minimum_cube_jacobian_determinant": min(row["minimum_jacobian_determinant"] for row in structures),
            "maximum_fold_raw_out_of_cube_fraction": max(fold_oog),
            "maximum_scale_equivariance_error_diagnostic": max(row["positive_scale_equivariance_maximum_absolute_error"] for row in structures),
        },
        "automatic_checks": checks,
        "capacity_passed": capacity_passed,
        "structure_passed": structure_passed,
        "automatic_pass": automatic_pass,
        "decision": decision,
        "source_nonmutation": source_nonmutation,
    }


def load_pairs(root: Path, config: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    source = config["paired_source"]
    for key in ("chart_pairs", "palette_pairs"):
        path = root / source[f"{key}_path"]
        if _sha256(path.read_bytes()) != source[f"{key}_sha256"]:
            raise ValueError(f"AO4R {key} file identity drift")
    return load_ao4m_pairs(
        root / source["chart_pairs_path"],
        root / source["palette_pairs_path"],
        root / source["loader_config_path"],
        {"paired_source": source},
    )


__all__ = [
    "GammaProPhotoPolynomialOperator",
    "evaluate_balica_root_polynomial",
    "gamma_prophoto_to_linear_srgb",
    "linear_srgb_to_gamma_prophoto",
    "load_pairs",
]
