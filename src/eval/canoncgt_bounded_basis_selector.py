"""Reference-only bounded Gaussian/lattice operator selection preflight."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.canoncgt_extended_quadratic_inverse import (
    _encode,
    range_escape_magnitude,
)
from src.eval.canoncgt_fixed_atlas_shared_lut import (
    _load_model_reference_only,
    _tensor_from_image,
)
from src.eval.canoncgt_reference_condition import CanonCGTReferenceError, _resolve
from src.eval.canoncgt_self_canonical_inverse import uniform_sample_indexes
from src.eval.spcp_global_logit_affine import srgb_code_to_oklab


@dataclass(frozen=True)
class BoundedBasisOperator:
    """An explicit bounded logit operator with one fixed spatially agnostic basis."""

    coefficients: np.ndarray
    representation: str
    asinh_scale: float
    coordinate_input_minimum: float
    coordinate_input_maximum: float
    basis_side: int
    gaussian_sigma: float

    def apply(self, values: np.ndarray) -> np.ndarray:
        features = basis_features(
            values,
            representation=self.representation,
            asinh_scale=self.asinh_scale,
            coordinate_input_minimum=self.coordinate_input_minimum,
            coordinate_input_maximum=self.coordinate_input_maximum,
            basis_side=self.basis_side,
            gaussian_sigma=self.gaussian_sigma,
        )
        logits = features @ self.coefficients
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


def _cube(side: int) -> np.ndarray:
    if side < 2:
        raise CanonCGTReferenceError("basis side must be at least two")
    axis = np.linspace(0.0, 1.0, side, dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )


def _normalized_coordinate(
    values: np.ndarray,
    *,
    asinh_scale: float,
    coordinate_input_minimum: float,
    coordinate_input_maximum: float,
) -> tuple[np.ndarray, np.ndarray]:
    if coordinate_input_minimum >= coordinate_input_maximum:
        raise CanonCGTReferenceError("invalid extended coordinate domain")
    encoded = _encode(values, asinh_scale).reshape(-1, 3)
    low = float(_encode(np.full((1, 3), coordinate_input_minimum), asinh_scale)[0, 0])
    high = float(_encode(np.full((1, 3), coordinate_input_maximum), asinh_scale)[0, 0])
    return encoded, (encoded - low) / (high - low)


def basis_features(
    values: np.ndarray,
    *,
    representation: str,
    asinh_scale: float,
    coordinate_input_minimum: float,
    coordinate_input_maximum: float,
    basis_side: int,
    gaussian_sigma: float,
) -> np.ndarray:
    encoded, coordinate = _normalized_coordinate(
        values,
        asinh_scale=asinh_scale,
        coordinate_input_minimum=coordinate_input_minimum,
        coordinate_input_maximum=coordinate_input_maximum,
    )
    base = np.column_stack((np.ones(encoded.shape[0]), encoded))
    if representation == "affine":
        return base
    centers = _cube(basis_side)
    if representation == "gaussian":
        if gaussian_sigma <= 0.0:
            raise CanonCGTReferenceError("Gaussian sigma must be positive")
        squared = np.sum((coordinate[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        weights = np.exp(-squared / (2.0 * gaussian_sigma**2))
        weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), 1.0e-15)
    elif representation == "lattice":
        distance = np.abs(coordinate[:, None, :] - centers[None, :, :])
        weights = np.prod(np.maximum(1.0 - distance * (basis_side - 1), 0.0), axis=2)
    else:
        raise CanonCGTReferenceError(f"unsupported representation: {representation}")
    return np.concatenate((base, weights), axis=1)


def _fit_coefficients(
    features: np.ndarray, target_logit: np.ndarray, ridge: float
) -> np.ndarray:
    gram = features.T @ features
    regularizer = np.eye(features.shape[1], dtype=np.float64) * ridge
    regularizer[0, 0] = 0.0
    return np.linalg.solve(gram + regularizer, features.T @ target_logit)


def _errors(
    operator: BoundedBasisOperator, source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    prediction = operator.apply(source)
    delta = srgb_code_to_oklab(prediction) - srgb_code_to_oklab(target)
    return np.linalg.norm(delta, axis=1)


def _error_summary(errors: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
    }


def fit_candidates(
    canonical: np.ndarray,
    reference: np.ndarray,
    *,
    maximum_rows: int,
    ridge_alpha: float,
    asinh_scale: float,
    coordinate_input_minimum: float,
    coordinate_input_maximum: float,
    basis_side: int,
    gaussian_sigma: float,
    target_logit_epsilon: float,
) -> tuple[dict[str, BoundedBasisOperator], dict[str, Any]]:
    if canonical.shape != reference.shape or canonical.ndim != 3:
        raise CanonCGTReferenceError("bounded basis geometry mismatch")
    indexes = uniform_sample_indexes(
        canonical.shape[0] * canonical.shape[1], maximum_rows
    )
    source = canonical.reshape(-1, 3)[indexes]
    target = reference.reshape(-1, 3)[indexes]
    position = np.arange(indexes.size)
    masks = {
        "fit": position % 3 == 0,
        "selection": position % 3 == 1,
        "audit": position % 3 == 2,
    }
    clipped = np.clip(target, target_logit_epsilon, 1.0 - target_logit_epsilon)
    target_logit = np.log(clipped / (1.0 - clipped))
    operators: dict[str, BoundedBasisOperator] = {}
    summaries: dict[str, Any] = {}
    for representation in ("affine", "gaussian", "lattice"):
        features = basis_features(
            source,
            representation=representation,
            asinh_scale=asinh_scale,
            coordinate_input_minimum=coordinate_input_minimum,
            coordinate_input_maximum=coordinate_input_maximum,
            basis_side=basis_side,
            gaussian_sigma=gaussian_sigma,
        )
        coefficients = _fit_coefficients(
            features[masks["fit"]], target_logit[masks["fit"]], ridge_alpha
        )
        operator = BoundedBasisOperator(
            coefficients=coefficients,
            representation=representation,
            asinh_scale=asinh_scale,
            coordinate_input_minimum=coordinate_input_minimum,
            coordinate_input_maximum=coordinate_input_maximum,
            basis_side=basis_side,
            gaussian_sigma=gaussian_sigma,
        )
        operators[representation] = operator
        summaries[representation] = {
            name: _error_summary(_errors(operator, source[mask], target[mask]))
            for name, mask in masks.items()
        }
        summaries[representation]["coefficient_absolute_maximum"] = float(
            np.max(np.abs(coefficients))
        )
    summaries["partition_counts"] = {
        name: int(np.sum(mask)) for name, mask in masks.items()
    }
    summaries["source"] = source
    summaries["target"] = target
    summaries["masks"] = masks
    return operators, summaries


def jacobian_diagnostics(
    operator: BoundedBasisOperator,
    *,
    grid_size: int,
    input_minimum: float,
    input_maximum: float,
) -> dict[str, float]:
    axis = np.linspace(input_minimum, input_maximum, grid_size, dtype=np.float64)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    epsilon = 1.0e-5
    columns = []
    for channel in range(3):
        positive = points.copy()
        negative = points.copy()
        positive[:, channel] += epsilon
        negative[:, channel] -= epsilon
        columns.append(
            (operator.apply(positive) - operator.apply(negative)) / (2.0 * epsilon)
        )
    matrices = np.stack(columns, axis=2)
    determinants = np.linalg.det(matrices)
    singular = np.linalg.svd(matrices, compute_uv=False)
    conditions = singular[:, 0] / np.maximum(singular[:, -1], 1.0e-15)
    return {
        "sample_count": int(points.shape[0]),
        "minimum_determinant": float(np.min(determinants)),
        "minimum_singular_value": float(np.min(singular[:, -1])),
        "maximum_condition_number": float(np.max(conditions)),
    }


def _verify_binding(root: Path, binding: Mapping[str, Any]) -> None:
    path = _resolve(root, str(binding["path"]))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != str(binding["sha256"]):
        raise CanonCGTReferenceError(f"binding mismatch: {binding['path']}")


def run_preflight(
    *,
    root: Path,
    config: Mapping[str, Any],
    output_path: Path,
    device: str,
    reverse: bool = False,
) -> dict[str, Any]:
    _verify_binding(root, config["parent_contract"])
    _verify_binding(root, config["matched_control_contract"])
    for binding in config["representation_context"]:
        _verify_binding(root, binding)
    torch, model, parent, model_audit = _load_model_reference_only(root, config, device)
    rows = list(parent["references"])
    if reverse:
        rows.reverse()
    build = config["build"]
    spec = config["structural_gates"]
    results = []
    with torch.inference_mode():
        for row in rows:
            reference = _tensor_from_image(
                _resolve(root, str(row["local_path"])), torch, device
            )
            condition = model.Embedding_Net(reference)
            canonical_result = model.Canonicalizer(reference, condition=condition)
            canonical = (
                canonical_result["result"][0].permute(1, 2, 0).detach().cpu().numpy()
            )
            reference_np = reference[0].permute(1, 2, 0).detach().cpu().numpy()
            operators, metrics = fit_candidates(
                canonical,
                reference_np,
                maximum_rows=int(build["maximum_sample_rows"]),
                ridge_alpha=float(build["ridge_alpha"]),
                asinh_scale=float(build["asinh_scale"]),
                coordinate_input_minimum=float(build["coordinate_input_minimum"]),
                coordinate_input_maximum=float(build["coordinate_input_maximum"]),
                basis_side=int(build["basis_side"]),
                gaussian_sigma=float(build["gaussian_sigma"]),
                target_logit_epsilon=float(build["target_logit_epsilon"]),
            )
            selected = min(
                ("gaussian", "lattice"),
                key=lambda name: (
                    metrics[name]["selection"]["mean"],
                    name != "lattice",
                ),
            )
            audit_best = min(
                ("gaussian", "lattice"),
                key=lambda name: (metrics[name]["audit"]["mean"], name != "lattice"),
            )
            operator = operators[selected]
            jacobian = jacobian_diagnostics(
                operator,
                grid_size=int(build["jacobian_grid_size"]),
                input_minimum=float(build["jacobian_input_minimum"]),
                input_maximum=float(build["jacobian_input_maximum"]),
            )
            source = metrics.pop("source")
            metrics.pop("target")
            masks = metrics.pop("masks")
            prediction = operator.apply(source[masks["audit"]])
            epsilon = 1.0 / 65535.0
            source_boundary = (source[masks["audit"]] <= epsilon) | (
                source[masks["audit"]] >= 1.0 - epsilon
            )
            output_boundary = (prediction <= epsilon) | (prediction >= 1.0 - epsilon)
            new_boundary = float(np.mean(output_boundary & ~source_boundary))
            selected_audit = metrics[selected]["audit"]
            affine_audit = metrics["affine"]["audit"]
            p95_regression = selected_audit["p95"] - affine_audit["p95"]
            oracle_mean = min(
                metrics[name]["audit"]["mean"] for name in ("gaussian", "lattice")
            )
            oracle_ratio = selected_audit["mean"] / max(oracle_mean, 1.0e-12)
            escape = range_escape_magnitude(canonical)
            gates = {
                "extended_input_support": escape
                <= float(spec["maximum_canonical_range_escape_magnitude"]),
                "audit_median": selected_audit["median"]
                <= float(spec["maximum_audit_oklab_error_median"]),
                "audit_p95": selected_audit["p95"]
                <= float(spec["maximum_audit_oklab_error_p95"]),
                "fit_audit_gap": selected_audit["mean"]
                / max(metrics[selected]["fit"]["mean"], 1.0e-12)
                <= float(spec["maximum_audit_to_fit_mean_error_ratio"]),
                "jacobian_determinant": jacobian["minimum_determinant"]
                >= float(spec["minimum_grid_jacobian_determinant"]),
                "jacobian_singular_value": jacobian["minimum_singular_value"]
                >= float(spec["minimum_grid_jacobian_singular_value"]),
                "jacobian_condition": jacobian["maximum_condition_number"]
                <= float(spec["maximum_grid_jacobian_condition_number"]),
                "coefficient_bound": metrics[selected]["coefficient_absolute_maximum"]
                <= float(spec["maximum_coefficient_absolute_value"]),
                "control_nonregression": p95_regression
                <= float(
                    spec["maximum_per_reference_p95_error_regression_vs_matched_affine"]
                ),
                "new_boundary": new_boundary
                <= float(spec["maximum_new_boundary_fraction"]),
            }
            results.append(
                {
                    "reference_id": row["reference_id"],
                    "canonical_range_escape_magnitude": escape,
                    "selected_representation": selected,
                    "audit_best_representation": audit_best,
                    "selection_audit_agreement": selected == audit_best,
                    "selected_to_audit_oracle_mean_error_ratio": oracle_ratio,
                    "p95_error_regression_vs_matched_affine": p95_regression,
                    "new_boundary_fraction": new_boundary,
                    "partition_counts": metrics.pop("partition_counts"),
                    "representations": metrics,
                    "selected_operator": {
                        "representation": selected,
                        "coefficients": operator.coefficients.tolist(),
                        "asinh_scale": operator.asinh_scale,
                        "coordinate_input_minimum": operator.coordinate_input_minimum,
                        "coordinate_input_maximum": operator.coordinate_input_maximum,
                        "basis_side": operator.basis_side,
                        "gaussian_sigma": operator.gaussian_sigma,
                    },
                    "jacobian": jacobian,
                    "gates": gates,
                    "passed": all(gates.values()),
                }
            )
    results.sort(key=lambda item: str(item["reference_id"]))
    selected_p95 = np.asarray(
        [
            row["representations"][row["selected_representation"]]["audit"]["p95"]
            for row in results
        ]
    )
    affine_p95 = np.asarray(
        [row["representations"]["affine"]["audit"]["p95"] for row in results]
    )
    reduction = float(
        np.median((affine_p95 - selected_p95) / np.maximum(affine_p95, 1.0e-12))
    )
    agreement = sum(bool(row["selection_audit_agreement"]) for row in results)
    worst_oracle_ratio = max(
        float(row["selected_to_audit_oracle_mean_error_ratio"]) for row in results
    )
    global_gates = {
        "median_control_gain": reduction
        >= float(spec["minimum_median_p95_error_reduction_vs_matched_affine"]),
        "selection_audit_agreement": agreement
        >= int(spec["minimum_selection_audit_agreement_references"]),
        "selection_audit_regret": worst_oracle_ratio
        <= float(spec["maximum_worst_selected_to_audit_oracle_mean_error_ratio"]),
        "all_references": sum(bool(row["passed"]) for row in results)
        >= int(spec["required_passing_references"]),
    }
    passed = all(global_gates.values())
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "phase": "reference_only_bounded_basis_selection_preflight",
        "application_source_reads": 0,
        "model": model_audit,
        "reference_count": len(results),
        "passing_references": sum(bool(row["passed"]) for row in results),
        "selected_representation_counts": {
            name: sum(row["selected_representation"] == name for row in results)
            for name in ("gaussian", "lattice")
        },
        "selection_audit_agreement_references": agreement,
        "maximum_selected_to_audit_oracle_mean_error_ratio": worst_oracle_ratio,
        "median_p95_error_reduction_vs_matched_affine": reduction,
        "global_gates": global_gates,
        "decision": "PASS_OPEN_SHARED_APPLICATION_D0"
        if passed
        else "FAIL_CLOSED_BOUNDED_BASIS_SELECTOR",
        "references": results,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded)
    return {
        "report": report,
        "path": output_path,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


__all__ = [
    "BoundedBasisOperator",
    "basis_features",
    "fit_candidates",
    "jacobian_diagnostics",
    "run_preflight",
]
