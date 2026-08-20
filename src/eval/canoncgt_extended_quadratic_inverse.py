"""Extended-domain explicit inverse preflight for CanonCGT references."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.canoncgt_fixed_atlas_shared_lut import (
    _load_model_reference_only,
    _tensor_from_image,
)
from src.eval.canoncgt_reference_condition import CanonCGTReferenceError, _resolve
from src.eval.canoncgt_self_canonical_inverse import (
    fit_inverse_operator,
    uniform_sample_indexes,
)
from src.eval.spcp_global_logit_affine import srgb_code_to_oklab


@dataclass(frozen=True)
class ExtendedQuadraticOperator:
    coefficients: np.ndarray
    asinh_scale: float

    def apply(self, values: np.ndarray) -> np.ndarray:
        features = quadratic_features(values, self.asinh_scale)
        logits = features @ self.coefficients
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


def _encode(values: np.ndarray, scale: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1] != 3 or not np.all(np.isfinite(array)) or scale <= 0.0:
        raise CanonCGTReferenceError("invalid extended quadratic input")
    return np.arcsinh(scale * (array - 0.5))


def quadratic_features(values: np.ndarray, scale: float) -> np.ndarray:
    z = _encode(values, scale).reshape(-1, 3)
    z0, z1, z2 = z.T
    return np.column_stack(
        (
            np.ones(z.shape[0]),
            z0,
            z1,
            z2,
            z0 * z0,
            z1 * z1,
            z2 * z2,
            z0 * z1,
            z0 * z2,
            z1 * z2,
        )
    )


def fit_extended_quadratic(
    canonical: np.ndarray,
    reference: np.ndarray,
    *,
    maximum_rows: int,
    ridge_alpha: float,
    asinh_scale: float,
    target_logit_epsilon: float,
) -> tuple[ExtendedQuadraticOperator, dict[str, float]]:
    if canonical.shape != reference.shape or canonical.ndim != 3:
        raise CanonCGTReferenceError("extended inverse geometry mismatch")
    indexes = uniform_sample_indexes(canonical.shape[0] * canonical.shape[1], maximum_rows)
    source = canonical.reshape(-1, 3)[indexes]
    target = reference.reshape(-1, 3)[indexes]
    fit_mask = np.arange(indexes.size) % 2 == 0
    features = quadratic_features(source, asinh_scale)
    clipped = np.clip(target, target_logit_epsilon, 1.0 - target_logit_epsilon)
    target_logit = np.log(clipped / (1.0 - clipped))
    gram = features[fit_mask].T @ features[fit_mask]
    regularizer = np.eye(features.shape[1], dtype=np.float64) * ridge_alpha
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        gram + regularizer, features[fit_mask].T @ target_logit[fit_mask]
    )
    operator = ExtendedQuadraticOperator(coefficients=coefficients, asinh_scale=asinh_scale)

    def errors(mask: np.ndarray) -> np.ndarray:
        prediction = operator.apply(source[mask])
        delta = srgb_code_to_oklab(prediction) - srgb_code_to_oklab(target[mask])
        return np.linalg.norm(delta, axis=1)

    fit_error = errors(fit_mask)
    holdout_error = errors(~fit_mask)
    return operator, {
        "sample_count": int(indexes.size),
        "fit_count": int(np.sum(fit_mask)),
        "holdout_count": int(np.sum(~fit_mask)),
        "fit_error_mean": float(np.mean(fit_error)),
        "holdout_error_mean": float(np.mean(holdout_error)),
        "holdout_error_median": float(np.median(holdout_error)),
        "holdout_error_p95": float(np.percentile(holdout_error, 95)),
        "holdout_to_fit_mean_ratio": float(
            np.mean(holdout_error) / max(float(np.mean(fit_error)), 1.0e-12)
        ),
    }


def jacobian_diagnostics(
    operator: ExtendedQuadraticOperator,
    *,
    grid_size: int,
    input_minimum: float,
    input_maximum: float,
) -> dict[str, float]:
    axis = np.linspace(input_minimum, input_maximum, grid_size, dtype=np.float64)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    z = _encode(points, operator.asinh_scale)
    c = operator.coefficients
    logits = quadratic_features(points, operator.asinh_scale) @ c
    output = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    dzdx = operator.asinh_scale / np.sqrt(
        1.0 + (operator.asinh_scale * (points - 0.5)) ** 2
    )
    jacobians = []
    for row, encoded, encoded_slope, rendered in zip(points, z, dzdx, output, strict=True):
        del row
        z0, z1, z2 = encoded
        dlogit_dz = np.vstack(
            (
                c[1] + 2.0 * z0 * c[4] + z1 * c[7] + z2 * c[8],
                c[2] + 2.0 * z1 * c[5] + z0 * c[7] + z2 * c[9],
                c[3] + 2.0 * z2 * c[6] + z0 * c[8] + z1 * c[9],
            )
        ).T
        jacobians.append(
            (rendered * (1.0 - rendered))[:, None]
            * dlogit_dz
            * encoded_slope[None, :]
        )
    matrices = np.asarray(jacobians)
    determinants = np.linalg.det(matrices)
    singular = np.linalg.svd(matrices, compute_uv=False)
    conditions = singular[:, 0] / np.maximum(singular[:, -1], 1.0e-15)
    return {
        "sample_count": int(points.shape[0]),
        "minimum_determinant": float(np.min(determinants)),
        "minimum_singular_value": float(np.min(singular[:, -1])),
        "maximum_condition_number": float(np.max(conditions)),
    }


def range_escape_magnitude(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(max(0.0, -float(np.min(array)), float(np.max(array)) - 1.0))


def run_preflight(
    *,
    root: Path,
    config: Mapping[str, Any],
    output_path: Path,
    device: str,
    reverse: bool = False,
) -> dict[str, Any]:
    torch, model, parent, model_audit = _load_model_reference_only(root, config, device)
    rows = list(parent["references"])
    if reverse:
        rows.reverse()
    build = config["build"]
    spec = config["structural_gates"]
    results = []
    with torch.inference_mode():
        for row in rows:
            reference = _tensor_from_image(_resolve(root, str(row["local_path"])), torch, device)
            condition = model.Embedding_Net(reference)
            canonical_result = model.Canonicalizer(reference, condition=condition)
            canonical = canonical_result["result"][0].permute(1, 2, 0).detach().cpu().numpy()
            reference_np = reference[0].permute(1, 2, 0).detach().cpu().numpy()
            operator, fit = fit_extended_quadratic(
                canonical,
                reference_np,
                maximum_rows=int(build["maximum_sample_rows"]),
                ridge_alpha=float(build["ridge_alpha"]),
                asinh_scale=float(build["asinh_scale"]),
                target_logit_epsilon=float(build["target_logit_epsilon"]),
            )
            _, baseline = fit_inverse_operator(
                canonical,
                reference_np,
                maximum_rows=int(build["maximum_sample_rows"]),
                ridge_alpha=float(build["ridge_alpha"]),
            )
            jacobian = jacobian_diagnostics(
                operator,
                grid_size=int(build["jacobian_grid_size"]),
                input_minimum=float(build["jacobian_input_minimum"]),
                input_maximum=float(build["jacobian_input_maximum"]),
            )
            escape = range_escape_magnitude(canonical)
            regression = fit["holdout_error_p95"] - baseline["holdout_error_p95"]
            gates = {
                "extended_input_support": escape <= float(spec["maximum_canonical_range_escape_magnitude"]),
                "holdout_median": fit["holdout_error_median"] <= float(spec["maximum_holdout_oklab_error_median"]),
                "holdout_p95": fit["holdout_error_p95"] <= float(spec["maximum_holdout_oklab_error_p95"]),
                "fit_holdout_gap": fit["holdout_to_fit_mean_ratio"] <= float(spec["maximum_holdout_to_fit_mean_error_ratio"]),
                "jacobian_determinant": jacobian["minimum_determinant"] >= float(spec["minimum_grid_jacobian_determinant"]),
                "jacobian_singular_value": jacobian["minimum_singular_value"] >= float(spec["minimum_grid_jacobian_singular_value"]),
                "jacobian_condition": jacobian["maximum_condition_number"] <= float(spec["maximum_grid_jacobian_condition_number"]),
                "coefficient_bound": float(np.max(np.abs(operator.coefficients))) <= float(spec["maximum_coefficient_absolute_value"]),
                "control_nonregression": regression <= float(spec["maximum_per_reference_p95_error_regression"]),
            }
            results.append(
                {
                    "reference_id": row["reference_id"],
                    "canonical_range_escape_magnitude": escape,
                    "operator": {"coefficients": operator.coefficients.tolist(), "asinh_scale": operator.asinh_scale},
                    "fit": fit,
                    "logit_affine_control": baseline,
                    "p95_error_regression_vs_control": regression,
                    "jacobian": jacobian,
                    "gates": gates,
                    "passed": all(gates.values()),
                }
            )
    results.sort(key=lambda item: str(item["reference_id"]))
    p95 = np.asarray([row["fit"]["holdout_error_p95"] for row in results])
    control_p95 = np.asarray([row["logit_affine_control"]["holdout_error_p95"] for row in results])
    reduction = float(np.median((control_p95 - p95) / np.maximum(control_p95, 1.0e-12)))
    control_gate = reduction >= float(spec["minimum_median_p95_error_reduction_vs_logit_affine"])
    passing = sum(bool(row["passed"]) for row in results)
    passed = passing >= int(spec["required_passing_references"]) and control_gate
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "phase": "reference_only_extended_quadratic_preflight",
        "application_source_reads": 0,
        "model": model_audit,
        "reference_count": len(results),
        "passing_references": passing,
        "required_passing_references": int(spec["required_passing_references"]),
        "median_p95_error_reduction_vs_logit_affine": reduction,
        "matched_control_gate": control_gate,
        "decision": "PASS_OPEN_SHARED_APPLICATION_D0" if passed else "FAIL_CLOSED_EXTENDED_QUADRATIC_INVERSE",
        "references": results,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded)
    return {"report": report, "path": output_path, "sha256": hashlib.sha256(encoded).hexdigest()}


__all__ = [
    "ExtendedQuadraticOperator",
    "fit_extended_quadratic",
    "jacobian_diagnostics",
    "quadratic_features",
    "range_escape_magnitude",
    "run_preflight",
]
