"""Reference-only extended-domain triangular-logit transport preflight."""

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
from src.eval.canoncgt_self_canonical_inverse import (
    fit_inverse_operator,
    uniform_sample_indexes,
)
from src.eval.spcp_global_logit_affine import srgb_code_to_oklab
from src.roll2film.triangular_logit_transport import (
    TriangularLogitTransport,
    fit_triangular_logit_transport,
)


@dataclass(frozen=True)
class ExtendedTriangularTransport:
    """Analytic extended coordinate followed by a cube-bounded transport."""

    transport: TriangularLogitTransport
    asinh_scale: float
    input_minimum: float
    input_maximum: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.asinh_scale)
            or self.asinh_scale <= 0.0
            or not np.isfinite(self.input_minimum)
            or not np.isfinite(self.input_maximum)
            or self.input_minimum >= self.input_maximum
            or self.transport.dose != 1.0
        ):
            raise CanonCGTReferenceError("invalid extended triangular contract")

    @property
    def _encoded_bounds(self) -> tuple[float, float]:
        low = float(
            _encode(np.full((1, 3), self.input_minimum), self.asinh_scale)[0, 0]
        )
        high = float(
            _encode(np.full((1, 3), self.input_maximum), self.asinh_scale)[0, 0]
        )
        return low, high

    def coordinate(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if (
            array.shape[-1] != 3
            or not np.all(np.isfinite(array))
            or np.any(array < self.input_minimum - 1.0e-12)
            or np.any(array > self.input_maximum + 1.0e-12)
        ):
            raise CanonCGTReferenceError("input lies outside frozen extended support")
        low, high = self._encoded_bounds
        coordinate = (_encode(array, self.asinh_scale) - low) / (high - low)
        if np.any(coordinate < -1.0e-12) or np.any(coordinate > 1.0 + 1.0e-12):
            raise CanonCGTReferenceError("extended coordinate escaped cube")
        return np.where(
            np.abs(coordinate) <= 1.0e-12,
            0.0,
            np.where(np.abs(coordinate - 1.0) <= 1.0e-12, 1.0, coordinate),
        )

    def apply(self, values: np.ndarray) -> np.ndarray:
        return self.transport.apply(self.coordinate(values))

    def inverse(self, values: np.ndarray) -> np.ndarray:
        coordinate = self.transport.inverse(values)
        low, high = self._encoded_bounds
        encoded = low + coordinate * (high - low)
        result = 0.5 + np.sinh(encoded) / self.asinh_scale
        if not np.all(np.isfinite(result)):
            raise CanonCGTReferenceError("extended inverse is non-finite")
        return result


def _error_summary(errors: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
    }


def _errors(
    operator: ExtendedTriangularTransport, source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    prediction = operator.apply(source)
    delta = srgb_code_to_oklab(prediction) - srgb_code_to_oklab(target)
    return np.linalg.norm(delta, axis=1)


def fit_extended_triangular_transport(
    canonical: np.ndarray,
    reference: np.ndarray,
    *,
    maximum_rows: int,
    asinh_scale: float,
    input_minimum: float,
    input_maximum: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    identity_shrinkage: float,
    maximum_evaluations: int,
) -> tuple[
    ExtendedTriangularTransport, dict[str, Any], np.ndarray, np.ndarray, np.ndarray
]:
    if canonical.shape != reference.shape or canonical.ndim != 3:
        raise CanonCGTReferenceError("extended triangular geometry mismatch")
    indexes = uniform_sample_indexes(
        canonical.shape[0] * canonical.shape[1], maximum_rows
    )
    source = canonical.reshape(-1, 3)[indexes]
    target = reference.reshape(-1, 3)[indexes]
    fit_mask = np.arange(indexes.size) % 2 == 0
    identity_transport = TriangularLogitTransport(np.zeros(14, dtype=np.float64))
    coordinate_operator = ExtendedTriangularTransport(
        identity_transport, asinh_scale, input_minimum, input_maximum
    )
    coordinate = coordinate_operator.coordinate(source)
    parameters, converged = fit_triangular_logit_transport(
        coordinate[fit_mask, None, :],
        target[fit_mask, None, :],
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        sample_stride=1,
        identity_shrinkage=identity_shrinkage,
        maximum_evaluations=maximum_evaluations,
    )
    operator = ExtendedTriangularTransport(
        TriangularLogitTransport(parameters),
        asinh_scale,
        input_minimum,
        input_maximum,
    )
    fit_errors = _errors(operator, source[fit_mask], target[fit_mask])
    holdout_errors = _errors(operator, source[~fit_mask], target[~fit_mask])
    metrics = {
        "sample_count": int(indexes.size),
        "fit_count": int(np.sum(fit_mask)),
        "holdout_count": int(np.sum(~fit_mask)),
        "converged": bool(converged),
        "fit": _error_summary(fit_errors),
        "holdout": _error_summary(holdout_errors),
        "holdout_to_fit_mean_ratio": float(
            np.mean(holdout_errors) / max(float(np.mean(fit_errors)), 1.0e-12)
        ),
    }
    return operator, metrics, source, target, fit_mask


def jacobian_diagnostics(
    operator: ExtendedTriangularTransport,
    *,
    grid_size: int,
    finite_difference: float,
) -> dict[str, float | int]:
    margin = max(0.02, finite_difference * 2.0)
    axis = np.linspace(
        operator.input_minimum + margin,
        operator.input_maximum - margin,
        grid_size,
        dtype=np.float64,
    )
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    columns = []
    for channel in range(3):
        positive = points.copy()
        negative = points.copy()
        positive[:, channel] += finite_difference
        negative[:, channel] -= finite_difference
        columns.append(
            (operator.apply(positive) - operator.apply(negative))
            / (2.0 * finite_difference)
        )
    matrices = np.stack(columns, axis=2)
    determinants = np.linalg.det(matrices)
    singular = np.linalg.svd(matrices, compute_uv=False)
    conditions = singular[:, 0] / np.maximum(singular[:, -1], 1.0e-15)
    output = operator.apply(points)
    restored = operator.inverse(output)
    return {
        "sample_count": int(points.shape[0]),
        "minimum_determinant": float(np.min(determinants)),
        "minimum_singular_value": float(np.min(singular[:, -1])),
        "maximum_condition_number": float(np.max(conditions)),
        "maximum_inverse_roundtrip_error": float(np.max(np.abs(restored - points))),
    }


def _verify_binding(root: Path, binding: Mapping[str, Any]) -> None:
    path = _resolve(root, str(binding["path"]))
    if hashlib.sha256(path.read_bytes()).hexdigest() != str(binding["sha256"]):
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
    _verify_binding(root, config["representation_context"])
    core_binding = {
        "path": config["representation_context"]["core_path"],
        "sha256": config["representation_context"]["core_sha256"],
    }
    _verify_binding(root, core_binding)
    torch, model, parent, model_audit = _load_model_reference_only(root, config, device)
    rows = list(parent["references"])
    if reverse:
        rows.reverse()
    build = config["build"]
    spec = config["structural_gates"]
    lower = np.asarray(build["parameter_lower_bounds"], dtype=np.float64)
    upper = np.asarray(build["parameter_upper_bounds"], dtype=np.float64)
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
            operator, fit, source, _, fit_mask = fit_extended_triangular_transport(
                canonical,
                reference_np,
                maximum_rows=int(build["maximum_sample_rows"]),
                asinh_scale=float(build["asinh_scale"]),
                input_minimum=float(build["coordinate_input_minimum"]),
                input_maximum=float(build["coordinate_input_maximum"]),
                lower_bounds=lower,
                upper_bounds=upper,
                identity_shrinkage=float(build["identity_shrinkage"]),
                maximum_evaluations=int(build["maximum_fit_evaluations"]),
            )
            _, baseline = fit_inverse_operator(
                canonical,
                reference_np,
                maximum_rows=int(build["maximum_sample_rows"]),
                ridge_alpha=0.0001,
            )
            jacobian = jacobian_diagnostics(
                operator,
                grid_size=int(build["jacobian_grid_size"]),
                finite_difference=float(build["jacobian_finite_difference"]),
            )
            holdout_prediction = operator.apply(source[~fit_mask])
            epsilon = 1.0 / 65535.0
            source_boundary = (source[~fit_mask] <= epsilon) | (
                source[~fit_mask] >= 1.0 - epsilon
            )
            output_boundary = (holdout_prediction <= epsilon) | (
                holdout_prediction >= 1.0 - epsilon
            )
            new_boundary = float(np.mean(output_boundary & ~source_boundary))
            parameters = operator.transport.parameters
            parameter_bounds = bool(
                np.all(parameters >= lower) and np.all(parameters <= upper)
            )
            p95_regression = fit["holdout"]["p95"] - baseline["holdout_error_p95"]
            escape = range_escape_magnitude(canonical)
            gates = {
                "converged": bool(fit["converged"]),
                "extended_input_support": escape
                <= float(spec["maximum_canonical_range_escape_magnitude"]),
                "holdout_median": fit["holdout"]["median"]
                <= float(spec["maximum_holdout_oklab_error_median"]),
                "holdout_p95": fit["holdout"]["p95"]
                <= float(spec["maximum_holdout_oklab_error_p95"]),
                "fit_holdout_gap": fit["holdout_to_fit_mean_ratio"]
                <= float(spec["maximum_holdout_to_fit_mean_error_ratio"]),
                "jacobian_determinant": jacobian["minimum_determinant"]
                >= float(spec["minimum_grid_jacobian_determinant"]),
                "jacobian_singular_value": jacobian["minimum_singular_value"]
                >= float(spec["minimum_grid_jacobian_singular_value"]),
                "jacobian_condition": jacobian["maximum_condition_number"]
                <= float(spec["maximum_grid_jacobian_condition_number"]),
                "inverse_roundtrip": jacobian["maximum_inverse_roundtrip_error"]
                <= float(spec["maximum_inverse_roundtrip_error"]),
                "parameter_bounds": parameter_bounds,
                "new_boundary": new_boundary
                <= float(spec["maximum_new_boundary_fraction"]),
                "control_nonregression": p95_regression
                <= float(spec["maximum_per_reference_p95_error_regression"]),
            }
            results.append(
                {
                    "reference_id": row["reference_id"],
                    "canonical_range_escape_magnitude": escape,
                    "parameters": parameters.tolist(),
                    "fit": fit,
                    "logit_affine_control": baseline,
                    "p95_error_regression_vs_control": p95_regression,
                    "new_boundary_fraction": new_boundary,
                    "jacobian": jacobian,
                    "gates": gates,
                    "passed": all(gates.values()),
                }
            )
    results.sort(key=lambda item: str(item["reference_id"]))
    p95 = np.asarray([row["fit"]["holdout"]["p95"] for row in results])
    control_p95 = np.asarray(
        [row["logit_affine_control"]["holdout_error_p95"] for row in results]
    )
    reduction = float(np.median((control_p95 - p95) / np.maximum(control_p95, 1.0e-12)))
    global_gates = {
        "median_control_gain": reduction
        >= float(spec["minimum_median_p95_error_reduction_vs_logit_affine"]),
        "all_references": sum(bool(row["passed"]) for row in results)
        >= int(spec["required_passing_references"]),
    }
    passed = all(global_gates.values())
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "phase": "reference_only_extended_triangular_transport_preflight",
        "application_source_reads": 0,
        "model": model_audit,
        "reference_count": len(results),
        "passing_references": sum(bool(row["passed"]) for row in results),
        "median_p95_error_reduction_vs_logit_affine": reduction,
        "global_gates": global_gates,
        "decision": "PASS_OPEN_SHARED_APPLICATION_D0"
        if passed
        else "FAIL_CLOSED_EXTENDED_TRIANGULAR_TRANSPORT",
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
    "ExtendedTriangularTransport",
    "fit_extended_triangular_transport",
    "jacobian_diagnostics",
    "run_preflight",
]
