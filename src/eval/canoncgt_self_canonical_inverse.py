"""Reference self-canonicalization inverse-operator preflight."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.canoncgt_fixed_atlas_shared_lut import (
    _load_model_reference_only,
    _tensor_from_image,
)
from src.eval.canoncgt_reference_condition import (
    CanonCGTReferenceError,
    _resolve,
    out_of_range_fraction,
)
from src.eval.spcp_global_logit_affine import (
    LogitAffineOperator,
    apply_operator,
    fit_operator_rows,
    matrix_diagnostics,
    srgb_code_to_oklab,
)


def uniform_sample_indexes(pixel_count: int, maximum: int) -> np.ndarray:
    if pixel_count < 2 or maximum < 2:
        raise CanonCGTReferenceError("insufficient inverse-fit samples")
    count = min(pixel_count, maximum)
    indexes = np.linspace(0, pixel_count - 1, count, dtype=np.int64)
    indexes = np.unique(indexes)
    if indexes.size < 2:
        raise CanonCGTReferenceError("degenerate inverse-fit sample")
    return indexes


def fit_inverse_operator(
    canonical: np.ndarray,
    reference: np.ndarray,
    *, maximum_rows: int,
    ridge_alpha: float,
) -> tuple[LogitAffineOperator, dict[str, float]]:
    if canonical.shape != reference.shape or canonical.ndim != 3 or canonical.shape[2] != 3:
        raise CanonCGTReferenceError("self-canonical pair geometry mismatch")
    indexes = uniform_sample_indexes(canonical.shape[0] * canonical.shape[1], maximum_rows)
    source = canonical.reshape(-1, 3)[indexes]
    target = reference.reshape(-1, 3)[indexes]
    fit_mask = np.arange(indexes.size) % 2 == 0
    matrix, bias = fit_operator_rows(
        source[fit_mask], target[fit_mask], ridge_alpha=ridge_alpha
    )
    operator = LogitAffineOperator(matrix=matrix, bias=bias, dose=1.0)

    def errors(mask: np.ndarray) -> np.ndarray:
        predicted = apply_operator(source[mask], operator)
        delta = srgb_code_to_oklab(predicted) - srgb_code_to_oklab(target[mask])
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


def run_preflight(
    *, root: Path, config: Mapping[str, Any], output_path: Path,
    device: str, reverse: bool = False,
) -> dict[str, Any]:
    torch, model, parent, model_audit = _load_model_reference_only(
        root, config, device
    )
    rows = list(parent["references"])
    if reverse:
        rows.reverse()
    build = config["build"]
    gates_spec = config["structural_gates"]
    results = []
    with torch.inference_mode():
        for row in rows:
            reference = _tensor_from_image(
                _resolve(root, str(row["local_path"])), torch, device
            )
            condition = model.Embedding_Net(reference)
            canonical_result = model.Canonicalizer(reference, condition=condition)
            canonical = canonical_result["result"][0].permute(1, 2, 0).detach().cpu().numpy()
            reference_np = reference[0].permute(1, 2, 0).detach().cpu().numpy()
            operator, fit = fit_inverse_operator(
                canonical,
                reference_np,
                maximum_rows=int(build["maximum_sample_rows"]),
                ridge_alpha=float(build["ridge_alpha"]),
            )
            matrix = matrix_diagnostics(operator)
            raw_range = out_of_range_fraction(canonical)
            gates = {
                "canonical_raw_range": raw_range <= float(gates_spec["maximum_reference_canonical_raw_out_of_range_fraction"]),
                "holdout_median": fit["holdout_error_median"] <= float(gates_spec["maximum_holdout_oklab_error_median"]),
                "holdout_p95": fit["holdout_error_p95"] <= float(gates_spec["maximum_holdout_oklab_error_p95"]),
                "fit_holdout_gap": fit["holdout_to_fit_mean_ratio"] <= float(gates_spec["maximum_holdout_to_fit_mean_error_ratio"]),
                "determinant": matrix["determinant"] >= float(gates_spec["minimum_matrix_determinant"]),
                "minimum_singular_value": matrix["minimum_singular_value"] >= float(gates_spec["minimum_matrix_singular_value"]),
                "condition_number": matrix["condition_number"] <= float(gates_spec["maximum_matrix_condition_number"]),
            }
            results.append(
                {
                    "reference_id": row["reference_id"],
                    "canonical_raw_out_of_range_fraction": raw_range,
                    "operator": {
                        "matrix": operator.matrix.tolist(),
                        "bias": operator.bias.tolist(),
                    },
                    "matrix_diagnostics": matrix,
                    "fit": fit,
                    "gates": gates,
                    "passed": all(gates.values()),
                }
            )
    results.sort(key=lambda row: str(row["reference_id"]))
    passing = sum(bool(row["passed"]) for row in results)
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "phase": "reference_only_structural_preflight",
        "application_source_reads": 0,
        "model": model_audit,
        "reference_count": len(results),
        "passing_references": passing,
        "required_passing_references": int(gates_spec["required_passing_references"]),
        "decision": (
            "PASS_OPEN_SHARED_APPLICATION_D0"
            if passing >= int(gates_spec["required_passing_references"])
            else "FAIL_CLOSED_SELF_CANONICAL_INVERSE"
        ),
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


__all__ = ["fit_inverse_operator", "run_preflight", "uniform_sample_indexes"]
