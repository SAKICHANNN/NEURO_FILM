"""Off-diagonal evaluator Oracle for a bank of bounded explicit operators."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.roll2film.triangular_logit_transport import (
    PARAMETER_COUNT,
    TriangularLogitTransport,
    fit_triangular_logit_transport,
    select_safe_transport,
)
from src.eval.fivek_unseen_content_confirmation import load_srgb16


class FiveKCasebankOracleError(ValueError):
    """Raised when the case-bank experiment violates its fixed boundary."""


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_population(
    manifest: Mapping[str, Any],
    *,
    split: str,
    target_variant: str,
    maximum_side: int,
) -> list[dict[str, Any]]:
    """Load one frozen split while validating every normalized asset hash."""

    target_fields = {
        "filtered": ("target_path", "target_sha256"),
        "aligned_expert": (
            "aligned_expert_path",
            "aligned_expert_sha256",
        ),
    }
    if split not in {"development", "confirmation"}:
        raise FiveKCasebankOracleError("invalid population split")
    if target_variant not in target_fields:
        raise FiveKCasebankOracleError("invalid target variant")
    target_path_field, target_hash_field = target_fields[target_variant]
    loaded = []
    for row in manifest.get("rows", []):
        if row.get("split") != split:
            continue
        source_path = Path(str(row["source_path"]))
        target_path = Path(str(row[target_path_field]))
        if (
            not source_path.is_file()
            or not target_path.is_file()
            or _file_sha256(source_path) != str(row["source_sha256"])
            or _file_sha256(target_path) != str(row[target_hash_field])
        ):
            raise FiveKCasebankOracleError("normalized asset drift")
        source = load_srgb16(source_path, maximum_side)
        target = load_srgb16(target_path, maximum_side)
        if source.shape != target.shape:
            raise FiveKCasebankOracleError("normalized pair shape drift")
        loaded.append(
            {
                "pair_id": str(row["pair_id"]),
                "group": str(row["camera_model"]),
                "source": source,
                "target": target,
            }
        )
    if not loaded:
        raise FiveKCasebankOracleError("empty normalized split")
    return sorted(loaded, key=lambda row: row["pair_id"])


def _rgb(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 3
        or array.shape[-1] != 3
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
        or np.any(array > 1.0)
    ):
        raise FiveKCasebankOracleError("invalid bounded RGB")
    return array


def _even_samples(rgb: np.ndarray, count: int) -> np.ndarray:
    flat = _rgb(rgb).reshape(-1, 3)
    if count < 1:
        raise FiveKCasebankOracleError("sample count must be positive")
    if len(flat) <= count:
        return flat
    indices = np.linspace(0, len(flat) - 1, count, dtype=np.int64)
    return flat[indices]


def _rmse(first: np.ndarray, second: np.ndarray) -> float:
    delta = np.asarray(first, dtype=np.float64) - np.asarray(
        second, dtype=np.float64
    )
    return float(np.sqrt(np.mean(delta * delta)))


def _safe_operator(
    parameters: np.ndarray, operator: Mapping[str, Any]
) -> tuple[TriangularLogitTransport, dict[str, float | int]]:
    return select_safe_transport(
        parameters=parameters,
        grid_size=int(operator["safe_dose_grid_size"]),
        finite_difference=float(operator["safe_dose_finite_difference"]),
        minimum_jacobian_determinant=float(
            operator["minimum_jacobian_determinant"]
        ),
        maximum_jacobian_condition=float(
            operator["maximum_jacobian_condition"]
        ),
        bisection_iterations=int(operator["safe_dose_bisection_iterations"]),
    )


def _fit_one(
    source: np.ndarray,
    target: np.ndarray,
    operator: Mapping[str, Any],
) -> tuple[np.ndarray, TriangularLogitTransport, dict[str, float | int]]:
    sample_count = int(operator["fit_samples_per_image"])
    source_samples = _even_samples(source, sample_count)
    target_samples = _even_samples(target, sample_count)
    parameters, success = fit_triangular_logit_transport(
        source_samples[:, None, :],
        target_samples[:, None, :],
        lower_bounds=np.asarray(operator["lower_bounds"], dtype=np.float64),
        upper_bounds=np.asarray(operator["upper_bounds"], dtype=np.float64),
        sample_stride=1,
        identity_shrinkage=float(operator["identity_shrinkage"]),
        maximum_evaluations=int(operator["maximum_fit_evaluations"]),
    )
    if not success:
        raise FiveKCasebankOracleError("explicit operator fit failed")
    safe, diagnostics = _safe_operator(parameters, operator)
    return parameters, safe, diagnostics


def _group_bootstrap(
    global_errors: np.ndarray,
    oracle_errors: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    repetitions: int,
) -> list[float]:
    unique = np.unique(groups)
    if len(unique) < 2 or repetitions < 1:
        raise FiveKCasebankOracleError("bootstrap requires multiple groups")
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        baseline = float(np.mean(global_errors[selected]))
        values.append(
            (baseline - float(np.mean(oracle_errors[selected])))
            / max(baseline, 1.0e-12)
        )
    return values


def evaluate_offdiagonal_oracle(
    development_rows: Sequence[Mapping[str, Any]],
    confirmation_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit development cases and evaluate them only on disjoint rows."""

    development = sorted(development_rows, key=lambda row: str(row["pair_id"]))
    confirmation = sorted(
        confirmation_rows, key=lambda row: str(row["pair_id"])
    )
    development_ids = {str(row["pair_id"]) for row in development}
    confirmation_ids = {str(row["pair_id"]) for row in confirmation}
    if (
        not development
        or not confirmation
        or len(development_ids) != len(development)
        or len(confirmation_ids) != len(confirmation)
        or development_ids & confirmation_ids
    ):
        raise FiveKCasebankOracleError("invalid off-diagonal populations")

    operator_config = config["operator"]
    if (
        int(operator_config["parameter_count"]) != PARAMETER_COUNT
        or operator_config.get("family")
        != "monotone_triangular_logit_transport"
        or config.get("router_training_allowed")
        or config.get("final_rgb_learning_allowed")
    ):
        raise FiveKCasebankOracleError("operator or Oracle boundary drift")

    cases = []
    pooled_source = []
    pooled_target = []
    for row in development:
        source = _rgb(row["source"])
        target = _rgb(row["target"])
        if source.shape != target.shape:
            raise FiveKCasebankOracleError("paired shape mismatch")
        parameters, safe, diagnostics = _fit_one(
            source, target, operator_config
        )
        cases.append(
            {
                "pair_id": str(row["pair_id"]),
                "parameters": parameters,
                "operator": safe,
                "diagnostics": diagnostics,
            }
        )
        pooled_source.append(
            _even_samples(source, int(operator_config["pooled_samples_per_image"]))
        )
        pooled_target.append(
            _even_samples(target, int(operator_config["pooled_samples_per_image"]))
        )
    pooled_parameters, pooled_safe, pooled_diagnostics = _fit_one(
        np.concatenate(pooled_source)[:, None, :],
        np.concatenate(pooled_target)[:, None, :],
        {**operator_config, "fit_samples_per_image": sum(map(len, pooled_source))},
    )

    evaluation = config["evaluation"]
    sample_count = int(evaluation["samples_per_confirmation_image"])
    strength_doses = [float(value) for value in evaluation["strength_doses"]]
    if not strength_doses or min(strength_doses) < 0.0 or max(strength_doses) > 1.0:
        raise FiveKCasebankOracleError("invalid strength control")
    global_errors = []
    strength_errors = []
    oracle_errors = []
    random_errors = []
    selected_case_ids = []
    rows = []
    random_seed = int(evaluation["random_case_seed"])
    for row in confirmation:
        source = _even_samples(_rgb(row["source"]), sample_count)
        target = _even_samples(_rgb(row["target"]), sample_count)
        global_error = _rmse(pooled_safe.apply(source), target)
        strength_error = min(
            _rmse(
                TriangularLogitTransport(
                    pooled_parameters,
                    dose=pooled_safe.dose * dose,
                ).apply(source),
                target,
            )
            for dose in strength_doses
        )
        case_errors = np.asarray(
            [_rmse(case["operator"].apply(source), target) for case in cases],
            dtype=np.float64,
        )
        oracle_index = int(np.argmin(case_errors))
        digest = hashlib.sha256(
            f"{random_seed}\0{row['pair_id']}".encode("utf-8")
        ).digest()
        random_index = int.from_bytes(digest[:8], "big") % len(cases)
        global_errors.append(global_error)
        strength_errors.append(strength_error)
        oracle_errors.append(float(case_errors[oracle_index]))
        random_errors.append(float(case_errors[random_index]))
        selected_case_ids.append(cases[oracle_index]["pair_id"])
        rows.append(
            {
                "pair_id": str(row["pair_id"]),
                "group": str(row["group"]),
                "global_rmse": global_error,
                "strength_oracle_rmse": strength_error,
                "case_oracle_rmse": float(case_errors[oracle_index]),
                "random_case_rmse": float(case_errors[random_index]),
                "selected_case_id": cases[oracle_index]["pair_id"],
                "random_case_id": cases[random_index]["pair_id"],
            }
        )

    global_array = np.asarray(global_errors)
    strength_array = np.asarray(strength_errors)
    oracle_array = np.asarray(oracle_errors)
    random_array = np.asarray(random_errors)
    groups = np.asarray([str(row["group"]) for row in confirmation], dtype=object)
    bootstrap = _group_bootstrap(
        global_array,
        oracle_array,
        groups,
        seed=int(evaluation["bootstrap_seed"]),
        repetitions=int(evaluation["bootstrap_repetitions"]),
    )
    improvement = (global_array - oracle_array) / np.maximum(
        global_array, 1.0e-12
    )
    strength_improvement = (strength_array - oracle_array) / np.maximum(
        strength_array, 1.0e-12
    )
    counts = {
        case_id: selected_case_ids.count(case_id)
        for case_id in sorted(set(selected_case_ids))
    }
    metrics = {
        "mean_improvement_over_global": float(
            (global_array.mean() - oracle_array.mean())
            / max(global_array.mean(), 1.0e-12)
        ),
        "win_fraction_over_global": float(np.mean(oracle_array < global_array)),
        "p95_ratio_to_global": float(
            np.quantile(oracle_array, 0.95)
            / max(np.quantile(global_array, 0.95), 1.0e-12)
        ),
        "worst_ratio_to_global": float(
            np.max(oracle_array) / max(np.max(global_array), 1.0e-12)
        ),
        "mean_improvement_over_strength_oracle": float(
            (strength_array.mean() - oracle_array.mean())
            / max(strength_array.mean(), 1.0e-12)
        ),
        "win_fraction_over_strength_oracle": float(
            np.mean(oracle_array < strength_array)
        ),
        "random_case_mean_ratio_to_global": float(
            random_array.mean() / max(global_array.mean(), 1.0e-12)
        ),
        "mean_improvement_over_random_case": float(
            (random_array.mean() - oracle_array.mean())
            / max(random_array.mean(), 1.0e-12)
        ),
        "win_fraction_over_random_case": float(
            np.mean(oracle_array < random_array)
        ),
        "median_per_row_improvement_over_global": float(np.median(improvement)),
        "median_per_row_improvement_over_strength": float(
            np.median(strength_improvement)
        ),
        "group_bootstrap_improvement_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "distinct_selected_cases": len(counts),
        "maximum_selected_case_share": max(counts.values()) / len(confirmation),
    }
    thresholds = evaluation["gates"]
    gates = {
        "mean": metrics["mean_improvement_over_global"]
        >= thresholds["minimum_mean_improvement_over_global"],
        "wins": metrics["win_fraction_over_global"]
        >= thresholds["minimum_win_fraction_over_global"],
        "p95": metrics["p95_ratio_to_global"]
        <= thresholds["maximum_p95_ratio_to_global"],
        "worst": metrics["worst_ratio_to_global"]
        <= thresholds["maximum_worst_ratio_to_global"],
        "beyond_strength": metrics["mean_improvement_over_strength_oracle"]
        >= thresholds["minimum_mean_improvement_over_strength_oracle"],
        "strength_wins": metrics["win_fraction_over_strength_oracle"]
        >= thresholds["minimum_win_fraction_over_strength_oracle"],
        "random_case": metrics["mean_improvement_over_random_case"]
        >= thresholds["minimum_mean_improvement_over_random_case"],
        "random_wins": metrics["win_fraction_over_random_case"]
        >= thresholds["minimum_win_fraction_over_random_case"],
        "bootstrap": metrics["group_bootstrap_improvement_ci95"][0]
        > thresholds["minimum_bootstrap_lower_improvement"],
        "case_support": metrics["distinct_selected_cases"]
        >= thresholds["minimum_distinct_selected_cases"],
        "case_concentration": metrics["maximum_selected_case_share"]
        <= thresholds["maximum_selected_case_share"],
    }
    return {
        "schema": "neuro_film.u5_r2bq1_offdiagonal_casebank_oracle.v1",
        "development_rows": len(development),
        "confirmation_rows": len(confirmation),
        "development_groups": len(
            set(str(row["group"]) for row in development)
        ),
        "confirmation_groups": len(set(groups.tolist())),
        "pooled_operator": {
            "parameters": pooled_parameters.tolist(),
            "parameter_sha256": _array_sha256(pooled_parameters),
            "dose": float(pooled_safe.dose),
            "diagnostics": pooled_diagnostics,
        },
        "case_bank": [
            {
                "pair_id": case["pair_id"],
                "parameters": case["parameters"].tolist(),
                "parameter_sha256": _array_sha256(case["parameters"]),
                "dose": float(case["operator"].dose),
                "diagnostics": case["diagnostics"],
            }
            for case in cases
        ],
        "selected_case_counts": counts,
        "metrics": metrics,
        "gates": gates,
        "automatic_pass": all(gates.values()),
        "router_training_allowed": False,
        "rows": rows,
    }


__all__ = [
    "FiveKCasebankOracleError",
    "evaluate_offdiagonal_oracle",
    "load_split_population",
]
