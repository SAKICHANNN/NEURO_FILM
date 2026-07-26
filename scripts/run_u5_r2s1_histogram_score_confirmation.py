#!/usr/bin/env python
"""Run frozen U5.R2S1C untouched histogram-score confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.histogram_case_retrieval import (  # noqa: E402
    HistogramCaseBank,
    canonical_rgb_histogram,
)
from scripts.run_u5_r2s1_histogram_case_retrieval_development import (  # noqa: E402
    _evaluate_method,
    _generator_kwargs,
    _grid,
    _make_population,
    _predict_grids,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _runtime_method(method: dict[str, Any]) -> dict[str, Any]:
    identifier = str(method["id"])
    if identifier == "query_kde_0p12":
        return {
            "id": identifier,
            "kind": "nonlearned_query_density_control",
            "bandwidth": float(method["bandwidth"]),
        }
    if identifier == "hard_1nn_hellinger":
        return {
            "id": identifier,
            "kind": "nonparametric_ml",
            "distance": str(method["distance"]),
        }
    if identifier == "global_mean_velocity":
        return {
            "id": identifier,
            "kind": "style_averaging_control",
        }
    raise ValueError(f"unexpected confirmatory method: {identifier}")


def evaluate_confirmation_gates(
    methods: dict[str, Any],
    *,
    histogram_permutation_error: float,
    gates: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, float]]:
    primary = methods["query_kde_0p12"]["metrics"]
    hard = methods["hard_1nn_hellinger"]["metrics"]
    global_mean = methods["global_mean_velocity"]["metrics"]
    primary_median = float(primary["operator_output_rmse"]["median"])
    hard_median = float(hard["operator_output_rmse"]["median"])
    global_median = float(global_mean["operator_output_rmse"]["median"])
    improvements = {
        "median_error_improvement_over_hard_1nn_fraction": 1.0
        - primary_median / hard_median,
        "median_error_improvement_over_global_fraction": 1.0
        - primary_median / global_median,
    }
    structure = primary["structure"]
    results = {
        "operator_rmse_median": primary_median
        <= float(gates["maximum_operator_output_rmse_median"]),
        "operator_rmse_p90": float(primary["operator_output_rmse"]["p90"])
        <= float(gates["maximum_operator_output_rmse_p90"]),
        "direction_cosine": float(
            primary["velocity_direction_cosine"]["median"]
        )
        >= float(gates["minimum_velocity_direction_cosine_median"]),
        "style_retention_minimum": float(
            primary["identity_rmse_retention_ratio"]["median"]
        )
        >= float(gates["minimum_identity_rmse_retention_ratio_median"]),
        "style_retention_maximum": float(
            primary["identity_rmse_retention_ratio"]["median"]
        )
        <= float(gates["maximum_identity_rmse_retention_ratio_median"]),
        "non_affine_retention": float(
            primary["best_affine_residual_retention_ratio"]["median"]
        )
        >= float(
            gates["minimum_best_affine_residual_retention_ratio_median"]
        ),
        "palette_attraction": float(
            primary["query_palette_log_density_gain"]["median"]
        )
        >= float(gates["minimum_query_palette_log_density_gain_median"]),
        "reference_separation": float(
            primary["reference_separation_ratio"]["median"]
        )
        >= float(gates["minimum_reference_separation_ratio_median"]),
        "improvement_over_hard": improvements[
            "median_error_improvement_over_hard_1nn_fraction"
        ]
        >= float(
            gates["minimum_median_error_improvement_over_hard_1nn_fraction"]
        ),
        "improvement_over_global": improvements[
            "median_error_improvement_over_global_fraction"
        ]
        >= float(
            gates["minimum_median_error_improvement_over_global_fraction"]
        ),
        "histogram_permutation": histogram_permutation_error
        <= float(gates["maximum_histogram_permutation_error"]),
        "operator_permutation": float(structure["permutation_operator_error"])
        <= float(gates["maximum_operator_permutation_error"]),
        "range": (
            float(structure["minimum_output"])
            >= float(gates["minimum_output"])
            and float(structure["maximum_output"])
            <= float(gates["maximum_output"])
        ),
        "positive_jacobian": float(
            structure["minimum_jacobian_determinant"]
        )
        > float(gates["minimum_jacobian_determinant_exclusive"]),
        "bounded_jacobian_norm": float(
            structure["maximum_jacobian_spectral_norm"]
        )
        <= float(gates["maximum_jacobian_spectral_norm"]),
        "inverse": float(structure["maximum_inverse_error"])
        <= float(gates["maximum_inverse_error"]),
        "replay": float(structure["maximum_replay_error"])
        <= float(gates["maximum_replay_error"]),
        "coefficient_bound": float(
            structure["maximum_coefficient_vector_norm"]
        )
        <= float(gates["maximum_coefficient_vector_norm"]),
    }
    results["all_except_repeat"] = bool(all(results.values()))
    return results, improvements


def run_confirmation(
    config: dict[str, Any],
    *,
    config_sha256: str,
    software_commit: str,
) -> dict[str, Any]:
    generator = config["synthetic_palette_generator"]
    observations = config["observations"]
    operator_config = config["operator"]
    common = {
        "sample_count": int(observations["samples_per_histogram"]),
        "histogram_axis_size": int(observations["histogram_axis_size"]),
        "velocity_grid_axis_size": int(
            operator_config["velocity_grid_axis_size"]
        ),
        "coefficient_vector_norm_cap": float(
            operator_config["coefficient_vector_norm_cap"]
        ),
        "generator_kwargs": _generator_kwargs(config),
    }
    _, bank_histograms, bank_grids, _ = _make_population(
        count=int(generator["case_bank_count"]),
        seed=int(generator["case_bank_seed"]),
        **common,
    )
    palettes, query_histograms, oracle_grids, query_samples = _make_population(
        count=int(generator["confirmation_query_count"]),
        seed=int(generator["confirmation_seed"]),
        **common,
    )
    bank = HistogramCaseBank(bank_histograms, bank_grids)
    points = _grid(int(operator_config["confirmation_grid_axis_size"]))
    shuffled = query_samples[0][
        np.random.default_rng(99174).permutation(len(query_samples[0]))
    ]
    permuted_histogram = canonical_rgb_histogram(
        shuffled, axis_size=int(observations["histogram_axis_size"])
    )
    histogram_permutation_error = float(
        np.max(np.abs(permuted_histogram - query_histograms[0]))
    )

    methods: dict[str, Any] = {}
    for fixed_method in config["methods"]:
        method = _runtime_method(fixed_method)
        grids = _predict_grids(
            method=method,
            bank=bank,
            query_histograms=query_histograms,
            histogram_axis_size=int(observations["histogram_axis_size"]),
            velocity_grid_axis_size=int(
                operator_config["velocity_grid_axis_size"]
            ),
            coefficient_vector_norm_cap=float(
                operator_config["coefficient_vector_norm_cap"]
            ),
        )
        permutation_grid = _predict_grids(
            method=method,
            bank=bank,
            query_histograms=permuted_histogram[None, :],
            histogram_axis_size=int(observations["histogram_axis_size"]),
            velocity_grid_axis_size=int(
                operator_config["velocity_grid_axis_size"]
            ),
            coefficient_vector_norm_cap=float(
                operator_config["coefficient_vector_norm_cap"]
            ),
        )[0]
        methods[str(fixed_method["id"])] = {
            "specification": fixed_method,
            "metrics": _evaluate_method(
                grids=grids,
                oracle_grids=oracle_grids,
                palettes=palettes,
                points=points,
                integration_steps=int(operator_config["integration_steps"]),
                permutation_grid=permutation_grid,
            ),
        }
    gate_results, improvements = evaluate_confirmation_gates(
        methods,
        histogram_permutation_error=histogram_permutation_error,
        gates=config["gates"],
    )
    if gate_results["all_except_repeat"]:
        branch = "pending_repeat_all_metric_gates_pass"
    else:
        failures = {
            name
            for name, passed in gate_results.items()
            if name != "all_except_repeat" and not passed
        }
        if failures & {
            "range",
            "positive_jacobian",
            "bounded_jacobian_norm",
            "inverse",
            "replay",
            "coefficient_bound",
        }:
            branch = "structure_fails"
        elif failures & {
            "improvement_over_hard",
            "improvement_over_global",
        }:
            branch = "control_improvement_fails"
        else:
            branch = "accuracy_or_retention_fails"
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "config_sha256": config_sha256,
        "software_commit": software_commit,
        "case_bank_seed": int(generator["case_bank_seed"]),
        "confirmation_seed": int(generator["confirmation_seed"]),
        "confirmation_query_count": len(query_histograms),
        "histogram_permutation_error": histogram_permutation_error,
        "methods": methods,
        "relative_improvements": improvements,
        "gates": config["gates"],
        "gate_results": gate_results,
        "decision_branch_before_repeat": branch,
        "repeat_report_required": True,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "u5_r2s1_histogram_score_confirmation_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    software_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = run_confirmation(
        config,
        config_sha256=_sha256(config_bytes),
        software_commit=software_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
