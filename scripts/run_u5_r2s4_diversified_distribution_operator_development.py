#!/usr/bin/env python
"""Run frozen U5.R2S4D diversified-distribution development."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_u5_r2s1_histogram_case_retrieval_development import (  # noqa: E402
    _evaluate_method,
    _grid,
)
from scripts.run_u5_r2s2_content_palette_nuisance_development import (  # noqa: E402
    _generator_kwargs,
    _make_styles,
)
from src.roll2film.cube_diffeomorphic_flow import (  # noqa: E402
    CubeDiffeomorphicColourFlow,
)
from src.roll2film.palette_score_flow import (  # noqa: E402
    DiagonalGaussianMixturePalette,
)
from src.roll2film.histogram_case_retrieval import sample_palette  # noqa: E402
from src.roll2film.unpaired_distribution_flow import (  # noqa: E402
    evaluate_grouped_distribution_loss,
    fit_grouped_unpaired_distribution_flow,
    make_distribution_loss_assets,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_activation(
    config: dict[str, Any], parent_decision: dict[str, Any]
) -> None:
    gate = config["activation_gate"]
    if config["status"] != "pending_u5_r2s3_repeat_closure":
        raise ValueError("unexpected S4 contract status")
    if parent_decision.get("decision_branch") != gate["requires_u5_r2s3_branch"]:
        raise RuntimeError("S4 activation rejected by the S3 decision branch")
    if parent_decision.get("repeat_report_sha256_equal") is not True:
        raise RuntimeError("S4 activation requires an exact S3 report repeat")


def _condition_palette(
    rng: np.random.Generator,
    centre: np.ndarray,
    config: dict[str, Any],
) -> DiagonalGaussianMixturePalette:
    counts = tuple(int(value) for value in config["component_counts"])
    count = counts[int(rng.integers(0, len(counts)))]
    means = np.clip(
        centre[None, :]
        + rng.normal(
            scale=float(config["component_mean_jitter_standard_deviation"]),
            size=(count, 3),
        ),
        float(config["component_mean_minimum"]),
        float(config["component_mean_maximum"]),
    )
    return DiagonalGaussianMixturePalette(
        weights=rng.dirichlet(
            np.full(count, float(config["weight_dirichlet_alpha"]))
        ),
        means=means,
        standard_deviations=rng.uniform(
            float(config["component_standard_deviation_minimum"]),
            float(config["component_standard_deviation_maximum"]),
            size=(count, 3),
        ),
    )


def _make_condition_distributions(
    style_grids: np.ndarray,
    *,
    seed: int,
    content_config: dict[str, Any],
    integration_steps: int,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    rng = np.random.default_rng(seed)
    centres = np.asarray(
        content_config["condition_palette_centres"], dtype=np.float64
    )
    if centres.shape != (int(content_config["condition_count"]), 3):
        raise ValueError("condition centres do not match condition_count")
    neutral_styles, styled_styles = [], []
    for grid in style_grids:
        operator = CubeDiffeomorphicColourFlow(
            grid, integration_steps=integration_steps
        )
        neutral_conditions, styled_conditions = [], []
        for centre in centres:
            neutral_scenes, styled_scenes = [], []
            for _ in range(
                int(content_config["scenes_per_condition_per_domain"])
            ):
                palette = _condition_palette(rng, centre, content_config)
                neutral_scenes.append(
                    sample_palette(
                        palette,
                        sample_count=int(content_config["samples_per_scene"]),
                        rng=rng,
                    )
                )
            for _ in range(
                int(content_config["scenes_per_condition_per_domain"])
            ):
                palette = _condition_palette(rng, centre, content_config)
                source = sample_palette(
                    palette,
                    sample_count=int(content_config["samples_per_scene"]),
                    rng=rng,
                )
                styled_scenes.append(operator.apply(source))
            neutral_conditions.append(np.concatenate(neutral_scenes))
            styled_conditions.append(np.concatenate(styled_scenes))
        neutral_styles.append(neutral_conditions)
        styled_styles.append(styled_conditions)
    return neutral_styles, styled_styles


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _method_groups(
    method: dict[str, Any],
    sources: list[np.ndarray],
    targets: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    method_id = str(method["id"])
    if method_id == "pooled_rff_mmd_192":
        return [np.concatenate(sources)], [np.concatenate(targets)]
    if method_id == "conditional_rff_mmd_192":
        return sources, targets
    if method_id == "shuffled_condition_rff_mmd_192":
        permutation = [int(index) for index in method["target_condition_permutation"]]
        if sorted(permutation) != list(range(len(targets))):
            raise ValueError("target condition permutation is invalid")
        return sources, [targets[index] for index in permutation]
    raise ValueError(f"unsupported S4 method: {method_id}")


def _decision_branch(
    primary_gate_results: dict[str, bool], *, shuffled_fails: bool
) -> str:
    if primary_gate_results["all_except_repeat"]:
        return "pending_repeat_primary_all_gates_pass"
    if not all(
        primary_gate_results[name]
        for name in (
            "range",
            "jacobian",
            "norm",
            "inverse",
            "replay",
            "coefficient",
        )
    ):
        return "structure_or_repeat_fails"
    if not shuffled_fails:
        return "shuffled_negative_passes"
    if primary_gate_results["heldout_distribution"] and not all(
        primary_gate_results[name]
        for name in ("oracle_median", "oracle_p90", "replicate")
    ):
        return "primary_matches_distribution_but_operator_fails"
    return "primary_fails_or_does_not_beat_pooled"


def _fit_method(
    method: dict[str, Any],
    neutral: list[list[np.ndarray]],
    styled: list[list[np.ndarray]],
    *,
    config: dict[str, Any],
    method_index: int,
    device: str,
) -> tuple[list[CubeDiffeomorphicColourFlow], list[dict[str, float]]]:
    optimization = config["optimization"]
    operator_config = config["operator"]
    assets = make_distribution_loss_assets(config["loss"])
    operators, traces = [], []
    for style_index, (sources, targets) in enumerate(
        zip(neutral, styled, strict=True)
    ):
        fit_sources, fit_targets = _method_groups(method, sources, targets)
        operator, trace = fit_grouped_unpaired_distribution_flow(
            fit_sources,
            fit_targets,
            loss_assets=assets,
            axis_size=int(operator_config["velocity_grid_axis_size"]),
            integration_steps=int(operator_config["integration_steps"]),
            coefficient_vector_norm_cap=float(
                operator_config["coefficient_vector_norm_cap"]
            ),
            steps=int(optimization["steps"]),
            learning_rate=float(optimization["learning_rate"]),
            coefficient_l2=float(optimization["coefficient_l2"]),
            velocity_smoothness_l2=float(
                optimization["velocity_smoothness_l2"]
            ),
            gradient_clip_norm=float(optimization["gradient_clip_norm"]),
            seed=int(optimization["seed"])
            + 1000 * method_index
            + style_index,
            device=device,
            deterministic_algorithms=bool(
                optimization["deterministic_algorithms"]
            ),
            optimization_dtype=str(optimization["dtype"]),
        )
        operators.append(operator)
        traces.append(trace)
    return operators, traces


def run_development(
    config: dict[str, Any],
    parent_decision: dict[str, Any],
    *,
    config_sha256: str,
    software_commit: str,
) -> dict[str, Any]:
    _validate_activation(config, parent_decision)
    style_config = config["style_generator"]
    content_config = config["content_condition_generator"]
    operator_config = config["operator"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    palettes, oracle_grids = _make_styles(
        count=int(style_config["development_style_count"]),
        seed=int(style_config["development_style_seed"]),
        generator_kwargs=_generator_kwargs(style_config),
        velocity_grid_axis_size=int(operator_config["velocity_grid_axis_size"]),
        coefficient_vector_norm_cap=float(
            operator_config["coefficient_vector_norm_cap"]
        ),
    )
    neutral_a, styled_a = _make_condition_distributions(
        oracle_grids,
        seed=int(content_config["development_observation_seed_a"]),
        content_config=content_config,
        integration_steps=int(operator_config["integration_steps"]),
    )
    neutral_b, styled_b = _make_condition_distributions(
        oracle_grids,
        seed=int(content_config["development_observation_seed_b"]),
        content_config=content_config,
        integration_steps=int(operator_config["integration_steps"]),
    )
    points = _grid(int(operator_config["evaluation_grid_axis_size"]))
    assets = make_distribution_loss_assets(config["loss"])
    reports: dict[str, Any] = {}
    for method_index, method in enumerate(config["methods"]):
        fit_a, traces_a = _fit_method(
            method,
            neutral_a,
            styled_a,
            config=config,
            method_index=2 * method_index,
            device=device,
        )
        fit_b, traces_b = _fit_method(
            method,
            neutral_b,
            styled_b,
            config=config,
            method_index=2 * method_index + 1,
            device=device,
        )
        grids_a = np.stack([operator.velocity_grid for operator in fit_a])
        metrics = _evaluate_method(
            grids=grids_a,
            oracle_grids=oracle_grids,
            palettes=palettes,
            points=points,
            integration_steps=int(operator_config["integration_steps"]),
            permutation_grid=grids_a[0],
        )
        heldout_improvements, replicate_errors = [], []
        for index, (left, right) in enumerate(zip(fit_a, fit_b, strict=True)):
            for operator, sources, targets in (
                (left, neutral_b[index], styled_b[index]),
                (right, neutral_a[index], styled_a[index]),
            ):
                identity_loss, fitted_loss = evaluate_grouped_distribution_loss(
                    operator, sources, targets, assets
                )
                heldout_improvements.append(
                    1.0 - fitted_loss / max(identity_loss, 1e-30)
                )
            replicate_errors.append(
                float(
                    np.sqrt(
                        np.mean((left.apply(points) - right.apply(points)) ** 2)
                    )
                )
            )
        traces = traces_a + traces_b
        metrics["heldout_correct_conditional_loss_improvement_fraction"] = (
            _summary(heldout_improvements)
        )
        metrics["same_style_fit_replicate_operator_rmse"] = _summary(
            replicate_errors
        )
        metrics["optimization_initial_loss"] = _summary(
            [trace["initial_distribution_loss"] for trace in traces]
        )
        metrics["optimization_final_loss"] = _summary(
            [trace["final_distribution_loss"] for trace in traces]
        )
        reports[str(method["id"])] = {
            "specification": method,
            "metrics": metrics,
        }
    primary = reports["conditional_rff_mmd_192"]["metrics"]
    pooled = reports["pooled_rff_mmd_192"]["metrics"]
    shuffled = reports["shuffled_condition_rff_mmd_192"]["metrics"]
    gates = config["gates"]
    primary_error = float(primary["operator_output_rmse"]["median"])
    pooled_improvement = 1.0 - primary_error / max(
        float(pooled["operator_output_rmse"]["median"]), 1e-30
    )
    shuffled_improvement = 1.0 - primary_error / max(
        float(shuffled["operator_output_rmse"]["median"]), 1e-30
    )
    structure = primary["structure"]
    primary_gate_results = {
        "heldout_distribution": primary[
            "heldout_correct_conditional_loss_improvement_fraction"
        ]["median"]
        >= float(
            gates[
                "minimum_heldout_conditional_distribution_loss_improvement_fraction"
            ]
        ),
        "oracle_median": primary_error
        <= float(gates["maximum_oracle_operator_output_rmse_median"]),
        "oracle_p90": primary["operator_output_rmse"]["p90"]
        <= float(gates["maximum_oracle_operator_output_rmse_p90"]),
        "replicate": primary["same_style_fit_replicate_operator_rmse"]["median"]
        <= float(
            gates["maximum_same_style_fit_replicate_operator_rmse_median"]
        ),
        "beats_pooled": pooled_improvement
        >= float(
            gates["minimum_primary_oracle_error_improvement_over_pooled_fraction"]
        ),
        "beats_shuffled": shuffled_improvement
        >= float(
            gates[
                "minimum_primary_oracle_error_improvement_over_shuffled_fraction"
            ]
        ),
        "style_min": primary["identity_rmse_retention_ratio"]["median"]
        >= float(gates["minimum_identity_rmse_retention_ratio_median"]),
        "style_max": primary["identity_rmse_retention_ratio"]["median"]
        <= float(gates["maximum_identity_rmse_retention_ratio_median"]),
        "range": structure["minimum_output"] >= gates["minimum_output"]
        and structure["maximum_output"] <= gates["maximum_output"],
        "jacobian": structure["minimum_jacobian_determinant"]
        > gates["minimum_jacobian_determinant_exclusive"],
        "norm": structure["maximum_jacobian_spectral_norm"]
        <= gates["maximum_jacobian_spectral_norm"],
        "inverse": structure["maximum_inverse_error"]
        <= gates["maximum_inverse_error"],
        "replay": structure["maximum_replay_error"]
        <= gates["maximum_replay_error"],
        "coefficient": structure["maximum_coefficient_vector_norm"]
        <= gates["maximum_coefficient_vector_norm"],
    }
    shuffled_fails = (
        shuffled["operator_output_rmse"]["median"]
        > float(gates["maximum_oracle_operator_output_rmse_median"])
        or shuffled["operator_output_rmse"]["p90"]
        > float(gates["maximum_oracle_operator_output_rmse_p90"])
        or shuffled["same_style_fit_replicate_operator_rmse"]["median"]
        > float(gates["maximum_same_style_fit_replicate_operator_rmse_median"])
    )
    primary_gate_results["shuffled_negative"] = bool(shuffled_fails)
    primary_gate_results["all_except_repeat"] = bool(
        all(primary_gate_results.values())
    )
    branch = _decision_branch(
        primary_gate_results, shuffled_fails=shuffled_fails
    )
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "config_sha256": config_sha256,
        "software_commit": software_commit,
        "device": device,
        "optimization_dtype": config["optimization"]["dtype"],
        "reserved_confirmation_seed_accessed": False,
        "methods": reports,
        "comparisons": {
            "primary_oracle_error_improvement_over_pooled_fraction": pooled_improvement,
            "primary_oracle_error_improvement_over_shuffled_fraction": shuffled_improvement,
        },
        "primary_gate_results": primary_gate_results,
        "decision_branch_before_repeat": branch,
        "gates": gates,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "u5_r2s4_diversified_distribution_operator_development_v1.json",
    )
    parser.add_argument(
        "--parent-decision",
        type=Path,
        default=None,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    parent_decision_path = args.parent_decision
    if parent_decision_path is None:
        parent_decision_path = ROOT / config["activation_gate"][
            "required_decision_path"
        ]
    if not parent_decision_path.is_file():
        raise SystemExit(
            f"S4 activation evidence is absent: {parent_decision_path}"
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = run_development(
        config,
        json.loads(parent_decision_path.read_text(encoding="utf-8")),
        config_sha256=_sha256(config_bytes),
        software_commit=commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
