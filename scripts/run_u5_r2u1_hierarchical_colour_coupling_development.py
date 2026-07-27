#!/usr/bin/env python
"""Run frozen U5.R2U1D hierarchical colour coupling development."""

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
from scripts.run_u5_r2s4_diversified_distribution_operator_development import (  # noqa: E402
    _make_condition_distributions,
)
from src.roll2film.cube_diffeomorphic_flow import (  # noqa: E402
    CubeDiffeomorphicColourFlow,
)
from src.roll2film.hierarchical_colour_coupling import (  # noqa: E402
    fit_paired_cube_diffeomorphic_flow,
    hierarchical_colour_coupling,
    random_colour_coupling,
)
from src.roll2film.unpaired_distribution_flow import (  # noqa: E402
    evaluate_grouped_distribution_loss,
    make_distribution_loss_assets,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_activation(
    config: dict[str, Any], parent_decision: dict[str, Any]
) -> None:
    if config["status"] != "pending_u5_r2s4_repeated_adjudication":
        raise ValueError("unexpected U1 contract status")
    gate = config["activation_gate"]
    if (
        parent_decision.get("repeat_report_sha256_equal") is not True
        or parent_decision.get("decision_branch")
        not in gate["allowed_parent_branches"]
    ):
        raise RuntimeError("U1 activation rejected by the repeated S4 decision")


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _couple_method(
    method: dict[str, Any],
    sources: list[np.ndarray],
    targets: list[np.ndarray],
    *,
    coupling_config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    method_id = str(method["id"])
    if len(sources) != len(targets) or not sources:
        raise ValueError("coupling groups must be equal and non-empty")
    if method_id == "pooled_hcc":
        pairs = hierarchical_colour_coupling(
            np.concatenate(sources),
            np.concatenate(targets),
            maximum_depth=int(coupling_config["hierarchical_maximum_depth"]),
            seed=seed,
        )
        return pairs.source, pairs.target, [len(pairs.source)]
    if method_id == "shuffled_condition_hcc":
        permutation = [
            int(value)
            for value in coupling_config[
                "shuffled_target_condition_permutation"
            ]
        ]
        if sorted(permutation) != list(range(len(targets))):
            raise ValueError("shuffled target condition permutation is invalid")
        method_targets = [targets[index] for index in permutation]
    else:
        method_targets = targets
    coupled_sources, coupled_targets, counts = [], [], []
    for index, (source, target) in enumerate(
        zip(sources, method_targets, strict=True)
    ):
        if method_id == "random_correct_condition":
            pairs = random_colour_coupling(source, target, seed=seed + index)
        elif method_id in {
            "correct_condition_hcc",
            "shuffled_condition_hcc",
        }:
            pairs = hierarchical_colour_coupling(
                source,
                target,
                maximum_depth=int(
                    coupling_config["hierarchical_maximum_depth"]
                ),
                seed=seed + index,
            )
        else:
            raise ValueError(f"unsupported U1 method: {method_id}")
        coupled_sources.append(pairs.source)
        coupled_targets.append(pairs.target)
        counts.append(len(pairs.source))
    return (
        np.concatenate(coupled_sources),
        np.concatenate(coupled_targets),
        counts,
    )


def _fit_method(
    method: dict[str, Any],
    neutral: list[list[np.ndarray]],
    styled: list[list[np.ndarray]],
    *,
    config: dict[str, Any],
    pair_seed: int,
    method_index: int,
    device: str,
) -> tuple[
    list[CubeDiffeomorphicColourFlow],
    list[dict[str, float]],
    list[list[int]],
]:
    optimization = config["optimization"]
    operator_config = config["operator"]
    operators, traces, pair_counts = [], [], []
    for style_index, (sources, targets) in enumerate(
        zip(neutral, styled, strict=True)
    ):
        source_pairs, target_pairs, counts = _couple_method(
            method,
            sources,
            targets,
            coupling_config=config["coupling"],
            seed=pair_seed + 100 * style_index,
        )
        operator, trace = fit_paired_cube_diffeomorphic_flow(
            source_pairs,
            target_pairs,
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
        pair_counts.append(counts)
    return operators, traces, pair_counts


def _decision_branch(
    gate_results: dict[str, bool], *, shuffled_fails: bool
) -> str:
    if gate_results["all_except_repeat"]:
        return "pending_repeat_primary_all_gates_pass"
    if not all(
        gate_results[name]
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
    if (
        gate_results["pair_fit"]
        and not (
            gate_results["oracle_median"] and gate_results["oracle_p90"]
        )
    ):
        return "operator_fails_despite_pair_fit"
    return "primary_does_not_beat_controls"


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
    coupling_config = config["coupling"]
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
    loss_assets = make_distribution_loss_assets(
        config["evaluation"]["heldout_distribution_loss_specification"]
    )
    reports: dict[str, Any] = {}
    for method_index, method in enumerate(config["methods"]):
        fit_a, traces_a, counts_a = _fit_method(
            method,
            neutral_a,
            styled_a,
            config=config,
            pair_seed=int(coupling_config["development_pair_seed_a"]),
            method_index=2 * method_index,
            device=device,
        )
        fit_b, traces_b, counts_b = _fit_method(
            method,
            neutral_b,
            styled_b,
            config=config,
            pair_seed=int(coupling_config["development_pair_seed_b"]),
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
                    operator, sources, targets, loss_assets
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
        metrics["pair_fit_loss_improvement_fraction"] = _summary(
            [
                1.0
                - trace["final_pair_mse"]
                / max(trace["initial_pair_mse"], 1e-30)
                for trace in traces
            ]
        )
        metrics["optimization_initial_pair_mse"] = _summary(
            [trace["initial_pair_mse"] for trace in traces]
        )
        metrics["optimization_final_pair_mse"] = _summary(
            [trace["final_pair_mse"] for trace in traces]
        )
        metrics["constructed_pair_count"] = _summary(
            [float(sum(counts)) for counts in counts_a + counts_b]
        )
        reports[str(method["id"])] = {
            "specification": method,
            "metrics": metrics,
        }
    primary = reports["correct_condition_hcc"]["metrics"]
    random_control = reports["random_correct_condition"]["metrics"]
    pooled = reports["pooled_hcc"]["metrics"]
    shuffled = reports["shuffled_condition_hcc"]["metrics"]
    gates = config["gates"]
    primary_error = float(primary["operator_output_rmse"]["median"])

    def improvement(control: dict[str, Any]) -> float:
        return 1.0 - primary_error / max(
            float(control["operator_output_rmse"]["median"]), 1e-30
        )

    improvements = {
        "primary_oracle_error_improvement_over_random_fraction": improvement(
            random_control
        ),
        "primary_oracle_error_improvement_over_pooled_fraction": improvement(
            pooled
        ),
        "primary_oracle_error_improvement_over_shuffled_fraction": improvement(
            shuffled
        ),
    }
    structure = primary["structure"]
    gate_results = {
        "pair_fit": primary["pair_fit_loss_improvement_fraction"]["median"]
        > 0.0,
        "oracle_median": primary_error
        <= float(gates["maximum_oracle_operator_output_rmse_median"]),
        "oracle_p90": primary["operator_output_rmse"]["p90"]
        <= float(gates["maximum_oracle_operator_output_rmse_p90"]),
        "replicate": primary["same_style_fit_replicate_operator_rmse"]["median"]
        <= float(
            gates["maximum_same_style_fit_replicate_operator_rmse_median"]
        ),
        "beats_random": improvements[
            "primary_oracle_error_improvement_over_random_fraction"
        ]
        >= float(
            gates[
                "minimum_primary_oracle_error_improvement_over_random_fraction"
            ]
        ),
        "beats_pooled": improvements[
            "primary_oracle_error_improvement_over_pooled_fraction"
        ]
        >= float(
            gates[
                "minimum_primary_oracle_error_improvement_over_pooled_fraction"
            ]
        ),
        "beats_shuffled": improvements[
            "primary_oracle_error_improvement_over_shuffled_fraction"
        ]
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
    gate_results["shuffled_negative"] = bool(shuffled_fails)
    gate_results["all_except_repeat"] = bool(all(gate_results.values()))
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
        "comparisons": improvements,
        "primary_gate_results": gate_results,
        "decision_branch_before_repeat": _decision_branch(
            gate_results, shuffled_fails=shuffled_fails
        ),
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
        / "u5_r2u1_hierarchical_colour_coupling_development_v1.json",
    )
    parser.add_argument("--parent-decision", type=Path, default=None)
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
            f"U1 activation evidence is absent: {parent_decision_path}"
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
