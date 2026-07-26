#!/usr/bin/env python
"""Run frozen U5.R2S3D unpaired distribution/operator pilot."""

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

from src.roll2film.cube_diffeomorphic_flow import CubeDiffeomorphicColourFlow  # noqa: E402
from src.roll2film.histogram_case_retrieval import (  # noqa: E402
    generate_synthetic_palette,
    sample_palette,
)
from src.roll2film.unpaired_distribution_flow import (  # noqa: E402
    evaluate_distribution_loss,
    fit_unpaired_distribution_flow,
    make_distribution_loss_assets,
)
from scripts.run_u5_r2s1_histogram_case_retrieval_development import (  # noqa: E402
    _evaluate_method,
    _grid,
)
from scripts.run_u5_r2s2_content_palette_nuisance_development import (  # noqa: E402
    _generator_kwargs,
    _make_styles,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _make_distributions(
    style_grids: np.ndarray,
    *,
    seed: int,
    content_config: dict[str, Any],
    integration_steps: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    kwargs = _generator_kwargs(content_config)
    neutral_groups, styled_groups = [], []
    for grid in style_grids:
        operator = CubeDiffeomorphicColourFlow(
            grid, integration_steps=integration_steps
        )
        neutral, styled = [], []
        for _ in range(int(content_config["scenes_per_distribution"])):
            palette = generate_synthetic_palette(rng, **kwargs)
            neutral.append(
                sample_palette(
                    palette,
                    sample_count=int(content_config["samples_per_scene"]),
                    rng=rng,
                )
            )
        for _ in range(int(content_config["scenes_per_distribution"])):
            palette = generate_synthetic_palette(rng, **kwargs)
            source = sample_palette(
                palette,
                sample_count=int(content_config["samples_per_scene"]),
                rng=rng,
            )
            styled.append(operator.apply(source))
        neutral_groups.append(np.concatenate(neutral))
        styled_groups.append(np.concatenate(styled))
    return neutral_groups, styled_groups


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def run_pilot(
    config: dict[str, Any], *, config_sha256: str, software_commit: str
) -> dict[str, Any]:
    style_config = config["style_generator"]
    content_config = config["content_generator"]
    operator_config = config["operator"]
    optimization = config["optimization"]
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
    neutral_a, styled_a = _make_distributions(
        oracle_grids,
        seed=int(content_config["development_observation_seed_a"]),
        content_config=content_config,
        integration_steps=int(operator_config["integration_steps"]),
    )
    neutral_b, styled_b = _make_distributions(
        oracle_grids,
        seed=int(content_config["development_observation_seed_b"]),
        content_config=content_config,
        integration_steps=int(operator_config["integration_steps"]),
    )
    points = _grid(int(operator_config["evaluation_grid_axis_size"]))
    reports = {}
    for method_index, method in enumerate(config["candidate_losses"]):
        assets = make_distribution_loss_assets(method)
        fit_a, fit_b, traces = [], [], []
        for style_index in range(len(oracle_grids)):
            pair = []
            for replicate, (source, target) in enumerate(
                (
                    (neutral_a[style_index], styled_a[style_index]),
                    (neutral_b[style_index], styled_b[style_index]),
                )
            ):
                operator, trace = fit_unpaired_distribution_flow(
                    source,
                    target,
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
                    gradient_clip_norm=float(
                        optimization["gradient_clip_norm"]
                    ),
                    seed=int(optimization["seed"])
                    + 1000 * method_index
                    + 2 * style_index
                    + replicate,
                    device=device,
                    deterministic_algorithms=bool(
                        optimization["deterministic_algorithms"]
                    ),
                )
                pair.append(operator)
                traces.append(trace)
            fit_a.append(pair[0])
            fit_b.append(pair[1])
        grids_a = np.stack([operator.velocity_grid for operator in fit_a])
        grids_b = np.stack([operator.velocity_grid for operator in fit_b])
        base = _evaluate_method(
            grids=grids_a,
            oracle_grids=oracle_grids,
            palettes=palettes,
            points=points,
            integration_steps=int(operator_config["integration_steps"]),
            permutation_grid=grids_a[0],
        )
        heldout_improvements, replicate_errors = [], []
        for index, (left, right) in enumerate(zip(fit_a, fit_b, strict=True)):
            for operator, source, target in (
                (left, neutral_b[index], styled_b[index]),
                (right, neutral_a[index], styled_a[index]),
            ):
                identity_loss, fitted_loss = evaluate_distribution_loss(
                    operator, source, target, assets
                )
                heldout_improvements.append(
                    1.0 - fitted_loss / max(identity_loss, 1e-30)
                )
            replicate_errors.append(
                float(
                    np.sqrt(
                        np.mean(
                            (
                                left.apply(points)
                                - right.apply(points)
                            )
                            ** 2
                        )
                    )
                )
            )
        base["heldout_distribution_loss_improvement_fraction"] = _summary(
            heldout_improvements
        )
        base["same_style_fit_replicate_operator_rmse"] = _summary(
            replicate_errors
        )
        base["optimization_initial_loss"] = _summary(
            [trace["initial_distribution_loss"] for trace in traces]
        )
        base["optimization_final_loss"] = _summary(
            [trace["final_distribution_loss"] for trace in traces]
        )
        gates = config["gates"]
        structure = base["structure"]
        gate_results = {
            "distribution": base[
                "heldout_distribution_loss_improvement_fraction"
            ]["median"]
            >= float(
                gates[
                    "minimum_heldout_distribution_loss_improvement_fraction"
                ]
            ),
            "oracle_median": base["operator_output_rmse"]["median"]
            <= float(gates["maximum_oracle_operator_output_rmse_median"]),
            "oracle_p90": base["operator_output_rmse"]["p90"]
            <= float(gates["maximum_oracle_operator_output_rmse_p90"]),
            "replicate": base[
                "same_style_fit_replicate_operator_rmse"
            ]["median"]
            <= float(
                gates[
                    "maximum_same_style_fit_replicate_operator_rmse_median"
                ]
            ),
            "style_min": base["identity_rmse_retention_ratio"]["median"]
            >= float(gates["minimum_identity_rmse_retention_ratio_median"]),
            "style_max": base["identity_rmse_retention_ratio"]["median"]
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
        gate_results["all_except_repeat"] = bool(all(gate_results.values()))
        reports[str(method["id"])] = {
            "specification": method,
            "metrics": base,
            "gate_results": gate_results,
        }
    distribution_pass = any(
        report["gate_results"]["distribution"] for report in reports.values()
    )
    all_pass = any(
        report["gate_results"]["all_except_repeat"] for report in reports.values()
    )
    branch = (
        "pending_repeat_distribution_and_operator_gates_pass"
        if all_pass
        else "distribution_pass_operator_fail"
        if distribution_pass
        else "distribution_fail"
    )
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "config_sha256": config_sha256,
        "software_commit": software_commit,
        "device": device,
        "reserved_confirmation_seed_accessed": False,
        "methods": reports,
        "decision_branch_before_repeat": branch,
        "gates": config["gates"],
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "u5_r2s3_unpaired_distribution_operator_pilot_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = run_pilot(
        config, config_sha256=_sha256(config_bytes), software_commit=commit
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
