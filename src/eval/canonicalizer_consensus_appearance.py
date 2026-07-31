"""Hard canonicalizer-consensus policy for film-inspired appearance matching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.roll2film.cube_diffeomorphic_flow import CubeDiffeomorphicColourFlow
from src.roll2film.filmset_recipe_explainability import evaluate_flow_structure
from src.roll2film.histogram_case_retrieval import (
    generate_synthetic_palette,
    sample_palette,
)
from src.roll2film.unpaired_distribution_flow import (
    DistributionLossAssets,
    evaluate_distribution_loss,
    fit_unpaired_distribution_flow,
    make_distribution_loss_assets,
)


SCHEMA = "neuro_film.u5_r2bk21_canonicalizer_consensus_appearance_report.v1"


class CanonicalizerConsensusError(ValueError):
    """Raised when the frozen BK21 contract or evidence is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("schema")
        != "neuro_film.u5_r2bk21_canonicalizer_consensus_appearance.v1"
        or config.get("status") != "contract_frozen"
    ):
        raise CanonicalizerConsensusError("BK21 contract is not frozen")
    parent = json.loads(
        (root / config["parent"]["decision"]).read_text(encoding="utf-8")
    )
    if parent.get("decision") != config["parent"]["required_decision"]:
        raise CanonicalizerConsensusError("BK21 parent decision mismatch")
    if len(config["canonicalizers"]) != 2:
        raise CanonicalizerConsensusError("BK21 requires exactly two canonicalizers")
    return config


def _grid(axis_size: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, axis_size, dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )


def _style_operator(config: dict[str, Any]) -> CubeDiffeomorphicColourFlow:
    spec = config["style"]
    axis_size = int(spec["velocity_grid_axis_size"])
    # A fixed smooth per-channel logistic direction. It is deliberately simple
    # enough that two appearance objectives can agree without paired samples.
    direction = np.asarray([0.92, -0.54, 0.71], dtype=np.float64)
    direction *= float(spec["coefficient_vector_norm_cap"]) / np.linalg.norm(
        direction
    )
    grid = np.broadcast_to(direction, (axis_size, axis_size, axis_size, 3)).copy()
    return CubeDiffeomorphicColourFlow(
        grid,
        integration_steps=int(spec["integration_steps"]),
    )


def _palette_kwargs(population: dict[str, Any]) -> dict[str, Any]:
    return {
        "component_counts": population["component_counts"],
        "weight_dirichlet_alpha": float(population["weight_dirichlet_alpha"]),
        "mean_minimum": float(population["mean_minimum"]),
        "mean_maximum": float(population["mean_maximum"]),
        "standard_deviation_minimum": float(
            population["standard_deviation_minimum"]
        ),
        "standard_deviation_maximum": float(
            population["standard_deviation_maximum"]
        ),
    }


def _sample_population(
    palettes: list[Any],
    *,
    seed: int,
    samples_per_scene: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            sample_palette(palette, sample_count=samples_per_scene, rng=rng)
            for palette in palettes
        ],
        axis=0,
    )


def make_scenario_distributions(
    config: dict[str, Any],
    scenario: dict[str, Any],
    *,
    scenario_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    population = config["population"]
    palette_seed = int(config["style"]["seed"]) + 100 * scenario_index
    rng = np.random.default_rng(palette_seed)
    palettes = [
        generate_synthetic_palette(rng, **_palette_kwargs(population))
        for _ in range(int(population["scenes_per_split"]))
    ]
    samples = int(population["samples_per_scene"])
    source_a = _sample_population(
        palettes, seed=int(scenario["seed_a"]), samples_per_scene=samples
    )
    source_b = _sample_population(
        palettes, seed=int(scenario["seed_b"]), samples_per_scene=samples
    )

    kind = scenario["kind"]
    if kind == "same_population_bounded_style":
        operator = _style_operator(config)
        target_a = operator.apply(
            _sample_population(
                palettes,
                seed=int(scenario["seed_a"]) + 1_000_000,
                samples_per_scene=samples,
            )
        )
        target_b = operator.apply(
            _sample_population(
                palettes,
                seed=int(scenario["seed_b"]) + 1_000_000,
                samples_per_scene=samples,
            )
        )
    elif kind == "identical_distribution":
        target_a = _sample_population(
            palettes,
            seed=int(scenario["seed_a"]) + 1_000_000,
            samples_per_scene=samples,
        )
        target_b = _sample_population(
            palettes,
            seed=int(scenario["seed_b"]) + 1_000_000,
            samples_per_scene=samples,
        )
    elif kind == "different_population":
        target_rng = np.random.default_rng(palette_seed + 50_000)
        target_palettes = [
            generate_synthetic_palette(target_rng, **_palette_kwargs(population))
            for _ in range(int(population["scenes_per_split"]))
        ]
        shift = np.asarray(
            population["different_population_shift"], dtype=np.float64
        )
        target_a = np.clip(
            _sample_population(
                target_palettes,
                seed=int(scenario["seed_a"]) + 1_000_000,
                samples_per_scene=samples,
            )[:, [2, 0, 1]]
            + shift,
            0.0,
            1.0,
        )
        target_b = np.clip(
            _sample_population(
                target_palettes,
                seed=int(scenario["seed_b"]) + 1_000_000,
                samples_per_scene=samples,
            )[:, [2, 0, 1]]
            + shift,
            0.0,
            1.0,
        )
    else:
        raise CanonicalizerConsensusError(f"unsupported BK21 scenario: {kind}")
    return source_a, target_a, source_b, target_b


def _normalized_cross_objective_score(
    operator: CubeDiffeomorphicColourFlow,
    source: np.ndarray,
    target: np.ndarray,
    assets: dict[str, DistributionLossAssets],
) -> tuple[float, dict[str, dict[str, float]]]:
    rows: dict[str, dict[str, float]] = {}
    ratios = []
    for name, loss_assets in assets.items():
        identity, candidate = evaluate_distribution_loss(
            operator, source, target, loss_assets
        )
        ratio = candidate / max(identity, 1e-30)
        ratios.append(ratio)
        rows[name] = {
            "identity_loss": identity,
            "candidate_loss": candidate,
            "residual_ratio": ratio,
            "improvement_fraction": 1.0 - ratio,
        }
    return float(np.mean(ratios)), rows


def evaluate(
    config: dict[str, Any],
    *,
    config_sha256: str,
    software_commit: str,
) -> dict[str, Any]:
    optimization = config["optimization"]
    operator_spec = config["operator"]
    policy = config["policy"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assets = {
        spec["id"]: make_distribution_loss_assets(spec)
        for spec in config["canonicalizers"]
    }
    evaluation_points = _grid(int(operator_spec["evaluation_grid_axis_size"]))
    style = _style_operator(config)
    style_identity_rmse = float(
        np.sqrt(np.mean(np.square(style.apply(evaluation_points) - evaluation_points)))
    )
    scenarios: dict[str, Any] = {}
    retained_operators: list[CubeDiffeomorphicColourFlow] = []

    for scenario_index, scenario in enumerate(config["scenarios"]):
        source_a, target_a, source_b, target_b = make_scenario_distributions(
            config, scenario, scenario_index=scenario_index
        )
        candidates: dict[str, CubeDiffeomorphicColourFlow] = {}
        candidate_rows: dict[str, Any] = {}
        for canonicalizer_index, spec in enumerate(config["canonicalizers"]):
            operator, trace = fit_unpaired_distribution_flow(
                source_a,
                target_a,
                loss_assets=assets[spec["id"]],
                axis_size=int(operator_spec["velocity_grid_axis_size"]),
                integration_steps=int(operator_spec["integration_steps"]),
                coefficient_vector_norm_cap=float(
                    operator_spec["coefficient_vector_norm_cap"]
                ),
                steps=int(optimization["steps"]),
                learning_rate=float(optimization["learning_rate"]),
                coefficient_l2=float(optimization["coefficient_l2"]),
                velocity_smoothness_l2=float(
                    optimization["velocity_smoothness_l2"]
                ),
                gradient_clip_norm=float(optimization["gradient_clip_norm"]),
                seed=int(optimization["seed"])
                + 100 * scenario_index
                + canonicalizer_index,
                device=device,
                deterministic_algorithms=bool(
                    optimization["deterministic_algorithms"]
                ),
                optimization_dtype=str(optimization["dtype"]),
            )
            score, objective_rows = _normalized_cross_objective_score(
                operator, source_b, target_b, assets
            )
            candidates[spec["id"]] = operator
            candidate_rows[spec["id"]] = {
                "validation_mean_normalized_residual_ratio": score,
                "validation_cross_objectives": objective_rows,
                "fit_trace": trace,
            }

        names = tuple(candidates)
        disagreement = float(
            np.sqrt(
                np.mean(
                    np.square(
                        candidates[names[0]].apply(evaluation_points)
                        - candidates[names[1]].apply(evaluation_points)
                    )
                )
            )
        )
        identity_losses = []
        for loss_assets in assets.values():
            identity, _ = evaluate_distribution_loss(
                CubeDiffeomorphicColourFlow.identity(
                    axis_size=int(operator_spec["velocity_grid_axis_size"]),
                    integration_steps=int(operator_spec["integration_steps"]),
                ),
                source_b,
                target_b,
                loss_assets,
            )
            identity_losses.append(identity)
        combined_identity_loss = float(np.mean(identity_losses))
        selected_name = min(
            candidate_rows,
            key=lambda name: candidate_rows[name][
                "validation_mean_normalized_residual_ratio"
            ],
        )
        selected_ratio = float(
            candidate_rows[selected_name][
                "validation_mean_normalized_residual_ratio"
            ]
        )
        selected_improvement = 1.0 - selected_ratio
        fallback_reasons = []
        if combined_identity_loss <= float(
            policy["weak_reference_combined_identity_loss_maximum"]
        ):
            fallback_reasons.append("weak-reference-change")
        if disagreement > float(policy["maximum_canonicalizer_operator_rmse"]):
            fallback_reasons.append("canonicalizer-disagreement")
        if selected_improvement < float(
            policy["minimum_selected_cross_objective_improvement_fraction"]
        ):
            fallback_reasons.append("insufficient-cross-objective-improvement")
        if fallback_reasons:
            policy_result = "identity_fallback"
            output_operator = CubeDiffeomorphicColourFlow.identity(
                axis_size=int(operator_spec["velocity_grid_axis_size"]),
                integration_steps=int(operator_spec["integration_steps"]),
            )
        else:
            policy_result = "hard_explicit_candidate"
            output_operator = candidates[selected_name]
            retained_operators.append(output_operator)
        scenarios[scenario["id"]] = {
            "kind": scenario["kind"],
            "expected_policy": scenario["expected_policy"],
            "combined_identity_loss": combined_identity_loss,
            "canonicalizer_operator_rmse": disagreement,
            "selected_canonicalizer": selected_name,
            "selected_cross_objective_improvement_fraction": selected_improvement,
            "policy_result": policy_result,
            "fallback_reasons": fallback_reasons,
            "candidate_metrics": candidate_rows,
            "policy_output_identity_rmse": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            output_operator.apply(evaluation_points)
                            - evaluation_points
                        )
                    )
                )
            ),
        }

    structure = evaluate_flow_structure(
        retained_operators
        or [
            CubeDiffeomorphicColourFlow.identity(
                axis_size=int(operator_spec["velocity_grid_axis_size"]),
                integration_steps=int(operator_spec["integration_steps"]),
            )
        ],
        coefficient_cap=float(operator_spec["coefficient_vector_norm_cap"]),
    )
    gates = config["gates"]
    checks = [
        {
            "name": "true_style_is_material",
            "passed": style_identity_rmse
            >= float(config["style"]["minimum_true_style_identity_rmse"]),
        },
        {
            "name": "material_scenario_selects_hard_candidate",
            "passed": scenarios["material_shared_look"]["policy_result"]
            == "hard_explicit_candidate",
        },
        {
            "name": "material_scenario_improves_cross_objective_appearance",
            "passed": scenarios["material_shared_look"][
                "selected_cross_objective_improvement_fraction"
            ]
            >= float(
                gates["material_minimum_cross_objective_improvement_fraction"]
            ),
        },
        {
            "name": "already_matched_returns_identity",
            "passed": scenarios["appearance_already_matched"]["policy_result"]
            == "identity_fallback",
        },
        {
            "name": "content_confounded_returns_identity",
            "passed": scenarios["content_confounded_reference"]["policy_result"]
            == "identity_fallback",
        },
        {
            "name": "retained_output_in_cube",
            "passed": structure["minimum_output"] >= float(gates["minimum_output"])
            and structure["maximum_output"] <= float(gates["maximum_output"]),
        },
        {
            "name": "retained_positive_jacobian",
            "passed": structure["minimum_jacobian_determinant"]
            > float(gates["minimum_jacobian_determinant_exclusive"]),
        },
        {
            "name": "retained_bounded_jacobian_norm",
            "passed": structure["maximum_jacobian_spectral_norm"]
            <= float(gates["maximum_jacobian_spectral_norm"]),
        },
        {
            "name": "retained_inverse",
            "passed": structure["maximum_inverse_error"]
            <= float(gates["maximum_inverse_error"]),
        },
        {
            "name": "retained_replay",
            "passed": structure["maximum_replay_error"]
            <= float(gates["maximum_replay_error"]),
        },
    ]
    automatic_pass = all(check["passed"] for check in checks)
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": config_sha256,
        "software_commit": software_commit,
        "device": device,
        "true_style_identity_rmse": style_identity_rmse,
        "scenarios": scenarios,
        "retained_structure": structure,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            config["decision_if_all_gates_pass"]
            if automatic_pass
            else config["decision_if_any_gate_fails"]
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


def run(
    root: Path,
    config_path: Path,
    output_path: Path,
    *,
    software_commit: str,
) -> dict[str, Any]:
    config = load_config(root, config_path)
    report = evaluate(
        config,
        config_sha256=sha256_file(config_path),
        software_commit=software_commit,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(report))
    return report


__all__ = [
    "CanonicalizerConsensusError",
    "canonical_json_bytes",
    "evaluate",
    "load_config",
    "make_scenario_distributions",
    "run",
]
