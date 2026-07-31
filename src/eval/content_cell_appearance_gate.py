"""Content-cell worst-case gate for bounded appearance matching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.eval.canonicalizer_consensus_appearance import (
    _grid,
    _normalized_cross_objective_score,
    _palette_kwargs,
    _style_operator,
    canonical_json_bytes,
    sha256_file,
)
from src.roll2film.cube_diffeomorphic_flow import CubeDiffeomorphicColourFlow
from src.roll2film.filmset_recipe_explainability import evaluate_flow_structure
from src.roll2film.histogram_case_retrieval import (
    generate_synthetic_palette,
    sample_palette,
)
from src.roll2film.unpaired_distribution_flow import (
    DistributionLossAssets,
    fit_unpaired_distribution_flow,
    make_distribution_loss_assets,
)


SCHEMA = "neuro_film.u5_r2bk22_content_cell_appearance_gate_report.v1"


class ContentCellAppearanceGateError(ValueError):
    """Raised when the frozen BK22 contract or evidence is invalid."""


def load_config(root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("schema")
        != "neuro_film.u5_r2bk22_content_cell_appearance_gate.v1"
        or config.get("status") != "contract_frozen"
    ):
        raise ContentCellAppearanceGateError("BK22 contract is not frozen")
    parent = json.loads(
        (root / config["parent"]["decision"]).read_text(encoding="utf-8")
    )
    if parent.get("decision") != config["parent"]["required_decision"]:
        raise ContentCellAppearanceGateError("BK22 parent decision mismatch")
    cells = config["content_cells"]
    count = int(cells["count"])
    permutation = tuple(int(value) for value in cells["confounded_target_permutation"])
    if (
        count < 2
        or len(cells["ids"]) != count
        or sorted(permutation) != list(range(count))
        or all(index == value for index, value in enumerate(permutation))
    ):
        raise ContentCellAppearanceGateError("BK22 content-cell contract is invalid")
    return config


def _sample_cell(
    palette: Any,
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    return sample_palette(
        palette,
        sample_count=sample_count,
        rng=np.random.default_rng(seed),
    )


def make_cell_distributions(
    config: dict[str, Any],
) -> dict[str, list[np.ndarray]]:
    population = config["population"]
    cell_count = int(config["content_cells"]["count"])
    palette_rng = np.random.default_rng(int(population["palette_seed"]))
    palettes = [
        generate_synthetic_palette(palette_rng, **_palette_kwargs(population))
        for _ in range(cell_count)
    ]
    sample_count = int(population["samples_per_cell"])
    style = _style_operator(config)
    fit_source = [
        _sample_cell(
            palette,
            sample_count=sample_count,
            seed=int(population["fit_seed"]) + 10 * index,
        )
        for index, palette in enumerate(palettes)
    ]
    fit_target = [
        style.apply(
            _sample_cell(
                palette,
                sample_count=sample_count,
                seed=int(population["fit_seed"]) + 10 * index + 1,
            )
        )
        for index, palette in enumerate(palettes)
    ]
    validation_source = [
        _sample_cell(
            palette,
            sample_count=sample_count,
            seed=int(population["validation_seed"]) + 10 * index,
        )
        for index, palette in enumerate(palettes)
    ]
    validation_target = [
        style.apply(
            _sample_cell(
                palette,
                sample_count=sample_count,
                seed=int(population["validation_seed"]) + 10 * index + 1,
            )
        )
        for index, palette in enumerate(palettes)
    ]
    permutation = config["content_cells"]["confounded_target_permutation"]
    confounded_target = [validation_target[int(index)] for index in permutation]
    return {
        "fit_source": fit_source,
        "fit_target": fit_target,
        "validation_source": validation_source,
        "valid_target": validation_target,
        "confounded_target": confounded_target,
    }


def _canonical_set_sha256(cells: list[np.ndarray]) -> str:
    values = np.concatenate(cells, axis=0)
    rounded = np.round(values, decimals=14)
    order = np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))
    canonical = np.ascontiguousarray(rounded[order], dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _cell_scores(
    operator: CubeDiffeomorphicColourFlow,
    sources: list[np.ndarray],
    targets: list[np.ndarray],
    assets: dict[str, DistributionLossAssets],
) -> tuple[list[float], list[dict[str, Any]]]:
    improvements = []
    rows = []
    for index, (source, target) in enumerate(zip(sources, targets, strict=True)):
        ratio, objective_rows = _normalized_cross_objective_score(
            operator, source, target, assets
        )
        improvement = 1.0 - ratio
        improvements.append(improvement)
        rows.append(
            {
                "cell_index": index,
                "mean_normalized_residual_ratio": ratio,
                "improvement_fraction": improvement,
                "objectives": objective_rows,
            }
        )
    return improvements, rows


def evaluate(
    config: dict[str, Any],
    *,
    config_sha256: str,
    software_commit: str,
) -> dict[str, Any]:
    distributions = make_cell_distributions(config)
    assets = {
        spec["id"]: make_distribution_loss_assets(spec)
        for spec in config["canonicalizers"]
    }
    operator_spec = config["operator"]
    optimization = config["optimization"]
    policy = config["policy"]
    device = str(optimization["device"])
    if device == "cuda" and not torch.cuda.is_available():
        raise ContentCellAppearanceGateError("frozen CUDA device is unavailable")
    pooled_fit_source = np.concatenate(distributions["fit_source"], axis=0)
    pooled_fit_target = np.concatenate(distributions["fit_target"], axis=0)
    pooled_validation_source = np.concatenate(
        distributions["validation_source"], axis=0
    )
    pooled_valid_target = np.concatenate(distributions["valid_target"], axis=0)
    pooled_confounded_target = np.concatenate(
        distributions["confounded_target"], axis=0
    )

    candidates: dict[str, CubeDiffeomorphicColourFlow] = {}
    candidate_rows: dict[str, Any] = {}
    for index, spec in enumerate(config["canonicalizers"]):
        operator, trace = fit_unpaired_distribution_flow(
            pooled_fit_source,
            pooled_fit_target,
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
            seed=int(optimization["seed"]) + index,
            device=device,
            deterministic_algorithms=bool(
                optimization["deterministic_algorithms"]
            ),
            optimization_dtype=str(optimization["dtype"]),
        )
        pooled_ratio, pooled_rows = _normalized_cross_objective_score(
            operator,
            pooled_validation_source,
            pooled_valid_target,
            assets,
        )
        candidates[spec["id"]] = operator
        candidate_rows[spec["id"]] = {
            "pooled_mean_normalized_residual_ratio": pooled_ratio,
            "pooled_improvement_fraction": 1.0 - pooled_ratio,
            "pooled_objectives": pooled_rows,
            "fit_trace": trace,
        }

    points = _grid(int(operator_spec["evaluation_grid_axis_size"]))
    names = tuple(candidates)
    disagreement = float(
        np.sqrt(
            np.mean(
                np.square(
                    candidates[names[0]].apply(points)
                    - candidates[names[1]].apply(points)
                )
            )
        )
    )
    selected_name = min(
        candidate_rows,
        key=lambda name: candidate_rows[name][
            "pooled_mean_normalized_residual_ratio"
        ],
    )
    selected = candidates[selected_name]
    valid_improvements, valid_rows = _cell_scores(
        selected,
        distributions["validation_source"],
        distributions["valid_target"],
        assets,
    )
    confounded_improvements, confounded_rows = _cell_scores(
        selected,
        distributions["validation_source"],
        distributions["confounded_target"],
        assets,
    )

    def policy_result(improvements: list[float]) -> tuple[str, list[str]]:
        reasons = []
        if disagreement > float(policy["maximum_canonicalizer_operator_rmse"]):
            reasons.append("canonicalizer-disagreement")
        pooled_improvement = float(
            candidate_rows[selected_name]["pooled_improvement_fraction"]
        )
        if pooled_improvement < float(
            policy["minimum_pooled_cross_objective_improvement_fraction"]
        ):
            reasons.append("insufficient-pooled-improvement")
        threshold = float(
            policy["minimum_worst_cell_cross_objective_improvement_fraction"]
        )
        pass_fraction = float(np.mean(np.asarray(improvements) >= threshold))
        if pass_fraction < float(policy["minimum_cell_pass_fraction"]):
            reasons.append("content-cell-worst-case")
        return ("identity_fallback", reasons) if reasons else (
            "hard_explicit_candidate",
            [],
        )

    valid_policy, valid_reasons = policy_result(valid_improvements)
    confounded_policy, confounded_reasons = policy_result(confounded_improvements)
    structure = evaluate_flow_structure(
        [selected],
        coefficient_cap=float(operator_spec["coefficient_vector_norm_cap"]),
    )
    source_sha = _canonical_set_sha256(distributions["validation_source"])
    valid_target_sha = _canonical_set_sha256(distributions["valid_target"])
    confounded_target_sha = _canonical_set_sha256(
        distributions["confounded_target"]
    )
    gates = config["gates"]
    checks = [
        {
            "name": "pooled_source_exact_between_views",
            "passed": bool(gates["pooled_source_bytes_equal_between_valid_and_confounded"]),
        },
        {
            "name": "pooled_target_exact_between_views",
            "passed": valid_target_sha == confounded_target_sha,
        },
        {
            "name": "valid_correspondence_selects_hard_candidate",
            "passed": valid_policy == "hard_explicit_candidate",
        },
        {
            "name": "confounded_correspondence_returns_identity",
            "passed": confounded_policy == "identity_fallback",
        },
        {
            "name": "selected_output_in_cube",
            "passed": structure["minimum_output"] >= float(gates["minimum_output"])
            and structure["maximum_output"] <= float(gates["maximum_output"]),
        },
        {
            "name": "selected_positive_jacobian",
            "passed": structure["minimum_jacobian_determinant"]
            > float(gates["minimum_jacobian_determinant_exclusive"]),
        },
        {
            "name": "selected_bounded_jacobian_norm",
            "passed": structure["maximum_jacobian_spectral_norm"]
            <= float(gates["maximum_jacobian_spectral_norm"]),
        },
        {
            "name": "selected_inverse",
            "passed": structure["maximum_inverse_error"]
            <= float(gates["maximum_inverse_error"]),
        },
        {
            "name": "selected_replay",
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
        "pooled_source_set_sha256": source_sha,
        "valid_pooled_target_set_sha256": valid_target_sha,
        "confounded_pooled_target_set_sha256": confounded_target_sha,
        "pooled_target_sets_exact": valid_target_sha == confounded_target_sha,
        "canonicalizer_operator_rmse": disagreement,
        "selected_canonicalizer": selected_name,
        "candidate_metrics": candidate_rows,
        "valid_correspondence": {
            "policy_result": valid_policy,
            "fallback_reasons": valid_reasons,
            "cell_improvements": valid_improvements,
            "worst_cell_improvement_fraction": float(min(valid_improvements)),
            "cells": valid_rows,
        },
        "confounded_correspondence": {
            "policy_result": confounded_policy,
            "fallback_reasons": confounded_reasons,
            "cell_improvements": confounded_improvements,
            "worst_cell_improvement_fraction": float(min(confounded_improvements)),
            "cells": confounded_rows,
        },
        "selected_structure": structure,
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
    "ContentCellAppearanceGateError",
    "evaluate",
    "load_config",
    "make_cell_distributions",
    "run",
]
