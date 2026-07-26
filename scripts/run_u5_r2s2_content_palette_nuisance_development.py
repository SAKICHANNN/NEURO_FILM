#!/usr/bin/env python
"""Run frozen U5.R2S2D synthetic content-palette nuisance development."""

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

from src.roll2film.content_palette_nuisance import (  # noqa: E402
    content_residual_features,
    density_ratio_velocity_grid,
    fit_bounded_multi_output_ridge,
)
from src.roll2film.cube_diffeomorphic_flow import (  # noqa: E402
    CubeDiffeomorphicColourFlow,
)
from src.roll2film.histogram_case_retrieval import (  # noqa: E402
    canonical_rgb_histogram,
    generate_synthetic_palette,
    histogram_kde_velocity_grid,
    sample_palette,
)
from src.roll2film.palette_score_flow import (  # noqa: E402
    DiagonalGaussianMixturePalette,
    palette_score_velocity_grid,
)
from scripts.run_u5_r2s1_histogram_case_retrieval_development import (  # noqa: E402
    _evaluate_method,
    _grid,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _generator_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "component_counts": payload["component_counts"],
        "weight_dirichlet_alpha": float(payload["weight_dirichlet_alpha"]),
        "mean_minimum": float(payload["mean_minimum"]),
        "mean_maximum": float(payload["mean_maximum"]),
        "standard_deviation_minimum": float(
            payload["standard_deviation_minimum"]
        ),
        "standard_deviation_maximum": float(
            payload["standard_deviation_maximum"]
        ),
    }


def _make_styles(
    *,
    count: int,
    seed: int,
    generator_kwargs: dict[str, Any],
    velocity_grid_axis_size: int,
    coefficient_vector_norm_cap: float,
) -> tuple[list[DiagonalGaussianMixturePalette], np.ndarray]:
    rng = np.random.default_rng(seed)
    palettes = []
    grids = []
    for _ in range(count):
        palette = generate_synthetic_palette(rng, **generator_kwargs)
        palettes.append(palette)
        grids.append(
            palette_score_velocity_grid(
                palette,
                axis_size=velocity_grid_axis_size,
                coefficient_vector_norm_cap=coefficient_vector_norm_cap,
            )
        )
    return palettes, np.stack(grids)


def _make_observations(
    style_grids: np.ndarray,
    *,
    seed: int,
    content_config: dict[str, Any],
    histogram_axis_size: int,
    integration_steps: int,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    content_kwargs = _generator_kwargs(content_config)
    styled_histograms = []
    neutral_histograms = []
    first_rows: tuple[np.ndarray, np.ndarray] | None = None
    for style_grid in style_grids:
        operator = CubeDiffeomorphicColourFlow(
            style_grid, integration_steps=integration_steps
        )
        styled_rows = []
        neutral_rows = []
        for _ in range(int(content_config["styled_scenes_per_style"])):
            content = generate_synthetic_palette(rng, **content_kwargs)
            source = sample_palette(
                content,
                sample_count=int(content_config["samples_per_scene"]),
                rng=rng,
            )
            styled_rows.append(operator.apply(source))
        for _ in range(int(content_config["neutral_control_scenes_per_style"])):
            content = generate_synthetic_palette(rng, **content_kwargs)
            neutral_rows.append(
                sample_palette(
                    content,
                    sample_count=int(content_config["samples_per_scene"]),
                    rng=rng,
                )
            )
        styled = np.concatenate(styled_rows)
        neutral = np.concatenate(neutral_rows)
        if first_rows is None:
            first_rows = (styled.copy(), neutral.copy())
        styled_histograms.append(
            canonical_rgb_histogram(styled, axis_size=histogram_axis_size)
        )
        neutral_histograms.append(
            canonical_rgb_histogram(neutral, axis_size=histogram_axis_size)
        )
    if first_rows is None:
        raise ValueError("at least one style is required")
    return (
        np.stack(styled_histograms),
        np.stack(neutral_histograms),
        first_rows,
    )


def _features(styled: np.ndarray, neutral: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            content_residual_features(left, right)
            for left, right in zip(styled, neutral, strict=True)
        ]
    )


def _project_candidate(
    method: dict[str, Any],
    *,
    styled: np.ndarray,
    neutral: np.ndarray,
    histogram_axis_size: int,
    velocity_grid_axis_size: int,
    coefficient_vector_norm_cap: float,
    global_mean: np.ndarray,
    ridge_models: dict[str, Any],
) -> np.ndarray:
    kind = str(method["kind"])
    if kind == "content_confounded_control":
        return np.stack(
            [
                histogram_kde_velocity_grid(
                    histogram,
                    histogram_axis_size=histogram_axis_size,
                    bandwidth=float(method["bandwidth"]),
                    velocity_grid_axis_size=velocity_grid_axis_size,
                    coefficient_vector_norm_cap=coefficient_vector_norm_cap,
                )
                for histogram in styled
            ]
        )
    if kind == "nonlearned_distribution_ratio":
        return np.stack(
            [
                density_ratio_velocity_grid(
                    left,
                    right,
                    histogram_axis_size=histogram_axis_size,
                    bandwidth=float(method["bandwidth"]),
                    velocity_grid_axis_size=velocity_grid_axis_size,
                    score_difference_scale=float(
                        method["score_difference_scale"]
                    ),
                    coefficient_vector_norm_cap=coefficient_vector_norm_cap,
                )
                for left, right in zip(styled, neutral, strict=True)
            ]
        )
    if kind == "bounded_parameter_ml":
        return ridge_models[str(method["id"])].predict(
            _features(styled, neutral)
        )
    if kind == "style_averaging_control":
        return np.repeat(global_mean[None, ...], len(styled), axis=0)
    raise ValueError(f"unsupported S2 method kind: {kind}")


def _same_style_metrics(
    first: np.ndarray,
    second: np.ndarray,
    *,
    points: np.ndarray,
    integration_steps: int,
) -> dict[str, float]:
    grid_rmse = np.sqrt(np.mean((first - second) ** 2, axis=(1, 2, 3, 4)))
    output_rmse = []
    for left, right in zip(first, second, strict=True):
        left_output = CubeDiffeomorphicColourFlow(
            left, integration_steps=integration_steps
        ).apply(points)
        right_output = CubeDiffeomorphicColourFlow(
            right, integration_steps=integration_steps
        ).apply(points)
        output_rmse.append(
            float(np.sqrt(np.mean((left_output - right_output) ** 2)))
        )
    pairs = np.arange(0, len(first) - 1, 2)
    between = []
    for index in pairs:
        left_output = CubeDiffeomorphicColourFlow(
            first[index], integration_steps=integration_steps
        ).apply(points)
        right_output = CubeDiffeomorphicColourFlow(
            first[index + 1], integration_steps=integration_steps
        ).apply(points)
        between.append(
            float(np.sqrt(np.mean((left_output - right_output) ** 2)))
        )
    median_same = float(np.median(output_rmse))
    median_between = float(np.median(between))
    return {
        "velocity_grid_rmse_median": float(np.median(grid_rmse)),
        "operator_output_rmse_median": median_same,
        "operator_output_rmse_p90": float(np.quantile(output_rmse, 0.9)),
        "between_style_operator_separation_median": median_between,
        "same_to_between_operator_ratio": median_same
        / max(median_between, 1e-15),
    }


def _identity_negative_rmse(
    grids: np.ndarray, *, points: np.ndarray, integration_steps: int
) -> float:
    values = []
    for grid in grids:
        output = CubeDiffeomorphicColourFlow(
            grid, integration_steps=integration_steps
        ).apply(points)
        values.append(float(np.sqrt(np.mean((output - points) ** 2))))
    return float(np.median(values))


def run_development(
    config: dict[str, Any],
    *,
    config_sha256: str,
    software_commit: str,
) -> dict[str, Any]:
    style_config = config["style_generator"]
    content_config = config["content_generator"]
    observations = config["observations"]
    operator_config = config["operator"]
    reserved = int(style_config["reserved_confirmation_style_seed"])
    used_seeds = {
        int(style_config["training_style_seed"]),
        int(style_config["development_style_seed"]),
        int(content_config["training_observation_seed"]),
        int(content_config["development_observation_seed_a"]),
        int(content_config["development_observation_seed_b"]),
    }
    if reserved in used_seeds:
        raise ValueError("reserved confirmation seed overlaps development")
    histogram_axis_size = int(observations["histogram_axis_size"])
    grid_axis_size = int(operator_config["velocity_grid_axis_size"])
    cap = float(operator_config["coefficient_vector_norm_cap"])
    integration_steps = int(operator_config["integration_steps"])
    style_kwargs = _generator_kwargs(style_config)
    _, train_grids = _make_styles(
        count=int(style_config["training_style_count"]),
        seed=int(style_config["training_style_seed"]),
        generator_kwargs=style_kwargs,
        velocity_grid_axis_size=grid_axis_size,
        coefficient_vector_norm_cap=cap,
    )
    dev_palettes, dev_grids = _make_styles(
        count=int(style_config["development_style_count"]),
        seed=int(style_config["development_style_seed"]),
        generator_kwargs=style_kwargs,
        velocity_grid_axis_size=grid_axis_size,
        coefficient_vector_norm_cap=cap,
    )
    train_styled, train_neutral, _ = _make_observations(
        train_grids,
        seed=int(content_config["training_observation_seed"]),
        content_config=content_config,
        histogram_axis_size=histogram_axis_size,
        integration_steps=integration_steps,
    )
    dev_styled_a, dev_neutral_a, first_rows = _make_observations(
        dev_grids,
        seed=int(content_config["development_observation_seed_a"]),
        content_config=content_config,
        histogram_axis_size=histogram_axis_size,
        integration_steps=integration_steps,
    )
    dev_styled_b, dev_neutral_b, _ = _make_observations(
        dev_grids,
        seed=int(content_config["development_observation_seed_b"]),
        content_config=content_config,
        histogram_axis_size=histogram_axis_size,
        integration_steps=integration_steps,
    )
    train_features = _features(train_styled, train_neutral)
    ridge_models = {}
    for method in config["candidate_methods"]:
        if method["kind"] == "bounded_parameter_ml":
            ridge_models[str(method["id"])] = fit_bounded_multi_output_ridge(
                train_features,
                train_grids,
                alpha=float(method["alpha"]),
                maximum_vector_norm=cap,
            )
    global_mean = np.mean(train_grids, axis=0)
    points = _grid(int(operator_config["evaluation_grid_axis_size"]))

    shuffled_styled = first_rows[0][
        np.random.default_rng(99201).permutation(len(first_rows[0]))
    ]
    shuffled_neutral = first_rows[1][
        np.random.default_rng(99202).permutation(len(first_rows[1]))
    ]
    perm_styled = canonical_rgb_histogram(
        shuffled_styled, axis_size=histogram_axis_size
    )
    perm_neutral = canonical_rgb_histogram(
        shuffled_neutral, axis_size=histogram_axis_size
    )
    histogram_permutation_error = float(
        max(
            np.max(np.abs(perm_styled - dev_styled_a[0])),
            np.max(np.abs(perm_neutral - dev_neutral_a[0])),
        )
    )

    methods: dict[str, Any] = {}
    predictions_a: dict[str, np.ndarray] = {}
    for method in config["candidate_methods"]:
        identifier = str(method["id"])
        grids_a = _project_candidate(
            method,
            styled=dev_styled_a,
            neutral=dev_neutral_a,
            histogram_axis_size=histogram_axis_size,
            velocity_grid_axis_size=grid_axis_size,
            coefficient_vector_norm_cap=cap,
            global_mean=global_mean,
            ridge_models=ridge_models,
        )
        grids_b = _project_candidate(
            method,
            styled=dev_styled_b,
            neutral=dev_neutral_b,
            histogram_axis_size=histogram_axis_size,
            velocity_grid_axis_size=grid_axis_size,
            coefficient_vector_norm_cap=cap,
            global_mean=global_mean,
            ridge_models=ridge_models,
        )
        perm_grid = _project_candidate(
            method,
            styled=perm_styled[None, :],
            neutral=perm_neutral[None, :],
            histogram_axis_size=histogram_axis_size,
            velocity_grid_axis_size=grid_axis_size,
            coefficient_vector_norm_cap=cap,
            global_mean=global_mean,
            ridge_models=ridge_models,
        )[0]
        predictions_a[identifier] = grids_a
        evaluation = _evaluate_method(
            grids=grids_a,
            oracle_grids=dev_grids,
            palettes=dev_palettes,
            points=points,
            integration_steps=integration_steps,
            permutation_grid=perm_grid,
        )
        replicate = _same_style_metrics(
            grids_a,
            grids_b,
            points=points,
            integration_steps=integration_steps,
        )
        identity_grids = _project_candidate(
            method,
            styled=dev_neutral_a[:24],
            neutral=dev_neutral_a[:24],
            histogram_axis_size=histogram_axis_size,
            velocity_grid_axis_size=grid_axis_size,
            coefficient_vector_norm_cap=cap,
            global_mean=global_mean,
            ridge_models=ridge_models,
        )
        content_stable = (
            replicate["operator_output_rmse_median"]
            < replicate["between_style_operator_separation_median"]
        )
        evaluation["content_replicate"] = replicate
        evaluation["identity_style_negative_output_rmse_median"] = (
            _identity_negative_rmse(
                identity_grids,
                points=points,
                integration_steps=integration_steps,
            )
        )
        evaluation["content_stable"] = bool(content_stable)
        evaluation["eligible_with_content"] = bool(
            evaluation["eligible"] and content_stable
        )
        methods[identifier] = {
            "specification": method,
            "metrics": evaluation,
        }

    raw_median = methods["raw_styled_kde_0p12"]["metrics"][
        "operator_output_rmse"
    ]["median"]
    global_median = methods["global_mean_velocity"]["metrics"][
        "operator_output_rmse"
    ]["median"]
    for report in methods.values():
        median = report["metrics"]["operator_output_rmse"]["median"]
        report["metrics"]["median_error_improvement_over_raw_kde_fraction"] = (
            1.0 - median / raw_median
        )
        report["metrics"]["median_error_improvement_over_global_fraction"] = (
            1.0 - median / global_median
        )

    eligible = [
        name
        for name, report in methods.items()
        if report["metrics"]["eligible_with_content"]
    ]
    simple_order = config["selection_rule"]["simpler_method_order"]

    def simplicity(name: str) -> int:
        kind = str(methods[name]["specification"]["kind"])
        family = {
            "nonlearned_distribution_ratio": "nonlearned_distribution_ratio",
            "bounded_parameter_ml": "ridge",
            "content_confounded_control": "raw_kde",
            "style_averaging_control": "global_mean",
        }[kind]
        return simple_order.index(family)

    ranked = sorted(
        eligible,
        key=lambda name: (
            methods[name]["metrics"]["operator_output_rmse"]["median"],
            methods[name]["metrics"]["content_replicate"][
                "operator_output_rmse_median"
            ],
            methods[name]["metrics"]["operator_output_rmse"]["p90"],
            -methods[name]["metrics"]["reference_separation_ratio"]["median"],
            simplicity(name),
        ),
    )
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "config_sha256": config_sha256,
        "software_commit": software_commit,
        "used_seeds": sorted(used_seeds),
        "reserved_confirmation_seed_accessed": False,
        "training_style_count": len(train_grids),
        "development_style_count": len(dev_grids),
        "histogram_permutation_error": histogram_permutation_error,
        "methods": methods,
        "eligible_methods": eligible,
        "development_ranking": ranked,
        "selected_development_primary": ranked[0] if ranked else None,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "u5_r2s2_content_palette_nuisance_development_v1.json",
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
    report = run_development(
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
