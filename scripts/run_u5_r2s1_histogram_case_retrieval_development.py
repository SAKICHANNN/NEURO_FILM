#!/usr/bin/env python
"""Run frozen U5.R2S1D synthetic canonical-histogram development."""

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

from src.roll2film.cube_diffeomorphic_flow import (  # noqa: E402
    CubeDiffeomorphicColourFlow,
    finite_difference_jacobians,
)
from src.roll2film.histogram_case_retrieval import (  # noqa: E402
    HistogramCaseBank,
    canonical_rgb_histogram,
    generate_synthetic_palette,
    histogram_kde_velocity_grid,
    sample_palette,
)
from src.roll2film.palette_score_flow import (  # noqa: E402
    DiagonalGaussianMixturePalette,
    palette_score_velocity_grid,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _grid(axis_size: int) -> np.ndarray:
    axis = (np.arange(axis_size, dtype=np.float64) + 0.5) / axis_size
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def _best_affine_residual(source: np.ndarray, target: np.ndarray) -> float:
    design = np.column_stack((np.ones(len(source)), source))
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return _rmse(design @ coefficients, target)


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _generator_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    payload = config["synthetic_palette_generator"]
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


def _make_population(
    *,
    count: int,
    seed: int,
    sample_count: int,
    histogram_axis_size: int,
    velocity_grid_axis_size: int,
    coefficient_vector_norm_cap: float,
    generator_kwargs: dict[str, Any],
) -> tuple[
    list[DiagonalGaussianMixturePalette],
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
]:
    rng = np.random.default_rng(seed)
    palettes = []
    histograms = []
    grids = []
    samples = []
    for _ in range(count):
        palette = generate_synthetic_palette(rng, **generator_kwargs)
        observed = sample_palette(
            palette, sample_count=sample_count, rng=rng
        )
        palettes.append(palette)
        samples.append(observed)
        histograms.append(
            canonical_rgb_histogram(
                observed, axis_size=histogram_axis_size
            )
        )
        grids.append(
            palette_score_velocity_grid(
                palette,
                axis_size=velocity_grid_axis_size,
                coefficient_vector_norm_cap=coefficient_vector_norm_cap,
            )
        )
    return palettes, np.stack(histograms), np.stack(grids), samples


def _predict_grids(
    *,
    method: dict[str, Any],
    bank: HistogramCaseBank,
    query_histograms: np.ndarray,
    histogram_axis_size: int,
    velocity_grid_axis_size: int,
    coefficient_vector_norm_cap: float,
) -> np.ndarray:
    identifier = str(method["id"])
    if identifier == "global_mean_velocity":
        mean = np.mean(bank.velocity_grids, axis=0)
        return np.repeat(mean[None, ...], len(query_histograms), axis=0)
    if method["kind"] == "nonparametric_ml":
        grids = [
            bank.hard_retrieve(
                histogram, distance=str(method["distance"])
            )[1]
            for histogram in query_histograms
        ]
        return np.stack(grids)
    if method["kind"] == "sparse_blend_control":
        grids = [
            bank.inverse_distance_blend(
                histogram,
                distance=str(method["distance"]),
                neighbors=int(method["neighbors"]),
                epsilon=float(method["distance_epsilon"]),
            )[1]
            for histogram in query_histograms
        ]
        return np.stack(grids)
    if method["kind"] == "nonlearned_query_density_control":
        grids = [
            histogram_kde_velocity_grid(
                histogram,
                histogram_axis_size=histogram_axis_size,
                bandwidth=float(method["bandwidth"]),
                velocity_grid_axis_size=velocity_grid_axis_size,
                coefficient_vector_norm_cap=coefficient_vector_norm_cap,
            )
            for histogram in query_histograms
        ]
        return np.stack(grids)
    raise ValueError(f"unsupported fixed method: {identifier}")


def _evaluate_method(
    *,
    grids: np.ndarray,
    oracle_grids: np.ndarray,
    palettes: list[DiagonalGaussianMixturePalette],
    points: np.ndarray,
    integration_steps: int,
    permutation_grid: np.ndarray,
) -> dict[str, Any]:
    predicted_outputs = []
    oracle_outputs = []
    velocity_rmse = []
    velocity_cosine = []
    style_ratio = []
    non_affine_ratio = []
    attraction = []
    for grid, oracle_grid, palette in zip(
        grids, oracle_grids, palettes, strict=True
    ):
        operator = CubeDiffeomorphicColourFlow(
            grid, integration_steps=integration_steps
        )
        oracle = CubeDiffeomorphicColourFlow(
            oracle_grid, integration_steps=integration_steps
        )
        output = operator.apply(points)
        oracle_output = oracle.apply(points)
        predicted_outputs.append(output)
        oracle_outputs.append(oracle_output)
        velocity_rmse.append(_rmse(grid, oracle_grid))
        flat = grid.reshape(-1)
        oracle_flat = oracle_grid.reshape(-1)
        velocity_cosine.append(
            float(
                np.dot(flat, oracle_flat)
                / max(
                    np.linalg.norm(flat) * np.linalg.norm(oracle_flat),
                    1e-15,
                )
            )
        )
        style_ratio.append(
            _rmse(output, points) / max(_rmse(oracle_output, points), 1e-15)
        )
        non_affine_ratio.append(
            _best_affine_residual(points, output)
            / max(_best_affine_residual(points, oracle_output), 1e-15)
        )
        attraction.append(
            float(
                np.mean(
                    palette.log_density(output)
                    - palette.log_density(points)
                )
            )
        )
    predicted = np.stack(predicted_outputs)
    oracle = np.stack(oracle_outputs)
    operator_rmse = np.sqrt(np.mean((predicted - oracle) ** 2, axis=(1, 2)))

    pair_indices = np.arange(0, len(grids) - 1, 2)
    predicted_separation = np.sqrt(
        np.mean(
            (predicted[pair_indices] - predicted[pair_indices + 1]) ** 2,
            axis=(1, 2),
        )
    )
    oracle_separation = np.sqrt(
        np.mean(
            (oracle[pair_indices] - oracle[pair_indices + 1]) ** 2,
            axis=(1, 2),
        )
    )
    separation_ratio = predicted_separation / np.maximum(
        oracle_separation, 1e-15
    )

    structural_indices = np.linspace(
        0, len(grids) - 1, num=min(12, len(grids)), dtype=np.int64
    )
    jacobian_axis = np.linspace(0.08, 0.92, 5, dtype=np.float64)
    jacobian_points = np.stack(
        np.meshgrid(
            jacobian_axis, jacobian_axis, jacobian_axis, indexing="ij"
        ),
        axis=-1,
    ).reshape(-1, 3)
    minimum_output = 1.0
    maximum_output = 0.0
    minimum_determinant = np.inf
    maximum_norm = 0.0
    maximum_inverse = 0.0
    maximum_replay = 0.0
    for index in structural_indices:
        operator = CubeDiffeomorphicColourFlow(
            grids[index], integration_steps=integration_steps
        )
        output = operator.apply(jacobian_points)
        jacobians = finite_difference_jacobians(
            operator, jacobian_points, step=1e-6
        )
        minimum_output = min(minimum_output, float(np.min(output)))
        maximum_output = max(maximum_output, float(np.max(output)))
        minimum_determinant = min(
            minimum_determinant, float(np.min(np.linalg.det(jacobians)))
        )
        maximum_norm = max(
            maximum_norm,
            float(np.max(np.linalg.svd(jacobians, compute_uv=False)[:, 0])),
        )
        maximum_inverse = max(
            maximum_inverse,
            float(
                np.max(
                    np.abs(operator.inverse(output) - jacobian_points)
                )
            ),
        )
        replay = CubeDiffeomorphicColourFlow.from_dict(
            json.loads(json.dumps(operator.to_dict(), sort_keys=True))
        )
        maximum_replay = max(
            maximum_replay,
            float(np.max(np.abs(replay.apply(points) - operator.apply(points)))),
        )
    maximum_coefficient_norm = float(
        np.max(np.linalg.norm(grids, axis=-1))
    )
    permutation_error = float(np.max(np.abs(grids[0] - permutation_grid)))
    eligible = bool(
        permutation_error == 0.0
        and maximum_coefficient_norm <= 2.0
        and minimum_output >= 0.0
        and maximum_output <= 1.0
        and minimum_determinant > 0.0
        and maximum_replay == 0.0
    )
    return {
        "velocity_grid_rmse": _summary(np.asarray(velocity_rmse)),
        "operator_output_rmse": _summary(operator_rmse),
        "velocity_direction_cosine": _summary(
            np.asarray(velocity_cosine)
        ),
        "identity_rmse_retention_ratio": _summary(np.asarray(style_ratio)),
        "best_affine_residual_retention_ratio": _summary(
            np.asarray(non_affine_ratio)
        ),
        "query_palette_log_density_gain": _summary(np.asarray(attraction)),
        "reference_separation_ratio": _summary(separation_ratio),
        "structure": {
            "sampled_operator_count": int(len(structural_indices)),
            "minimum_output": minimum_output,
            "maximum_output": maximum_output,
            "minimum_jacobian_determinant": minimum_determinant,
            "maximum_jacobian_spectral_norm": maximum_norm,
            "maximum_inverse_error": maximum_inverse,
            "maximum_replay_error": maximum_replay,
            "maximum_coefficient_vector_norm": maximum_coefficient_norm,
            "permutation_operator_error": permutation_error,
        },
        "eligible": eligible,
    }


def run_development(
    config: dict[str, Any],
    *,
    config_sha256: str,
    software_commit: str,
) -> dict[str, Any]:
    generator = config["synthetic_palette_generator"]
    observations = config["observations"]
    operator_config = config["operator"]
    reserved_seed = int(generator["reserved_untouched_confirmation_seed"])
    used_seeds = {
        int(generator["case_bank_seed"]),
        int(generator["development_seed"]),
    }
    if reserved_seed in used_seeds:
        raise ValueError("reserved confirmation seed overlaps development")

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
        count=int(generator["development_query_count"]),
        seed=int(generator["development_seed"]),
        **common,
    )
    bank = HistogramCaseBank(bank_histograms, bank_grids)
    points = _grid(int(operator_config["confirmation_grid_axis_size"]))
    shuffled = query_samples[0][
        np.random.default_rng(99173).permutation(len(query_samples[0]))
    ]
    permuted_histogram = canonical_rgb_histogram(
        shuffled, axis_size=int(observations["histogram_axis_size"])
    )
    histogram_permutation_error = float(
        np.max(np.abs(permuted_histogram - query_histograms[0]))
    )

    methods: dict[str, Any] = {}
    predicted: dict[str, np.ndarray] = {}
    for method in config["candidate_methods"]:
        identifier = str(method["id"])
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
        permuted_grid = _predict_grids(
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
        predicted[identifier] = grids
        methods[identifier] = {
            "specification": method,
            "metrics": _evaluate_method(
                grids=grids,
                oracle_grids=oracle_grids,
                palettes=palettes,
                points=points,
                integration_steps=int(operator_config["integration_steps"]),
                permutation_grid=permuted_grid,
            ),
        }

    eligible = [
        identifier
        for identifier, report in methods.items()
        if bool(report["metrics"]["eligible"])
    ]
    simple_order = config["development_selection_rule"][
        "simpler_method_order"
    ]

    def simplicity(identifier: str) -> int:
        kind = str(methods[identifier]["specification"]["kind"])
        family = (
            "query_kde"
            if kind == "nonlearned_query_density_control"
            else "hard_1nn"
            if kind == "nonparametric_ml"
            else "top3_inverse_blend"
            if kind == "sparse_blend_control"
            else "global_mean"
        )
        return (
            simple_order.index(family)
            if family in simple_order
            else len(simple_order)
        )

    ranked = sorted(
        eligible,
        key=lambda identifier: (
            methods[identifier]["metrics"]["operator_output_rmse"]["median"],
            methods[identifier]["metrics"]["operator_output_rmse"]["p90"],
            -methods[identifier]["metrics"][
                "best_affine_residual_retention_ratio"
            ]["median"],
            simplicity(identifier),
        ),
    )
    primary = ranked[0] if ranked else None
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "config_sha256": config_sha256,
        "software_commit": software_commit,
        "used_seeds": sorted(used_seeds),
        "reserved_confirmation_seed_accessed": False,
        "histogram_permutation_error": histogram_permutation_error,
        "case_bank_count": len(bank_histograms),
        "development_query_count": len(query_histograms),
        "methods": methods,
        "eligible_methods": eligible,
        "development_ranking": ranked,
        "selected_development_primary": primary,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "u5_r2s1_histogram_case_retrieval_development_v1.json",
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
