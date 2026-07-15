"""Run fixed-budget Roll2Film L2 nonlinear falsification controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.evaluation import (  # noqa: E402
    classify_fixed_budget_l2,
    paired_improvement_summary,
)
from src.roll2film.identification import estimate_affine_spline_transport_operator  # noqa: E402
from src.roll2film.lut import bake_dense_lut  # noqa: E402
from src.roll2film.simulator import (  # noqa: E402
    PseudoRollConfig,
    alternate_l2_truth_operator,
    default_l2_truth_operator,
    mix_target_frames,
    partition_pixels,
    sample_neutral_prior,
    scanner_truth_operator,
    shuffle_frame_boundaries,
    simulate_pseudo_roll,
)
from src.roll2film.splines import AffineMonotoneSplineOperator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "roll2film_e0_l2_fixed_budget.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "e0_l2_fixed_budget" / "report.json",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    standard_error = float(array.std(ddof=1) / np.sqrt(len(array)))
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "ci95_low": float(array.mean() - 1.96 * standard_error),
        "ci95_high": float(array.mean() + 1.96 * standard_error),
    }


def _rmse(estimate: Any, truth: Any, holdout: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate.apply(holdout) - truth.apply(holdout)) ** 2)))


def _grid(size: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, size)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def run(config: dict[str, Any]) -> dict[str, Any]:
    sizes = [int(value) for value in config["group_sizes"]]
    total_pixels = int(config["total_target_pixels"])
    if any(total_pixels % size for size in sizes):
        raise ValueError("every group size must divide total_target_pixels")
    gate_size = int(config["gate"]["gate_group_size"])
    if gate_size not in sizes:
        raise ValueError("gate_group_size must be listed in group_sizes")
    iterations = int(config["estimator_iterations"])
    truth = default_l2_truth_operator()
    alternate = alternate_l2_truth_operator()
    scanner = scanner_truth_operator()
    composite = AffineMonotoneSplineOperator(scanner.then(truth.affine), truth.splines)
    operator_grid = _grid(int(config["operator_grid_size"]))
    arms = (
        "independent",
        "repeated_support",
        "mixed_operator",
        "nuisance_correct_boundaries",
        "nuisance_random_boundaries",
    )
    errors = {arm: {size: [] for size in sizes} for arm in arms}
    partition_grid_max_abs: list[float] = []
    prior_nominal_errors: list[float] = []
    prior_swapped_errors: list[float] = []
    scanner_vs_film_errors: list[float] = []
    scanner_vs_composite_errors: list[float] = []
    clean_raw_errors: list[float] = []
    clean_flexible_normalization_errors: list[float] = []
    nuisance = config["nuisance"]
    prior_offset = np.asarray(config["prior_swap_offset"], dtype=np.float64)

    for replicate in range(int(config["replicates"])):
        seed = int(config["seed"]) + replicate * 100_000
        neutral = sample_neutral_prior(int(config["neutral_prior_pixels"]), seed + 1)
        holdout = sample_neutral_prior(int(config["holdout_pixels"]), seed + 2)
        fixed_roll = simulate_pseudo_roll(
            PseudoRollConfig(frames=1, pixels_per_frame=total_pixels, seed=seed + 3),
            truth,
        )
        reference_grid = None
        for size in sizes:
            partitioned = partition_pixels(fixed_roll.target_pixels, size)
            estimate = estimate_affine_spline_transport_operator(
                neutral,
                partitioned,
                iterations=iterations,
            )
            rendered_grid = estimate.apply(operator_grid)
            if reference_grid is None:
                reference_grid = rendered_grid
            partition_grid_max_abs.append(float(np.max(np.abs(reference_grid - rendered_grid))))

            pixels_per_frame = total_pixels // size
            independent = simulate_pseudo_roll(
                PseudoRollConfig(frames=size, pixels_per_frame=pixels_per_frame, seed=seed + size * 20),
                truth,
            )
            other = simulate_pseudo_roll(
                PseudoRollConfig(
                    frames=size,
                    pixels_per_frame=pixels_per_frame,
                    seed=seed + size * 20 + 1,
                ),
                alternate,
            )
            candidates = {
                "independent": independent.target_frames,
                "repeated_support": tuple(independent.target_frames[0] for _ in range(size)),
                "mixed_operator": mix_target_frames(independent, other),
            }
            for arm, frames in candidates.items():
                estimate = estimate_affine_spline_transport_operator(
                    neutral,
                    frames,
                    iterations=iterations,
                )
                errors[arm][size].append(_rmse(estimate, truth, holdout))

            nuisance_roll = simulate_pseudo_roll(
                PseudoRollConfig(
                    frames=size,
                    pixels_per_frame=pixels_per_frame,
                    exposure_sigma=float(nuisance["exposure_sigma"]),
                    white_balance_sigma=float(nuisance["white_balance_sigma"]),
                    scene_mean_sigma=float(nuisance["scene_mean_sigma"]),
                    sensor_noise_sigma=float(nuisance["sensor_noise_sigma"]),
                    seed=seed + size * 20 + 2,
                ),
                truth,
            )
            nuisance_frames = {
                "nuisance_correct_boundaries": nuisance_roll.target_frames,
                "nuisance_random_boundaries": shuffle_frame_boundaries(
                    nuisance_roll.target_frames,
                    seed=seed + size * 20 + 3,
                ),
            }
            for arm, frames in nuisance_frames.items():
                estimate = estimate_affine_spline_transport_operator(
                    neutral,
                    frames,
                    iterations=iterations,
                    normalize_frame_photometric=True,
                )
                errors[arm][size].append(_rmse(estimate, truth, holdout))

        gate_roll = simulate_pseudo_roll(
            PseudoRollConfig(
                frames=gate_size,
                pixels_per_frame=total_pixels // gate_size,
                seed=seed + 90_000,
            ),
            truth,
        )
        nominal = estimate_affine_spline_transport_operator(
            neutral,
            gate_roll.target_frames,
            iterations=iterations,
        )
        swapped = estimate_affine_spline_transport_operator(
            neutral + prior_offset,
            gate_roll.target_frames,
            iterations=iterations,
        )
        prior_nominal_errors.append(_rmse(nominal, truth, holdout))
        prior_swapped_errors.append(_rmse(swapped, truth, holdout))
        clean_raw_errors.append(_rmse(nominal, truth, holdout))
        flexible = estimate_affine_spline_transport_operator(
            neutral,
            gate_roll.target_frames,
            iterations=iterations,
            normalize_frame_photometric=True,
        )
        clean_flexible_normalization_errors.append(_rmse(flexible, truth, holdout))

        scanner_roll = simulate_pseudo_roll(
            PseudoRollConfig(
                frames=gate_size,
                pixels_per_frame=total_pixels // gate_size,
                seed=seed + 91_000,
            ),
            composite,
        )
        scanner_estimate = estimate_affine_spline_transport_operator(
            neutral,
            scanner_roll.target_frames,
            iterations=iterations,
        )
        scanner_vs_film_errors.append(_rmse(scanner_estimate, truth, holdout))
        scanner_vs_composite_errors.append(_rmse(scanner_estimate, composite, holdout))

    summaries = {
        arm: {str(size): _summary(values) for size, values in by_size.items()}
        for arm, by_size in errors.items()
    }
    bootstrap = int(config["bootstrap_resamples"])
    paired = {
        "nuisance_boundary": paired_improvement_summary(
            errors["nuisance_random_boundaries"][gate_size],
            errors["nuisance_correct_boundaries"][gate_size],
            seed=int(config["seed"]) + 700_001,
            bootstrap_resamples=bootstrap,
        ),
        "independent_support": paired_improvement_summary(
            errors["repeated_support"][gate_size],
            errors["independent"][gate_size],
            seed=int(config["seed"]) + 700_002,
            bootstrap_resamples=bootstrap,
        ),
        "mixed_operator": paired_improvement_summary(
            errors["mixed_operator"][gate_size],
            errors["independent"][gate_size],
            seed=int(config["seed"]) + 700_003,
            bootstrap_resamples=bootstrap,
        ),
    }
    diagnostics = {
        "prior_swap_nominal_over_shifted_improvement": paired_improvement_summary(
            prior_swapped_errors,
            prior_nominal_errors,
            seed=int(config["seed"]) + 700_004,
            bootstrap_resamples=bootstrap,
        ),
        "scanner_composite_over_film_truth_improvement": paired_improvement_summary(
            scanner_vs_film_errors,
            scanner_vs_composite_errors,
            seed=int(config["seed"]) + 700_005,
            bootstrap_resamples=bootstrap,
        ),
        "clean_raw_over_flexible_frame_normalization_improvement": paired_improvement_summary(
            clean_flexible_normalization_errors,
            clean_raw_errors,
            seed=int(config["seed"]) + 700_006,
            bootstrap_resamples=bootstrap,
        ),
    }
    decision = classify_fixed_budget_l2(
        partition_operator_grid_max_abs=max(partition_grid_max_abs),
        nuisance_boundary=paired["nuisance_boundary"],
        independent_support=paired["independent_support"],
        mixed_operator=paired["mixed_operator"],
        partition_tolerance=float(config["gate"]["partition_operator_grid_max_abs"]),
        relative_improvement_min=float(config["gate"]["paired_relative_improvement_min"]),
    )
    lut33 = bake_dense_lut(truth, 33)
    lut65 = bake_dense_lut(truth, 65)
    bake_probe = sample_neutral_prior(4096, int(config["seed"]) + 999_000)
    bake_probe = np.clip(bake_probe, 0.0, 1.0)
    analytic = truth.apply(bake_probe)
    return {
        "fixed_budget": {
            "total_target_pixels": total_pixels,
            "group_sizes": sizes,
            "pixels_per_frame": {str(size): total_pixels // size for size in sizes},
        },
        "summaries": summaries,
        "paired_controls_at_gate_size": {"gate_group_size": gate_size, **paired},
        "partition_equivalence": {"operator_grid_max_abs": max(partition_grid_max_abs)},
        "operator_contract": {
            "jacobian_min_on_grid": float(truth.jacobian_determinant(operator_grid).min()),
            "lut33_max_abs": float(np.max(np.abs(lut33.apply(bake_probe) - analytic))),
            "lut65_max_abs": float(np.max(np.abs(lut65.apply(bake_probe) - analytic))),
        },
        "diagnostics_not_used_for_pass": diagnostics,
        "decision": decision,
    }


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = run(config)
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "software_commit": _commit(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "operator_truth": default_l2_truth_operator().to_dict(),
        "result": result,
        "claim_boundary": config["claim_boundary"],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(
        json.dumps(
            {
                "output": str(output),
                "method_control_decision": result["decision"]["method_control_decision"],
                "roll_information_decision": result["decision"]["roll_information_decision"],
            },
            indent=2,
        )
    )
    return 0 if result["decision"]["method_control_decision"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
