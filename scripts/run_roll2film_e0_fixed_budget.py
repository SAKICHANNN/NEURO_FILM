"""Run fixed-budget Roll2Film E0 affine falsification controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.evaluation import (  # noqa: E402
    classify_fixed_budget_e0,
    paired_improvement_summary,
    recovery_metrics,
)
from src.roll2film.identification import estimate_gaussian_transport_operator  # noqa: E402
from src.roll2film.simulator import (  # noqa: E402
    PseudoRollConfig,
    alternate_truth_operator,
    default_truth_operator,
    mix_target_frames,
    partition_pixels,
    sample_neutral_prior,
    scanner_truth_operator,
    shuffle_frame_boundaries,
    simulate_pseudo_roll,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "roll2film_e0_fixed_budget.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "e0_fixed_budget" / "report.json",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    standard_error = float(array.std(ddof=1) / np.sqrt(len(array)))
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "ci95_low": float(array.mean() - 1.96 * standard_error),
        "ci95_high": float(array.mean() + 1.96 * standard_error),
    }


def _parameter_max_abs(first: Any, second: Any) -> float:
    return float(
        max(
            np.max(np.abs(first.matrix - second.matrix)),
            np.max(np.abs(first.bias - second.bias)),
        )
    )


def run(config: dict[str, Any]) -> dict[str, Any]:
    sizes = [int(value) for value in config["group_sizes"]]
    total_pixels = int(config["total_target_pixels"])
    if any(total_pixels % size for size in sizes):
        raise ValueError("every group size must divide total_target_pixels")
    gate_size = int(config["gate"]["gate_group_size"])
    if gate_size not in sizes:
        raise ValueError("gate_group_size must be listed in group_sizes")

    truth = default_truth_operator()
    alternate = alternate_truth_operator()
    scanner = scanner_truth_operator()
    composite = truth.then(scanner)
    arms = (
        "independent",
        "repeated_support",
        "mixed_operator",
        "nuisance_correct_boundaries",
        "nuisance_random_boundaries",
    )
    errors: dict[str, dict[int, list[float]]] = {
        arm: {size: [] for size in sizes} for arm in arms
    }
    partition_parameter_max_abs: list[float] = []
    prior_nominal_errors: list[float] = []
    prior_swapped_errors: list[float] = []
    scanner_vs_film_errors: list[float] = []
    scanner_vs_composite_errors: list[float] = []

    nuisance = config["nuisance"]
    offset = np.asarray(config["prior_swap_offset"], dtype=np.float64)
    for replicate in range(int(config["replicates"])):
        seed = int(config["seed"]) + replicate * 100_000
        neutral = sample_neutral_prior(int(config["neutral_prior_pixels"]), seed + 1)
        holdout = sample_neutral_prior(int(config["holdout_pixels"]), seed + 2)

        fixed_pool_roll = simulate_pseudo_roll(
            PseudoRollConfig(frames=1, pixels_per_frame=total_pixels, seed=seed + 3),
            truth,
        )
        reference_estimate = None
        for size in sizes:
            partitioned = partition_pixels(fixed_pool_roll.target_pixels, size)
            estimate = estimate_gaussian_transport_operator(neutral, partitioned)
            if reference_estimate is None:
                reference_estimate = estimate
            partition_parameter_max_abs.append(_parameter_max_abs(reference_estimate, estimate))

            pixels_per_frame = total_pixels // size
            independent_roll = simulate_pseudo_roll(
                PseudoRollConfig(frames=size, pixels_per_frame=pixels_per_frame, seed=seed + size * 10),
                truth,
            )
            alternate_roll = simulate_pseudo_roll(
                PseudoRollConfig(
                    frames=size,
                    pixels_per_frame=pixels_per_frame,
                    seed=seed + size * 10 + 1,
                ),
                alternate,
            )
            repeated_frames = tuple(independent_roll.target_frames[0] for _ in range(size))
            candidate_frames = {
                "independent": independent_roll.target_frames,
                "repeated_support": repeated_frames,
                "mixed_operator": mix_target_frames(independent_roll, alternate_roll),
            }
            for arm, frames in candidate_frames.items():
                estimate = estimate_gaussian_transport_operator(neutral, frames)
                errors[arm][size].append(
                    recovery_metrics(estimate, truth, holdout).holdout_rgb_rmse
                )

            nuisance_roll = simulate_pseudo_roll(
                PseudoRollConfig(
                    frames=size,
                    pixels_per_frame=pixels_per_frame,
                    exposure_sigma=float(nuisance["exposure_sigma"]),
                    scene_mean_sigma=float(nuisance["scene_mean_sigma"]),
                    sensor_noise_sigma=float(nuisance["sensor_noise_sigma"]),
                    seed=seed + size * 10 + 2,
                ),
                truth,
            )
            nuisance_frames = {
                "nuisance_correct_boundaries": nuisance_roll.target_frames,
                "nuisance_random_boundaries": shuffle_frame_boundaries(
                    nuisance_roll.target_frames,
                    seed=seed + size * 10 + 3,
                ),
            }
            for arm, frames in nuisance_frames.items():
                estimate = estimate_gaussian_transport_operator(
                    neutral,
                    frames,
                    normalize_frame_exposure=True,
                )
                errors[arm][size].append(
                    recovery_metrics(estimate, truth, holdout).holdout_rgb_rmse
                )

        gate_roll = simulate_pseudo_roll(
            PseudoRollConfig(
                frames=gate_size,
                pixels_per_frame=total_pixels // gate_size,
                seed=seed + 90_000,
            ),
            truth,
        )
        nominal = estimate_gaussian_transport_operator(neutral, gate_roll.target_frames)
        swapped = estimate_gaussian_transport_operator(neutral + offset, gate_roll.target_frames)
        prior_nominal_errors.append(recovery_metrics(nominal, truth, holdout).holdout_rgb_rmse)
        prior_swapped_errors.append(recovery_metrics(swapped, truth, holdout).holdout_rgb_rmse)

        scanner_roll = simulate_pseudo_roll(
            PseudoRollConfig(
                frames=gate_size,
                pixels_per_frame=total_pixels // gate_size,
                seed=seed + 91_000,
            ),
            composite,
        )
        scanner_estimate = estimate_gaussian_transport_operator(neutral, scanner_roll.target_frames)
        scanner_vs_film_errors.append(
            recovery_metrics(scanner_estimate, truth, holdout).holdout_rgb_rmse
        )
        scanner_vs_composite_errors.append(
            recovery_metrics(scanner_estimate, composite, holdout).holdout_rgb_rmse
        )

    summaries = {
        arm: {str(size): _summary(values) for size, values in by_size.items()}
        for arm, by_size in errors.items()
    }
    bootstrap_resamples = int(config["bootstrap_resamples"])
    nuisance_boundary = paired_improvement_summary(
        errors["nuisance_random_boundaries"][gate_size],
        errors["nuisance_correct_boundaries"][gate_size],
        seed=int(config["seed"]) + 700_001,
        bootstrap_resamples=bootstrap_resamples,
    )
    independent_support = paired_improvement_summary(
        errors["repeated_support"][gate_size],
        errors["independent"][gate_size],
        seed=int(config["seed"]) + 700_002,
        bootstrap_resamples=bootstrap_resamples,
    )
    mixed_operator = paired_improvement_summary(
        errors["mixed_operator"][gate_size],
        errors["independent"][gate_size],
        seed=int(config["seed"]) + 700_003,
        bootstrap_resamples=bootstrap_resamples,
    )
    prior_swap = paired_improvement_summary(
        prior_swapped_errors,
        prior_nominal_errors,
        seed=int(config["seed"]) + 700_004,
        bootstrap_resamples=bootstrap_resamples,
    )
    scanner_confound = paired_improvement_summary(
        scanner_vs_film_errors,
        scanner_vs_composite_errors,
        seed=int(config["seed"]) + 700_005,
        bootstrap_resamples=bootstrap_resamples,
    )
    max_partition_error = max(partition_parameter_max_abs)
    gate = classify_fixed_budget_e0(
        partition_parameter_max_abs=max_partition_error,
        nuisance_boundary=nuisance_boundary,
        independent_support=independent_support,
        mixed_operator=mixed_operator,
        partition_tolerance=float(config["gate"]["partition_parameter_max_abs"]),
        relative_improvement_min=float(config["gate"]["paired_relative_improvement_min"]),
    )
    return {
        "fixed_budget": {
            "total_target_pixels": total_pixels,
            "group_sizes": sizes,
            "pixels_per_frame": {str(size): total_pixels // size for size in sizes},
        },
        "summaries": summaries,
        "paired_controls_at_gate_size": {
            "gate_group_size": gate_size,
            "nuisance_boundary": nuisance_boundary,
            "independent_support": independent_support,
            "mixed_operator": mixed_operator,
        },
        "partition_equivalence": {
            "parameter_max_abs": max_partition_error,
        },
        "diagnostics_not_used_for_pass": {
            "prior_swap_nominal_over_shifted_improvement": prior_swap,
            "scanner_composite_over_film_truth_improvement": scanner_confound,
            "interpretation": (
                "A positive prior-swap value shows source-prior sensitivity. A positive scanner value "
                "shows the estimator recovers the composite look better than the film component, "
                "which is expected non-identifiability rather than success."
            ),
        },
        "decision": gate,
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
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "operator_truth": default_truth_operator().to_dict(),
        "scanner_truth": scanner_truth_operator().to_dict(),
        "result": result,
        "claim_boundary": config["claim_boundary"],
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "method_control_decision": result["decision"]["method_control_decision"],
                "roll_information_decision": result["decision"]["roll_information_decision"],
            },
            indent=2,
        )
    )
    return 0 if result["decision"]["method_control_decision"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
