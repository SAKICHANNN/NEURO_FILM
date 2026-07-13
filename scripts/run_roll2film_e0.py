"""Run the data-independent Roll2Film known-operator identifiability gate."""

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

from src.roll2film.evaluation import classify_e0, recovery_metrics  # noqa: E402
from src.roll2film.identification import estimate_gaussian_transport_operator  # noqa: E402
from src.roll2film.simulator import (  # noqa: E402
    PseudoRollConfig,
    alternate_truth_operator,
    default_truth_operator,
    mix_target_frames,
    sample_neutral_prior,
    simulate_pseudo_roll,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "roll2film_e0.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "roll2film" / "e0" / "report.json")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    standard_error = float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "ci95_low": float(array.mean() - 1.96 * standard_error),
        "ci95_high": float(array.mean() + 1.96 * standard_error),
    }


def run_condition(config: dict[str, Any], condition: dict[str, Any], base_seed: int) -> dict[str, Any]:
    truth = default_truth_operator()
    alternate = alternate_truth_operator()
    sizes = [int(value) for value in config["group_sizes"]]
    metric_names = ("matrix_rmse", "bias_rmse", "holdout_rgb_rmse", "style_ratio")
    records: dict[str, dict[int, dict[str, list[float]]]] = {
        "correct": {size: {name: [] for name in metric_names} for size in sizes},
        "shuffled": {size: {name: [] for name in metric_names} for size in sizes},
    }
    for replicate in range(int(config["replicates"])):
        neutral = sample_neutral_prior(int(config["neutral_prior_pixels"]), base_seed + 100_000 + replicate)
        holdout = sample_neutral_prior(int(config["holdout_pixels"]), base_seed + 200_000 + replicate)
        for size in sizes:
            common = dict(
                frames=size,
                pixels_per_frame=int(config["pixels_per_frame"]),
                exposure_sigma=float(condition["exposure_sigma"]),
                scene_mean_sigma=float(condition["scene_mean_sigma"]),
                sensor_noise_sigma=float(condition["sensor_noise_sigma"]),
            )
            roll_seed = base_seed + replicate * 10_000 + size * 10
            correct_roll = simulate_pseudo_roll(PseudoRollConfig(seed=roll_seed, **common), truth)
            alternate_roll = simulate_pseudo_roll(PseudoRollConfig(seed=roll_seed + 1, **common), alternate)
            candidate_frames = {
                "correct": correct_roll.target_frames,
                "shuffled": mix_target_frames(correct_roll, alternate_roll),
            }
            for label, frames in candidate_frames.items():
                estimate = estimate_gaussian_transport_operator(
                    neutral,
                    frames,
                    normalize_frame_exposure=bool(condition["normalize_frame_exposure"]),
                )
                metrics = recovery_metrics(estimate, truth, holdout).to_dict()
                for name, value in metrics.items():
                    records[label][size][name].append(value)
    summaries: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for label in records:
        summaries[label] = {
            str(size): {name: _summary(values) for name, values in records[label][size].items()}
            for size in sizes
        }
    correct_gate_values = {
        size: summaries["correct"][str(size)]["holdout_rgb_rmse"]["mean"] for size in sizes
    }
    shuffled_gate_values = {
        size: summaries["shuffled"][str(size)]["holdout_rgb_rmse"]["mean"] for size in sizes
    }
    gate = config["gate"]
    return {
        "condition": condition,
        "summaries": summaries,
        "diagnostic_gate": classify_e0(
            correct_gate_values,
            shuffled_gate_values,
            endpoint_relative_gain_min=float(gate["endpoint_relative_gain_min"]),
            correct_vs_shuffled_final_gap_min=float(gate["correct_vs_shuffled_final_gap_min"]),
        ),
    }


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    conditions = {
        name: run_condition(config, condition, int(config["seed"]) + index * 1_000_000)
        for index, (name, condition) in enumerate(config["conditions"].items())
    }
    primary_name = str(config["gate"]["primary_condition"])
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
        "conditions": conditions,
        "primary_decision": conditions[primary_name]["diagnostic_gate"],
        "claim_boundary": config["claim_boundary"],
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "primary_decision": report["primary_decision"]}, indent=2))
    return 0 if report["primary_decision"]["decision"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
