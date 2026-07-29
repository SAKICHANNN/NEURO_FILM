#!/usr/bin/env python
"""Run frozen bounded Bernstein recorder proxy evaluation."""

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

from scripts.run_u5_r2an0_paired_positive_film_recovery import (  # noqa: E402
    _atomic_write,
    _canonical_json,
)
from scripts.run_u5_r2aq2_colorreference_velvia100f_held_group_baselines import (  # noqa: E402
    load_pair_table,
)
from scripts.run_u5_r2aq3i_colorreference_joint_in_domain_proxy_baselines import (  # noqa: E402
    fold_masks,
    source_colour_folds,
)
from src.roll2film.colorreference_bounded_bernstein import (  # noqa: E402
    apply_bounded_bernstein,
    fit_bounded_bernstein,
)
from src.roll2film.colorreference_proxy_baselines import (  # noqa: E402
    xyz_to_lab_d50,
)


CONFIG_SHA256 = "c8e99600ff112406a19271e502763e12cdf6733b2192e826fecf199c680d2597"
EXPERIMENT_ID = "u5.r2aq3b-colorreference-bounded-bernstein-proxy-v1"
REPORT_SCHEMA = "neuro-film.u5.r2aq3b.bounded-bernstein-proxy.v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _require_clean_tracked_worktree() -> None:
    for command in (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("AQ3B requires a clean tracked worktree")


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AQ3B config hash mismatch")
    config = json.loads(raw)
    family = config["candidate_family"]
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or family["degrees"] != [1, 2, 3]
        or family["post_fit_clipping"]
        or not config["fit_allowed"]
        or config["training_allowed"]
        or config["render_allowed"]
        or config["production_integration_allowed"]
    ):
        raise ValueError("AQ3B frozen contract mismatch")
    return config


def _load_parent_report(config: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / config["parent"]["report"]
    payload = path.read_bytes()
    if _sha256(payload) != config["parent"]["report_sha256"]:
        raise ValueError("AQ3B parent report hash mismatch")
    report = json.loads(payload)
    if (
        report["decision"]
        != "open_one_separately_frozen_bounded_nonlinear_explicit_proxy_family"
        or not report["unsafe_quadratic_mechanism_pass"]
        or report["bounded_champion"] is not None
    ):
        raise ValueError("AQ3B parent decision mismatch")
    return report


def _candidate_summary(
    folds: list[dict[str, Any]],
    model_id: str,
    *,
    baseline_key: str,
) -> dict[str, Any]:
    metrics = [fold["models"][model_id]["confirmation"] for fold in folds]
    weights = np.asarray(
        [metric["row_count"] for metric in metrics], dtype=np.float64
    )
    means = np.asarray(
        [metric["mean_deltae76"] for metric in metrics], dtype=np.float64
    )
    baseline_means = np.asarray(
        [fold["baselines"][baseline_key] for fold in folds],
        dtype=np.float64,
    )
    return {
        "fold_count": len(folds),
        "row_count": int(weights.sum()),
        "mean_deltae76": float(np.average(means, weights=weights)),
        "median_fold_mean_deltae76": float(np.median(means)),
        "worst_fold_mean_deltae76": float(np.max(means)),
        "every_fold_improves_over_nonnegative_affine": bool(
            np.all(means < baseline_means)
        ),
        "raw_xyz_negative_component_fraction": float(
            np.average(
                [
                    metric["raw_xyz_negative_component_fraction"]
                    for metric in metrics
                ],
                weights=weights,
            )
        ),
        "raw_xyz_above_1_2_component_fraction": float(
            np.average(
                [
                    metric["raw_xyz_above_1_2_component_fraction"]
                    for metric in metrics
                ],
                weights=weights,
            )
        ),
    }


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    parent = _load_parent_report(config)
    table = load_pair_table(
        ROOT / config["parent"]["pair_table"],
        expected_sha256=config["parent"]["pair_table_sha256"],
    )
    split = config["split_contract"]
    folds = source_colour_folds(
        table["source"], salt=split["source_colour_salt"]
    )
    parent_folds = {
        (
            int(row["held_target_set"]),
            int(row["held_source_colour_fold"]),
        ): row
        for row in parent["joint_folds"]
    }
    numerics = config["numerics"]
    white = np.asarray(
        config["target_space"]["reference_white_xyz"],
        dtype=np.float64,
    )
    joint = []
    for held_set in split["target_sets"]:
        for held_fold in range(split["source_colour_fold_count"]):
            development, confirmation = fold_masks(
                table,
                folds,
                held_set=held_set,
                held_fold=held_fold,
            )
            if (
                int(development.sum())
                != split["expected_development_rows_by_source_fold"][
                    held_fold
                ]
                or int(confirmation.sum())
                != split["expected_confirmation_rows_by_source_fold"][
                    held_fold
                ]
            ):
                raise RuntimeError("AQ3B fold size mismatch")
            model_rows = {}
            for degree in config["candidate_family"]["degrees"]:
                model = fit_bounded_bernstein(
                    table["source"][development],
                    table["xyz"][development],
                    degree=degree,
                    lower_bound=numerics["coefficient_lower_bound"],
                    upper_bound=numerics["coefficient_upper_bound"],
                    tolerance=numerics["lsq_linear_tolerance"],
                    maximum_iterations=numerics[
                        "lsq_linear_maximum_iterations"
                    ],
                )
                model_id = model.model_id
                model_rows[model_id] = {
                    "model": model.to_dict(),
                    "confirmation": _evaluate_bernstein(
                        model,
                        table["source"][confirmation],
                        table["xyz"][confirmation],
                        table["lab"][confirmation],
                        white,
                    ),
                }
            parent_fold = parent_folds[(held_set, held_fold)]
            joint.append(
                {
                    "held_target_set": held_set,
                    "held_source_colour_fold": held_fold,
                    "development_rows": int(development.sum()),
                    "confirmation_rows": int(confirmation.sum()),
                    "models": model_rows,
                    "baselines": {
                        "nonnegative_affine": parent_fold["models"][
                            "nonnegative_affine"
                        ]["confirmation"]["mean_deltae76"],
                        "unsafe_quadratic": parent_fold["models"][
                            "quadratic_full"
                        ]["confirmation"]["mean_deltae76"],
                    },
                }
            )
    model_ids = config["candidate_family"]["model_ids"]
    summaries = {
        model_id: _candidate_summary(
            joint,
            model_id,
            baseline_key="nonnegative_affine",
        )
        for model_id in model_ids
    }
    gates = config["model_selection"]["eligibility"]
    nonnegative_mean = config["model_selection"]["frozen_baselines"][
        "nonnegative_affine_mean_deltae76"
    ]
    unsafe_mean = config["model_selection"]["frozen_baselines"][
        "unsafe_quadratic_mean_deltae76"
    ]
    eligibility = {}
    for model_id in model_ids:
        row = summaries[model_id]
        checks = [
            {
                "name": "every_fold_improves_nonnegative_affine",
                "passed": row[
                    "every_fold_improves_over_nonnegative_affine"
                ],
            },
            {
                "name": "mean_deltae76",
                "passed": row["mean_deltae76"]
                <= gates["mean_deltae76_max"],
            },
            {
                "name": "worst_fold_mean_deltae76",
                "passed": row["worst_fold_mean_deltae76"]
                <= gates["worst_fold_mean_deltae76_max"],
            },
            {
                "name": "improvement_over_nonnegative_affine",
                "passed": 1.0
                - row["mean_deltae76"] / nonnegative_mean
                >= gates["mean_improvement_over_nonnegative_affine_min"],
            },
            {
                "name": "penalty_relative_to_unsafe_quadratic",
                "passed": row["mean_deltae76"] / unsafe_mean - 1.0
                <= gates["mean_penalty_relative_to_unsafe_quadratic_max"],
            },
            {
                "name": "raw_xyz_negative_fraction",
                "passed": row["raw_xyz_negative_component_fraction"]
                <= gates["raw_xyz_negative_component_fraction_max"],
            },
            {
                "name": "raw_xyz_high_fraction",
                "passed": row["raw_xyz_above_1_2_component_fraction"]
                <= gates["raw_xyz_above_1_2_component_fraction_max"],
            },
        ]
        eligibility[model_id] = {
            "checks": checks,
            "eligible": all(check["passed"] for check in checks),
        }
    eligible = [
        model_id
        for model_id in model_ids
        if eligibility[model_id]["eligible"]
    ]
    champion = None
    if eligible:
        best = min(summaries[model]["mean_deltae76"] for model in eligible)
        near_best = {
            model
            for model in eligible
            if summaries[model]["mean_deltae76"]
            <= best
            * (
                1.0
                + config["model_selection"][
                    "simpler_degree_tie_relative"
                ]
            )
        }
        champion = next(
            model
            for model in config["model_selection"]["simplicity_order"]
            if model in near_best
        )
    automatic_pass = champion is not None
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": _git_commit(),
        "config_sha256": CONFIG_SHA256,
        "pair_table_sha256": config["parent"]["pair_table_sha256"],
        "parent_report_sha256": config["parent"]["report_sha256"],
        "joint_folds": joint,
        "joint_summary": summaries,
        "eligibility": eligibility,
        "eligible_models": eligible,
        "champion": champion,
        "automatic_pass": automatic_pass,
        "decision": config["decision_branches"][
            "pass" if automatic_pass else "fail"
        ],
        "post_fit_clipping": False,
        "render_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def _evaluate_bernstein(
    model: Any,
    source: np.ndarray,
    target_xyz: np.ndarray,
    target_lab: np.ndarray,
    white: np.ndarray,
) -> dict[str, float | int]:
    predicted_xyz = apply_bounded_bernstein(model, source)
    predicted_lab = xyz_to_lab_d50(
        predicted_xyz, reference_white=white
    )
    delta_e = np.linalg.norm(predicted_lab - target_lab, axis=1)
    return {
        "row_count": int(source.shape[0]),
        "xyz_rmse": float(
            np.sqrt(np.mean((predicted_xyz - target_xyz) ** 2))
        ),
        "mean_deltae76": float(np.mean(delta_e)),
        "median_deltae76": float(np.median(delta_e)),
        "p95_deltae76": float(np.percentile(delta_e, 95.0)),
        "maximum_deltae76": float(np.max(delta_e)),
        "raw_xyz_negative_component_fraction": float(
            np.mean(predicted_xyz < 0.0)
        ),
        "raw_xyz_above_1_2_component_fraction": float(
            np.mean(predicted_xyz > 1.2)
        ),
    }


def run(config_path: Path, output_root: Path) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    config = load_config(config_path, CONFIG_SHA256)
    report = run_experiment(config)
    report_bytes = _canonical_json(report)
    _atomic_write(output_root / "report.json", report_bytes)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "champion": report["champion"],
                "decision": report["decision"],
                "eligible_models": report["eligible_models"],
                "joint_summary": report["joint_summary"],
                "report_sha256": _sha256(report_bytes),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2aq3b_colorreference_bounded_bernstein_proxy_v1.json",
    )
    parser.add_argument(
        "--expected-config-sha256",
        default=CONFIG_SHA256,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "outputs/experiments/u5_r2aq3b_colorreference_bounded_bernstein_proxy_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    load_config(args.config, args.expected_config_sha256)
    run(args.config, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
