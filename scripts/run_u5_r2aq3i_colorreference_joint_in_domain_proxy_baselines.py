#!/usr/bin/env python
"""Evaluate explicit recorder proxies on held set and colour folds."""

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
    _evaluate,
    _summaries,
    load_pair_table,
)
from scripts.run_u5_r2aq3t_colorreference_calibration_suite_topology import (  # noqa: E402
    colour_fold,
)
from src.roll2film.colorreference_proxy_baselines import (  # noqa: E402
    fit_proxy_model,
    xyz_to_lab_d50,
)


CONFIG_SHA256 = "e5840fd2dccc83c7cd97fb6b5201420e5e8cf7b60f942201068783ee877e1d52"
EXPERIMENT_ID = "u5.r2aq3i-colorreference-joint-in-domain-proxy-baselines-v1"
REPORT_SCHEMA = "neuro-film.u5.r2aq3i.joint-in-domain-proxy-baselines.v1"


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
            raise RuntimeError("AQ3I requires a clean tracked worktree")


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AQ3I config hash mismatch")
    config = json.loads(raw)
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["split_contract"]["primary_fold_count"] != 30
        or not config["fit_allowed"]
        or config["training_allowed"]
        or config["render_allowed"]
        or config["production_integration_allowed"]
    ):
        raise ValueError("AQ3I frozen contract mismatch")
    return config


def source_colour_folds(
    source: np.ndarray,
    *,
    salt: str,
) -> np.ndarray:
    values = np.asarray(source, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("AQ3I source must be finite normalized RGB")
    codes = np.rint(values * 255.0).astype(np.uint8)
    if not np.allclose(values, codes.astype(np.float64) / 255.0):
        raise ValueError("AQ3I source is not exact uint8-derived RGB")
    return colour_fold(codes, salt=salt)


def fold_masks(
    table: dict[str, np.ndarray],
    folds: np.ndarray,
    *,
    held_set: int,
    held_fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    sets = table["test_set"]
    if folds.shape != sets.shape:
        raise ValueError("AQ3I fold shape mismatch")
    development = (sets != held_set) & (folds != held_fold)
    confirmation = (sets == held_set) & (folds == held_fold)
    if np.any(development & confirmation):
        raise RuntimeError("AQ3I split leakage")
    return development, confirmation


def _fit_and_evaluate_fold(
    table: dict[str, np.ndarray],
    folds: np.ndarray,
    *,
    held_set: int,
    held_fold: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    development, confirmation = fold_masks(
        table,
        folds,
        held_set=held_set,
        held_fold=held_fold,
    )
    split = config["split_contract"]
    expected_development = split[
        "expected_development_rows_by_source_fold"
    ][held_fold]
    expected_confirmation = split[
        "expected_confirmation_rows_by_source_fold"
    ][held_fold]
    if (
        int(development.sum()) != expected_development
        or int(confirmation.sum()) != expected_confirmation
    ):
        raise RuntimeError("AQ3I fold size mismatch")
    numerics = config["numerics"]
    white = np.asarray(
        config["target_space"]["reference_white_xyz"],
        dtype=np.float64,
    )
    models = {}
    for candidate in config["candidate_models"]:
        model_id = candidate["id"]
        model = fit_proxy_model(
            table["source"][development],
            table["xyz"][development],
            model_id=model_id,
            quadratic_ridge=numerics["quadratic_ridge"],
            nonnegative_maximum_iterations=numerics[
                "nonnegative_maximum_iterations"
            ],
        )
        models[model_id] = {
            "model": model.to_dict(),
            "development": _evaluate(
                model,
                table["source"][development],
                table["xyz"][development],
                table["lab"][development],
                white,
            ),
            "confirmation": _evaluate(
                model,
                table["source"][confirmation],
                table["xyz"][confirmation],
                table["lab"][confirmation],
                white,
            ),
        }
    return {
        "held_target_set": held_set,
        "held_source_colour_fold": held_fold,
        "development_rows": int(development.sum()),
        "confirmation_rows": int(confirmation.sum()),
        "models": models,
    }


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    table = load_pair_table(
        ROOT / config["parent"]["pair_table"],
        expected_sha256=config["parent"]["pair_table_sha256"],
    )
    split = config["split_contract"]
    folds = source_colour_folds(
        table["source"], salt=split["source_colour_salt"]
    )
    counts_per_set = []
    for test_set in split["target_sets"]:
        mask = table["test_set"] == test_set
        counts_per_set.append(
            np.bincount(folds[mask], minlength=5).tolist()
        )
    if any(
        counts != split["source_colour_fold_row_counts"]
        for counts in counts_per_set
    ):
        raise RuntimeError("AQ3I source fold counts drifted")
    coverage = np.zeros(table["source"].shape[0], dtype=np.int64)
    for test_set in split["target_sets"]:
        for held_fold in range(split["source_colour_fold_count"]):
            _, confirmation = fold_masks(
                table,
                folds,
                held_set=test_set,
                held_fold=held_fold,
            )
            coverage += confirmation.astype(np.int64)
    if not np.all(coverage == 1):
        raise RuntimeError("AQ3I confirmation coverage mismatch")
    white = np.asarray(
        config["target_space"]["reference_white_xyz"],
        dtype=np.float64,
    )
    consistency = np.linalg.norm(
        xyz_to_lab_d50(table["xyz"], reference_white=white)
        - table["lab"],
        axis=1,
    )
    target_checks = [
        {
            "name": "provided_xyz_lab_median_consistency",
            "passed": float(np.median(consistency))
            <= config["target_space"][
                "provided_xyz_lab_consistency_median_deltae76_max"
            ],
        },
        {
            "name": "provided_xyz_lab_p95_consistency",
            "passed": float(np.percentile(consistency, 95.0))
            <= config["target_space"][
                "provided_xyz_lab_consistency_p95_deltae76_max"
            ],
        },
    ]
    joint = [
        _fit_and_evaluate_fold(
            table,
            folds,
            held_set=test_set,
            held_fold=held_fold,
            config=config,
        )
        for test_set in split["target_sets"]
        for held_fold in range(split["source_colour_fold_count"])
    ]
    model_ids = [
        candidate["id"] for candidate in config["candidate_models"]
    ]
    summary = _summaries(joint, model_ids)
    selection = config["model_selection"]
    eligible = []
    for model_id in model_ids:
        if model_id == "identity":
            continue
        row = summary[model_id]
        if (
            row["every_fold_improves_over_identity"]
            and row["raw_xyz_negative_component_fraction"] <= 0.005
            and row["raw_xyz_above_1_2_component_fraction"] <= 0.005
            and row["worst_fold_mean_deltae76"] <= 15.0
        ):
            eligible.append(model_id)
    champion = None
    bounded_relative_checks: list[dict[str, Any]] = []
    if eligible:
        best = min(summary[model]["mean_deltae76"] for model in eligible)
        near_best = {
            model
            for model in eligible
            if summary[model]["mean_deltae76"]
            <= best * (1.0 + selection["simpler_model_tie_relative"])
        }
        champion = next(
            model
            for model in selection["simplicity_order"]
            if model in near_best
        )
        relative = selection["bounded_champion_relative_gates"]
        champion_mean = summary[champion]["mean_deltae76"]
        bounded_relative_checks = [
            {
                "name": "bounded_champion_improves_over_identity",
                "passed": 1.0
                - champion_mean / summary["identity"]["mean_deltae76"]
                >= relative["mean_deltae76_improvement_over_identity_min"],
            },
            {
                "name": "bounded_champion_improves_over_diagonal",
                "passed": 1.0
                - champion_mean
                / summary["diagonal_affine"]["mean_deltae76"]
                >= relative[
                    "mean_deltae76_improvement_over_diagonal_affine_min"
                ],
            },
        ]
    bounded_pass = champion is not None and all(
        check["passed"] for check in bounded_relative_checks
    )
    quadratic = summary["quadratic_full"]
    nonnegative = summary["nonnegative_affine"]
    quadratic_fold_means = np.asarray(
        [
            fold["models"]["quadratic_full"]["confirmation"][
                "mean_deltae76"
            ]
            for fold in joint
        ]
    )
    nonnegative_fold_means = np.asarray(
        [
            fold["models"]["nonnegative_affine"]["confirmation"][
                "mean_deltae76"
            ]
            for fold in joint
        ]
    )
    mechanism = selection["unsafe_quadratic_mechanism_gate"]
    mechanism_checks = [
        {
            "name": "quadratic_mean_accuracy",
            "passed": quadratic["mean_deltae76"]
            <= mechanism["mean_deltae76_max"],
        },
        {
            "name": "quadratic_worst_fold_accuracy",
            "passed": quadratic["worst_fold_mean_deltae76"]
            <= mechanism["worst_fold_mean_deltae76_max"],
        },
        {
            "name": "quadratic_improves_nonnegative_mean",
            "passed": 1.0
            - quadratic["mean_deltae76"]
            / nonnegative["mean_deltae76"]
            >= mechanism["mean_improvement_over_nonnegative_affine_min"],
        },
        {
            "name": "quadratic_negative_fraction_bounded_for_diagnosis",
            "passed": quadratic["raw_xyz_negative_component_fraction"]
            <= mechanism["raw_xyz_negative_component_fraction_max"],
        },
        {
            "name": "quadratic_high_fraction_bounded_for_diagnosis",
            "passed": quadratic["raw_xyz_above_1_2_component_fraction"]
            <= mechanism["raw_xyz_above_1_2_component_fraction_max"],
        },
        {
            "name": "quadratic_every_fold_improves_nonnegative",
            "passed": bool(
                np.all(quadratic_fold_means < nonnegative_fold_means)
            ),
        },
    ]
    mechanism_pass = all(
        check["passed"] for check in mechanism_checks
    )
    target_pass = all(check["passed"] for check in target_checks)
    automatic_pass = target_pass and (bounded_pass or mechanism_pass)
    if target_pass and bounded_pass:
        decision = config["decision_branches"]["bounded_champion"]
    elif target_pass and mechanism_pass:
        decision = config["decision_branches"][
            "unsafe_quadratic_mechanism_pass"
        ]
    else:
        decision = config["decision_branches"][
            "no_supported_model_or_mechanism"
        ]
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": _git_commit(),
        "config_sha256": CONFIG_SHA256,
        "pair_table_sha256": config["parent"]["pair_table_sha256"],
        "source_colour_fold_counts_per_target_set": counts_per_set,
        "confirmation_coverage_minimum": int(coverage.min()),
        "confirmation_coverage_maximum": int(coverage.max()),
        "target_xyz_lab_consistency": {
            "median_deltae76": float(np.median(consistency)),
            "p95_deltae76": float(np.percentile(consistency, 95.0)),
            "maximum_deltae76": float(np.max(consistency)),
        },
        "target_checks": target_checks,
        "joint_folds": joint,
        "joint_summary": summary,
        "eligible_bounded_models": eligible,
        "bounded_champion": champion,
        "bounded_relative_checks": bounded_relative_checks,
        "bounded_champion_pass": bounded_pass,
        "unsafe_quadratic_mechanism_checks": mechanism_checks,
        "unsafe_quadratic_mechanism_pass": mechanism_pass,
        "automatic_pass": automatic_pass,
        "decision": decision,
        "aq2_reopened": False,
        "render_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
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
                "bounded_champion": report["bounded_champion"],
                "decision": report["decision"],
                "report_sha256": _sha256(report_bytes),
                "joint_summary": report["joint_summary"],
                "unsafe_quadratic_mechanism_pass": report[
                    "unsafe_quadratic_mechanism_pass"
                ],
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
        / "configs/u5_r2aq3i_colorreference_joint_in_domain_proxy_baselines_v1.json",
    )
    parser.add_argument(
        "--expected-config-sha256",
        default=CONFIG_SHA256,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "outputs/experiments/u5_r2aq3i_colorreference_joint_in_domain_proxy_baselines_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    load_config(args.config, args.expected_config_sha256)
    run(args.config, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
