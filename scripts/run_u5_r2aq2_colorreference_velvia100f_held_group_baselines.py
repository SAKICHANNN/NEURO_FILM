#!/usr/bin/env python
"""Run frozen held-group ColorReference explicit proxy baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
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
from src.roll2film.colorreference_proxy_baselines import (  # noqa: E402
    ProxyModel,
    apply_proxy_model,
    fit_proxy_model,
    xyz_to_lab_d50,
)


CONFIG_SHA256 = "596ba1379aa9689dd0c02f4b359fd2a9513a59e19af503939d7bb33ab4da34f2"
REPORT_SCHEMA = "neuro-film.u5.r2aq2.held-group-proxy-baselines.v1"


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
            raise RuntimeError("AQ2 requires a clean tracked worktree")


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AQ2 config hash mismatch")
    config = json.loads(raw)
    if (
        config["experiment_id"]
        != "u5.r2aq2-colorreference-velvia100f-held-group-baselines-v1"
        or config["split_contract"]["primary_fold_count"] != 30
        or not config["fit_allowed"]
        or config["training_allowed"]
        or config["render_allowed"]
        or config["production_integration_allowed"]
    ):
        raise ValueError("AQ2 frozen contract mismatch")
    return config


def load_pair_table(
    path: Path, *, expected_sha256: str
) -> dict[str, np.ndarray]:
    payload = path.read_bytes()
    if _sha256(payload) != expected_sha256:
        raise ValueError("AQ2 pair-table hash mismatch")
    reader = csv.DictReader(
        io.StringIO(payload.decode("utf-8")), delimiter="\t"
    )
    rows = list(reader)
    if len(rows) != 8640:
        raise ValueError("AQ2 pair-table row count mismatch")
    return {
        "test_set": np.asarray(
            [int(row["test_set"]) for row in rows], dtype=np.int64
        ),
        "slide_index": np.asarray(
            [int(row["slide_index"]) for row in rows], dtype=np.int64
        ),
        "source": np.asarray(
            [
                [
                    float(row["source_r"]),
                    float(row["source_g"]),
                    float(row["source_b"]),
                ]
                for row in rows
            ],
            dtype=np.float64,
        ),
        "xyz": 0.01
        * np.asarray(
            [
                [
                    float(row["xyz_x"]),
                    float(row["xyz_y"]),
                    float(row["xyz_z"]),
                ]
                for row in rows
            ],
            dtype=np.float64,
        ),
        "lab": np.asarray(
            [
                [
                    float(row["lab_l"]),
                    float(row["lab_a"]),
                    float(row["lab_b"]),
                ]
                for row in rows
            ],
            dtype=np.float64,
        ),
    }


def _fold_masks(
    table: dict[str, np.ndarray],
    split_kind: str,
    held_set: int | None,
    held_slide: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    sets = table["test_set"]
    slides = table["slide_index"]
    if split_kind == "joint":
        development = (sets != held_set) & (slides != held_slide)
        confirmation = (sets == held_set) & (slides == held_slide)
    elif split_kind == "held_set":
        development = sets != held_set
        confirmation = sets == held_set
    elif split_kind == "held_slide":
        development = slides != held_slide
        confirmation = slides == held_slide
    else:
        raise ValueError("unsupported split kind")
    if np.any(development & confirmation):
        raise RuntimeError("AQ2 split leakage")
    return development, confirmation


def _evaluate(
    model: ProxyModel,
    source: np.ndarray,
    target_xyz: np.ndarray,
    target_lab: np.ndarray,
    reference_white: np.ndarray,
) -> dict[str, float | int]:
    predicted_xyz = apply_proxy_model(model, source)
    predicted_lab = xyz_to_lab_d50(
        predicted_xyz, reference_white=reference_white
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


def _fit_and_evaluate_fold(
    table: dict[str, np.ndarray],
    *,
    split_kind: str,
    held_set: int | None,
    held_slide: int | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    development, confirmation = _fold_masks(
        table, split_kind, held_set, held_slide
    )
    expected = config["split_contract"]
    if split_kind == "joint" and (
        int(development.sum())
        != expected["primary_expected_development_rows"]
        or int(confirmation.sum())
        != expected["primary_expected_confirmation_rows"]
    ):
        raise RuntimeError("AQ2 primary fold size mismatch")
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
        "split_kind": split_kind,
        "held_test_set": held_set,
        "held_slide_index": held_slide,
        "development_rows": int(development.sum()),
        "confirmation_rows": int(confirmation.sum()),
        "models": models,
    }


def _summaries(
    folds: list[dict[str, Any]], model_ids: list[str]
) -> dict[str, dict[str, Any]]:
    summaries = {}
    identity_fold_means = np.asarray(
        [
            fold["models"]["identity"]["confirmation"][
                "mean_deltae76"
            ]
            for fold in folds
        ]
    )
    for model_id in model_ids:
        metrics = [
            fold["models"][model_id]["confirmation"]
            for fold in folds
        ]
        weights = np.asarray(
            [metric["row_count"] for metric in metrics],
            dtype=np.float64,
        )
        mean_de = np.asarray(
            [metric["mean_deltae76"] for metric in metrics]
        )
        summaries[model_id] = {
            "fold_count": len(folds),
            "row_count": int(weights.sum()),
            "mean_deltae76": float(
                np.average(mean_de, weights=weights)
            ),
            "median_fold_mean_deltae76": float(np.median(mean_de)),
            "worst_fold_mean_deltae76": float(np.max(mean_de)),
            "every_fold_improves_over_identity": bool(
                np.all(mean_de < identity_fold_means)
            )
            if model_id != "identity"
            else False,
            "identity_mean_improvement_fraction": float(
                1.0
                - np.average(mean_de, weights=weights)
                / np.average(identity_fold_means, weights=weights)
            ),
            "raw_xyz_negative_component_fraction": float(
                np.average(
                    [
                        metric[
                            "raw_xyz_negative_component_fraction"
                        ]
                        for metric in metrics
                    ],
                    weights=weights,
                )
            ),
            "raw_xyz_above_1_2_component_fraction": float(
                np.average(
                    [
                        metric[
                            "raw_xyz_above_1_2_component_fraction"
                        ]
                        for metric in metrics
                    ],
                    weights=weights,
                )
            ),
        }
    return summaries


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    table_path = ROOT / config["parent"]["pair_table"]
    table = load_pair_table(
        table_path,
        expected_sha256=config["parent"]["pair_table_sha256"],
    )
    white = np.asarray(
        config["target_space"]["reference_white_xyz"],
        dtype=np.float64,
    )
    reconstructed_lab = xyz_to_lab_d50(
        table["xyz"], reference_white=white
    )
    consistency = np.linalg.norm(
        reconstructed_lab - table["lab"], axis=1
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
            split_kind="joint",
            held_set=test_set,
            held_slide=slide,
            config=config,
        )
        for test_set in (1, 2, 3, 4, 5, 9)
        for slide in (1, 2, 3, 4, 5)
    ]
    held_set = [
        _fit_and_evaluate_fold(
            table,
            split_kind="held_set",
            held_set=test_set,
            held_slide=None,
            config=config,
        )
        for test_set in (1, 2, 3, 4, 5, 9)
    ]
    held_slide = [
        _fit_and_evaluate_fold(
            table,
            split_kind="held_slide",
            held_set=None,
            held_slide=slide,
            config=config,
        )
        for slide in (1, 2, 3, 4, 5)
    ]
    model_ids = [
        candidate["id"] for candidate in config["candidate_models"]
    ]
    joint_summary = _summaries(joint, model_ids)
    selection = config["model_selection"]
    eligible = []
    for model_id in model_ids:
        if model_id == "identity":
            continue
        summary = joint_summary[model_id]
        if (
            summary["every_fold_improves_over_identity"]
            and summary["raw_xyz_negative_component_fraction"]
            <= 0.005
            and summary["raw_xyz_above_1_2_component_fraction"]
            <= 0.005
            and summary["worst_fold_mean_deltae76"] <= 15.0
        ):
            eligible.append(model_id)
    champion = None
    if eligible:
        best = min(
            joint_summary[model_id]["mean_deltae76"]
            for model_id in eligible
        )
        tie = selection["simpler_model_tie_relative"]
        near_best = {
            model_id
            for model_id in eligible
            if joint_summary[model_id]["mean_deltae76"]
            <= best * (1.0 + tie)
        }
        champion = next(
            model_id
            for model_id in selection["simplicity_order"]
            if model_id in near_best
        )
    relative = selection["relative_gates"]
    relative_checks = []
    if champion is not None:
        champion_mean = joint_summary[champion]["mean_deltae76"]
        identity_mean = joint_summary["identity"]["mean_deltae76"]
        diagonal_mean = joint_summary["diagonal_affine"][
            "mean_deltae76"
        ]
        relative_checks = [
            {
                "name": "champion_improves_over_identity",
                "passed": 1.0 - champion_mean / identity_mean
                >= relative[
                    "champion_mean_deltae76_improvement_over_identity_min"
                ],
            },
            {
                "name": "champion_improves_over_diagonal",
                "passed": 1.0 - champion_mean / diagonal_mean
                >= relative[
                    "champion_mean_deltae76_improvement_over_diagonal_affine_min"
                ],
            },
        ]
    automatic_pass = (
        all(check["passed"] for check in target_checks)
        and champion is not None
        and all(check["passed"] for check in relative_checks)
    )
    quadratic_gain = (
        1.0
        - joint_summary["quadratic_full"]["mean_deltae76"]
        / joint_summary["full_affine"]["mean_deltae76"]
    )
    if not automatic_pass:
        decision = config["decision_branches"]["no_eligible_model"]
    elif (
        champion == "quadratic_full"
        and quadratic_gain
        >= relative[
            "quadratic_improvement_over_full_affine_to_open_iterative_nonlinear_family_min"
        ]
    ):
        decision = config["decision_branches"][
            "quadratic_material_gain"
        ]
    else:
        decision = config["decision_branches"][
            "simple_model_champion"
        ]
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": _git_commit(),
        "config_sha256": CONFIG_SHA256,
        "pair_table_sha256": config["parent"]["pair_table_sha256"],
        "target_xyz_lab_consistency": {
            "median_deltae76": float(np.median(consistency)),
            "p95_deltae76": float(
                np.percentile(consistency, 95.0)
            ),
            "maximum_deltae76": float(np.max(consistency)),
        },
        "target_checks": target_checks,
        "joint_folds": joint,
        "held_set_folds": held_set,
        "held_slide_folds": held_slide,
        "joint_summary": joint_summary,
        "held_set_summary": _summaries(held_set, model_ids),
        "held_slide_summary": _summaries(held_slide, model_ids),
        "eligible_models": eligible,
        "champion": champion,
        "quadratic_improvement_over_full_affine": quadratic_gain,
        "relative_checks": relative_checks,
        "automatic_pass": automatic_pass,
        "decision": decision,
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
                "champion": report["champion"],
                "decision": report["decision"],
                "report_sha256": _sha256(report_bytes),
                "joint_summary": report["joint_summary"],
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
        / "configs/u5_r2aq2_colorreference_velvia100f_held_group_baselines_v1.json",
    )
    parser.add_argument(
        "--expected-config-sha256", default=CONFIG_SHA256
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "outputs/experiments/u5_r2aq2_colorreference_velvia100f_held_group_baselines_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    load_config(args.config, args.expected_config_sha256)
    run(args.config, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
