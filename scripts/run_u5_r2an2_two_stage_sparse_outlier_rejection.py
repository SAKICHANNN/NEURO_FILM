#!/usr/bin/env python
"""Run the frozen U5.R2AN2 two-stage sparse-outlier experiment."""

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
    _error_metrics,
    _fit_record,
    _structure_metrics,
    generate_paired_design,
    split_masks,
)
from scripts.run_u5_r2an1_robust_paired_positive_film_recovery import (  # noqa: E402
    perturb_development_pairs,
)
from src.roll2film.positive_film import (  # noqa: E402
    positive_film_operator_from_config,
)
from src.roll2film.positive_film_fitting import (  # noqa: E402
    fit_positive_film_response_operator,
)
from src.roll2film.positive_film_sparse_rejection import (  # noqa: E402
    fit_two_stage_sparse_rejection_operator,
)


CONFIG_SHA256 = "a19a16f2d91bf8c85bd84889cd75fc4459884dfd6b9400a24460c95f8f995abf"
REPORT_SCHEMA = "neuro-film.u5.r2an2.two-stage-sparse-rejection-report.v1"
REPEAT_SCHEMA = "neuro-film.u5.r2an2.repeat-decision.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
            raise RuntimeError("AN2 requires a clean tracked worktree")


def load_config(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AN2 config hash mismatch")
    config = json.loads(raw)
    if (
        config["experiment_id"]
        != "u5.r2an2-two-stage-sparse-outlier-rejection-v1"
        or config["parent"]["report_sha256"]
        != "ec5309e8f61388d8dc04b04162f865d404c6675d575252668423e8d40185bb08"
        or config["parent"]["automatic_pass"]
        or config["parent"]["failed_check_count"] != 1
        or config["algorithm_may_read_true_outlier_rows"]
        or config["real_film_pixel_allowed"]
        or config["proxy_pair_allowed"]
        or config["photographic_render_allowed"]
        or config["production_integration_allowed"]
    ):
        raise ValueError("AN2 frozen contract mismatch")
    return config


def retained_support_metrics(
    retained_mask: np.ndarray,
    patch: np.ndarray,
    illuminant: np.ndarray,
    exposure: np.ndarray,
) -> dict[str, Any]:
    """Summarize retained development support without changing selection."""

    retained = np.asarray(retained_mask, dtype=bool)
    patch_values = np.asarray(patch)
    illuminant_values = np.asarray(illuminant)
    exposure_values = np.asarray(exposure)
    if (
        retained.ndim != 1
        or patch_values.shape != retained.shape
        or illuminant_values.shape != retained.shape
        or exposure_values.shape != retained.shape
    ):
        raise ValueError("AN2 retained-support arrays must align")
    retained_patch = patch_values[retained]
    retained_illuminant = illuminant_values[retained]
    retained_exposure = exposure_values[retained]
    patch_counts = [
        int(np.count_nonzero(retained_patch == value))
        for value in np.unique(patch_values)
    ]
    cell_counts = [
        int(
            np.count_nonzero(
                retained
                & (illuminant_values == illuminant_value)
                & (exposure_values == exposure_value)
            )
        )
        for illuminant_value in np.unique(illuminant_values)
        for exposure_value in np.unique(exposure_values)
    ]
    return {
        "retained_row_count": int(np.count_nonzero(retained)),
        "rejected_row_count": int(np.count_nonzero(~retained)),
        "distinct_base_patch_count": int(np.unique(retained_patch).size),
        "distinct_illuminant_count": int(np.unique(retained_illuminant).size),
        "distinct_exposure_count": int(np.unique(retained_exposure).size),
        "minimum_retained_rows_per_base_patch": min(patch_counts),
        "minimum_retained_rows_per_illuminant_exposure_cell": min(cell_counts),
    }


def _support_checks(
    witness: str,
    scenario: str,
    support: dict[str, Any],
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    prefix = f"{witness}.{scenario}.support"
    return [
        {
            "name": f"{prefix}.total_rows",
            "passed": support["retained_row_count"]
            >= int(contract["minimum_retained_total_rows"]),
        },
        {
            "name": f"{prefix}.cell_rows",
            "passed": support[
                "minimum_retained_rows_per_illuminant_exposure_cell"
            ]
            >= int(
                contract[
                    "minimum_retained_rows_per_development_illuminant_exposure_cell"
                ]
            ),
        },
        {
            "name": f"{prefix}.patch_rows",
            "passed": support["minimum_retained_rows_per_base_patch"]
            >= int(contract["minimum_retained_rows_per_base_patch"]),
        },
        {
            "name": f"{prefix}.patch_count",
            "passed": support["distinct_base_patch_count"]
            >= int(contract["minimum_distinct_base_patches"]),
        },
        {
            "name": f"{prefix}.illuminant_count",
            "passed": support["distinct_illuminant_count"]
            >= int(contract["minimum_distinct_illuminants"]),
        },
        {
            "name": f"{prefix}.exposure_count",
            "passed": support["distinct_exposure_count"]
            >= int(contract["minimum_distinct_exposures"]),
        },
    ]


def run_child(config: dict[str, Any], *, software_commit: str) -> dict[str, Any]:
    parent_path = ROOT / str(config["parent"]["config_path"])
    parent_raw = parent_path.read_bytes()
    if _sha256(parent_raw) != config["parent"]["config_sha256"]:
        raise ValueError("AN2 parent config hash mismatch")
    an1 = json.loads(parent_raw)
    an0_path = ROOT / str(an1["parent"]["config_path"])
    an0_raw = an0_path.read_bytes()
    if _sha256(an0_raw) != an1["parent"]["config_sha256"]:
        raise ValueError("AN2 grandparent config hash mismatch")
    an0 = json.loads(an0_raw)
    source, patch, illuminant, exposure = generate_paired_design(an0)
    masks = split_masks(an0, patch, illuminant, exposure)
    truth_raw = (ROOT / str(an0["truth_operator_config"])).read_bytes()
    if _sha256(truth_raw) != an0["truth_operator_config_sha256"]:
        raise ValueError("AN2 truth config hash mismatch")
    truths = json.loads(truth_raw)
    source_before = source.copy()
    development_mask = masks["development"]
    development_source = source[development_mask]
    development_patch = patch[development_mask]
    development_illuminant = illuminant[development_mask]
    development_exposure = exposure[development_mask]
    confirmation_source = source[masks["confirmation"]]
    fit = config["two_stage_fit"]
    common = {
        "identity_mixture": float(an0["models"]["matrix_identity_mixture"]),
        "restart_count": int(fit["restart_count"]),
        "maximum_function_evaluations": int(
            fit["maximum_function_evaluations"]
        ),
        "function_tolerance": float(fit["function_tolerance"]),
        "parameter_tolerance": float(fit["parameter_tolerance"]),
        "gradient_tolerance": float(fit["gradient_tolerance"]),
        "seed": int(an0["synthetic_design"]["base_patch_seed"]),
    }
    gates = config["gates"]
    all_checks: list[dict[str, Any]] = []
    witness_records: dict[str, Any] = {}
    for witness_index, witness_name in enumerate(config["truth_witnesses"]):
        truth = positive_film_operator_from_config(
            truths["witnesses"][witness_name]
        )
        clean_target = truth.apply(source)
        development_target = clean_target[development_mask]
        confirmation_target = clean_target[masks["confirmation"]]
        scenarios: dict[str, Any] = {}
        for scenario_index, (scenario_name, scenario) in enumerate(
            config["scenarios"].items()
        ):
            perturbation_seed = (
                int(config["perturbation_seed"])
                + 100 * witness_index
                + scenario_index
            )
            noisy_source, noisy_target, outlier_rows = perturb_development_pairs(
                development_source,
                development_target,
                scenario,
                seed=perturbation_seed,
            )
            candidate = fit_two_stage_sparse_rejection_operator(
                noisy_source,
                noisy_target,
                rejected_row_fraction=float(fit["rejected_row_fraction"]),
                stage_one_loss_scale=float(fit["stage_one_loss_scale"]),
                stage_two_loss_scale=float(fit["stage_two_loss_scale"]),
                **common,
            )
            linear = fit_positive_film_response_operator(
                noisy_source,
                noisy_target,
                model="two_matrix",
                loss="linear",
                loss_scale=float(fit["stage_two_loss_scale"]),
                **common,
            )
            one_matrix = fit_positive_film_response_operator(
                noisy_source,
                noisy_target,
                model="one_matrix",
                loss="soft_l1",
                loss_scale=float(fit["stage_one_loss_scale"]),
                **common,
            )
            candidate_metrics = _error_metrics(
                candidate.stage_two.operator,
                confirmation_source,
                confirmation_target,
            )
            stage_one_metrics = _error_metrics(
                candidate.stage_one.operator,
                confirmation_source,
                confirmation_target,
            )
            linear_metrics = _error_metrics(
                linear.operator, confirmation_source, confirmation_target
            )
            one_metrics = _error_metrics(
                one_matrix.operator,
                confirmation_source,
                confirmation_target,
            )
            gain_over_linear = 1.0 - (
                candidate_metrics["rgb_rmse"] / linear_metrics["rgb_rmse"]
            )
            gain_over_one = 1.0 - (
                candidate_metrics["rgb_rmse"] / one_metrics["rgb_rmse"]
            )
            gain_over_stage_one = 1.0 - (
                candidate_metrics["rgb_rmse"]
                / stage_one_metrics["rgb_rmse"]
            )
            relative_loss_to_stage_one = (
                candidate_metrics["rgb_rmse"]
                / stage_one_metrics["rgb_rmse"]
            ) - 1.0
            support = retained_support_metrics(
                candidate.retained_row_mask,
                development_patch,
                development_illuminant,
                development_exposure,
            )
            rejected_set = set(candidate.rejected_row_indices.tolist())
            outlier_set = set(outlier_rows.tolist())
            true_positive = len(rejected_set & outlier_set)
            outlier_recall = (
                true_positive / len(outlier_set) if outlier_set else None
            )
            outlier_precision = (
                true_positive / len(rejected_set) if outlier_set else None
            )
            structure = _structure_metrics(candidate.stage_two.operator, an0)
            checks = [
                {
                    "name": f"{witness_name}.{scenario_name}.candidate_rmse",
                    "passed": candidate_metrics["rgb_rmse"]
                    <= float(
                        gates["candidate_confirmation_rgb_rmse_maximum"]
                    ),
                },
                {
                    "name": f"{witness_name}.{scenario_name}.candidate_max_error",
                    "passed": candidate_metrics["maximum_absolute_error"]
                    <= float(
                        gates[
                            "candidate_confirmation_maximum_absolute_error"
                        ]
                    ),
                },
                {
                    "name": f"{witness_name}.{scenario_name}.gain_over_one_matrix",
                    "passed": gain_over_one
                    >= float(gates["minimum_gain_over_robust_one_matrix"]),
                },
                {
                    "name": f"{witness_name}.{scenario_name}.positive_jacobian",
                    "passed": structure["minimum_jacobian_determinant"]
                    > float(
                        gates[
                            "minimum_finite_difference_jacobian_determinant"
                        ]
                    ),
                },
            ]
            checks.extend(
                _support_checks(
                    witness_name,
                    scenario_name,
                    support,
                    config["retained_support"],
                )
            )
            if scenario_name == "sparse_correspondence_outliers":
                checks.extend(
                    [
                        {
                            "name": f"{witness_name}.{scenario_name}.gain_over_linear",
                            "passed": gain_over_linear
                            >= float(
                                gates[
                                    "minimum_outlier_gain_over_linear_two_matrix"
                                ]
                            ),
                        },
                        {
                            "name": f"{witness_name}.{scenario_name}.gain_over_stage_one",
                            "passed": gain_over_stage_one
                            >= float(
                                gates[
                                    "minimum_outlier_gain_over_stage_one_soft_l1"
                                ]
                            ),
                        },
                        {
                            "name": f"{witness_name}.{scenario_name}.outlier_recall",
                            "passed": outlier_recall is not None
                            and outlier_recall
                            >= float(
                                gates[
                                    "minimum_outlier_recall_for_diagnostic"
                                ]
                            ),
                        },
                        {
                            "name": f"{witness_name}.{scenario_name}.outlier_precision",
                            "passed": outlier_precision is not None
                            and outlier_precision
                            >= float(
                                gates[
                                    "minimum_outlier_precision_for_diagnostic"
                                ]
                            ),
                        },
                    ]
                )
            else:
                checks.append(
                    {
                        "name": f"{witness_name}.{scenario_name}.relative_loss_to_stage_one",
                        "passed": relative_loss_to_stage_one
                        <= float(
                            gates[
                                "maximum_noise_only_relative_loss_to_stage_one_soft_l1"
                            ]
                        ),
                    }
                )
            all_checks.extend(checks)
            scenarios[scenario_name] = {
                "perturbation_seed": perturbation_seed,
                "noisy_source_f64_sha256": _sha256(
                    noisy_source.tobytes(order="C")
                ),
                "noisy_target_f64_sha256": _sha256(
                    noisy_target.tobytes(order="C")
                ),
                "true_outlier_row_count_for_diagnostic_only": int(
                    outlier_rows.size
                ),
                "true_outlier_rows_i64_sha256_for_diagnostic_only": _sha256(
                    outlier_rows.tobytes(order="C")
                ),
                "stage_one_fit": _fit_record(candidate.stage_one),
                "stage_two_fit": _fit_record(candidate.stage_two),
                "linear_two_matrix_fit": _fit_record(linear),
                "robust_one_matrix_fit": _fit_record(one_matrix),
                "residual_scores_f64_sha256": _sha256(
                    candidate.residual_scores.tobytes(order="C")
                ),
                "rejected_rows_i64_sha256": _sha256(
                    candidate.rejected_row_indices.tobytes(order="C")
                ),
                "retained_mask_bool_sha256": _sha256(
                    candidate.retained_row_mask.tobytes(order="C")
                ),
                "retained_support": support,
                "outlier_diagnostic": {
                    "true_positive_count": true_positive,
                    "recall": outlier_recall,
                    "precision": outlier_precision,
                },
                "clean_confirmation": {
                    "row_count": int(confirmation_source.shape[0]),
                    "candidate": candidate_metrics,
                    "stage_one_soft_l1": stage_one_metrics,
                    "linear_two_matrix": linear_metrics,
                    "robust_one_matrix": one_metrics,
                    "candidate_gain_over_linear_two_matrix": gain_over_linear,
                    "candidate_gain_over_robust_one_matrix": gain_over_one,
                    "candidate_gain_over_stage_one_soft_l1": (
                        gain_over_stage_one
                    ),
                    "candidate_relative_loss_to_stage_one_soft_l1": (
                        relative_loss_to_stage_one
                    ),
                },
                "candidate_structure": structure,
                "automatic_checks": checks,
            }
        witness_records[witness_name] = {"scenarios": scenarios}
    source_nonmutation = np.array_equal(source, source_before)
    all_checks.append(
        {"name": "source_nonmutation", "passed": source_nonmutation}
    )
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": CONFIG_SHA256,
        "parent_report_sha256": config["parent"]["report_sha256"],
        "design_source_f64_sha256": _sha256(source.tobytes(order="C")),
        "development_row_count": int(np.count_nonzero(development_mask)),
        "confirmation_row_count": int(
            np.count_nonzero(masks["confirmation"])
        ),
        "algorithm_received_true_outlier_rows": False,
        "witnesses": witness_records,
        "automatic_checks": all_checks,
        "automatic_pass": all(bool(check["passed"]) for check in all_checks),
        "claim_ceiling": config["claim_ceiling"],
    }


def _child_command(
    config_path: Path, output_path: Path, software_commit: str
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--expected-config-sha256",
        CONFIG_SHA256,
        "--child-output",
        str(output_path),
        "--software-commit",
        software_commit,
    ]


def run_parent(config_path: Path, output_dir: Path) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    config = load_config(config_path, expected_sha256=CONFIG_SHA256)
    software_commit = _git_commit()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / "report_a.json", output_dir / "report_b.json"]
    for path in paths:
        subprocess.run(
            _child_command(config_path, path, software_commit),
            cwd=ROOT,
            check=True,
        )
    payloads = [path.read_bytes() for path in paths]
    reports = [json.loads(payload) for payload in payloads]
    byte_identical = payloads[0] == payloads[1]
    automatic_pass = byte_identical and all(
        bool(report["automatic_pass"]) for report in reports
    )
    decision = {
        "schema": REPEAT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": CONFIG_SHA256,
        "report_sha256": [_sha256(payload) for payload in payloads],
        "reports_byte_identical": byte_identical,
        "automatic_pass": automatic_pass,
        "decision": (
            "advance_two_stage_sparse_outlier_rejection"
            if automatic_pass
            else "close_two_stage_sparse_outlier_rejection"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    decision_bytes = _canonical_json(decision)
    _atomic_write(output_dir / "repeat_decision.json", decision_bytes)
    print(
        json.dumps(
            {
                "automatic_pass": automatic_pass,
                "decision": decision["decision"],
                "report_sha256": decision["report_sha256"],
                "repeat_decision_sha256": _sha256(decision_bytes),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return decision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2an2_two_stage_sparse_outlier_rejection_v1.json",
    )
    parser.add_argument(
        "--expected-config-sha256", default=CONFIG_SHA256
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--child-output", type=Path)
    parser.add_argument("--software-commit")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config(
        args.config, expected_sha256=args.expected_config_sha256
    )
    if args.child_output is not None:
        if not args.software_commit:
            raise ValueError("AN2 child requires --software-commit")
        report = run_child(config, software_commit=args.software_commit)
        _atomic_write(args.child_output, _canonical_json(report))
        return 0
    if args.output_dir is None:
        raise ValueError("AN2 parent requires --output-dir")
    run_parent(args.config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
