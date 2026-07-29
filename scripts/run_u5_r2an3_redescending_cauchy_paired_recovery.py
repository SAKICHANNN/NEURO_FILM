#!/usr/bin/env python
"""Run the frozen U5.R2AN3 redescending Cauchy paired-recovery test."""

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


CONFIG_SHA256 = "7fc6f03d6d48f9f140f9116ab98775ec4d83133b26326add9c3a80582a2a7e10"
REPORT_SCHEMA = "neuro-film.u5.r2an3.redescending-cauchy-report.v1"
REPEAT_SCHEMA = "neuro-film.u5.r2an3.repeat-decision.v1"


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
            raise RuntimeError("AN3 requires a clean tracked worktree")


def load_config(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AN3 config hash mismatch")
    config = json.loads(raw)
    if (
        config["experiment_id"]
        != "u5.r2an3-redescending-cauchy-paired-recovery-v1"
        or config["parents"]["an1"]["report_sha256"]
        != "ec5309e8f61388d8dc04b04162f865d404c6675d575252668423e8d40185bb08"
        or config["parents"]["an2"]["report_sha256"]
        != "d5490ff406b7c26c0dcf56ab5957e4093b6e95cfb0892a6e62d880399ae0bcdb"
        or config["fit"]["candidate_loss"] != "cauchy"
        or config["fit"]["candidate_loss_scale"] != 0.005
        or config["fit"]["loss_or_scale_sweep_allowed"]
        or config["real_film_pixel_allowed"]
        or config["proxy_pair_allowed"]
        or config["photographic_render_allowed"]
        or config["production_integration_allowed"]
    ):
        raise ValueError("AN3 frozen contract mismatch")
    return config


def run_child(config: dict[str, Any], *, software_commit: str) -> dict[str, Any]:
    an1_path = ROOT / str(config["parents"]["an1"]["config_path"])
    an1_raw = an1_path.read_bytes()
    if _sha256(an1_raw) != config["parents"]["an1"]["config_sha256"]:
        raise ValueError("AN3 AN1 config hash mismatch")
    an1 = json.loads(an1_raw)
    an0_path = ROOT / str(an1["parent"]["config_path"])
    an0_raw = an0_path.read_bytes()
    if _sha256(an0_raw) != an1["parent"]["config_sha256"]:
        raise ValueError("AN3 AN0 config hash mismatch")
    an0 = json.loads(an0_raw)
    source, patch, illuminant, exposure = generate_paired_design(an0)
    masks = split_masks(an0, patch, illuminant, exposure)
    truth_raw = (ROOT / str(an0["truth_operator_config"])).read_bytes()
    if _sha256(truth_raw) != an0["truth_operator_config_sha256"]:
        raise ValueError("AN3 truth config hash mismatch")
    truths = json.loads(truth_raw)
    source_before = source.copy()
    development_source = source[masks["development"]]
    confirmation_source = source[masks["confirmation"]]
    fit = config["fit"]
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
        development_target = clean_target[masks["development"]]
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
            candidate = fit_positive_film_response_operator(
                noisy_source,
                noisy_target,
                model=str(fit["candidate_model"]),
                loss=str(fit["candidate_loss"]),
                loss_scale=float(fit["candidate_loss_scale"]),
                **common,
            )
            soft_l1 = fit_positive_film_response_operator(
                noisy_source,
                noisy_target,
                model="two_matrix",
                loss=str(fit["soft_l1_control_loss"]),
                loss_scale=float(fit["candidate_loss_scale"]),
                **common,
            )
            linear = fit_positive_film_response_operator(
                noisy_source,
                noisy_target,
                model="two_matrix",
                loss=str(fit["linear_control_loss"]),
                loss_scale=float(fit["candidate_loss_scale"]),
                **common,
            )
            one_matrix = fit_positive_film_response_operator(
                noisy_source,
                noisy_target,
                model="one_matrix",
                loss=str(fit["robust_one_matrix_control_loss"]),
                loss_scale=float(fit["candidate_loss_scale"]),
                **common,
            )
            candidate_metrics = _error_metrics(
                candidate.operator, confirmation_source, confirmation_target
            )
            soft_l1_metrics = _error_metrics(
                soft_l1.operator, confirmation_source, confirmation_target
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
            gain_over_soft_l1 = 1.0 - (
                candidate_metrics["rgb_rmse"] / soft_l1_metrics["rgb_rmse"]
            )
            gain_over_one = 1.0 - (
                candidate_metrics["rgb_rmse"] / one_metrics["rgb_rmse"]
            )
            relative_loss_to_soft_l1 = (
                candidate_metrics["rgb_rmse"] / soft_l1_metrics["rgb_rmse"]
            ) - 1.0
            structure = _structure_metrics(candidate.operator, an0)
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
                            "name": f"{witness_name}.{scenario_name}.gain_over_soft_l1",
                            "passed": gain_over_soft_l1
                            >= float(
                                gates[
                                    "minimum_outlier_gain_over_soft_l1_two_matrix"
                                ]
                            ),
                        },
                    ]
                )
            else:
                checks.append(
                    {
                        "name": f"{witness_name}.{scenario_name}.relative_loss_to_soft_l1",
                        "passed": relative_loss_to_soft_l1
                        <= float(
                            gates[
                                "maximum_noise_only_relative_loss_to_soft_l1"
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
                "outlier_row_count_for_diagnostic_only": int(
                    outlier_rows.size
                ),
                "outlier_rows_i64_sha256_for_diagnostic_only": _sha256(
                    outlier_rows.tobytes(order="C")
                ),
                "candidate_cauchy_fit": _fit_record(candidate),
                "soft_l1_two_matrix_fit": _fit_record(soft_l1),
                "linear_two_matrix_fit": _fit_record(linear),
                "robust_one_matrix_fit": _fit_record(one_matrix),
                "clean_confirmation": {
                    "row_count": int(confirmation_source.shape[0]),
                    "candidate_cauchy": candidate_metrics,
                    "soft_l1_two_matrix": soft_l1_metrics,
                    "linear_two_matrix": linear_metrics,
                    "robust_one_matrix": one_metrics,
                    "candidate_gain_over_linear_two_matrix": gain_over_linear,
                    "candidate_gain_over_soft_l1_two_matrix": (
                        gain_over_soft_l1
                    ),
                    "candidate_gain_over_robust_one_matrix": gain_over_one,
                    "candidate_relative_loss_to_soft_l1_two_matrix": (
                        relative_loss_to_soft_l1
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
        "parent_report_sha256": {
            "an1": config["parents"]["an1"]["report_sha256"],
            "an2": config["parents"]["an2"]["report_sha256"],
        },
        "design_source_f64_sha256": _sha256(source.tobytes(order="C")),
        "development_row_count": int(
            np.count_nonzero(masks["development"])
        ),
        "confirmation_row_count": int(
            np.count_nonzero(masks["confirmation"])
        ),
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
            "advance_redescending_cauchy_paired_recovery"
            if automatic_pass
            else "close_redescending_cauchy_paired_recovery"
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
        / "configs/u5_r2an3_redescending_cauchy_paired_recovery_v1.json",
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
            raise ValueError("AN3 child requires --software-commit")
        report = run_child(config, software_commit=args.software_commit)
        _atomic_write(args.child_output, _canonical_json(report))
        return 0
    if args.output_dir is None:
        raise ValueError("AN3 parent requires --output-dir")
    run_parent(args.config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
