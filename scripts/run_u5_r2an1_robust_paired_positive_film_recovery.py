#!/usr/bin/env python
"""Run the frozen U5.R2AN1 robust paired-recovery experiment."""

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
    _canonical_json,
    _error_metrics,
    _fit_record,
    _structure_metrics,
    _atomic_write,
    generate_paired_design,
    split_masks,
)
from src.roll2film.positive_film import (  # noqa: E402
    positive_film_operator_from_config,
)
from src.roll2film.positive_film_fitting import (  # noqa: E402
    fit_positive_film_response_operator,
)


CONFIG_SHA256 = "d5b5658586661351d7ca8c803858a667446dcb03f7a9541c785cea1211873975"
REPORT_SCHEMA = "neuro-film.u5.r2an1.robust-paired-recovery-report.v1"
REPEAT_SCHEMA = "neuro-film.u5.r2an1.repeat-decision.v1"


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
            raise RuntimeError("AN1 requires a clean tracked worktree")


def load_config(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AN1 config hash mismatch")
    config = json.loads(raw)
    if (
        config["experiment_id"]
        != "u5.r2an1-robust-paired-positive-film-recovery-v1"
        or config["parent"]["report_sha256"]
        != "22fc3542ed2b7a5304ac558c761683617414c4fed0af103b90789d35aedea36d"
        or not config["parent"]["automatic_pass"]
        or config["real_film_pixel_allowed"]
        or config["production_integration_allowed"]
        or config["stock_identity_claim_allowed"]
    ):
        raise ValueError("AN1 frozen contract mismatch")
    return config


def perturb_development_pairs(
    source: np.ndarray,
    target: np.ndarray,
    scenario: dict[str, Any],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply deterministic independent patch-mean noise and sparse outliers."""

    source_values = np.asarray(source, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if (
        source_values.ndim != 2
        or source_values.shape[1] != 3
        or target_values.shape != source_values.shape
        or not np.all(np.isfinite(source_values))
        or not np.all(np.isfinite(target_values))
    ):
        raise ValueError("AN1 perturbation requires finite paired RGB rows")
    rng = np.random.default_rng(seed)
    noisy_source = np.clip(
        source_values
        + rng.normal(
            0.0,
            float(scenario["source_noise_standard_deviation"]),
            source_values.shape,
        ),
        0.0,
        1.0,
    )
    noisy_target = np.clip(
        target_values
        + rng.normal(
            0.0,
            float(scenario["target_noise_standard_deviation"]),
            target_values.shape,
        ),
        0.0,
        1.0,
    )
    outlier_count = int(
        round(
            float(scenario["target_outlier_row_fraction"])
            * target_values.shape[0]
        )
    )
    if outlier_count:
        outlier_rows = np.sort(
            rng.choice(target_values.shape[0], size=outlier_count, replace=False)
        )
        noisy_target[outlier_rows] = np.clip(
            noisy_target[outlier_rows]
            + rng.uniform(
                -float(scenario["target_outlier_uniform_magnitude"]),
                float(scenario["target_outlier_uniform_magnitude"]),
                size=(outlier_count, 3),
            ),
            0.0,
            1.0,
        )
    else:
        outlier_rows = np.empty(0, dtype=np.int64)
    return noisy_source, noisy_target, outlier_rows


def run_child(
    config: dict[str, Any],
    *,
    software_commit: str,
) -> dict[str, Any]:
    parent_path = ROOT / str(config["parent"]["config_path"])
    parent_raw = parent_path.read_bytes()
    if _sha256(parent_raw) != config["parent"]["config_sha256"]:
        raise ValueError("AN1 parent config hash mismatch")
    parent = json.loads(parent_raw)
    source, patch, illuminant, exposure = generate_paired_design(parent)
    masks = split_masks(parent, patch, illuminant, exposure)
    truth_raw = (ROOT / str(parent["truth_operator_config"])).read_bytes()
    if _sha256(truth_raw) != parent["truth_operator_config_sha256"]:
        raise ValueError("AN1 truth config hash mismatch")
    truths = json.loads(truth_raw)
    source_before = source.copy()
    fit = config["fit"]
    common_options = {
        "identity_mixture": float(parent["models"]["matrix_identity_mixture"]),
        "restart_count": int(fit["restart_count"]),
        "maximum_function_evaluations": int(
            fit["maximum_function_evaluations"]
        ),
        "function_tolerance": float(fit["function_tolerance"]),
        "parameter_tolerance": float(fit["parameter_tolerance"]),
        "gradient_tolerance": float(fit["gradient_tolerance"]),
        "seed": int(parent["synthetic_design"]["base_patch_seed"]),
    }
    development_source = source[masks["development"]]
    confirmation_source = source[masks["confirmation"]]
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
            robust = fit_positive_film_response_operator(
                noisy_source,
                noisy_target,
                model="two_matrix",
                loss=str(fit["robust_loss"]),
                loss_scale=float(fit["robust_loss_scale"]),
                **common_options,
            )
            linear = fit_positive_film_response_operator(
                noisy_source,
                noisy_target,
                model="two_matrix",
                loss=str(fit["linear_loss"]),
                loss_scale=float(fit["robust_loss_scale"]),
                **common_options,
            )
            one_matrix = fit_positive_film_response_operator(
                noisy_source,
                noisy_target,
                model="one_matrix",
                loss=str(fit["robust_loss"]),
                loss_scale=float(fit["robust_loss_scale"]),
                **common_options,
            )
            robust_metrics = _error_metrics(
                robust.operator, confirmation_source, confirmation_target
            )
            linear_metrics = _error_metrics(
                linear.operator, confirmation_source, confirmation_target
            )
            one_matrix_metrics = _error_metrics(
                one_matrix.operator, confirmation_source, confirmation_target
            )
            gain_over_linear = 1.0 - (
                robust_metrics["rgb_rmse"] / linear_metrics["rgb_rmse"]
            )
            gain_over_one = 1.0 - (
                robust_metrics["rgb_rmse"] / one_matrix_metrics["rgb_rmse"]
            )
            relative_loss_to_linear = (
                robust_metrics["rgb_rmse"] / linear_metrics["rgb_rmse"]
            ) - 1.0
            structure = _structure_metrics(robust.operator, parent)
            checks = [
                {
                    "name": f"{witness_name}.{scenario_name}.robust_rmse",
                    "passed": robust_metrics["rgb_rmse"]
                    <= float(gates["robust_confirmation_rgb_rmse_maximum"]),
                },
                {
                    "name": f"{witness_name}.{scenario_name}.robust_max_error",
                    "passed": robust_metrics["maximum_absolute_error"]
                    <= float(
                        gates["robust_confirmation_maximum_absolute_error"]
                    ),
                },
                {
                    "name": f"{witness_name}.{scenario_name}.gain_over_one_matrix",
                    "passed": gain_over_one
                    >= float(gates["minimum_gain_over_robust_one_matrix"]),
                },
                {
                    "name": f"{witness_name}.{scenario_name}.positive_jacobian",
                    "passed": structure["minimum_jacobian_determinant"] > 0.0,
                },
            ]
            if scenario_name == "sparse_correspondence_outliers":
                checks.append(
                    {
                        "name": f"{witness_name}.{scenario_name}.gain_over_linear",
                        "passed": gain_over_linear
                        >= float(
                            gates[
                                "minimum_outlier_gain_over_linear_two_matrix"
                            ]
                        ),
                    }
                )
            else:
                checks.append(
                    {
                        "name": f"{witness_name}.{scenario_name}.relative_loss_to_linear",
                        "passed": relative_loss_to_linear
                        <= float(
                            gates["maximum_noise_only_relative_loss_to_linear"]
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
                "outlier_row_count": int(outlier_rows.size),
                "outlier_rows_i64_sha256": _sha256(
                    outlier_rows.tobytes(order="C")
                ),
                "robust_two_matrix_fit": _fit_record(robust),
                "linear_two_matrix_fit": _fit_record(linear),
                "robust_one_matrix_fit": _fit_record(one_matrix),
                "clean_confirmation": {
                    "row_count": int(confirmation_source.shape[0]),
                    "robust_two_matrix": robust_metrics,
                    "linear_two_matrix": linear_metrics,
                    "robust_one_matrix": one_matrix_metrics,
                    "robust_gain_over_linear_two_matrix": gain_over_linear,
                    "robust_gain_over_one_matrix": gain_over_one,
                    "robust_relative_loss_to_linear_two_matrix": (
                        relative_loss_to_linear
                    ),
                },
                "robust_structure": structure,
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
        "development_row_count": int(np.count_nonzero(masks["development"])),
        "confirmation_row_count": int(np.count_nonzero(masks["confirmation"])),
        "witnesses": witness_records,
        "automatic_checks": all_checks,
        "automatic_pass": all(bool(check["passed"]) for check in all_checks),
        "claim_ceiling": config["claim_ceiling"],
    }


def _child_command(
    config_path: Path,
    output_path: Path,
    software_commit: str,
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
            "advance_robust_paired_positive_film_recovery"
            if automatic_pass
            else "close_or_diagnose_robust_paired_positive_film_recovery"
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
        / "configs/u5_r2an1_robust_paired_positive_film_recovery_v1.json",
    )
    parser.add_argument(
        "--expected-config-sha256",
        default=CONFIG_SHA256,
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
            raise ValueError("AN1 child requires --software-commit")
        report = run_child(config, software_commit=args.software_commit)
        _atomic_write(args.child_output, _canonical_json(report))
        return 0
    if args.output_dir is None:
        raise ValueError("AN1 parent requires --output-dir")
    run_parent(args.config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
