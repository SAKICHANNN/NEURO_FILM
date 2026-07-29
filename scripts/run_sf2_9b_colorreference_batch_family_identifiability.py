#!/usr/bin/env python
"""Run the frozen SF2.9B batch-held-out target-family pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_sf2_9b_colorreference_batch_family_source import _assets
from src.real_film.it8_batch_family import (
    exact_stratified_fixed_score_p,
    evaluate_fixed_split,
    load_charge,
    make_family_pipeline,
    residual_features,
    select_c_leave_one_year_out,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nuisance_pipeline(c_value: float, maximum_iterations: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=maximum_iterations,
                    random_state=0,
                ),
            ),
        ]
    )


def _fit_view(
    *,
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    years: np.ndarray,
    development: np.ndarray,
    confirmatory: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    model_config = config["family_model"]
    c_grid = [float(value) for value in model_config["c_grid"]]
    maximum_iterations = int(model_config["maximum_iterations"])
    factory = lambda c: make_family_pipeline(c, maximum_iterations)
    chosen_c, development_scores = select_c_leave_one_year_out(
        factory,
        x[development],
        y[development],
        years[development],
        c_grid,
    )
    model = factory(chosen_c)
    result = evaluate_fixed_split(model, x, y, development, confirmatory)
    fitted = clone(model).fit(x[development], y[development])
    pca = fitted.named_steps["pca"]
    p_value, assignments = exact_stratified_fixed_score_p(
        np.asarray(result["scores"]), np.asarray(result["truth"])
    )
    result.update(
        {
            "view": name,
            "selected_c": chosen_c,
            "development_leave_one_year_out_balanced_accuracy": development_scores,
            "pca_components": int(pca.n_components_),
            "pca_retained_variance_ratio": pca.retained_variance_ratio_,
            "exact_fixed_score_stratified_label_permutation_p": p_value,
            "exact_permutation_assignments": assignments,
        }
    )
    return result


def _fit_nuisance(
    *,
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    years: np.ndarray,
    development: np.ndarray,
    confirmatory: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    model_config = config["family_model"]
    c_grid = [float(value) for value in model_config["c_grid"]]
    maximum_iterations = int(model_config["maximum_iterations"])
    factory = lambda c: _nuisance_pipeline(c, maximum_iterations)
    chosen_c, development_scores = select_c_leave_one_year_out(
        factory,
        x[development],
        y[development],
        years[development],
        c_grid,
    )
    result = evaluate_fixed_split(
        factory(chosen_c), x, y, development, confirmatory
    )
    result.update(
        {
            "view": name,
            "selected_c": chosen_c,
            "development_leave_one_year_out_balanced_accuracy": development_scores,
        }
    )
    return result


def build_report(
    config: dict[str, Any],
    config_sha256: str,
    source_config: dict[str, Any],
    charges,
) -> dict[str, Any]:
    years = np.asarray([charge.year for charge in charges], dtype=np.int64)
    y = np.asarray(
        [1 if charge.family == "V" else 0 for charge in charges],
        dtype=np.int64,
    )
    confirm_digits = set(
        int(value)
        for value in config["split"]["confirmatory_year_last_digits"]
    )
    confirmatory = np.asarray(
        [index for index, year in enumerate(years) if year % 10 in confirm_digits],
        dtype=np.int64,
    )
    development = np.asarray(
        [index for index, year in enumerate(years) if year % 10 not in confirm_digits],
        dtype=np.int64,
    )
    expected_dev_digits = set(
        int(value)
        for value in config["split"]["development_year_last_digits"]
    )
    if expected_dev_digits | confirm_digits != set(range(10)):
        raise ValueError("development/confirmatory year rules do not partition digits")
    if expected_dev_digits & confirm_digits:
        raise ValueError("development/confirmatory year rules overlap")
    names = [charge.name for charge in charges]
    sample_ids = charges[0].sample_ids
    if any(charge.sample_ids != sample_ids for charge in charges):
        raise ValueError("charge sample identities differ")

    spectral_x = residual_features(
        charges, development, view="spectral"
    )
    lab_x = residual_features(charges, development, view="lab")
    spectral = _fit_view(
        name="spectral_primary",
        x=spectral_x,
        y=y,
        years=years,
        development=development,
        confirmatory=confirmatory,
        config=config,
    )
    lab = _fit_view(
        name="lab_secondary",
        x=lab_x,
        y=y,
        years=years,
        development=development,
        confirmatory=confirmatory,
        config=config,
    )
    centered_year = years.astype(np.float64) - float(
        np.mean(years[development])
    )
    year_x = np.column_stack((centered_year, centered_year**2))
    batch_error_x = np.asarray(
        [
            [charge.mean_de_median, charge.mean_de_p90]
            for charge in charges
        ],
        dtype=np.float64,
    )
    year_control = _fit_nuisance(
        name="calendar_year_only",
        x=year_x,
        y=y,
        years=years,
        development=development,
        confirmatory=confirmatory,
        config=config,
    )
    batch_control = _fit_nuisance(
        name="batch_error_only",
        x=batch_error_x,
        y=y,
        years=years,
        development=development,
        confirmatory=confirmatory,
        config=config,
    )
    best_nuisance = max(
        0.5,
        float(year_control["balanced_accuracy"]),
        float(batch_control["balanced_accuracy"]),
    )
    spectral_delta = float(spectral["balanced_accuracy"]) - best_nuisance
    confirm_counts = {
        family: int(
            np.sum(y[confirmatory] == (1 if family == "V" else 0))
        )
        for family in ("E", "V")
    }
    gates = config["gates"]
    checks = {
        "minimum_confirmatory_charges_per_family": min(confirm_counts.values())
        >= int(config["split"]["minimum_confirmatory_charges_per_family"]),
        "spectral_balanced_accuracy": float(spectral["balanced_accuracy"])
        >= float(gates["spectral_balanced_accuracy_minimum"]),
        "spectral_roc_auc": float(spectral["roc_auc"])
        >= float(gates["spectral_roc_auc_minimum"]),
        "spectral_minus_best_nuisance": spectral_delta
        >= float(gates["spectral_minus_best_nuisance_minimum"]),
        "spectral_exact_permutation_p": float(
            spectral["exact_fixed_score_stratified_label_permutation_p"]
        )
        <= float(gates["spectral_exact_permutation_p_maximum"]),
        "lab_balanced_accuracy": float(lab["balanced_accuracy"])
        >= float(gates["lab_balanced_accuracy_minimum"]),
    }
    passed = all(checks.values())
    stable_evidence = {
        "config_sha256": config_sha256,
        "charge_names": names,
        "development_names": [names[index] for index in development],
        "confirmatory_names": [names[index] for index in confirmatory],
        "confirmatory_counts": confirm_counts,
        "spectral": spectral,
        "lab": lab,
        "year_control": year_control,
        "batch_error_control": batch_control,
        "best_nuisance_balanced_accuracy": best_nuisance,
        "spectral_minus_best_nuisance_balanced_accuracy": spectral_delta,
        "checks": checks,
    }
    stable_bytes = json.dumps(
        stable_evidence, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": config_sha256,
        "source_experiment_id": source_config["experiment_id"],
        "source_report_sha256": config["parent_source_report_sha256"],
        "source_stable_evidence_id": config["parent_stable_evidence_id"],
        "stable_evidence_id": _sha256(stable_bytes),
        **stable_evidence,
        "automatic_identifiability_gate_passed": passed,
        "decision": (
            "retain_cross_charge_target_family_physical_signal_only"
            if passed
            else "close_target_family_identifiability_under_frozen_pilot"
        ),
        "operator_fitting_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/sf2_9b_colorreference_batch_family_identifiability_v1.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    source_report = ROOT / config["parent_source_report"]
    if _sha256_file(source_report) != config["parent_source_report_sha256"]:
        raise ValueError("parent source report hash mismatch")
    parent_report = json.loads(source_report.read_bytes())
    if parent_report["stable_evidence_id"] != config["parent_stable_evidence_id"]:
        raise ValueError("parent source stable evidence identity mismatch")
    source_config_path = ROOT / config["parent_config"]
    source_config = json.loads(source_config_path.read_bytes())
    assets = _assets(source_config)
    data_root = ROOT / source_config["data_root"]
    charges = [
        load_charge(data_root / asset["path"], asset["family_code"])
        for asset in assets
    ]
    report = build_report(
        config, _sha256(config_bytes), source_config, charges
    )
    output = args.output or ROOT / source_config["output_root"] / "pilot.json"
    if not output.is_absolute():
        output = ROOT / output
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256(encoded),
                "stable_evidence_id": report["stable_evidence_id"],
                "automatic_identifiability_gate_passed": report[
                    "automatic_identifiability_gate_passed"
                ],
                "decision": report["decision"],
                "spectral_balanced_accuracy": report["spectral"][
                    "balanced_accuracy"
                ],
                "lab_balanced_accuracy": report["lab"]["balanced_accuracy"],
                "best_nuisance_balanced_accuracy": report[
                    "best_nuisance_balanced_accuracy"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
