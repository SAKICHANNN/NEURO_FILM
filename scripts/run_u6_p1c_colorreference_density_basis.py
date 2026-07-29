#!/usr/bin/env python
"""Evaluate frozen shared and family target-density spectral bases."""

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

from scripts.run_sf2_9b_colorreference_batch_family_source import _assets
from src.real_film.it8_batch_family import load_charge
from src.real_film.it8_density_basis import (
    FittedDensityBasis,
    fit_density_basis,
    optical_density,
    reconstruction_rmse,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _basis_record(basis: FittedDensityBasis) -> dict[str, Any]:
    return {
        "rank": basis.rank,
        "iterations": basis.iterations,
        "maximum_iterations": basis.maximum_iterations,
        "converged": basis.converged,
        "fit_reconstruction_error": basis.reconstruction_error,
        "components_sha256": basis.components_sha256,
        "components": basis.components.tolist(),
    }


def _relative_improvement(candidate: float, baseline: float) -> float:
    if baseline <= 0.0:
        raise ValueError("baseline RMSE must be positive")
    return (baseline - candidate) / baseline


def build_report(
    contract: dict[str, Any],
    contract_sha256: str,
    charges,
    source_config: dict[str, Any],
) -> dict[str, Any]:
    years = np.asarray([charge.year for charge in charges], dtype=np.int64)
    families = np.asarray([charge.family for charge in charges])
    fit_digits = set(int(value) for value in contract["split"]["fit_year_last_digits"])
    held_digits = set(
        int(value)
        for value in contract["split"]["held_charge_year_last_digits"]
    )
    if fit_digits | held_digits != set(range(10)) or fit_digits & held_digits:
        raise ValueError("fit/held year rules do not partition final digits")
    fit_indices = np.asarray(
        [index for index, year in enumerate(years) if year % 10 in fit_digits],
        dtype=np.int64,
    )
    held_indices = np.asarray(
        [index for index, year in enumerate(years) if year % 10 in held_digits],
        dtype=np.int64,
    )
    floor_percent = float(
        contract["representation"]["transmittance_floor_percent"]
    )
    density_rows = []
    floor_count = 0
    for charge in charges:
        density, charge_floor_count = optical_density(
            charge.spectra_pct, floor_percent=floor_percent
        )
        density_rows.append(density)
        floor_count += charge_floor_count
    density = np.stack(density_rows)
    nmf = contract["models"]["nmf"]
    fit_kwargs = {
        "init": nmf["init"],
        "solver": nmf["solver"],
        "tolerance": float(nmf["tolerance"]),
        "maximum_iterations": int(nmf["maximum_iterations"]),
        "random_state": int(nmf["random_state"]),
    }

    def fit_model(indices: np.ndarray, rank: int):
        rows = density[indices].reshape(-1, density.shape[-1])
        return fit_density_basis(rows, rank=rank, **fit_kwargs)

    shared3_basis, shared3_model = fit_model(fit_indices, 3)
    shared6_basis, shared6_model = fit_model(fit_indices, 6)
    family_models = {}
    family_bases = {}
    for family in ("E", "V"):
        indices = fit_indices[families[fit_indices] == family]
        basis, model = fit_model(indices, 3)
        family_bases[family] = basis
        family_models[family] = model

    held_rows = []
    for index in held_indices:
        family = str(families[index])
        wrong_family = "V" if family == "E" else "E"
        rows = density[index]
        correct = reconstruction_rmse(family_models[family], rows)
        wrong = reconstruction_rmse(family_models[wrong_family], rows)
        shared3 = reconstruction_rmse(shared3_model, rows)
        shared6 = reconstruction_rmse(shared6_model, rows)
        held_rows.append(
            {
                "name": charges[index].name,
                "family": family,
                "year": int(years[index]),
                "correct_family_rank3_rmse": correct,
                "wrong_family_rank3_rmse": wrong,
                "shared_rank3_rmse": shared3,
                "shared_rank6_rmse": shared6,
                "correct_vs_shared_rank3_relative_improvement": (
                    _relative_improvement(correct, shared3)
                ),
                "correct_vs_wrong_rank3_relative_improvement": (
                    _relative_improvement(correct, wrong)
                ),
                "correct_vs_shared_rank6_relative_improvement": (
                    _relative_improvement(correct, shared6)
                ),
            }
        )

    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in held_rows])

    shared3_improvement = values(
        "correct_vs_shared_rank3_relative_improvement"
    )
    wrong_improvement = values(
        "correct_vs_wrong_rank3_relative_improvement"
    )
    shared6_improvement = values(
        "correct_vs_shared_rank6_relative_improvement"
    )
    structural_wins = int(np.count_nonzero(shared3_improvement > 0.0))
    wrong_wins = int(np.count_nonzero(wrong_improvement > 0.0))
    utility_wins = int(np.count_nonzero(shared6_improvement > 0.0))
    family_win_fraction = {
        family: float(
            np.mean(
                shared3_improvement[
                    np.asarray([row["family"] == family for row in held_rows])
                ]
                > 0.0
            )
        )
        for family in ("E", "V")
    }
    basis_records = {
        "shared_rank3": _basis_record(shared3_basis),
        "shared_rank6": _basis_record(shared6_basis),
        "family_E_rank3": _basis_record(family_bases["E"]),
        "family_V_rank3": _basis_record(family_bases["V"]),
    }
    gates = contract["held_charge_gates"]
    checks = {
        "all_models_converge": all(
            record["converged"] for record in basis_records.values()
        ),
        "structural_correct_vs_shared_rank3_wins": structural_wins
        >= int(gates["structural_correct_vs_shared_rank3_minimum_wins"]),
        "structural_correct_vs_shared_rank3_median": float(
            np.median(shared3_improvement)
        )
        >= float(
            gates[
                "structural_correct_vs_shared_rank3_median_relative_improvement_minimum"
            ]
        ),
        "negative_correct_vs_wrong_rank3_wins": wrong_wins
        >= int(gates["negative_correct_vs_wrong_rank3_minimum_wins"]),
        "negative_correct_vs_wrong_rank3_median": float(
            np.median(wrong_improvement)
        )
        >= float(
            gates[
                "negative_correct_vs_wrong_rank3_median_relative_improvement_minimum"
            ]
        ),
        "each_family_correct_vs_shared_rank3": min(
            family_win_fraction.values()
        )
        >= float(
            gates[
                "each_family_correct_vs_shared_rank3_minimum_win_fraction"
            ]
        ),
        "utility_correct_vs_shared_rank6_wins": utility_wins
        >= int(gates["utility_correct_vs_shared_rank6_minimum_wins"]),
        "utility_correct_vs_shared_rank6_median": float(
            np.median(shared6_improvement)
        )
        >= float(
            gates[
                "utility_correct_vs_shared_rank6_median_relative_improvement_minimum"
            ]
        ),
    }
    structural_keys = [
        key
        for key in checks
        if key.startswith("structural_")
        or key.startswith("negative_")
        or key.startswith("each_family_")
        or key == "all_models_converge"
    ]
    utility_keys = [key for key in checks if key.startswith("utility_")]
    structural_pass = all(checks[key] for key in structural_keys)
    utility_pass = all(checks[key] for key in utility_keys)
    if structural_pass and utility_pass:
        decision = "retain_two_family_rank3_basis_bank_for_reference_research"
        branch = contract["branch_rules"]["structural_and_utility_pass"]
    elif structural_pass:
        decision = "retain_family_structure_but_prefer_shared_rank6"
        branch = contract["branch_rules"]["structural_pass_utility_fail"]
    else:
        decision = "close_family_specific_density_bases"
        branch = contract["branch_rules"]["structural_fail"]
    stable_core = {
        "contract_sha256": contract_sha256,
        "fit_names": [charges[index].name for index in fit_indices],
        "held_names": [charges[index].name for index in held_indices],
        "density_floor_count": floor_count,
        "density_value_count": int(density.size),
        "basis_records": basis_records,
        "held_rows": held_rows,
        "metrics": {
            "structural_correct_vs_shared_rank3_wins": structural_wins,
            "structural_correct_vs_shared_rank3_median_relative_improvement": float(
                np.median(shared3_improvement)
            ),
            "negative_correct_vs_wrong_rank3_wins": wrong_wins,
            "negative_correct_vs_wrong_rank3_median_relative_improvement": float(
                np.median(wrong_improvement)
            ),
            "utility_correct_vs_shared_rank6_wins": utility_wins,
            "utility_correct_vs_shared_rank6_median_relative_improvement": float(
                np.median(shared6_improvement)
            ),
            "family_structural_win_fraction": family_win_fraction,
        },
        "checks": checks,
        "structural_pass": structural_pass,
        "utility_pass": utility_pass,
        "decision": decision,
    }
    stable_bytes = json.dumps(
        stable_core, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "schema_version": 1,
        "schema": "neuro_film.u6_p1c_colorreference_density_basis_report.v1",
        "experiment_id": contract["experiment_id"],
        "node": contract["node"],
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_experiment_id": source_config["experiment_id"],
        "parent_pilot_report_sha256": contract["parent_pilot_report_sha256"],
        "stable_evidence_id": _sha256(stable_bytes),
        **stable_core,
        "branch": branch,
        "operator_fitting_allowed": False,
        "claim_ceiling": contract["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p1c_colorreference_density_basis_v1.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract_bytes = args.config.read_bytes()
    contract = json.loads(contract_bytes)
    parent_report = ROOT / contract["parent_pilot_report"]
    if _sha256_file(parent_report) != contract["parent_pilot_report_sha256"]:
        raise ValueError("parent pilot report hash mismatch")
    parent = json.loads(parent_report.read_bytes())
    if parent["stable_evidence_id"] != contract["parent_pilot_stable_evidence_id"]:
        raise ValueError("parent pilot stable evidence mismatch")
    parent_decision_path = ROOT / contract["parent_decision"]
    if _sha256_file(parent_decision_path) != contract["parent_decision_sha256"]:
        raise ValueError("parent decision hash mismatch")
    parent_decision = json.loads(parent_decision_path.read_bytes())
    source_config_path = (
        ROOT / "configs/sf2_9b_colorreference_batch_family_v1.json"
    )
    if _sha256_file(source_config_path) != contract["source_config_sha256"]:
        raise ValueError("source config hash mismatch")
    source_config = json.loads(source_config_path.read_bytes())
    assets = _assets(source_config)
    data_root = ROOT / source_config["data_root"]
    charges = [
        load_charge(data_root / asset["path"], asset["family_code"])
        for asset in assets
    ]
    if parent_decision["source"]["archives"] != len(charges):
        raise ValueError("parent decision archive count mismatch")
    report = build_report(
        contract, _sha256(contract_bytes), charges, source_config
    )
    output = args.output or ROOT / source_config["output_root"] / "density_basis.json"
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
                "structural_pass": report["structural_pass"],
                "utility_pass": report["utility_pass"],
                "decision": report["decision"],
                **report["metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
