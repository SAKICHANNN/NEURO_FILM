#!/usr/bin/env python
"""Evaluate the frozen shared rank-6 target-density reference basis."""

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
from src.film_physics.spectral_scanner import (
    apply_spectral_scanner,
    synthetic_profile_from_contract,
)
from src.real_film.it8_batch_family import load_charge
from src.real_film.it8_density_basis import (
    fit_density_basis,
    optical_density,
    reconstruct_density,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def build_report(
    contract: dict[str, Any],
    contract_sha256: str,
    charges,
    scanner_contract: dict[str, Any],
) -> dict[str, Any]:
    fit_year_maximum = int(contract["split"]["fit_year_maximum"])
    held_year_minimum = int(contract["split"]["forward_held_year_minimum"])
    fit_indices = np.asarray(
        [
            index
            for index, charge in enumerate(charges)
            if charge.year <= fit_year_maximum
        ],
        dtype=np.int64,
    )
    held_indices = np.asarray(
        [
            index
            for index, charge in enumerate(charges)
            if charge.year >= held_year_minimum
        ],
        dtype=np.int64,
    )
    if len(fit_indices) + len(held_indices) != len(charges):
        raise ValueError("forward split leaves an uncovered calendar year")
    held_counts = {
        family: sum(charges[index].family == family for index in held_indices)
        for family in ("E", "V")
    }
    if len(held_indices) != int(contract["split"]["expected_forward_held_charges"]):
        raise ValueError("forward held charge count mismatch")
    if min(held_counts.values()) != int(
        contract["split"]["expected_forward_held_charges_per_family"]
    ):
        raise ValueError("forward held family count mismatch")

    floor_percent = float(
        contract["representation"]["transmittance_floor_percent"]
    )
    density_rows = []
    floor_count = 0
    for charge in charges:
        density, count = optical_density(
            charge.spectra_pct, floor_percent=floor_percent
        )
        density_rows.append(density)
        floor_count += count
    density = np.stack(density_rows)
    nmf_config = contract["models"]["nmf"]
    fit_rows = density[fit_indices].reshape(-1, density.shape[-1])
    models = {}
    bases = {}
    for rank in [
        *contract["models"]["lower_rank_controls"],
        contract["models"]["candidate_rank"],
    ]:
        basis, model = fit_density_basis(
            fit_rows,
            rank=int(rank),
            init=nmf_config["init"],
            solver=nmf_config["solver"],
            tolerance=float(nmf_config["tolerance"]),
            maximum_iterations=int(nmf_config["maximum_iterations"]),
            random_state=int(nmf_config["random_state"]),
        )
        bases[int(rank)] = basis
        models[int(rank)] = model

    wavelength = np.arange(
        float(scanner_contract["wavelength_grid_nm"]["start"]),
        float(scanner_contract["wavelength_grid_nm"]["stop"]) + 0.5,
        float(scanner_contract["wavelength_grid_nm"]["step"]),
        dtype=np.float64,
    )
    profiles = {
        name: synthetic_profile_from_contract(wavelength, row)
        for name, row in scanner_contract["synthetic_profiles"].items()
    }
    source_start = int(contract["representation"]["wavelength_grid_nm"][0])
    source_step = int(contract["representation"]["wavelength_grid_nm"][2])
    observer_start = int(
        contract["representation"]["scanner_observer_grid_nm"][0]
    )
    observer_stop = int(
        contract["representation"]["scanner_observer_grid_nm"][1]
    )
    observer_slice = slice(
        (observer_start - source_start) // source_step,
        (observer_stop - source_start) // source_step + 1,
    )

    held_rows = []
    for index in held_indices:
        bounded_original = np.power(10.0, -density[index])
        rank_metrics = {}
        for rank, model in models.items():
            reconstructed_density = reconstruct_density(model, density[index])
            reconstructed_transmittance = np.power(
                10.0, -reconstructed_density
            )
            scanner_rmse = {}
            for name, profile in profiles.items():
                original_rgb = apply_spectral_scanner(
                    bounded_original[:, observer_slice], profile
                )
                reconstructed_rgb = apply_spectral_scanner(
                    reconstructed_transmittance[:, observer_slice], profile
                )
                scanner_rmse[name] = float(
                    np.sqrt(np.mean((original_rgb - reconstructed_rgb) ** 2))
                )
            rank_metrics[str(rank)] = {
                "density_rmse": float(
                    np.sqrt(
                        np.mean((density[index] - reconstructed_density) ** 2)
                    )
                ),
                "transmittance_percent_rmse": float(
                    100.0
                    * np.sqrt(
                        np.mean(
                            (
                                bounded_original
                                - reconstructed_transmittance
                            )
                            ** 2
                        )
                    )
                ),
                "scanner_rgb_rmse": scanner_rmse,
            }
        held_rows.append(
            {
                "name": charges[index].name,
                "family": charges[index].family,
                "year": charges[index].year,
                "ranks": rank_metrics,
            }
        )

    def rank_values(rank: int, metric: str) -> list[float]:
        return [float(row["ranks"][str(rank)][metric]) for row in held_rows]

    rank6_density = rank_values(6, "density_rmse")
    rank6_transmittance = rank_values(6, "transmittance_percent_rmse")
    rank5_density = rank_values(5, "density_rmse")
    rank3_density = rank_values(3, "density_rmse")
    scanner_p95 = {
        name: _percentile(
            [
                float(row["ranks"]["6"]["scanner_rgb_rmse"][name])
                for row in held_rows
            ],
            0.95,
        )
        for name in profiles
    }
    rank3_scanner_p95 = {
        name: _percentile(
            [
                float(row["ranks"]["3"]["scanner_rgb_rmse"][name])
                for row in held_rows
            ],
            0.95,
        )
        for name in profiles
    }
    each_family_density_p95 = {
        family: _percentile(
            [
                float(row["ranks"]["6"]["density_rmse"])
                for row in held_rows
                if row["family"] == family
            ],
            0.95,
        )
        for family in ("E", "V")
    }
    rank6_vs_rank5 = (
        np.median(rank5_density) - np.median(rank6_density)
    ) / np.median(rank5_density)
    metrics = {
        "rank6_density_rmse_median": float(np.median(rank6_density)),
        "rank6_density_rmse_p95": _percentile(rank6_density, 0.95),
        "rank6_transmittance_percent_rmse_median": float(
            np.median(rank6_transmittance)
        ),
        "rank6_transmittance_percent_rmse_p95": _percentile(
            rank6_transmittance, 0.95
        ),
        "rank6_scanner_rgb_rmse_p95": scanner_p95,
        "rank6_each_family_density_rmse_p95": each_family_density_p95,
        "rank6_vs_rank5_density_median_relative_improvement": float(
            rank6_vs_rank5
        ),
        "rank3_density_rmse_p95": _percentile(rank3_density, 0.95),
        "rank3_scanner_rgb_rmse_p95": rank3_scanner_p95,
    }
    gates = contract["automatic_gates"]
    checks = {
        "all_models_converge": all(basis.converged for basis in bases.values()),
        "rank6_density_median": metrics["rank6_density_rmse_median"]
        <= float(gates["rank6_density_rmse_median_maximum"]),
        "rank6_density_p95": metrics["rank6_density_rmse_p95"]
        <= float(gates["rank6_density_rmse_p95_maximum"]),
        "rank6_transmittance_median": metrics[
            "rank6_transmittance_percent_rmse_median"
        ]
        <= float(
            gates["rank6_transmittance_percent_rmse_median_maximum"]
        ),
        "rank6_transmittance_p95": metrics[
            "rank6_transmittance_percent_rmse_p95"
        ]
        <= float(gates["rank6_transmittance_percent_rmse_p95_maximum"]),
        "rank6_scanner_a": scanner_p95["scanner_a"]
        <= float(gates["rank6_scanner_a_rgb_rmse_p95_maximum"]),
        "rank6_scanner_b": scanner_p95["scanner_b"]
        <= float(gates["rank6_scanner_b_rgb_rmse_p95_maximum"]),
        "rank6_each_family": max(each_family_density_p95.values())
        <= float(gates["rank6_each_family_density_rmse_p95_maximum"]),
        "rank6_vs_rank5": rank6_vs_rank5
        >= float(
            gates[
                "rank6_vs_rank5_density_median_relative_improvement_minimum"
            ]
        ),
        "rank3_density_negative": metrics["rank3_density_rmse_p95"]
        >= float(gates["rank3_negative_density_rmse_p95_minimum"]),
        "rank3_scanner_negative": max(rank3_scanner_p95.values())
        >= float(gates["rank3_negative_scanner_rgb_rmse_p95_minimum"]),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    passed = all(checks.values())
    basis_records = {
        str(rank): {
            "rank": rank,
            "iterations": basis.iterations,
            "converged": basis.converged,
            "fit_reconstruction_error": basis.reconstruction_error,
            "components_sha256": basis.components_sha256,
        }
        for rank, basis in sorted(bases.items())
    }
    stable_core = {
        "contract_sha256": contract_sha256,
        "fit_names": [charges[index].name for index in fit_indices],
        "held_names": [charges[index].name for index in held_indices],
        "held_family_counts": held_counts,
        "density_floor_count": floor_count,
        "density_value_count": int(density.size),
        "basis_records": basis_records,
        "held_rows": held_rows,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": passed,
    }
    stable_bytes = json.dumps(
        stable_core, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "schema_version": 1,
        "schema": "neuro_film.u6_p1d_shared_density_basis_fidelity_report.v1",
        "experiment_id": contract["experiment_id"],
        "node": contract["node"],
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "stable_evidence_id": _sha256(stable_bytes),
        **stable_core,
        "decision": (
            "retain_shared_rank6_internal_reference_primitive"
            if passed
            else "close_shared_rank6_reference_primitive"
        ),
        "branch": contract["branch_rules"]["pass" if passed else "fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p1d_shared_density_basis_fidelity_v1.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract_bytes = args.config.read_bytes()
    contract = json.loads(contract_bytes)
    for path_key, sha_key in (
        ("parent_decision", "parent_decision_sha256"),
        ("scanner_reference_contract", "scanner_reference_contract_sha256"),
        ("source_config", "source_config_sha256"),
    ):
        if _sha256_file(ROOT / contract[path_key]) != contract[sha_key]:
            raise ValueError(f"{path_key} hash mismatch")
    source_config = json.loads((ROOT / contract["source_config"]).read_bytes())
    scanner_contract = json.loads(
        (ROOT / contract["scanner_reference_contract"]).read_bytes()
    )
    assets = _assets(source_config)
    data_root = ROOT / source_config["data_root"]
    charges = [
        load_charge(data_root / asset["path"], asset["family_code"])
        for asset in assets
    ]
    report = build_report(
        contract, _sha256(contract_bytes), charges, scanner_contract
    )
    output = args.output or ROOT / source_config["output_root"] / "shared_basis.json"
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
                "automatic_pass": report["automatic_pass"],
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
