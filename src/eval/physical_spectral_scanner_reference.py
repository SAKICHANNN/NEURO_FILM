"""U6.P6D synthetic spectral-scanner reference and metamer witness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.spectral_scanner import (
    apply_spectral_scanner,
    construct_primary_metamer_pair,
    synthetic_profile_from_contract,
)


SCHEMA = "neuro_film.u6_p6d_spectral_scanner_reference_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P6D contract")
    return payload


def _stable_id(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()


def _wavelength_grid(row: dict[str, Any]) -> np.ndarray:
    start = float(row["start"])
    stop = float(row["stop"])
    step = float(row["step"])
    if step <= 0.0 or stop <= start:
        raise ValueError("invalid wavelength grid")
    count = int(round((stop - start) / step))
    grid = start + np.arange(count + 1, dtype=np.float64) * step
    if abs(float(grid[-1]) - stop) > 1e-12:
        raise ValueError("wavelength stop must be exactly reachable")
    return grid


def evaluate_spectral_scanner_reference(
    contract: dict[str, Any],
) -> dict[str, Any]:
    wavelength = _wavelength_grid(contract["wavelength_grid_nm"])
    profiles = {
        name: synthetic_profile_from_contract(wavelength, row)
        for name, row in contract["synthetic_profiles"].items()
    }
    scanner_a = profiles["scanner_a"]
    scanner_b = profiles["scanner_b"]
    witnesses = contract["synthetic_witnesses"]
    gates = contract["automatic_gates"]

    clear = np.full(
        wavelength.shape, float(witnesses["clear_transmittance"]), np.float64
    )
    neutral = np.full(
        wavelength.shape, float(witnesses["neutral_transmittance"]), np.float64
    )
    clear_responses = {
        name: apply_spectral_scanner(clear, profile)
        for name, profile in profiles.items()
    }
    neutral_responses = {
        name: apply_spectral_scanner(neutral, profile)
        for name, profile in profiles.items()
    }
    wedge_density = np.asarray(witnesses["density_wedge"], dtype=np.float64)
    wedge = np.power(10.0, -wedge_density[:, None]) * np.ones_like(
        wavelength[None, :]
    )
    wedge_outputs = {
        name: apply_spectral_scanner(wedge, profile)
        for name, profile in profiles.items()
    }

    metamer_first, metamer_second = construct_primary_metamer_pair(
        scanner_a,
        scanner_b,
        neutral_transmittance=float(witnesses["neutral_transmittance"]),
        perturbation_max_abs=float(
            witnesses["metamer_perturbation_max_abs"]
        ),
    )
    a_first = apply_spectral_scanner(metamer_first, scanner_a)
    a_second = apply_spectral_scanner(metamer_second, scanner_a)
    b_first = apply_spectral_scanner(metamer_first, scanner_b)
    b_second = apply_spectral_scanner(metamer_second, scanner_b)
    a_difference = float(np.max(np.abs(a_first - a_second)))
    b_difference = float(np.linalg.norm(b_first - b_second))
    irreducible = b_difference * 0.5

    all_outputs = np.concatenate(
        [
            *(np.ravel(value) for value in clear_responses.values()),
            *(np.ravel(value) for value in neutral_responses.values()),
            *(np.ravel(value) for value in wedge_outputs.values()),
            a_first,
            a_second,
            b_first,
            b_second,
        ]
    )
    cross_response = float(
        max(
            np.max(
                np.abs(
                    apply_spectral_scanner(metamer_first, scanner_a)
                    - apply_spectral_scanner(metamer_first, scanner_b)
                )
            ),
            np.max(
                np.abs(
                    apply_spectral_scanner(metamer_second, scanner_a)
                    - apply_spectral_scanner(metamer_second, scanner_b)
                )
            ),
        )
    )
    identical_control = float(
        np.max(
            np.abs(
                apply_spectral_scanner(metamer_first, scanner_a)
                - apply_spectral_scanner(metamer_first.copy(), scanner_a)
            )
        )
    )
    same_observer_first, same_observer_second = construct_primary_metamer_pair(
        scanner_a,
        scanner_b,
        neutral_transmittance=float(witnesses["neutral_transmittance"]),
        perturbation_max_abs=float(
            witnesses["metamer_perturbation_max_abs"]
        ),
    )
    same_observer_difference = float(
        np.linalg.norm(
            apply_spectral_scanner(same_observer_first, scanner_a)
            - apply_spectral_scanner(same_observer_second, scanner_a)
        )
    )

    metrics = {
        "wavelength_count": int(wavelength.size),
        "wavelength_min_nm": float(wavelength[0]),
        "wavelength_max_nm": float(wavelength[-1]),
        "profile_sha256": {
            name: profile.profile_sha256 for name, profile in profiles.items()
        },
        "clear_response_max_abs_error": float(
            max(np.max(np.abs(value - 1.0)) for value in clear_responses.values())
        ),
        "neutral_channel_spread_max": float(
            max(np.ptp(value) for value in neutral_responses.values())
        ),
        "output_minimum": float(np.min(all_outputs)),
        "output_maximum": float(np.max(all_outputs)),
        "density_wedge_maximum_positive_step": float(
            max(np.max(np.diff(value, axis=0)) for value in wedge_outputs.values())
        ),
        "scanner_a_metamer_max_abs_difference": a_difference,
        "scanner_b_metamer_l2_difference": b_difference,
        "rgb_only_irreducible_l2_error": irreducible,
        "scanner_profile_cross_response_max_abs": cross_response,
        "identical_spectrum_control_max_abs": identical_control,
        "same_observer_metamer_l2_difference": same_observer_difference,
        "metamer_transmittance_minimum": float(
            min(np.min(metamer_first), np.min(metamer_second))
        ),
        "metamer_transmittance_maximum": float(
            max(np.max(metamer_first), np.max(metamer_second))
        ),
    }
    decisions = {
        "clear_response": metrics["clear_response_max_abs_error"]
        <= float(gates["clear_response_max_abs_error"]),
        "neutral_response": metrics["neutral_channel_spread_max"]
        <= float(gates["neutral_channel_spread_max"]),
        "bounded_output": metrics["output_minimum"]
        >= float(gates["output_minimum"])
        - float(gates["clear_response_max_abs_error"])
        and metrics["output_maximum"]
        <= float(gates["output_maximum"])
        + float(gates["clear_response_max_abs_error"]),
        "density_wedge_monotone": metrics[
            "density_wedge_maximum_positive_step"
        ]
        <= float(gates["density_wedge_maximum_positive_step"]),
        "primary_metamer": metrics["scanner_a_metamer_max_abs_difference"]
        <= float(gates["scanner_a_metamer_max_abs_difference"]),
        "observer_separation": metrics["scanner_b_metamer_l2_difference"]
        >= float(gates["scanner_b_metamer_l2_difference_min"]),
        "rgb_only_lower_bound": metrics["rgb_only_irreducible_l2_error"]
        >= float(gates["rgb_only_irreducible_l2_error_min"]),
        "profile_separation": metrics["scanner_profile_cross_response_max_abs"]
        >= float(gates["scanner_profile_cross_response_max_abs_min"]),
        "identical_control": metrics["identical_spectrum_control_max_abs"] == 0.0,
        "same_observer_control": metrics[
            "same_observer_metamer_l2_difference"
        ]
        <= float(gates["scanner_a_metamer_max_abs_difference"]),
    }
    core = {
        "schema": "neuro_film.u6_p6d_spectral_scanner_reference_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "metrics": metrics,
        "decisions": decisions,
        "automatic_pass": all(decisions.values()),
        "branch": (
            contract["branch_rule"]["pass"]
            if all(decisions.values())
            else contract["branch_rule"]["fail"]
        ),
    }
    return {**core, "stable_evidence_id": _stable_id(core)}


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "evaluate_spectral_scanner_reference",
    "load_contract",
    "write_report",
]
