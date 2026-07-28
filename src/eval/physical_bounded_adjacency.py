"""U6.P5C evaluation for a smooth analytically bounded adjacency candidate."""

from __future__ import annotations

from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_spatial_halo import evaluate_spatial_halo
from src.eval.physical_spatial_response import _profile, _slanted_edge
from src.film_physics import (
    apply_bounded_development_adjacency,
    apply_development_adjacency,
    density_to_scan_transmittance,
)


SCHEMA = "neuro_film.u6_p5c_bounded_adjacency_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P5C contract")
    return payload


def evaluate_bounded_adjacency(
    contract: dict[str, Any],
    p5b_contract: dict[str, Any],
    p5a_contract: dict[str, Any],
    sensitometry_config: dict[str, Any],
    *,
    diagnostic_path: Path,
) -> dict[str, Any]:
    candidate = contract["candidate"]
    profile = _profile(p5a_contract)
    if tuple(candidate["chemical_spread_sigma_um_rgb"]) != (
        profile.development_adjacency_sigma_um_rgb
    ):
        raise ValueError("candidate sigma does not match frozen P5A profile")
    if tuple(candidate["unbounded_gain_rgb"]) != (
        profile.development_adjacency_gain_rgb
    ):
        raise ValueError("candidate gain does not match frozen P5A profile")
    transmittance_bound = float(
        candidate["maximum_absolute_transmittance_delta"]
    )
    density_bound = float(candidate["maximum_absolute_density_delta"])
    bounded_apply = partial(
        apply_bounded_development_adjacency,
        maximum_absolute_transmittance_delta=transmittance_bound,
        maximum_absolute_density_delta=density_bound,
    )
    halo_report = evaluate_spatial_halo(
        p5b_contract,
        p5a_contract,
        sensitometry_config,
        diagnostic_path=diagnostic_path,
        adjacency_apply=bounded_apply,
    )

    peak_density_correction = 0.0
    peak_transmittance_correction = 0.0
    sign_preserved = True
    repeat_exact = True
    for pair in p5b_contract["charts"]["density_edges"]:
        density, _ = _slanted_edge(
            tuple(p5b_contract["charts"]["shape"]),
            float(p5b_contract["charts"]["slant_degrees"]),
            float(pair[0]),
            float(pair[1]),
        )
        unbounded = apply_development_adjacency(density, profile)
        bounded = bounded_apply(density, profile)
        repeated = bounded_apply(density, profile)
        raw_correction = unbounded - density
        correction = bounded - density
        active = np.abs(raw_correction) > 1e-14
        sign_preserved = sign_preserved and bool(
            np.all(raw_correction[active] * correction[active] >= 0.0)
        )
        repeat_exact = repeat_exact and np.array_equal(bounded, repeated)
        peak_density_correction = max(
            peak_density_correction, float(np.max(np.abs(correction)))
        )
        peak_transmittance_correction = max(
            peak_transmittance_correction,
            float(
                np.max(
                    np.abs(
                        density_to_scan_transmittance(bounded)
                        - density_to_scan_transmittance(density)
                    )
                )
            ),
        )

    density_metrics = [
        metric
        for row in halo_report["density_edge_metrics"]
        for metric in row["channels"]
    ]
    peak_normalized_effect = max(
        max(
            metric["normalized_overshoot"],
            metric["normalized_undershoot"],
        )
        for metric in density_metrics
    )
    gates = contract["evaluation"]
    candidate_decisions = {
        "unchanged_p5b_gates": all(halo_report["decisions"].values()),
        "nontrivial_normalized_effect": peak_normalized_effect
        >= float(gates["minimum_peak_normalized_adjacency_effect"]),
        "nontrivial_density_effect": peak_density_correction
        >= float(gates["minimum_peak_absolute_density_correction"]),
        "density_bound": peak_density_correction <= density_bound + 1e-12,
        "transmittance_bound": peak_transmittance_correction
        <= transmittance_bound + 1e-12,
        "sign_preserved": sign_preserved,
        "repeat": repeat_exact,
    }
    passed = all(candidate_decisions.values())
    core = {
        "schema": "neuro_film.u6_p5c_bounded_adjacency_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "candidate": candidate["name"],
        "p5b_halo_metrics": {
            "density_edge_metrics": halo_report["density_edge_metrics"],
            "exposure_edge_metrics": halo_report["exposure_edge_metrics"],
            "diagnostic_sha256": halo_report["diagnostic_sha256"],
            "wrong_order_max_abs": halo_report["wrong_order_max_abs"],
            "decisions": halo_report["decisions"],
        },
        "candidate_metrics": {
            "peak_normalized_adjacency_effect": peak_normalized_effect,
            "peak_absolute_density_correction": peak_density_correction,
            "peak_absolute_transmittance_correction": (
                peak_transmittance_correction
            ),
        },
        "candidate_decisions": candidate_decisions,
        "automatic_pass": passed,
        "branch": contract["branch_rule"]["pass" if passed else "fail"],
    }
    evidence_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": evidence_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
