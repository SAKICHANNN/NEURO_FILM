"""Frozen U6.P4DM colour-cloud/scanner combined ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.cross_layer_cloud_chart_artifact import _chart
from src.eval.physical_scanner_profile import _profile
from src.eval.sensitometry_cloud_capacity_v2 import evaluate as evaluate_capacity
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.cross_layer_cloud_runtime import (
    iter_target_density_cross_layer_cloud_rows,
)
from src.film_physics.scanner import apply_scanner_profile


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    parent = root / contract["parent"]["path"]
    evidence = json.loads(parent.read_text(encoding="utf-8"))
    if (
        _sha(parent) != contract["parent"]["sha256"]
        or evidence["decision"] != contract["parent"]["required_decision"]
    ):
        raise RuntimeError("P4DM parent drift")
    return contract


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    f = contract["fixture"]
    shape = (f["height"], f["width"])
    base, _ = _chart(shape)
    capacity = evaluate_capacity(
        root, root / "configs/u6_p4di_sensitometry_cloud_capacity_v2.json"
    )
    profile = CrossLayerCloudReferenceProfile.from_payload(capacity["compiled_profile"])
    maximum = np.asarray(profile.count_profile.marginal_rates_cmy) * np.asarray(
        profile.count_profile.mark_optical_density_cmy
    )
    luminance = np.sum(base * np.asarray([0.2126, 0.7152, 0.0722]), axis=-1)
    target = np.clip(0.25 + 0.5 * (1.0 - luminance), 0.05, 0.95)[..., None] * maximum
    rows = tuple(
        iter_target_density_cross_layer_cloud_rows(
            profile,
            target,
            maximum_developed_density_cmy=tuple(maximum),
            seed=f["seed"],
            row_tile_height=127,
        )
    )
    cloud_density = np.concatenate([result.density for _, result in rows])
    transmittance_mean = np.exp(-target)
    transmittance_cloud = np.exp(-cloud_density.astype(np.float64))
    scanner_contract = json.loads(
        (root / "configs/u6_p6a_scanner_profile_boundary_v1.json").read_text()
    )
    scanner_a = _profile(scanner_contract["profiles"]["scanner_a"])
    scanner_b = _profile(scanner_contract["profiles"]["scanner_b"])
    mean_a = apply_scanner_profile(
        transmittance_mean, scanner_a, pixel_pitch_um=f["pixel_pitch_um"]
    )
    cloud_a = apply_scanner_profile(
        transmittance_cloud, scanner_a, pixel_pitch_um=f["pixel_pitch_um"]
    )
    cloud_a_repeat = apply_scanner_profile(
        transmittance_cloud, scanner_a, pixel_pitch_um=f["pixel_pitch_um"]
    )
    mean_b = apply_scanner_profile(
        transmittance_mean, scanner_b, pixel_pitch_um=f["pixel_pitch_um"]
    )
    cloud_b = apply_scanner_profile(
        transmittance_cloud, scanner_b, pixel_pitch_um=f["pixel_pitch_um"]
    )
    margin = f["neutral_margin"]
    region = np.s_[margin:-margin, margin:-margin, :]
    cloud_before = _rms((transmittance_cloud - transmittance_mean)[region])
    cloud_after = _rms((cloud_a - mean_a)[region])
    scanner_nuisance = _rms((mean_a - transmittance_mean)[region])
    wrong_scanner_residual = _rms((cloud_b - mean_a)[region])
    correct_residual = _rms((cloud_a - mean_a)[region])
    attenuation = cloud_after / cloud_before
    wrong_ratio = wrong_scanner_residual / correct_residual
    new_boundary = float(
        np.mean(
            ((cloud_a <= 0.0) | (cloud_a >= 1.0)) & ~((mean_a <= 0.0) | (mean_a >= 1.0))
        )
    )
    gates = contract["gates"]
    results = {
        "cloud_survives": cloud_after
        >= gates["minimum_cloud_residual_rms_after_scanner"],
        "scanner_nuisance": scanner_nuisance
        >= gates["minimum_scanner_nuisance_rms_without_cloud"],
        "attenuation": attenuation <= gates["maximum_cloud_residual_attenuation_ratio"],
        "wrong_scanner": wrong_ratio >= gates["minimum_wrong_scanner_residual_ratio"],
        "boundary": new_boundary <= gates["maximum_combined_new_boundary_fraction"],
        "repeat": bool(np.array_equal(cloud_a, cloud_a_repeat)),
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "cloud_profile_identity": profile.identity(),
        "cloud_residual_rms_before_scanner": cloud_before,
        "cloud_residual_rms_after_scanner": cloud_after,
        "scanner_nuisance_rms_without_cloud": scanner_nuisance,
        "cloud_residual_attenuation_ratio": attenuation,
        "wrong_scanner_to_correct_residual_ratio": wrong_ratio,
        "combined_new_boundary_fraction": new_boundary,
        "cloud_scanner_a_sha256": hashlib.sha256(
            cloud_a.astype("<f8").tobytes()
        ).hexdigest(),
        "gates": results,
        "decision": contract["decision_if_pass"]
        if all(results.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4dm_cloud_scanner_combined_ablation_report.v1",
        "automatic_pass": all(results.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
