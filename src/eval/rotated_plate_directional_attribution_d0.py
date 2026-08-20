"""Registration-free directional attribution for perpendicular plate scans."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.rotated_plate_coherence_d0 import _load_u16

SCHEMA = "neuro-film.u6-p6ax-rotated-plate-directional-attribution-d0-contract.v1"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported P6AX contract")
    return value


def _rank_center_crop(values: np.ndarray, size: int) -> np.ndarray:
    if values.ndim != 2 or values.dtype != np.uint16:
        raise ValueError("P6AX rank input drift")
    if size <= 0 or size > min(values.shape):
        raise ValueError("P6AX crop size drift")
    counts = np.bincount(values.ravel(), minlength=65536)
    starts = np.cumsum(counts, dtype=np.int64) - counts
    mapping = (starts + 0.5 * counts) / values.size
    y0 = (values.shape[0] - size) // 2
    x0 = (values.shape[1] - size) // 2
    return mapping[values[y0 : y0 + size, x0 : x0 + size]]


def _directional_profile(values: np.ndarray, analysis: dict[str, Any]) -> dict[str, Any]:
    crop_size = int(analysis["center_crop_size"])
    patch_size = int(analysis["patch_size"])
    grid = [int(v) for v in analysis["patch_grid"]]
    if crop_size != patch_size * grid[0] or grid[0] != grid[1]:
        raise ValueError("P6AX patch geometry drift")
    crop = _rank_center_crop(values, crop_size)
    window_1d = np.hanning(patch_size + 1)[:-1]
    window = window_1d[:, None] * window_1d[None, :]
    frequency = np.fft.fftshift(np.fft.fftfreq(patch_size))
    yy, xx = np.meshgrid(frequency, frequency, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy) / 0.5
    mask = (radius >= float(analysis["minimum_normalized_radius"])) & (
        radius <= float(analysis["maximum_normalized_radius"])
    )
    radial_bins = int(analysis["radial_bins"])
    bin_index = np.minimum((radius * radial_bins).astype(np.int64), radial_bins - 1)
    profiles: list[np.ndarray] = []
    for gy in range(grid[0]):
        for gx in range(grid[1]):
            patch = crop[
                gy * patch_size : (gy + 1) * patch_size,
                gx * patch_size : (gx + 1) * patch_size,
            ]
            centered = (patch - float(np.mean(patch))) * window
            power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
            log_power = np.log1p(power)
            sums = np.bincount(
                bin_index.ravel(), weights=log_power.ravel(), minlength=radial_bins
            )
            counts = np.bincount(bin_index.ravel(), minlength=radial_bins)
            radial_mean = sums / counts
            profiles.append(log_power - radial_mean[bin_index])
    profile = np.mean(np.stack(profiles, axis=0), axis=0)
    selected = profile[mask]
    return {
        "profile": profile,
        "mask": mask,
        "patches": len(profiles),
        "valid_frequency_bins": int(np.count_nonzero(mask)),
        "residual_standard_deviation": float(np.std(selected)),
        "profile_sha256": _sha(np.ascontiguousarray(profile, dtype="<f8").tobytes()),
    }


def _correlation(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    a = first[mask].astype(np.float64, copy=False)
    b = second[mask].astype(np.float64, copy=False)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denominator = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return float("nan")
    return float(np.dot(a, b) / denominator)


def _classify_profiles(
    profiles: list[dict[str, np.ndarray]], minimum_margin: float
) -> list[dict[str, Any]]:
    if len(profiles) != 2:
        raise ValueError("P6AX requires two plates")
    rows: list[dict[str, Any]] = []
    for index, current in enumerate(profiles):
        other = profiles[1 - index]
        reference = current["reference"]
        rotated = current["rotated"]
        mask = current["mask"]
        plate_correlation = _correlation(reference, np.rot90(rotated), mask)
        scanner_correlation = _correlation(reference, rotated, mask)
        wrong_correlations = [
            _correlation(reference, other["reference"], mask),
            _correlation(reference, other["rotated"], mask),
            _correlation(reference, np.rot90(other["rotated"]), mask),
        ]
        wrong_correlation = max(wrong_correlations)
        plate_margin = plate_correlation - max(scanner_correlation, wrong_correlation)
        scanner_margin = scanner_correlation - max(plate_correlation, wrong_correlation)
        rows.append(
            {
                "plate_following_correlation": plate_correlation,
                "scanner_fixed_correlation": scanner_correlation,
                "wrong_plate_correlation": wrong_correlation,
                "plate_following_margin": plate_margin,
                "scanner_fixed_margin": scanner_margin,
                "plate_following_pass": bool(plate_margin >= minimum_margin),
                "scanner_fixed_pass": bool(scanner_margin >= minimum_margin),
            }
        )
    return rows


def _validate_parent(root: Path, specification: dict[str, Any]) -> None:
    payload = (root / specification["path"]).read_bytes()
    value = json.loads(payload)
    if _sha(payload) != specification["sha256"]:
        raise ValueError("P6AX parent hash drift")
    if value.get("status") != specification["required_status"]:
        raise ValueError("P6AX parent status drift")


def evaluate(contract: dict[str, Any], root: Path) -> dict[str, Any]:
    for specification in contract["parents"].values():
        _validate_parent(root, specification)
    analysis = contract["analysis"]
    gates = contract["gates"]
    plate_profiles: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for plate in contract["plates"]:
        role_profiles: dict[str, Any] = {}
        role_facts: dict[str, Any] = {}
        for role in ("reference", "rotated"):
            values, facts = _load_u16(root / plate[role]["path"], plate[role])
            result = _directional_profile(values, analysis)
            role_profiles[role] = result["profile"]
            role_facts[role] = {
                **facts,
                "patches": result["patches"],
                "valid_frequency_bins": result["valid_frequency_bins"],
                "residual_standard_deviation": result["residual_standard_deviation"],
                "profile_sha256": result["profile_sha256"],
            }
        plate_profiles.append(
            {
                "reference": role_profiles["reference"],
                "rotated": role_profiles["rotated"],
                "mask": _directional_profile_mask(analysis),
            }
        )
        source_rows.append({"id": plate["id"], "roles": role_facts})
    classifications = _classify_profiles(
        plate_profiles,
        float(gates["minimum_hypothesis_margin_over_competing_and_wrong_plate"]),
    )
    support_pass = True
    rows: list[dict[str, Any]] = []
    for source, classification in zip(source_rows, classifications, strict=True):
        for facts in source["roles"].values():
            support_pass = support_pass and bool(
                facts["patches"] == int(gates["required_patches_per_scan"])
                and facts["valid_frequency_bins"]
                >= int(gates["minimum_valid_frequency_bins"])
                and np.isfinite(facts["residual_standard_deviation"])
                and facts["residual_standard_deviation"]
                >= float(gates["minimum_residual_standard_deviation"])
            )
        finite = all(
            np.isfinite(value)
            for key, value in classification.items()
            if key.endswith(("correlation", "margin"))
        )
        support_pass = support_pass and finite
        rows.append({**source, **classification})
    plate_passes = sum(bool(row["plate_following_pass"]) for row in rows)
    scanner_passes = sum(bool(row["scanner_fixed_pass"]) for row in rows)
    if support_pass and plate_passes == int(
        gates["required_plate_following_passes_for_plate_decision"]
    ):
        status = "PASS_PLATE_FOLLOWING_DIRECTIONAL_STRUCTURE_D0"
        decision = contract["decision_if_plate_following"]
        automatic_pass = True
    elif support_pass and scanner_passes == int(
        gates["required_scanner_fixed_passes_for_scanner_decision"]
    ):
        status = "PASS_SCANNER_FIXED_DIRECTIONAL_STRUCTURE_D0"
        decision = contract["decision_if_scanner_fixed"]
        automatic_pass = True
    else:
        status = "FAIL_INCONCLUSIVE_DIRECTIONAL_ATTRIBUTION_D0"
        decision = contract["decision_if_inconclusive"]
        automatic_pass = False
    scientific = {
        "schema": "neuro-film.u6-p6ax-rotated-plate-directional-attribution-d0-report.v1",
        "experiment_id": contract["experiment_id"],
        "config_sha256": _sha(_canonical(contract)),
        "analysis": analysis,
        "rows": rows,
        "aggregate": {
            "plates": len(rows),
            "support_pass": support_pass,
            "plate_following_passes": plate_passes,
            "scanner_fixed_passes": scanner_passes,
        },
        "status": status,
        "automatic_pass": automatic_pass,
        "decision": decision,
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**scientific, "stable_evidence_id": _sha(_canonical(scientific))}


def _directional_profile_mask(analysis: dict[str, Any]) -> np.ndarray:
    patch_size = int(analysis["patch_size"])
    frequency = np.fft.fftshift(np.fft.fftfreq(patch_size))
    yy, xx = np.meshgrid(frequency, frequency, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy) / 0.5
    return (radius >= float(analysis["minimum_normalized_radius"])) & (
        radius <= float(analysis["maximum_normalized_radius"])
    )
