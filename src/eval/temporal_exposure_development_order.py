"""Frozen U6.P9F pre-development exposure ordering audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.sensitometry_primitive import build_operator
from src.eval.temporal_exposure_flicker import load_contract as load_p9e_contract
from src.film_physics.temporal_exposure import (
    TemporalExposureProfile,
    generate_temporal_exposure,
)


class TemporalExposureDevelopmentOrderError(RuntimeError):
    """Raised when P9F inputs or typed ordering drift."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TemporalExposureDevelopmentOrderError("contract must be an object")
    return payload


def _candidate_density(
    operator: Any, exposures: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    return np.stack(
        [operator.apply(exposures * np.exp2(offset)) for offset in offsets],
        axis=0,
    )


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != "neuro_film.u6_p9f_exposure_development_order_contract.v1":
        raise TemporalExposureDevelopmentOrderError("unsupported P9F contract")
    parents = contract["parents"]
    p9e_evidence_path = root / parents["p9e_evidence"]
    sensitometry_path = root / parents["sensitometry_config"]
    parent_hashes_exact = bool(
        _sha(p9e_evidence_path) == parents["p9e_evidence_sha256"]
        and _sha(sensitometry_path) == parents["sensitometry_config_sha256"]
    )
    if not parent_hashes_exact:
        raise TemporalExposureDevelopmentOrderError("parent evidence hash mismatch")
    p9e_evidence = json.loads(p9e_evidence_path.read_text(encoding="utf-8"))
    if p9e_evidence.get("automatic_pass") is not True:
        raise TemporalExposureDevelopmentOrderError("P9E does not admit composition")
    p9e_contract = load_p9e_contract(root / "configs/u6_p9e_temporal_exposure_flicker_v1.json")
    p9e_profile = {
        key: value for key, value in p9e_contract["profile"].items() if key != "schema"
    }
    frame_count = int(p9e_evidence["frame_count"])
    trajectory = generate_temporal_exposure(
        TemporalExposureProfile(**p9e_profile), frame_count
    )

    operator = build_operator(json.loads(sensitometry_path.read_text(encoding="utf-8")))
    experiment = contract["experiment"]
    scalar_exposures = np.geomspace(
        float(experiment["exposure_minimum"]),
        float(experiment["exposure_maximum"]),
        int(experiment["exposure_sample_count"]),
    )
    exposures = np.repeat(scalar_exposures[:, None], 3, axis=1)
    burn = int(experiment["burn_in_frames"])
    offsets = trajectory.offset_stops[burn:]
    correct = _candidate_density(operator, exposures, offsets)
    replay = _candidate_density(operator, exposures, offsets)
    replay_exact = np.array_equal(correct, replay)

    row_parts: list[np.ndarray] = []
    start = 0
    for count in experiment["row_partition_counts"]:
        stop = start + int(count)
        row_parts.append(_candidate_density(operator, exposures[start:stop], offsets))
        start = stop
    if start != exposures.shape[0]:
        raise TemporalExposureDevelopmentOrderError("row partitions do not cover ramp")
    partition_exact = np.array_equal(np.concatenate(row_parts, axis=1), correct)

    base_density = operator.apply(exposures)
    anchor = int(experiment["anchor_sample_index"])
    anchor_delta = correct[:, anchor, :] - base_density[anchor, :]
    wrong_density = base_density[None, :, :] + anchor_delta[:, None, :]
    boundary_violations = 0
    implied_wrong: list[np.ndarray] = []
    for frame_density in wrong_density:
        try:
            recovered = operator.inverse(frame_density)
        except ValueError:
            boundary_violations += 1
            continue
        implied_wrong.append(np.log2(recovered / exposures))
    if len(implied_wrong) != offsets.size:
        wrong = np.full((offsets.size, exposures.shape[0], 3), np.nan)
    else:
        wrong = np.stack(implied_wrong, axis=0)
    recovered_correct = np.stack(
        [operator.inverse(frame_density) for frame_density in correct], axis=0
    )
    implied_correct = np.log2(recovered_correct / exposures[None, :, :])
    correct_error = np.abs(implied_correct - offsets[:, None, None])
    wrong_span = np.ptp(wrong, axis=(1, 2))
    wrong_error = np.max(np.abs(wrong - offsets[:, None, None]), axis=(1, 2))
    zero_index = int(np.flatnonzero(trajectory.frame_indices == 0)[0])
    zero_density_exact = np.array_equal(
        operator.apply(exposures * trajectory.multiplier[zero_index]), base_density
    )
    finite = bool(
        np.all(np.isfinite(correct))
        and np.all(np.isfinite(wrong))
        and np.all(np.isfinite(implied_correct))
    )
    measurements = {
        "parent_hashes_exact": parent_hashes_exact,
        "correct_order_replay_byte_exact": replay_exact,
        "correct_order_partition_byte_exact": partition_exact,
        "maximum_correct_order_implied_stop_error": float(np.max(correct_error)),
        "wrong_order_p50_implied_stop_span": float(np.quantile(wrong_span, 0.50)),
        "wrong_order_p95_implied_stop_span": float(np.quantile(wrong_span, 0.95)),
        "wrong_order_maximum_implied_stop_span": float(np.max(wrong_span)),
        "wrong_order_p95_maximum_stop_error": float(np.quantile(wrong_error, 0.95)),
        "wrong_order_maximum_stop_error": float(np.max(wrong_error)),
        "zero_offset_density_identity_exact": zero_density_exact,
        "all_values_finite": finite,
        "density_boundary_violation_count": boundary_violations,
        "scanner_or_display_transform_count_zero": True,
    }
    gates = contract["automatic_gates"]
    gate_results = {
        "parent_hashes_exact": parent_hashes_exact is gates["parent_hashes_exact"],
        "correct_order_replay_byte_exact": replay_exact
        is gates["correct_order_replay_byte_exact"],
        "correct_order_partition_byte_exact": partition_exact
        is gates["correct_order_partition_byte_exact"],
        "maximum_correct_order_implied_stop_error": measurements[
            "maximum_correct_order_implied_stop_error"
        ]
        <= gates["maximum_correct_order_implied_stop_error"],
        "minimum_wrong_order_p50_implied_stop_span": measurements[
            "wrong_order_p50_implied_stop_span"
        ]
        >= gates["minimum_wrong_order_p50_implied_stop_span"],
        "minimum_wrong_order_p95_implied_stop_span": measurements[
            "wrong_order_p95_implied_stop_span"
        ]
        >= gates["minimum_wrong_order_p95_implied_stop_span"],
        "minimum_wrong_order_p95_maximum_stop_error": measurements[
            "wrong_order_p95_maximum_stop_error"
        ]
        >= gates["minimum_wrong_order_p95_maximum_stop_error"],
        "zero_offset_density_identity_exact": zero_density_exact
        is gates["zero_offset_density_identity_exact"],
        "all_values_finite": finite is gates["all_values_finite"],
        "density_boundary_violation_count_maximum": boundary_violations
        <= gates["density_boundary_violation_count_maximum"],
        "scanner_or_display_transform_count_zero": gates[
            "scanner_or_display_transform_count_zero"
        ]
        is True,
    }
    if set(gate_results) != set(gates):
        raise TemporalExposureDevelopmentOrderError("P9F gate vocabulary drift")
    passed = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p9f_exposure_development_order_report.v1",
        "contract_sha256": _sha(root / "configs/u6_p9f_exposure_development_order_v1.json"),
        "p9e_stable_evidence_id": p9e_evidence["stable_evidence_id"],
        "sensitometry_config_sha256": parents["sensitometry_config_sha256"],
        "frame_count_scored": int(offsets.size),
        "exposure_sample_count": int(exposures.shape[0]),
        "measurements": measurements,
        "gate_results": gate_results,
        "automatic_pass": passed,
        "decision": contract["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {**stable, "stable_evidence_id": hashlib.sha256(encoded).hexdigest()}


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode()).hexdigest()
