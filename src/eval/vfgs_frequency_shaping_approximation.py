"""U6.P8BR source-conformance gate for a VFGS spectral approximation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.real_uniform_grain_source import hash_file

SCHEMA = "neuro_film.u6_p8br_vfgs_frequency_shaping_approximation_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p8br_vfgs_frequency_shaping_approximation_report.v1"


class VfgsFrequencyShapingApproximationError(RuntimeError):
    """Raised when the frozen VFGS family is not portable or drifts."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(binding["path"])
    if not path.is_file() or hash_file(path, "sha256") != binding["sha256"]:
        raise VfgsFrequencyShapingApproximationError(f"parent hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise VfgsFrequencyShapingApproximationError("parent must be an object")
    return payload


def validate_contract(contract: Mapping[str, Any]) -> None:
    model = contract.get("model", {})
    source = contract.get("primary_sources", {}).get("vfgs", {})
    gates = contract.get("automatic_gates", {})
    if (
        contract.get("schema") != SCHEMA
        or model.get("cutoff_candidates_inclusive") != [2, 14]
        or model.get("soft_window_k") != 1.118
        or model.get("rectangular_control_k") != 0.0
        or model.get("window_bins") != 64
        or model.get("window_quantization_scale") != 128
        or model.get("learned_predictor_used") is not False
        or source.get("commit") != "fbf4bd95058e934fb7246edd5e7fb8d6c9ed0ec0"
        or source.get("core_sha256")
        != "8051fbefc6fef8395e678a4a105457fc799a16044e51a6123f53fd0a28e0946a"
        or gates.get("minimum_confirmation_nps_retained_thomas_improvement_fraction")
        != 0.75
        or gates.get("require_two_byte_identical_reports") is not True
        or contract.get("roles", {}).get("refit_rescale_or_cutoff_change_on_confirmation")
        is not False
    ):
        raise VfgsFrequencyShapingApproximationError("P8BR frozen contract drift")


def vfgs_frequency_window(*, cutoff: int, softness_k: float) -> np.ndarray:
    """Reproduce the public vfgs_init_window table or reject undefined access.

    The upstream C routine stores 64 cumulative-energy samples and searches for
    the first sample no greater than ``fc``.  Exhausting that table would make
    its following ``nrj[i]`` read undefined, so this clean-room implementation
    rejects rather than inventing an extrapolation.
    """

    if not isinstance(cutoff, int) or not 0 <= cutoff <= 15:
        raise VfgsFrequencyShapingApproximationError("invalid VFGS cutoff")
    k_value = float(softness_k)
    if not math.isfinite(k_value) or k_value < 0.0:
        raise VfgsFrequencyShapingApproximationError("invalid VFGS softness")
    bins = 64
    scale = 128
    if k_value < 1.0:
        output = np.asarray(
            [scale if index < (cutoff + 1) * 4 else 0 for index in range(bins)],
            dtype=np.float64,
        )
    else:
        energy = []
        for index in range(bins):
            t_value = index / bins
            value = (
                k_value * k_value
                if index == 0
                else k_value
                * k_value
                / 2.0
                * (1.0 + math.sin(math.pi * t_value) / (math.pi * t_value))
            )
            energy.append(value)
        fc_value = (cutoff + 1.0) / 16.0
        if fc_value <= k_value * k_value / 2.0:
            a_value = k_value * k_value / (2.0 * fc_value)
        else:
            found = next(
                (index for index, value in enumerate(energy) if value <= fc_value),
                None,
            )
            if found is None or found == 0:
                raise VfgsFrequencyShapingApproximationError(
                    f"VFGS energy-table search exhausts at cutoff {cutoff}"
                )
            a_value = (
                (fc_value - energy[found - 1])
                / (energy[found] - energy[found - 1])
                + found
                - 1
            ) / bins
        quantized = []
        for index in range(bins):
            frequency = (index + 0.5) / bins
            if a_value * frequency < 1.0:
                positive = k_value * math.cos(math.pi * a_value * frequency / 2.0)
                quantized.append(math.floor(positive * scale + 0.5))
            else:
                quantized.append(0)
        output = np.asarray(quantized, dtype=np.float64)
    output /= scale
    if not np.all(np.isfinite(output)) or np.any(output < 0.0):
        raise VfgsFrequencyShapingApproximationError("invalid VFGS window")
    output.setflags(write=False)
    return output


def evaluate_vfgs_frequency_shaping_approximation(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    """Run source conformance before any scan statistics may be read."""

    validate_contract(contract)
    parents = contract["parents"]
    p4bs = _load_bound(root, parents["p4bs_decision"])
    p4cf = _load_bound(root, parents["p4cf_decision"])
    if (
        p4bs.get("decision") != parents["p4bs_decision"]["required_decision"]
        or p4cf.get("decision") != parents["p4cf_decision"]["required_decision"]
    ):
        raise VfgsFrequencyShapingApproximationError("parent decision drift")

    model = contract["model"]
    first, last = (int(value) for value in model["cutoff_candidates_inclusive"])
    invalid_soft: list[int] = []
    soft_hashes: dict[str, str] = {}
    rectangular_hashes: dict[str, str] = {}
    for cutoff in range(first, last + 1):
        try:
            soft = vfgs_frequency_window(
                cutoff=cutoff, softness_k=float(model["soft_window_k"])
            )
        except VfgsFrequencyShapingApproximationError:
            invalid_soft.append(cutoff)
        else:
            soft_hashes[str(cutoff)] = hashlib.sha256(
                np.ascontiguousarray(soft.astype("<f8")).tobytes()
            ).hexdigest()
        rectangular = vfgs_frequency_window(
            cutoff=cutoff, softness_k=float(model["rectangular_control_k"])
        )
        rectangular_hashes[str(cutoff)] = hashlib.sha256(
            np.ascontiguousarray(rectangular.astype("<f8")).tobytes()
        ).hexdigest()

    source_conformant = not invalid_soft
    checks = {
        "all_frozen_soft_cutoffs_defined": source_conformant,
        "all_rectangular_controls_defined": len(rectangular_hashes) == last - first + 1,
        "uniform_scan_scoring_reached": False,
        "repeat_exact": True,
    }
    stable = {
        "schema": REPORT_SCHEMA,
        "contract_sha256": hash_file(
            root / "configs/u6_p8br_vfgs_frequency_shaping_approximation_v1.json",
            "sha256",
        ),
        "source_conformance": {
            "vfgs_commit": contract["primary_sources"]["vfgs"]["commit"],
            "core_sha256": contract["primary_sources"]["vfgs"]["core_sha256"],
            "soft_window_k": model["soft_window_k"],
            "invalid_soft_cutoffs": invalid_soft,
            "valid_soft_window_sha256": soft_hashes,
            "rectangular_window_sha256": rectangular_hashes,
            "failure": (
                "The public default-K energy-table search has no sample <= 0.625 "
                "for cutoff 9; the following C interpolation reads beyond nrj[63]."
                if invalid_soft == [9]
                else None
            ),
        },
        "development_source_reads": 0,
        "confirmation_source_reads": 0,
        "checks": checks,
        "automatic_pass": False,
        "decision": "close_default_vfgs_soft_window_before_scan_read",
        "next_leaf": (
            "Seek a source-defined portable frequency window or separately frozen "
            "standards baseline; do not skip cutoff 9 or extrapolate the table."
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {**stable, "stable_evidence_id": stable_id}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    encoded = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "VfgsFrequencyShapingApproximationError",
    "evaluate_vfgs_frequency_shaping_approximation",
    "validate_contract",
    "vfgs_frequency_window",
    "write_report",
]
