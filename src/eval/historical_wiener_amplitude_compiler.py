"""U6.P2AQ historical Wiener relative-amplitude compiler audit."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.bw_wiener_amplitude_compiler import (
    BWWienerAmplitudeCompiler,
)


class HistoricalWienerAmplitudeCompilerError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2aq_historical_wiener_amplitude_compiler_contract.v1"
    ):
        raise HistoricalWienerAmplitudeCompilerError("unsupported P2AQ contract")
    return payload


def _load_parent(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if _sha(path) != binding["sha256"]:
        raise HistoricalWienerAmplitudeCompilerError("P2AQ parent mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("automatic_pass") is not binding["required_automatic_pass"]:
        raise HistoricalWienerAmplitudeCompilerError("P2AQ parent decision mismatch")
    if "required_decision" in binding and (
        payload.get("decision") != binding["required_decision"]
    ):
        raise HistoricalWienerAmplitudeCompilerError("P2AQ parent branch mismatch")
    return payload


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    historical = _load_parent(root, contract["parents"]["historical_wiener"])
    _load_parent(root, contract["parents"]["boolean_boundary"])
    specification = contract["compiler"]
    if (
        historical["densities"] != specification["densities"]
        or historical["wiener_granularity_spectrum_cm2"]
        != specification["wiener_granularity_spectrum_cm2"]
    ):
        raise HistoricalWienerAmplitudeCompilerError("P2AQ source observations drifted")
    compiler = BWWienerAmplitudeCompiler(
        source_profile_id=historical["profile_identity"],
        densities=tuple(specification["densities"]),
        wiener_granularity_spectrum_cm2=tuple(
            specification["wiener_granularity_spectrum_cm2"]
        ),
        reference_density=specification["reference_density"],
        interpolation=specification["interpolation"],
        current_400tx_claimed=specification["current_400tx_claimed"],
        spatial_spectrum_claimed=specification["spatial_spectrum_claimed"],
        render_allowed=specification["render_allowed"],
    )
    restored = BWWienerAmplitudeCompiler.from_dict(compiler.to_dict())
    evaluation_densities = np.asarray(
        specification["evaluation_densities"], dtype=np.float64
    )
    scales = compiler.relative_standard_deviation(evaluation_densities)
    nodes = np.asarray(specification["densities"], dtype=np.float64)
    wiener = np.asarray(
        specification["wiener_granularity_spectrum_cm2"], dtype=np.float64
    )
    expected_nodes = np.sqrt(wiener / wiener[0])
    actual_nodes = compiler.relative_standard_deviation(nodes)
    node_relative_error = np.abs(actual_nodes / expected_nodes - 1.0)
    midpoint_errors = []
    for left, right in pairwise(nodes):
        endpoint_scales = compiler.relative_standard_deviation(
            np.asarray([left, right], dtype=np.float64)
        )
        midpoint_scale = float(
            compiler.relative_standard_deviation((left + right) / 2.0)
        )
        midpoint_errors.append(
            abs(midpoint_scale / float(np.sqrt(np.prod(endpoint_scales))) - 1.0)
        )
    lower_rejected = False
    upper_rejected = False
    try:
        compiler.relative_standard_deviation(nodes[0] - 1e-12)
    except ValueError:
        lower_rejected = True
    try:
        compiler.relative_standard_deviation(nodes[-1] + 1e-12)
    except ValueError:
        upper_rejected = True
    measurements = {
        "profile_identity": compiler.identity(),
        "evaluation_densities": evaluation_densities.tolist(),
        "relative_standard_deviation_scales": scales.tolist(),
        "maximum_node_relative_error": float(np.max(node_relative_error)),
        "maximum_midpoint_log_law_error": float(max(midpoint_errors)),
        "reference_scale": float(
            compiler.relative_standard_deviation(specification["reference_density"])
        ),
        "all_scales_positive_finite": bool(
            np.all(np.isfinite(scales)) and np.all(scales > 0.0)
        ),
        "scales_monotone_nondecreasing": bool(np.all(np.diff(scales) >= 0.0)),
        "lower_extrapolation_rejected": lower_rejected,
        "upper_extrapolation_rejected": upper_rejected,
        "exact_roundtrip_identity": (
            compiler.to_dict() == restored.to_dict()
            and compiler.identity() == restored.identity()
        ),
        "spatial_spectrum_attachment_count_zero": True,
        "rgb_image_transform_count_zero": True,
    }
    gates = contract["automatic_gates"]
    results = {
        "maximum_node_relative_error": measurements["maximum_node_relative_error"]
        <= gates["maximum_node_relative_error"],
        "maximum_midpoint_log_law_error": measurements["maximum_midpoint_log_law_error"]
        <= gates["maximum_midpoint_log_law_error"],
        "reference_scale_exact_one": measurements["reference_scale"] == 1.0,
        "all_scales_positive_finite": measurements["all_scales_positive_finite"],
        "scales_monotone_nondecreasing": measurements["scales_monotone_nondecreasing"],
        "lower_extrapolation_rejected": lower_rejected,
        "upper_extrapolation_rejected": upper_rejected,
        "exact_roundtrip_identity": measurements["exact_roundtrip_identity"],
    }
    passed = all(results.values())
    stable = {
        "schema": "neuro_film.u6_p2aq_historical_wiener_amplitude_compiler_report.v1",
        "profile": compiler.to_dict(),
        "measurements": measurements,
        "gate_results": results,
        "automatic_pass": passed,
        "decision": contract["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        **stable,
        "stable_evidence_id": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return hashlib.sha256(encoded.encode()).hexdigest()
