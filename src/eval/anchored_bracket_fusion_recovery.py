"""Anchored source-only bracket fusion for the frozen P3U recovery gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.bracket_eiv_mechanism_recovery import _regime_gates
from src.eval.external_exposure_anchor_sufficiency import (
    _gain_and_anchor_error,
    _target_cache,
)
from src.eval.external_exposure_anchor_sufficiency import (
    load_contract as load_p3w_contract,
)
from src.eval.mechanism_recovery_uncertainty import _evaluate_case, _observed_source
from src.eval.multiexposure_mechanism_recovery import _lens_delta
from src.eval.promist_halation_identifiability import _blur, build_rows

SCHEMA = "neuro_film.u6_p3x_anchored_bracket_fusion_recovery_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p3x_anchored_bracket_fusion_recovery_report.v1"


class AnchoredFusionError(ValueError):
    """Raised when the P3X contract or its parent chain drifts."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _verify_file(root: Path, binding: dict[str, Any], label: str) -> Path:
    path = root / binding["path"]
    if _sha256_bytes(path.read_bytes()) != binding["sha256"]:
        raise AnchoredFusionError(f"P3X {label} hash drift")
    return path


def load_contract(
    root: Path, path: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract.get("node") != "ULT > U6 > U6.P3 > U6.P3X"
    ):
        raise AnchoredFusionError("unsupported P3X contract")
    parents = contract["parents"]
    p3w_contract_path = _verify_file(root, parents["p3w_contract"], "parent contract")
    p3w_decision_path = _verify_file(root, parents["p3w_decision"], "parent decision")
    _verify_file(root, parents["p3w_evaluator"], "parent evaluator")
    p3w_contract, p3v_contract, p3u_contract, p3t_contract, p3s_contract = (
        load_p3w_contract(root, p3w_contract_path)
    )
    p3w_decision = json.loads(p3w_decision_path.read_text(encoding="utf-8"))
    if p3w_decision.get("decision") != parents["p3w_decision"]["required_decision"]:
        raise AnchoredFusionError("P3X parent decision drift")
    if contract["anchor_precision_levels_ppm"] != [0, 100]:
        raise AnchoredFusionError("P3X anchor level drift")
    if contract["fusion"]["target_reads"] is not False:
        raise AnchoredFusionError("P3X information-flow drift")
    return (
        contract,
        p3w_contract,
        p3v_contract,
        p3u_contract,
        p3t_contract,
        p3s_contract,
    )


def _inverse_variance_fuse(
    normalized: list[np.ndarray], variances: list[float]
) -> np.ndarray:
    if len(normalized) != 3 or len(variances) != 3:
        raise AnchoredFusionError("P3X requires exactly three bracket observations")
    weights = np.reciprocal(np.asarray(variances, dtype=np.float64))
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise AnchoredFusionError("P3X fusion weights are invalid")
    stacked = np.stack(normalized, axis=0)
    return np.sum(stacked * weights[:, None, None, None], axis=0) / float(
        np.sum(weights)
    )


def _fuse_source(
    rows: list[dict[str, Any]],
    observed: dict[str, Any],
    role: str,
    regime: dict[str, Any],
    p3u_contract: dict[str, Any],
    p3t_contract: dict[str, Any],
    anchor_seed: int,
    precision_ppm: int,
) -> dict[str, Any]:
    row_size = int(np.prod(rows[0]["exposure"].shape))
    quantization_step = 1.0 / float((1 << int(regime["bits"])) - 1)
    observation_variance = float(regime["source_noise_sigma"]) ** 2 + (
        quantization_step**2 / 12.0
    )
    corrected: list[np.ndarray] = []
    normalized_variances: list[float] = []
    anchor_rows: list[dict[str, Any]] = []
    bound = float(precision_ppm) * 1e-6
    bound_violations = 0
    for index, row in enumerate(rows):
        source = observed["source"][index * row_size : (index + 1) * row_size].reshape(
            row["exposure"].shape
        )
        true_gain, anchor_error = _gain_and_anchor_error(
            row,
            role,
            regime,
            p3u_contract,
            anchor_seed,
            precision_ppm,
        )
        measured_factor = 1.0 + true_gain + anchor_error
        if measured_factor <= 0.0:
            raise RuntimeError("P3X measured exposure factor is invalid")
        corrected.append(
            np.ascontiguousarray(source / measured_factor, dtype=np.float64)
        )
        scale = float(row["exposure_scale"])
        normalized_variances.append(
            observation_variance / (measured_factor**2 * scale**2)
        )
        if abs(anchor_error) > bound + 1e-18:
            bound_violations += 1
        anchor_rows.append(
            {
                "pattern": row["pattern"],
                "exposure_scale": scale,
                "base_sha256": row["base_sha256"],
                "true_gain_nuisance": true_gain,
                "anchor_error": anchor_error,
                "measured_factor": measured_factor,
                "normalized_variance": normalized_variances[-1],
            }
        )
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["pattern"], []).append(index)
    reconstructed: list[np.ndarray | None] = [None] * len(rows)
    for pattern, indices in groups.items():
        if len(indices) != 3:
            raise AnchoredFusionError(f"P3X pattern {pattern} lacks three brackets")
        latent = _inverse_variance_fuse(
            [
                corrected[index] / float(rows[index]["exposure_scale"])
                for index in indices
            ],
            [normalized_variances[index] for index in indices],
        )
        for index in indices:
            reconstructed[index] = np.ascontiguousarray(
                latent * float(rows[index]["exposure_scale"]), dtype=np.float64
            )
    images = [image for image in reconstructed if image is not None]
    if len(images) != len(rows):
        raise RuntimeError("P3X reconstruction inventory is incomplete")
    clipped_count = sum(
        int(np.count_nonzero((image < 0.0) | (image > 1.0))) for image in images
    )
    physical_contract = p3t_contract["candidate_families"]["physical-backing-return"]
    lens_contract = p3t_contract["candidate_families"]["lens-diffusion"]
    physical_sigmas = tuple(
        float(value) for value in physical_contract["sigma_grid_pixels"]
    )
    lens_sigmas = np.asarray(lens_contract["sigma_pixels"], dtype=np.float64)
    lens_shape = np.asarray(lens_contract["normalized_weight_shape"], dtype=np.float64)
    return {
        "source": np.concatenate([image.reshape(-1) for image in images]),
        "lens_delta": np.concatenate(
            [
                _lens_delta(image, lens_sigmas, lens_shape).reshape(-1)
                for image in images
            ]
        ),
        "physical_blurs": {
            sigma: np.concatenate([_blur(image, sigma).reshape(-1) for image in images])
            for sigma in physical_sigmas
        },
        "anchor_rows_sha256": _sha256_bytes(
            json.dumps(
                anchor_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("ascii")
        ),
        "maximum_absolute_realized_anchor_error": max(
            abs(row["anchor_error"]) for row in anchor_rows
        ),
        "anchor_bound_violation_count": bound_violations,
        "reconstruction_clipped_scalar_count": clipped_count,
        "fusion_target_read_count": 0,
    }


def _evaluate_level_regime(
    precision_ppm: int,
    regime: dict[str, Any],
    raw_development: dict[str, Any],
    raw_confirmation: dict[str, Any],
    development_targets: dict[str, np.ndarray],
    confirmation_targets: dict[str, np.ndarray],
    target_clipped_count: int,
    development_rows: list[dict[str, Any]],
    confirmation_rows: list[dict[str, Any]],
    anchor_seed: int,
    contract: dict[str, Any],
    p3u_contract: dict[str, Any],
    p3t_contract: dict[str, Any],
) -> dict[str, Any]:
    development = _fuse_source(
        development_rows,
        raw_development,
        "development",
        regime,
        p3u_contract,
        p3t_contract,
        anchor_seed,
        precision_ppm,
    )
    confirmation = _fuse_source(
        confirmation_rows,
        raw_confirmation,
        "confirmation",
        regime,
        p3u_contract,
        p3t_contract,
        anchor_seed,
        precision_ppm,
    )
    cases = [
        _evaluate_case(
            truth_case,
            development,
            confirmation,
            development_targets[truth_case["case_id"]],
            confirmation_targets[truth_case["case_id"]],
            p3t_contract,
        )
        for truth_case in p3t_contract["truth_cases"]
    ]
    clipped_count = (
        int(raw_development["clipped_scalar_count"])
        + int(raw_confirmation["clipped_scalar_count"])
        + target_clipped_count
    )
    gate_results, summary = _regime_gates(cases, regime, p3u_contract, clipped_count)
    target_reads = (
        development["fusion_target_read_count"]
        + confirmation["fusion_target_read_count"]
    )
    bound_violations = (
        development["anchor_bound_violation_count"]
        + confirmation["anchor_bound_violation_count"]
    )
    reconstruction_clips = (
        development["reconstruction_clipped_scalar_count"]
        + (confirmation["reconstruction_clipped_scalar_count"])
    )
    extra = contract["additional_gates"]
    additional = {
        "fusion_target_blind": target_reads
        <= int(extra["fusion_target_read_count_max"]),
        "anchor_bound": bound_violations
        <= int(extra["maximum_anchor_bound_violation_count"]),
        "reconstruction_no_clipping": reconstruction_clips
        <= int(extra["maximum_reconstruction_clipped_scalar_count"]),
    }
    return {
        "regime": regime,
        "fusion_facts": {
            "development_anchor_rows_sha256": development["anchor_rows_sha256"],
            "confirmation_anchor_rows_sha256": confirmation["anchor_rows_sha256"],
            "maximum_absolute_realized_anchor_error": max(
                development["maximum_absolute_realized_anchor_error"],
                confirmation["maximum_absolute_realized_anchor_error"],
            ),
            "target_read_count": target_reads,
            "anchor_bound_violation_count": bound_violations,
            "reconstruction_clipped_scalar_count": reconstruction_clips,
        },
        "case_results": cases,
        "summary": summary,
        "gate_results": {**gate_results, **additional},
        "passed": all((*gate_results.values(), *additional.values())),
    }


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract, p3w_contract, _, p3u_contract, p3t_contract, p3s_contract = load_contract(
        root, contract_path
    )
    development_rows = build_rows(p3s_contract, "development")
    confirmation_rows = build_rows(p3s_contract, "confirmation")
    pattern_overlap = len(
        {row["pattern"] for row in development_rows}
        & {row["pattern"] for row in confirmation_rows}
    )
    regime_inputs: dict[str, dict[str, Any]] = {}
    for regime in p3u_contract["regimes"]:
        raw_development = _observed_source(
            development_rows, "development", regime, p3u_contract, p3t_contract
        )
        raw_confirmation = _observed_source(
            confirmation_rows, "confirmation", regime, p3u_contract, p3t_contract
        )
        development_targets, dev_clipped = _target_cache(
            development_rows, "development", regime, p3u_contract, p3t_contract
        )
        confirmation_targets, con_clipped = _target_cache(
            confirmation_rows, "confirmation", regime, p3u_contract, p3t_contract
        )
        regime_inputs[regime["regime_id"]] = {
            "regime": regime,
            "raw_development": raw_development,
            "raw_confirmation": raw_confirmation,
            "development_targets": development_targets,
            "confirmation_targets": confirmation_targets,
            "target_clipped_count": dev_clipped + con_clipped,
        }
    level_results = []
    for precision_ppm in contract["anchor_precision_levels_ppm"]:
        regimes = [
            _evaluate_level_regime(
                int(precision_ppm),
                values["regime"],
                values["raw_development"],
                values["raw_confirmation"],
                values["development_targets"],
                values["confirmation_targets"],
                int(values["target_clipped_count"]),
                development_rows,
                confirmation_rows,
                int(p3w_contract["anchor_observation"]["seed"]),
                contract,
                p3u_contract,
                p3t_contract,
            )
            for values in regime_inputs.values()
        ]
        level_results.append(
            {
                "precision_ppm": int(precision_ppm),
                "regime_results": regimes,
                "passed_all_three_regimes": all(row["passed"] for row in regimes),
            }
        )
    passing = [
        row["precision_ppm"] for row in level_results if row["passed_all_three_regimes"]
    ]
    required = {int(value) for value in contract["required_passing_levels_ppm"]}
    role_separation = pattern_overlap <= int(
        contract["additional_gates"][
            "development_confirmation_pattern_overlap_count_max"
        ]
    )
    global_gates = {
        "both_anchor_levels_pass_all_three_regimes": required.issubset(set(passing)),
        "role_separation": role_separation,
    }
    core = {
        "schema": REPORT_SCHEMA,
        "node": contract["node"],
        "contract_sha256": _sha256_bytes(contract_path.read_bytes()),
        "parent_bindings": contract["parents"],
        "anchor_seed": p3w_contract["anchor_observation"]["seed"],
        "role_facts": {
            "development_rows": len(development_rows),
            "confirmation_rows": len(confirmation_rows),
            "development_confirmation_pattern_overlap_count": pattern_overlap,
            "fusion_reads_targets": False,
            "mechanism_fit_reads_confirmation": False,
        },
        "level_results": level_results,
        "summary": {"passing_anchor_levels_ppm": passing},
        "global_gate_results": global_gates,
        "passed": all(global_gates.values()),
        "decision": (
            "retain-anchored-bracket-fusion-recovery"
            if all(global_gates.values())
            else "close-anchored-bracket-fusion-recovery"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "scientific_stable_id": _sha256_bytes(_canonical_bytes(core))}


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = _canonical_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256_bytes(payload)


__all__ = [
    "AnchoredFusionError",
    "_inverse_variance_fuse",
    "evaluate",
    "load_contract",
    "write_report",
]
