"""Quantify external exposure-anchor precision for P3T/P3U recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.bracket_eiv_mechanism_recovery import (
    _regime_gates,
)
from src.eval.bracket_eiv_mechanism_recovery import (
    load_contract as load_p3v_contract,
)
from src.eval.mechanism_recovery_uncertainty import (
    _evaluate_case,
    _observed_source,
    _observed_target,
    _rng,
)
from src.eval.multiexposure_mechanism_recovery import _lens_delta
from src.eval.promist_halation_identifiability import _blur, build_rows

SCHEMA = "neuro_film.u6_p3w_external_exposure_anchor_sufficiency_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p3w_external_exposure_anchor_sufficiency_report.v1"


class ExposureAnchorError(ValueError):
    """Raised when the P3W contract or its frozen parent chain drifts."""


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
        raise ExposureAnchorError(f"P3W {label} hash drift")
    return path


def load_contract(
    root: Path, path: Path
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract.get("node") != "ULT > U6 > U6.P3 > U6.P3W"
    ):
        raise ExposureAnchorError("unsupported P3W contract")
    parents = contract["parents"]
    p3v_contract_path = _verify_file(root, parents["p3v_contract"], "parent contract")
    p3v_decision_path = _verify_file(root, parents["p3v_decision"], "parent decision")
    _verify_file(root, parents["p3v_evaluator"], "parent evaluator")
    p3v_contract, p3u_contract, p3t_contract, p3s_contract = load_p3v_contract(
        root, p3v_contract_path
    )
    p3v_decision = json.loads(p3v_decision_path.read_text(encoding="utf-8"))
    if p3v_decision.get("decision") != parents["p3v_decision"]["required_decision"]:
        raise ExposureAnchorError("P3W parent decision drift")
    levels = [int(value) for value in contract["anchor_precision_levels_ppm"]]
    if levels != [0, 100, 250, 500, 1000]:
        raise ExposureAnchorError("P3W anchor ladder drift")
    if contract["roles"]["anchor_reads_targets"] is not False:
        raise ExposureAnchorError("P3W anchor information-flow drift")
    return contract, p3v_contract, p3u_contract, p3t_contract, p3s_contract


def _gain_and_anchor_error(
    row: dict[str, Any],
    role: str,
    regime: dict[str, Any],
    p3u_contract: dict[str, Any],
    anchor_seed: int,
    precision_ppm: int,
) -> tuple[float, float]:
    source_generator = _rng(
        int(p3u_contract["uncertainty_seed"]),
        regime["regime_id"],
        role,
        row["pattern"],
        float(row["exposure_scale"]).hex(),
        row["base_sha256"],
        "source",
    )
    true_gain = float(
        source_generator.uniform(
            -float(regime["maximum_absolute_source_gain_jitter"]),
            float(regime["maximum_absolute_source_gain_jitter"]),
        )
    )
    bound = float(precision_ppm) * 1e-6
    if bound == 0.0:
        return true_gain, 0.0
    anchor_generator = _rng(
        anchor_seed,
        str(precision_ppm),
        regime["regime_id"],
        role,
        row["pattern"],
        float(row["exposure_scale"]).hex(),
        row["base_sha256"],
        "anchor",
    )
    return true_gain, float(anchor_generator.uniform(-bound, bound))


def _correct_source(
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
    corrected: list[np.ndarray] = []
    anchor_rows: list[dict[str, Any]] = []
    bound_violations = 0
    clipped_count = 0
    bound = float(precision_ppm) * 1e-6
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
        measured_gain = true_gain + anchor_error
        image = source / (1.0 + measured_gain)
        clipped_count += int(np.count_nonzero((image < 0.0) | (image > 1.0)))
        if abs(anchor_error) > bound + 1e-18:
            bound_violations += 1
        corrected.append(np.ascontiguousarray(image, dtype=np.float64))
        anchor_rows.append(
            {
                "pattern": row["pattern"],
                "exposure_scale": float(row["exposure_scale"]),
                "base_sha256": row["base_sha256"],
                "true_gain_nuisance": true_gain,
                "anchor_error": anchor_error,
                "measured_gain_nuisance": measured_gain,
            }
        )
    physical_contract = p3t_contract["candidate_families"]["physical-backing-return"]
    lens_contract = p3t_contract["candidate_families"]["lens-diffusion"]
    physical_sigmas = tuple(
        float(value) for value in physical_contract["sigma_grid_pixels"]
    )
    lens_sigmas = np.asarray(lens_contract["sigma_pixels"], dtype=np.float64)
    lens_shape = np.asarray(lens_contract["normalized_weight_shape"], dtype=np.float64)
    return {
        "source": np.concatenate([image.reshape(-1) for image in corrected]),
        "lens_delta": np.concatenate(
            [
                _lens_delta(image, lens_sigmas, lens_shape).reshape(-1)
                for image in corrected
            ]
        ),
        "physical_blurs": {
            sigma: np.concatenate(
                [_blur(image, sigma).reshape(-1) for image in corrected]
            )
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
        "source_correction_clipped_scalar_count": clipped_count,
        "anchor_target_read_count": 0,
    }


def _target_cache(
    rows: list[dict[str, Any]],
    role: str,
    regime: dict[str, Any],
    p3u_contract: dict[str, Any],
    p3t_contract: dict[str, Any],
) -> tuple[dict[str, np.ndarray], int]:
    targets: dict[str, np.ndarray] = {}
    clipped_count = 0
    for truth_case in p3t_contract["truth_cases"]:
        target, clipped = _observed_target(
            rows,
            role,
            truth_case,
            regime,
            p3u_contract,
            p3t_contract,
        )
        targets[truth_case["case_id"]] = target
        clipped_count += clipped
    return targets, clipped_count


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
    contract: dict[str, Any],
    p3u_contract: dict[str, Any],
    p3t_contract: dict[str, Any],
) -> dict[str, Any]:
    seed = int(contract["anchor_observation"]["seed"])
    development = _correct_source(
        development_rows,
        raw_development,
        "development",
        regime,
        p3u_contract,
        p3t_contract,
        seed,
        precision_ppm,
    )
    confirmation = _correct_source(
        confirmation_rows,
        raw_confirmation,
        "confirmation",
        regime,
        p3u_contract,
        p3t_contract,
        seed,
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
    extra = contract["additional_gates"]
    anchor_reads = (
        development["anchor_target_read_count"]
        + confirmation["anchor_target_read_count"]
    )
    bound_violations = (
        development["anchor_bound_violation_count"]
        + confirmation["anchor_bound_violation_count"]
    )
    correction_clips = (
        development["source_correction_clipped_scalar_count"]
        + confirmation["source_correction_clipped_scalar_count"]
    )
    additional = {
        "anchor_target_blind": anchor_reads
        <= int(extra["anchor_target_read_count_max"]),
        "anchor_bound": bound_violations
        <= int(extra["maximum_anchor_bound_violation_count"]),
        "source_correction_no_clipping": correction_clips
        <= int(extra["maximum_source_correction_clipped_scalar_count"]),
    }
    return {
        "regime": regime,
        "anchor_facts": {
            "development_rows_sha256": development["anchor_rows_sha256"],
            "confirmation_rows_sha256": confirmation["anchor_rows_sha256"],
            "maximum_absolute_realized_anchor_error": max(
                development["maximum_absolute_realized_anchor_error"],
                confirmation["maximum_absolute_realized_anchor_error"],
            ),
            "target_read_count": anchor_reads,
            "bound_violation_count": bound_violations,
            "source_correction_clipped_scalar_count": correction_clips,
        },
        "case_results": cases,
        "summary": summary,
        "gate_results": {**gate_results, **additional},
        "passed": all((*gate_results.values(), *additional.values())),
    }


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract, _, p3u_contract, p3t_contract, p3s_contract = load_contract(
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
    level_results: list[dict[str, Any]] = []
    for precision_ppm in contract["anchor_precision_levels_ppm"]:
        regimes = []
        for values in regime_inputs.values():
            regimes.append(
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
                    contract,
                    p3u_contract,
                    p3t_contract,
                )
            )
        level_results.append(
            {
                "precision_ppm": int(precision_ppm),
                "regime_results": regimes,
                "passed_all_three_regimes": all(row["passed"] for row in regimes),
            }
        )
    levels = [row["precision_ppm"] for row in level_results]
    passing = [
        row["precision_ppm"] for row in level_results if row["passed_all_three_regimes"]
    ]
    prefix = passing == levels[: len(passing)]
    required = {int(value) for value in contract["required_passing_levels_ppm"]}
    required_pass = required.issubset(set(passing))
    role_separation = pattern_overlap <= int(
        contract["additional_gates"][
            "development_confirmation_pattern_overlap_count_max"
        ]
    )
    global_gates = {
        "required_anchor_levels_pass_all_three_regimes": required_pass,
        "passing_anchor_levels_form_prefix": prefix,
        "role_separation": role_separation,
    }
    core = {
        "schema": REPORT_SCHEMA,
        "node": contract["node"],
        "contract_sha256": _sha256_bytes(contract_path.read_bytes()),
        "parent_bindings": contract["parents"],
        "role_facts": {
            "development_rows": len(development_rows),
            "confirmation_rows": len(confirmation_rows),
            "development_confirmation_pattern_overlap_count": pattern_overlap,
            "anchor_reads_targets": False,
            "mechanism_fit_reads_confirmation": False,
        },
        "level_results": level_results,
        "summary": {
            "passing_anchor_levels_ppm": passing,
            "largest_passing_tested_anchor_bound_ppm": max(passing)
            if passing
            else None,
        },
        "global_gate_results": global_gates,
        "passed": all(global_gates.values()),
        "decision": (
            "retain-tested-external-exposure-anchor-requirement"
            if all(global_gates.values())
            else "close-tested-external-exposure-anchor-ladder"
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
    "ExposureAnchorError",
    "_gain_and_anchor_error",
    "evaluate",
    "load_contract",
    "write_report",
]
