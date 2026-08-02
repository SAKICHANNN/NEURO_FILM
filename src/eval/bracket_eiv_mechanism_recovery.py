"""Source-only bracket normalization before the frozen P3U recovery gates."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.mechanism_recovery_uncertainty import (
    _evaluate_case,
    _observed_source,
    _observed_target,
)
from src.eval.mechanism_recovery_uncertainty import (
    load_contract as load_p3u_contract,
)
from src.eval.multiexposure_mechanism_recovery import _lens_delta
from src.eval.promist_halation_identifiability import _blur, build_rows

SCHEMA = "neuro_film.u6_p3v_bracket_eiv_mechanism_recovery_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p3v_bracket_eiv_mechanism_recovery_report.v1"


class BracketEIVError(ValueError):
    """Raised when the P3V contract or its frozen parent chain drifts."""


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
        raise BracketEIVError(f"P3V {label} hash drift")
    return path


def load_contract(
    root: Path, path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract.get("node") != "ULT > U6 > U6.P3 > U6.P3V"
    ):
        raise BracketEIVError("unsupported P3V contract")
    parents = contract["parents"]
    p3u_contract_path = _verify_file(root, parents["p3u_contract"], "parent contract")
    p3u_decision_path = _verify_file(root, parents["p3u_decision"], "parent decision")
    _verify_file(root, parents["p3u_evaluator"], "parent evaluator")
    p3u_contract, p3t_contract, p3s_contract = load_p3u_contract(
        root, p3u_contract_path
    )
    p3u_decision = json.loads(p3u_decision_path.read_text(encoding="utf-8"))
    if p3u_decision.get("decision") != parents["p3u_decision"]["required_decision"]:
        raise BracketEIVError("P3V parent decision drift")
    estimator = contract["estimator"]
    if int(estimator["iterations"]) != 8:
        raise BracketEIVError("P3V iteration count drift")
    if contract["roles"]["normalizer_reads_targets"] is not False:
        raise BracketEIVError("P3V information-flow drift")
    return contract, p3u_contract, p3t_contract, p3s_contract


def _zero_mean_bounded_logs(
    raw_logs: np.ndarray, lower: float, upper: float
) -> np.ndarray:
    low_shift = float(np.min(raw_logs - upper))
    high_shift = float(np.max(raw_logs - lower))
    for _ in range(80):
        shift = 0.5 * (low_shift + high_shift)
        mean = float(np.mean(np.clip(raw_logs - shift, lower, upper)))
        if mean > 0.0:
            low_shift = shift
        else:
            high_shift = shift
    result = np.clip(raw_logs - 0.5 * (low_shift + high_shift), lower, upper)
    if abs(float(np.mean(result))) > 1e-14:
        raise RuntimeError("P3V gain gauge did not converge")
    return result


def _normalize_brackets(
    rows: list[dict[str, Any]],
    observed: dict[str, Any],
    regime: dict[str, Any],
    p3t_contract: dict[str, Any],
    iterations: int,
) -> dict[str, Any]:
    row_size = int(np.prod(rows[0]["exposure"].shape))
    images = [
        observed["source"][index * row_size : (index + 1) * row_size].reshape(
            rows[index]["exposure"].shape
        )
        for index in range(len(rows))
    ]
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["pattern"], []).append(index)
    if any(len(indices) != 3 for indices in groups.values()):
        raise BracketEIVError("P3V requires three exposure rows per pattern")
    gains = np.ones(len(rows), dtype=np.float64)
    maximum_relative = 2.0 * float(regime["maximum_absolute_source_gain_jitter"])
    lower_log = math.log1p(-maximum_relative)
    upper_log = math.log1p(maximum_relative)
    for _ in range(iterations):
        bases: dict[str, np.ndarray] = {}
        for pattern, indices in groups.items():
            normalized = [
                images[index] / (float(rows[index]["exposure_scale"]) * gains[index])
                for index in indices
            ]
            bases[pattern] = np.mean(np.stack(normalized, axis=0), axis=0)
        raw_gains = np.empty_like(gains)
        for index, row in enumerate(rows):
            predictor = float(row["exposure_scale"]) * bases[row["pattern"]]
            denominator = float(np.sum(np.square(predictor), dtype=np.float64))
            if denominator <= 0.0:
                raise RuntimeError("P3V bracket predictor is degenerate")
            raw_gains[index] = float(
                np.sum(images[index] * predictor, dtype=np.float64) / denominator
            )
        if np.any(raw_gains <= 0.0) or not np.all(np.isfinite(raw_gains)):
            raise RuntimeError("P3V gain estimate is invalid")
        gains = np.exp(_zero_mean_bounded_logs(np.log(raw_gains), lower_log, upper_log))

    corrected = [images[index] / gains[index] for index in range(len(rows))]
    physical_contract = p3t_contract["candidate_families"]["physical-backing-return"]
    lens_contract = p3t_contract["candidate_families"]["lens-diffusion"]
    physical_sigmas = tuple(
        float(value) for value in physical_contract["sigma_grid_pixels"]
    )
    lens_sigmas = np.asarray(lens_contract["sigma_pixels"], dtype=np.float64)
    lens_shape = np.asarray(lens_contract["normalized_weight_shape"], dtype=np.float64)
    lens_deltas = [
        _lens_delta(image, lens_sigmas, lens_shape).reshape(-1) for image in corrected
    ]
    physical_blurs = {
        sigma: np.concatenate([_blur(image, sigma).reshape(-1) for image in corrected])
        for sigma in physical_sigmas
    }
    bound_violations = int(
        np.count_nonzero(
            (gains < 1.0 - maximum_relative - 1e-14)
            | (gains > 1.0 + maximum_relative + 1e-14)
        )
    )
    return {
        "source": np.concatenate([image.reshape(-1) for image in corrected]),
        "lens_delta": np.concatenate(lens_deltas),
        "physical_blurs": physical_blurs,
        "estimated_gains": [float(value) for value in gains],
        "gain_geometric_mean": float(np.exp(np.mean(np.log(gains)))),
        "gain_bound_violation_count": bound_violations,
        "target_read_count": 0,
    }


def _regime_gates(
    cases: list[dict[str, Any]],
    regime: dict[str, Any],
    p3u_contract: dict[str, Any],
    clipped_count: int,
) -> tuple[dict[str, bool], dict[str, Any]]:
    pure = [row for row in cases if row["truth_family"] != "mixed-abstain"]
    mixed = [row for row in cases if row["truth_family"] == "mixed-abstain"]
    common = p3u_contract["common_gates"]
    correct_count = sum(row["decision"] == row["truth_family"] for row in pure)
    sigma_count = sum(
        row["physical_sigma_exact"] is True
        for row in pure
        if row["truth_family"] == "physical-backing-return"
    )
    abstention_count = sum(row["decision"] == "abstain" for row in mixed)
    summary = {
        "pure_correct_family_count": correct_count,
        "physical_sigma_exact_count": sigma_count,
        "mixed_abstention_count": abstention_count,
        "maximum_pure_confirmation_rmse": max(
            row["best_confirmation_rmse"] for row in pure
        ),
        "minimum_wrong_to_correct_rmse_ratio": min(
            float(row["wrong_to_correct_rmse_ratio"]) for row in pure
        ),
        "maximum_relative_parameter_error": max(
            float(row["relative_parameter_error"]) for row in pure
        ),
        "minimum_mixed_best_confirmation_rmse": min(
            row["best_confirmation_rmse"] for row in mixed
        ),
        "total_clipped_scalar_count": clipped_count,
    }
    gates = {
        "pure_correct_family": correct_count
        >= int(common["pure_case_correct_family_count_min_per_regime"]),
        "pure_confirmation_error": summary["maximum_pure_confirmation_rmse"]
        <= float(regime["maximum_pure_confirmation_rmse"]),
        "wrong_family_separation": summary["minimum_wrong_to_correct_rmse_ratio"]
        >= float(common["minimum_wrong_to_correct_rmse_ratio_per_regime"]),
        "parameter_recovery": summary["maximum_relative_parameter_error"]
        <= float(regime["maximum_relative_parameter_error"]),
        "physical_sigma_recovery": sigma_count
        >= int(common["physical_sigma_exact_count_min_per_regime"]),
        "mixed_abstention": abstention_count
        >= int(common["mixed_case_abstention_count_min_per_regime"]),
        "mixed_residual_floor": summary["minimum_mixed_best_confirmation_rmse"]
        >= float(common["minimum_mixed_best_confirmation_rmse_per_regime"]),
        "no_quantization_clipping": clipped_count
        <= int(common["maximum_quantization_clipped_scalar_count"]),
    }
    return gates, summary


def _evaluate_regime(
    regime: dict[str, Any],
    development_rows: list[dict[str, Any]],
    confirmation_rows: list[dict[str, Any]],
    contract: dict[str, Any],
    p3u_contract: dict[str, Any],
    p3t_contract: dict[str, Any],
) -> dict[str, Any]:
    raw_development = _observed_source(
        development_rows, "development", regime, p3u_contract, p3t_contract
    )
    raw_confirmation = _observed_source(
        confirmation_rows, "confirmation", regime, p3u_contract, p3t_contract
    )
    iterations = int(contract["estimator"]["iterations"])
    development = _normalize_brackets(
        development_rows, raw_development, regime, p3t_contract, iterations
    )
    confirmation = _normalize_brackets(
        confirmation_rows, raw_confirmation, regime, p3t_contract, iterations
    )
    cases: list[dict[str, Any]] = []
    target_clipped_count = 0
    for truth_case in p3t_contract["truth_cases"]:
        development_target, dev_clipped = _observed_target(
            development_rows,
            "development",
            truth_case,
            regime,
            p3u_contract,
            p3t_contract,
        )
        confirmation_target, con_clipped = _observed_target(
            confirmation_rows,
            "confirmation",
            truth_case,
            regime,
            p3u_contract,
            p3t_contract,
        )
        target_clipped_count += dev_clipped + con_clipped
        cases.append(
            _evaluate_case(
                truth_case,
                development,
                confirmation,
                development_target,
                confirmation_target,
                p3t_contract,
            )
        )
    clipped_count = (
        int(raw_development["clipped_scalar_count"])
        + int(raw_confirmation["clipped_scalar_count"])
        + target_clipped_count
    )
    gate_results, summary = _regime_gates(cases, regime, p3u_contract, clipped_count)
    additional = {
        "normalizer_target_blind": development["target_read_count"]
        + confirmation["target_read_count"]
        <= int(contract["additional_gates"]["source_normalizer_target_read_count_max"]),
        "gain_bounds": development["gain_bound_violation_count"]
        + confirmation["gain_bound_violation_count"]
        <= int(contract["additional_gates"]["maximum_gain_bound_violation_count"]),
    }
    return {
        "regime": regime,
        "normalizer": {
            "development_estimated_gains": development["estimated_gains"],
            "confirmation_estimated_gains": confirmation["estimated_gains"],
            "development_gain_geometric_mean": development["gain_geometric_mean"],
            "confirmation_gain_geometric_mean": confirmation["gain_geometric_mean"],
            "target_read_count": development["target_read_count"]
            + confirmation["target_read_count"],
            "gain_bound_violation_count": development["gain_bound_violation_count"]
            + confirmation["gain_bound_violation_count"],
        },
        "case_results": cases,
        "summary": summary,
        "gate_results": {**gate_results, **additional},
        "passed": all((*gate_results.values(), *additional.values())),
    }


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract, p3u_contract, p3t_contract, p3s_contract = load_contract(
        root, contract_path
    )
    development_rows = build_rows(p3s_contract, "development")
    confirmation_rows = build_rows(p3s_contract, "confirmation")
    pattern_overlap = len(
        {row["pattern"] for row in development_rows}
        & {row["pattern"] for row in confirmation_rows}
    )
    regimes = [
        _evaluate_regime(
            regime,
            development_rows,
            confirmation_rows,
            contract,
            p3u_contract,
            p3t_contract,
        )
        for regime in p3u_contract["regimes"]
    ]
    role_separation = pattern_overlap <= int(
        contract["additional_gates"][
            "development_confirmation_pattern_overlap_count_max"
        ]
    )
    core = {
        "schema": REPORT_SCHEMA,
        "node": contract["node"],
        "contract_sha256": _sha256_bytes(contract_path.read_bytes()),
        "parent_bindings": contract["parents"],
        "role_facts": {
            "development_rows": len(development_rows),
            "confirmation_rows": len(confirmation_rows),
            "development_confirmation_pattern_overlap_count": pattern_overlap,
            "normalizer_reads_targets": False,
            "mechanism_fit_reads_confirmation": False,
        },
        "regime_results": regimes,
        "global_gate_results": {
            "all_three_p3u_regimes_pass": all(row["passed"] for row in regimes),
            "role_separation": role_separation,
        },
        "passed": all(row["passed"] for row in regimes) and role_separation,
        "decision": (
            "retain-source-only-bracket-eiv-recovery"
            if all(row["passed"] for row in regimes) and role_separation
            else "close-source-only-bracket-eiv-recovery"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "scientific_stable_id": _sha256_bytes(_canonical_bytes(core))}


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = _canonical_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256_bytes(payload)


__all__ = ["BracketEIVError", "evaluate", "load_contract", "write_report"]
