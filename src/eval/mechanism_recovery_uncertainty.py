"""Challenge P3T mechanism recovery under frozen observation uncertainty."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.multiexposure_mechanism_recovery import (
    _fit_scalar,
    _lens_delta,
    _predict,
    _prepare_role,
    _rmse,
)
from src.eval.multiexposure_mechanism_recovery import (
    load_contract as load_p3t_contract,
)
from src.eval.promist_halation_identifiability import _blur, _response, build_rows

SCHEMA = "neuro_film.u6_p3u_mechanism_recovery_uncertainty_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p3u_mechanism_recovery_uncertainty_report.v1"


class MechanismUncertaintyError(ValueError):
    """Raised when the P3U contract or a bound parent drifts."""


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
        raise MechanismUncertaintyError(f"P3U {label} hash drift")
    return path


def load_contract(
    root: Path, path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract.get("node") != "ULT > U6 > U6.P3 > U6.P3U"
    ):
        raise MechanismUncertaintyError("unsupported P3U contract")
    parents = contract["parents"]
    parent_contract_path = _verify_file(
        root, parents["p3t_contract"], "parent contract"
    )
    parent_decision_path = _verify_file(
        root, parents["p3t_decision"], "parent decision"
    )
    _verify_file(root, parents["p3t_evaluator"], "parent evaluator")
    parent_contract, p3s_contract = load_p3t_contract(root, parent_contract_path)
    parent_decision = json.loads(parent_decision_path.read_text(encoding="utf-8"))
    if parent_decision.get("decision") != parents["p3t_decision"]["required_decision"]:
        raise MechanismUncertaintyError("P3U parent decision drift")
    regimes = contract["regimes"]
    if [int(row["bits"]) for row in regimes] != [16, 12, 10]:
        raise MechanismUncertaintyError("P3U regime identity drift")
    if contract["quantization"]["clipping_allowed"] is not False:
        raise MechanismUncertaintyError("P3U clipping policy drift")
    return contract, parent_contract, p3s_contract


def _rng(seed: int, *tokens: str) -> np.random.Generator:
    payload = "|".join((str(seed), *tokens)).encode("ascii")
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(derived)


def _quantize(
    values: np.ndarray, bits: int, full_scale: float
) -> tuple[np.ndarray, int]:
    clipped = int(np.count_nonzero((values < 0.0) | (values > full_scale)))
    levels = float((1 << bits) - 1)
    bounded = np.clip(values, 0.0, full_scale)
    quantized = np.rint(bounded * (levels / full_scale)) * (full_scale / levels)
    return np.ascontiguousarray(quantized, dtype=np.float64), clipped


def _observed_source(
    rows: list[dict[str, Any]],
    role: str,
    regime: dict[str, Any],
    contract: dict[str, Any],
    parent_contract: dict[str, Any],
) -> dict[str, Any]:
    seed = int(contract["uncertainty_seed"])
    bits = int(regime["bits"])
    full_scale = float(contract["quantization"]["source_full_scale"])
    noise_sigma = float(regime["source_noise_sigma"])
    maximum_jitter = float(regime["maximum_absolute_source_gain_jitter"])
    physical_contract = parent_contract["candidate_families"]["physical-backing-return"]
    lens_contract = parent_contract["candidate_families"]["lens-diffusion"]
    physical_sigmas = tuple(
        float(value) for value in physical_contract["sigma_grid_pixels"]
    )
    lens_sigmas = np.asarray(lens_contract["sigma_pixels"], dtype=np.float64)
    lens_shape = np.asarray(lens_contract["normalized_weight_shape"], dtype=np.float64)
    sources: list[np.ndarray] = []
    lens_deltas: list[np.ndarray] = []
    physical_blurs: dict[float, list[np.ndarray]] = {
        sigma: [] for sigma in physical_sigmas
    }
    gains: list[float] = []
    clipped_count = 0
    for row in rows:
        generator = _rng(
            seed,
            regime["regime_id"],
            role,
            row["pattern"],
            float(row["exposure_scale"]).hex(),
            row["base_sha256"],
            "source",
        )
        gain = float(generator.uniform(-maximum_jitter, maximum_jitter))
        observed = np.asarray(row["exposure"], dtype=np.float64) * (1.0 + gain)
        observed = observed + generator.normal(0.0, noise_sigma, size=observed.shape)
        observed, row_clipped = _quantize(observed, bits, full_scale)
        clipped_count += row_clipped
        gains.append(gain)
        sources.append(observed.reshape(-1))
        lens_deltas.append(_lens_delta(observed, lens_sigmas, lens_shape).reshape(-1))
        for sigma in physical_sigmas:
            physical_blurs[sigma].append(_blur(observed, sigma).reshape(-1))
    return {
        "source": np.concatenate(sources),
        "lens_delta": np.concatenate(lens_deltas),
        "physical_blurs": {
            sigma: np.concatenate(images) for sigma, images in physical_blurs.items()
        },
        "clipped_scalar_count": clipped_count,
        "minimum_realized_gain_jitter": min(gains),
        "maximum_realized_gain_jitter": max(gains),
    }


def _observed_target(
    rows: list[dict[str, Any]],
    role: str,
    truth_case: dict[str, Any],
    regime: dict[str, Any],
    contract: dict[str, Any],
    parent_contract: dict[str, Any],
) -> tuple[np.ndarray, int]:
    prepared = _prepare_role(rows, role, truth_case, parent_contract)
    target = prepared["target"].copy()
    row_size = int(np.prod(rows[0]["exposure"].shape))
    seed = int(contract["uncertainty_seed"])
    noise_sigma = float(regime["target_noise_sigma"])
    bits = int(regime["bits"])
    full_scale = float(contract["quantization"]["target_full_scale"])
    result: list[np.ndarray] = []
    clipped_count = 0
    for index, row in enumerate(rows):
        segment = slice(index * row_size, (index + 1) * row_size)
        generator = _rng(
            seed,
            regime["regime_id"],
            truth_case["case_id"],
            role,
            row["pattern"],
            float(row["exposure_scale"]).hex(),
            row["base_sha256"],
            "target",
        )
        observed = target[segment] + generator.normal(
            0.0, noise_sigma, size=target[segment].shape
        )
        observed, row_clipped = _quantize(observed, bits, full_scale)
        result.append(observed)
        clipped_count += row_clipped
    return np.concatenate(result), clipped_count


def _evaluate_case(
    truth_case: dict[str, Any],
    development: dict[str, Any],
    confirmation: dict[str, Any],
    development_target: np.ndarray,
    confirmation_target: np.ndarray,
    parent_contract: dict[str, Any],
) -> dict[str, Any]:
    observation = parent_contract["observation"]
    toe = float(observation["response_toe"])
    density_scale = float(observation["response_density_scale"])
    physical_contract = parent_contract["candidate_families"]["physical-backing-return"]
    lens_contract = parent_contract["candidate_families"]["lens-diffusion"]
    physical_fits: list[dict[str, float]] = []
    for sigma in (float(value) for value in physical_contract["sigma_grid_pixels"]):
        fraction, development_rmse = _fit_scalar(
            development["source"],
            development["physical_blurs"][sigma],
            development_target,
            tuple(float(value) for value in physical_contract["fraction_bounds"]),
            toe,
            density_scale,
        )
        physical_fits.append(
            {
                "sigma_pixels": sigma,
                "fraction": fraction,
                "development_rmse": development_rmse,
            }
        )
    physical_fit = min(
        physical_fits, key=lambda row: (row["development_rmse"], row["sigma_pixels"])
    )
    physical_prediction = _predict(
        confirmation["source"],
        confirmation["physical_blurs"][physical_fit["sigma_pixels"]],
        physical_fit["fraction"],
        toe,
        density_scale,
    )
    physical_rmse = _rmse(physical_prediction, confirmation_target)
    lens_weight, lens_development_rmse = _fit_scalar(
        development["source"],
        development["lens_delta"],
        development_target,
        tuple(float(value) for value in lens_contract["total_weight_bounds"]),
        toe,
        density_scale,
    )
    lens_prediction = _predict(
        confirmation["source"],
        confirmation["lens_delta"],
        lens_weight,
        toe,
        density_scale,
    )
    lens_rmse = _rmse(lens_prediction, confirmation_target)
    identity_rmse = _rmse(
        _response(confirmation["source"], toe, density_scale), confirmation_target
    )
    family_rmse = {
        "physical-backing-return": physical_rmse,
        "lens-diffusion": lens_rmse,
    }
    selected = min(family_rmse, key=lambda key: (family_rmse[key], key))
    best = family_rmse[selected]
    other = max(family_rmse.values())
    is_mixed = truth_case["family"] == "mixed-abstain"
    decision = "abstain" if is_mixed and best >= 0.0002 else selected
    if truth_case["family"] == "physical-backing-return":
        parameter_error: float | None = abs(
            physical_fit["fraction"] - float(truth_case["fraction"])
        ) / float(truth_case["fraction"])
        sigma_exact: bool | None = physical_fit["sigma_pixels"] == float(
            truth_case["sigma_pixels"]
        )
    elif truth_case["family"] == "lens-diffusion":
        parameter_error = abs(lens_weight - float(truth_case["total_weight"])) / float(
            truth_case["total_weight"]
        )
        sigma_exact = None
    else:
        parameter_error = None
        sigma_exact = None
    return {
        "case_id": truth_case["case_id"],
        "truth_family": truth_case["family"],
        "physical_fit": {
            "sigma_pixels": physical_fit["sigma_pixels"],
            "fraction": physical_fit["fraction"],
            "development_rmse": physical_fit["development_rmse"],
            "confirmation_rmse": physical_rmse,
        },
        "lens_fit": {
            "total_weight": lens_weight,
            "development_rmse": lens_development_rmse,
            "confirmation_rmse": lens_rmse,
        },
        "identity_confirmation_rmse": identity_rmse,
        "selected_family": selected,
        "decision": decision,
        "best_confirmation_rmse": best,
        "wrong_to_correct_rmse_ratio": (None if is_mixed else other / max(best, 1e-30)),
        "relative_parameter_error": parameter_error,
        "physical_sigma_exact": sigma_exact,
    }


def _evaluate_regime(
    regime: dict[str, Any],
    development_rows: list[dict[str, Any]],
    confirmation_rows: list[dict[str, Any]],
    contract: dict[str, Any],
    parent_contract: dict[str, Any],
) -> dict[str, Any]:
    development = _observed_source(
        development_rows, "development", regime, contract, parent_contract
    )
    confirmation = _observed_source(
        confirmation_rows, "confirmation", regime, contract, parent_contract
    )
    cases: list[dict[str, Any]] = []
    target_clipped_count = 0
    for truth_case in parent_contract["truth_cases"]:
        development_target, dev_clipped = _observed_target(
            development_rows,
            "development",
            truth_case,
            regime,
            contract,
            parent_contract,
        )
        confirmation_target, con_clipped = _observed_target(
            confirmation_rows,
            "confirmation",
            truth_case,
            regime,
            contract,
            parent_contract,
        )
        target_clipped_count += dev_clipped + con_clipped
        cases.append(
            _evaluate_case(
                truth_case,
                development,
                confirmation,
                development_target,
                confirmation_target,
                parent_contract,
            )
        )
    pure = [row for row in cases if row["truth_family"] != "mixed-abstain"]
    mixed = [row for row in cases if row["truth_family"] == "mixed-abstain"]
    common = contract["common_gates"]
    correct_count = sum(row["decision"] == row["truth_family"] for row in pure)
    sigma_count = sum(
        row["physical_sigma_exact"] is True
        for row in pure
        if row["truth_family"] == "physical-backing-return"
    )
    abstention_count = sum(row["decision"] == "abstain" for row in mixed)
    clipped_count = (
        int(development["clipped_scalar_count"])
        + int(confirmation["clipped_scalar_count"])
        + target_clipped_count
    )
    gate_results = {
        "pure_correct_family": correct_count
        >= int(common["pure_case_correct_family_count_min_per_regime"]),
        "pure_confirmation_error": max(row["best_confirmation_rmse"] for row in pure)
        <= float(regime["maximum_pure_confirmation_rmse"]),
        "wrong_family_separation": min(
            float(row["wrong_to_correct_rmse_ratio"]) for row in pure
        )
        >= float(common["minimum_wrong_to_correct_rmse_ratio_per_regime"]),
        "parameter_recovery": max(
            float(row["relative_parameter_error"]) for row in pure
        )
        <= float(regime["maximum_relative_parameter_error"]),
        "physical_sigma_recovery": sigma_count
        >= int(common["physical_sigma_exact_count_min_per_regime"]),
        "mixed_abstention": abstention_count
        >= int(common["mixed_case_abstention_count_min_per_regime"]),
        "mixed_residual_floor": min(row["best_confirmation_rmse"] for row in mixed)
        >= float(common["minimum_mixed_best_confirmation_rmse_per_regime"]),
        "no_quantization_clipping": clipped_count
        <= int(common["maximum_quantization_clipped_scalar_count"]),
    }
    return {
        "regime": regime,
        "source_observation": {
            "development_clipped_scalar_count": development["clipped_scalar_count"],
            "confirmation_clipped_scalar_count": confirmation["clipped_scalar_count"],
            "development_realized_gain_jitter_range": [
                development["minimum_realized_gain_jitter"],
                development["maximum_realized_gain_jitter"],
            ],
            "confirmation_realized_gain_jitter_range": [
                confirmation["minimum_realized_gain_jitter"],
                confirmation["maximum_realized_gain_jitter"],
            ],
            "target_clipped_scalar_count": target_clipped_count,
        },
        "case_results": cases,
        "summary": {
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
        },
        "gate_results": gate_results,
        "passed": all(gate_results.values()),
    }


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract, parent_contract, p3s_contract = load_contract(root, contract_path)
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
            parent_contract,
        )
        for regime in contract["regimes"]
    ]
    role_separation = pattern_overlap <= int(
        contract["common_gates"]["development_confirmation_pattern_overlap_count_max"]
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
            "fit_reads_confirmation": False,
        },
        "regime_results": regimes,
        "global_gate_results": {
            "all_regimes_pass": all(row["passed"] for row in regimes),
            "role_separation": role_separation,
        },
        "passed": all(row["passed"] for row in regimes) and role_separation,
        "decision": (
            "retain-p3t-under-frozen-uncertainty-envelope"
            if all(row["passed"] for row in regimes) and role_separation
            else "close-failed-p3t-uncertainty-envelope"
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
    "MechanismUncertaintyError",
    "evaluate",
    "load_contract",
    "write_report",
]
