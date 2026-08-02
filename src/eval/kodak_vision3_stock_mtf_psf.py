"""U5.R2BU1 held-frequency positive-PSF stock-bank evaluator."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.kodak_vision3_mtf_diversity import (
    CHANNELS,
    STOCKS,
    _log_value_from_pixel,
    canonical_json,
    hash_file,
    load_trace,
)

SCHEMA = "neuro_film.u5_r2bu1_kodak_vision3_stock_mtf_psf_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2bu1_kodak_vision3_stock_mtf_psf_report.v1"
BUNDLE_SCHEMA = "neuro_film.vision3_stock_mtf_gaussian_psf_bundle.v1"


class StockMtfPsfError(RuntimeError):
    """Raised when the frozen BU1 evidence or protocol drifts."""


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise StockMtfPsfError("BU1 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload.get("parent", {})
    protocol = payload.get("protocol", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != "U5.R2BU1"
        or parent.get("decision_sha256")
        != "eae84870c4c3706228fecbe5696825c7d500020558dab94d8b6177baabe1848e"
        or parent.get("stable_evidence_id")
        != "7056a6888d4ce72106419c41041cd3fc5e9f54149ccf0d83609b042679f629c6"
        or parent.get("trace_sha256")
        != "d88c0e698c0e0969696224df50f1573ed78a6de0a894075f6502d9b3f658a794"
        or tuple(protocol.get("stocks", ())) != STOCKS
        or tuple(protocol.get("channels", ())) != CHANNELS
        or protocol.get("common_frequencies_cycles_per_mm")
        != [25.0, 30.0, 35.0, 40.0, 45.0, 55.0, 60.0, 65.0]
        or protocol.get("development_frequencies_cycles_per_mm")
        != [25.0, 35.0, 45.0, 60.0]
        or protocol.get("confirmation_frequencies_cycles_per_mm")
        != [30.0, 40.0, 55.0, 65.0]
        or protocol.get("sigma_grid_um")
        != {"minimum": 0.25, "maximum": 25.0, "step": 0.025}
        or gates
        != {
            "required_stock_count": 3,
            "required_channel_count": 3,
            "required_development_frequency_count": 4,
            "required_confirmation_frequency_count": 4,
            "maximum_candidate_confirmation_median_rmse": 0.05,
            "maximum_candidate_confirmation_worst_rmse": 0.08,
            "minimum_overall_improvement_over_shared_fraction": 0.1,
            "minimum_each_stock_improvement_over_shared_fraction": 0.0,
            "minimum_channels_beating_shared_per_stock": 2,
            "minimum_overall_improvement_over_wrong_stock_fraction": 0.1,
            "minimum_each_stock_improvement_over_wrong_stock_fraction": 0.0,
            "minimum_rows_beating_identity": 9,
            "require_all_sigmas_strictly_inside_grid": True,
            "minimum_mtf_response": 0.0,
            "maximum_mtf_response": 1.0,
            "two_byte_identical_audits": True,
        }
    ):
        raise StockMtfPsfError("BU1 frozen contract drift")
    for key in ("decision_path", "trace_path"):
        _relative_path(str(parent.get(key, "")))
    wrong = protocol.get("wrong_stock_control", {})
    if tuple(wrong) != STOCKS or any(wrong[stock] not in STOCKS for stock in STOCKS):
        raise StockMtfPsfError("BU1 wrong-stock control drift")
    return payload


def gaussian_mtf(frequencies_cycles_per_mm: np.ndarray, sigma_um: float) -> np.ndarray:
    frequencies = np.asarray(frequencies_cycles_per_mm, dtype=np.float64)
    if (
        frequencies.ndim != 1
        or not np.all(np.isfinite(frequencies))
        or np.any(frequencies < 0.0)
        or not math.isfinite(float(sigma_um))
        or sigma_um < 0.0
    ):
        raise StockMtfPsfError("BU1 invalid Gaussian MTF request")
    sigma_mm = float(sigma_um) * 1e-3
    return np.exp(-2.0 * math.pi**2 * np.square(sigma_mm * frequencies))


def _sigma_grid(protocol: Mapping[str, Any]) -> np.ndarray:
    spec = protocol["sigma_grid_um"]
    minimum = float(spec["minimum"])
    maximum = float(spec["maximum"])
    step = float(spec["step"])
    count = round((maximum - minimum) / step) + 1
    grid = minimum + step * np.arange(count, dtype=np.float64)
    grid[-1] = maximum
    if len(grid) != 991 or not np.all(np.diff(grid) > 0.0):
        raise StockMtfPsfError("BU1 invalid sigma grid")
    return grid


def fit_sigma(
    frequencies_cycles_per_mm: np.ndarray,
    target_response: np.ndarray,
    sigma_grid_um: np.ndarray,
) -> tuple[float, float]:
    frequencies = np.asarray(frequencies_cycles_per_mm, dtype=np.float64)
    target = np.asarray(target_response, dtype=np.float64)
    grid = np.asarray(sigma_grid_um, dtype=np.float64)
    if (
        frequencies.ndim != 1
        or target.shape != frequencies.shape
        or grid.ndim != 1
        or len(grid) < 2
        or not np.all(np.isfinite(target))
        or np.any(target < 0.0)
        or np.any(target > 1.1)
    ):
        raise StockMtfPsfError("BU1 invalid sigma fit")
    sigma_mm = grid[:, None] * 1e-3
    predictions = np.exp(-2.0 * math.pi**2 * np.square(sigma_mm * frequencies[None, :]))
    losses = np.mean(np.square(predictions - target[None, :]), axis=1)
    index = int(np.argmin(losses))
    return float(grid[index]), float(losses[index])


def _common_responses(
    trace: Mapping[str, Any], common: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    responses: dict[str, dict[str, np.ndarray]] = {}
    for stock in STOCKS:
        row = trace["stocks"][stock]
        axes = row["graph_axes"]
        responses[stock] = {}
        for channel in CHANNELS:
            coordinates = np.asarray(row["curves"][channel], dtype=np.float64)
            frequencies = np.asarray(
                [
                    _log_value_from_pixel(float(x), axes["x_value_pixels"])
                    for x in coordinates[:, 0]
                ],
                dtype=np.float64,
            )
            source_response = np.asarray(
                [
                    _log_value_from_pixel(float(y), axes["y_value_pixels"]) / 100.0
                    for y in coordinates[:, 1]
                ],
                dtype=np.float64,
            )
            common_response = np.exp(
                np.interp(
                    np.log10(common),
                    np.log10(frequencies),
                    np.log(source_response),
                )
            )
            if (
                not np.all(np.isfinite(common_response))
                or np.any(common_response <= 0.0)
                or np.any(common_response > 1.1)
            ):
                raise StockMtfPsfError(f"BU1 invalid source response: {stock}/{channel}")
            responses[stock][channel] = common_response
    return responses


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


def evaluate_stock_psf(
    config: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent = config["parent"]
    decision_path = root / _relative_path(str(parent["decision_path"]))
    trace_path = root / _relative_path(str(parent["trace_path"]))
    if (
        not decision_path.is_file()
        or hash_file(decision_path) != parent["decision_sha256"]
        or not trace_path.is_file()
        or hash_file(trace_path) != parent["trace_sha256"]
    ):
        raise StockMtfPsfError("BU1 parent evidence integrity mismatch")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        decision.get("formal_evidence", {}).get("stable_evidence_id")
        != parent["stable_evidence_id"]
        or decision.get("decision")
        != "retain_source_domain_mtf_signature_and_open_joint_physical_prior"
        or not decision.get("result", {}).get("diversity_pass")
    ):
        raise StockMtfPsfError("BU1 parent decision mismatch")
    trace = load_trace(trace_path)
    protocol = config["protocol"]
    gates = config["gates"]
    common = np.asarray(protocol["common_frequencies_cycles_per_mm"], dtype=np.float64)
    development = np.asarray(
        [common.tolist().index(value) for value in protocol["development_frequencies_cycles_per_mm"]],
        dtype=np.int64,
    )
    confirmation = np.asarray(
        [common.tolist().index(value) for value in protocol["confirmation_frequencies_cycles_per_mm"]],
        dtype=np.int64,
    )
    if len(np.intersect1d(development, confirmation)) != 0 or len(
        np.union1d(development, confirmation)
    ) != len(common):
        raise StockMtfPsfError("BU1 frequency split drift")
    responses = _common_responses(trace, common)
    sigma_grid = _sigma_grid(protocol)

    candidate_sigmas: dict[str, dict[str, float]] = {stock: {} for stock in STOCKS}
    development_losses: dict[str, dict[str, float]] = {stock: {} for stock in STOCKS}
    for stock in STOCKS:
        for channel in CHANNELS:
            sigma, loss = fit_sigma(
                common[development], responses[stock][channel][development], sigma_grid
            )
            candidate_sigmas[stock][channel] = sigma
            development_losses[stock][channel] = loss

    shared_sigmas: dict[str, float] = {}
    shared_losses: dict[str, float] = {}
    for channel in CHANNELS:
        frequencies = np.tile(common[development], len(STOCKS))
        target = np.concatenate(
            [responses[stock][channel][development] for stock in STOCKS]
        )
        sigma, loss = fit_sigma(frequencies, target, sigma_grid)
        shared_sigmas[channel] = sigma
        shared_losses[channel] = loss

    rows: list[dict[str, Any]] = []
    candidate_errors: list[float] = []
    shared_errors: list[float] = []
    wrong_errors: list[float] = []
    identity_errors: list[float] = []
    per_stock: dict[str, Any] = {}
    for stock in STOCKS:
        stock_candidate: list[float] = []
        stock_shared: list[float] = []
        stock_wrong: list[float] = []
        channels_beating_shared = 0
        wrong_stock = protocol["wrong_stock_control"][stock]
        for channel in CHANNELS:
            target = responses[stock][channel][confirmation]
            candidate_prediction = gaussian_mtf(
                common[confirmation], candidate_sigmas[stock][channel]
            )
            shared_prediction = gaussian_mtf(
                common[confirmation], shared_sigmas[channel]
            )
            wrong_prediction = gaussian_mtf(
                common[confirmation], candidate_sigmas[wrong_stock][channel]
            )
            identity_prediction = np.ones_like(target)
            candidate_rmse = _rmse(candidate_prediction, target)
            shared_rmse = _rmse(shared_prediction, target)
            wrong_rmse = _rmse(wrong_prediction, target)
            identity_rmse = _rmse(identity_prediction, target)
            channels_beating_shared += int(candidate_rmse < shared_rmse)
            stock_candidate.append(candidate_rmse)
            stock_shared.append(shared_rmse)
            stock_wrong.append(wrong_rmse)
            candidate_errors.append(candidate_rmse)
            shared_errors.append(shared_rmse)
            wrong_errors.append(wrong_rmse)
            identity_errors.append(identity_rmse)
            rows.append(
                {
                    "stock": stock,
                    "channel": channel,
                    "wrong_stock": wrong_stock,
                    "sigma_um": candidate_sigmas[stock][channel],
                    "shared_sigma_um": shared_sigmas[channel],
                    "development_mse": development_losses[stock][channel],
                    "target_confirmation": target.tolist(),
                    "candidate_confirmation": candidate_prediction.tolist(),
                    "candidate_rmse": candidate_rmse,
                    "shared_rmse": shared_rmse,
                    "wrong_stock_rmse": wrong_rmse,
                    "identity_rmse": identity_rmse,
                    "candidate_beats_shared": candidate_rmse < shared_rmse,
                    "candidate_beats_wrong_stock": candidate_rmse < wrong_rmse,
                    "candidate_beats_identity": candidate_rmse < identity_rmse,
                }
            )
        candidate_median = float(np.median(stock_candidate))
        shared_median = float(np.median(stock_shared))
        wrong_median = float(np.median(stock_wrong))
        per_stock[stock] = {
            "candidate_median_rmse": candidate_median,
            "shared_median_rmse": shared_median,
            "wrong_stock_median_rmse": wrong_median,
            "improvement_over_shared_fraction": 1.0 - candidate_median / shared_median,
            "improvement_over_wrong_stock_fraction": 1.0 - candidate_median / wrong_median,
            "channels_beating_shared": channels_beating_shared,
        }

    candidate_array = np.asarray(candidate_errors, dtype=np.float64)
    shared_array = np.asarray(shared_errors, dtype=np.float64)
    wrong_array = np.asarray(wrong_errors, dtype=np.float64)
    overall_shared_improvement = 1.0 - float(np.median(candidate_array)) / float(
        np.median(shared_array)
    )
    overall_wrong_improvement = 1.0 - float(np.median(candidate_array)) / float(
        np.median(wrong_array)
    )
    sigma_min = float(protocol["sigma_grid_um"]["minimum"])
    sigma_max = float(protocol["sigma_grid_um"]["maximum"])
    all_sigmas = [
        candidate_sigmas[stock][channel] for stock in STOCKS for channel in CHANNELS
    ] + list(shared_sigmas.values())
    probe_frequencies = np.linspace(0.0, float(common[-1]), 257, dtype=np.float64)
    probe_responses = np.concatenate(
        [gaussian_mtf(probe_frequencies, sigma) for sigma in all_sigmas]
    )
    gate_results = {
        "row_count": len(rows) == len(STOCKS) * len(CHANNELS),
        "candidate_median_rmse": float(np.median(candidate_array))
        <= float(gates["maximum_candidate_confirmation_median_rmse"]),
        "candidate_worst_rmse": float(np.max(candidate_array))
        <= float(gates["maximum_candidate_confirmation_worst_rmse"]),
        "overall_shared_improvement": overall_shared_improvement
        >= float(gates["minimum_overall_improvement_over_shared_fraction"]),
        "every_stock_shared_improvement": all(
            row["improvement_over_shared_fraction"]
            > float(gates["minimum_each_stock_improvement_over_shared_fraction"])
            for row in per_stock.values()
        ),
        "every_stock_shared_channel_count": all(
            row["channels_beating_shared"]
            >= int(gates["minimum_channels_beating_shared_per_stock"])
            for row in per_stock.values()
        ),
        "overall_wrong_stock_improvement": overall_wrong_improvement
        >= float(gates["minimum_overall_improvement_over_wrong_stock_fraction"]),
        "every_stock_wrong_improvement": all(
            row["improvement_over_wrong_stock_fraction"]
            > float(gates["minimum_each_stock_improvement_over_wrong_stock_fraction"])
            for row in per_stock.values()
        ),
        "identity_control": sum(row["candidate_beats_identity"] for row in rows)
        >= int(gates["minimum_rows_beating_identity"]),
        "sigma_grid_interior": all(sigma_min < sigma < sigma_max for sigma in all_sigmas),
        "positive_bounded_mtf": bool(
            np.all(probe_responses >= float(gates["minimum_mtf_response"]))
            and np.all(probe_responses <= float(gates["maximum_mtf_response"]))
        ),
    }
    passed = all(gate_results.values())
    bundle_core = {
        "schema": BUNDLE_SCHEMA,
        "source_decision_sha256": parent["decision_sha256"],
        "source_trace_sha256": parent["trace_sha256"],
        "domain": {
            "operator": "positive isotropic Gaussian PSF",
            "sigma_unit": "micrometre",
            "frequency_unit": "cycles/mm",
            "mtf_equation": "exp(-2*pi^2*(sigma_um*1e-3*f)^2)",
            "placement": "unresolved",
        },
        "stocks": {
            stock: {
                "measurement_context": trace["stocks"][stock]["measurement_context"],
                "channel_sigma_um": candidate_sigmas[stock],
            }
            for stock in STOCKS
        },
        "shared_control_sigma_um": shared_sigmas,
        "development_frequencies_cycles_per_mm": common[development].tolist(),
        "confirmation_frequencies_cycles_per_mm": common[confirmation].tolist(),
        "claim_ceiling": config["claim_ceiling"],
    }
    bundle = {
        **bundle_core,
        "bundle_id": hashlib.sha256(canonical_json(bundle_core)).hexdigest(),
    }
    bundle_sha256 = hashlib.sha256(canonical_json(bundle)).hexdigest()
    stable = {
        "experiment_id": config["experiment_id"],
        "parent_decision_sha256": parent["decision_sha256"],
        "trace_sha256": parent["trace_sha256"],
        "candidate_sigmas_um": candidate_sigmas,
        "shared_sigmas_um": shared_sigmas,
        "shared_development_mse": shared_losses,
        "confirmation_rows": rows,
        "per_stock": per_stock,
        "candidate_confirmation_median_rmse": float(np.median(candidate_array)),
        "candidate_confirmation_worst_rmse": float(np.max(candidate_array)),
        "shared_confirmation_median_rmse": float(np.median(shared_array)),
        "wrong_stock_confirmation_median_rmse": float(np.median(wrong_array)),
        "overall_improvement_over_shared_fraction": overall_shared_improvement,
        "overall_improvement_over_wrong_stock_fraction": overall_wrong_improvement,
        "identity_rows_beaten": sum(row["candidate_beats_identity"] for row in rows),
        "bundle_sha256": bundle_sha256,
        "gate_results": gate_results,
    }
    report = {
        "schema": REPORT_SCHEMA,
        **stable,
        "stock_psf_pass": passed,
        "stable_evidence_id": hashlib.sha256(canonical_json(stable)).hexdigest(),
        "decision": (
            "retain_isolated_stock_specific_positive_psf_source_prior"
            if passed
            else "close_one_gaussian_stock_mtf_bank_without_rescue"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    return report, bundle


__all__ = [
    "BUNDLE_SCHEMA",
    "REPORT_SCHEMA",
    "SCHEMA",
    "StockMtfPsfError",
    "evaluate_stock_psf",
    "fit_sigma",
    "gaussian_mtf",
    "load_contract",
]
