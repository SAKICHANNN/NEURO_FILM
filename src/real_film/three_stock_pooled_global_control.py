"""SF3.A2B pooled-global control for stock-specific K=1 operators."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.real_film.three_stock_k1_baseline import (
    StockFrameSamples,
    _fit,
    _frame_errors,
    _select_operator,
    _validate_population,
)
from src.real_film.three_stock_k1_baseline import (
    load_contract as load_k1_contract,
)

CONTRACT_SCHEMA = "neuro-film.sf3-a2b-three-stock-pooled-global-control-contract.v1"
REPORT_SCHEMA = "neuro-film.sf3-a2b-three-stock-pooled-global-control-report.v1"


class ThreeStockPooledGlobalControlError(ValueError):
    """Raised when the pooled-global comparison violates its frozen contract."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(
    path: Path, *, root: Path
) -> tuple[bytes, dict[str, Any], Path, dict[str, Any]]:
    raw = path.read_bytes()
    contract = json.loads(raw)
    if not isinstance(contract, dict) or contract.get("schema") != CONTRACT_SCHEMA:
        raise ThreeStockPooledGlobalControlError("unsupported SF3.A2B contract")
    parent = contract.get("parent", {}).get("k1_contract", {})
    parent_path = root / Path(str(parent.get("path", "")))
    if not parent_path.is_file() or _sha256_file(parent_path) != parent.get("sha256"):
        raise ThreeStockPooledGlobalControlError("SF3.A2 parent contract drift")
    _, k1_contract = load_k1_contract(parent_path, root=root)
    if contract.get("required_stocks") != k1_contract.get("required_stocks"):
        raise ThreeStockPooledGlobalControlError("stock order differs from SF3.A2")
    if (
        contract.get("pooled_selection", {}).get(
            "confirmation_target_access_before_all_selections_frozen"
        )
        != 0
    ):
        raise ThreeStockPooledGlobalControlError("confirmation access gate drift")
    return raw, contract, parent_path, k1_contract


def _pooled_development(
    stocks: Sequence[str],
    development: Mapping[str, Sequence[StockFrameSamples]],
) -> list[StockFrameSamples]:
    pooled: list[StockFrameSamples] = []
    for stock in stocks:
        for frame in development[stock]:
            pooled.append(
                StockFrameSamples(
                    scene_id=frame.scene_id,
                    frame_id=f"{stock}::{frame.frame_id}",
                    roll_id=f"{stock}::{frame.roll_id}",
                    source=frame.source,
                    target=frame.target,
                )
            )
    return pooled


def evaluate(
    contract_path: Path,
    *,
    root: Path,
    development: Mapping[str, Sequence[StockFrameSamples]],
    confirmation: Mapping[str, Sequence[StockFrameSamples]],
) -> dict[str, Any]:
    """Compare each SF3.A2 stock operator with one pooled global operator."""

    contract_raw, contract, k1_path, k1_contract = load_contract(
        contract_path, root=root
    )
    stocks = list(contract["required_stocks"])
    common_scenes = _validate_population(stocks, development, confirmation)

    stock_names: dict[str, str] = {}
    stock_selection_errors: dict[str, dict[str, float]] = {}
    stock_operators = {}
    for stock in stocks:
        name, errors = _select_operator(development[stock], k1_contract)
        stock_names[stock] = name
        stock_selection_errors[stock] = errors
        stock_operators[stock] = _fit(name, development[stock], k1_contract)

    pooled_development = _pooled_development(stocks, development)
    pooled_name, pooled_selection_errors = _select_operator(
        pooled_development, k1_contract
    )
    pooled_operator = _fit(pooled_name, pooled_development, k1_contract)

    gates = contract["gates"]
    stock_results: dict[str, Any] = {}
    for stock in stocks:
        frames = list(confirmation[stock])
        minimum = int(gates["minimum_confirmation_frames_per_stock"])
        if len(frames) < minimum:
            raise ThreeStockPooledGlobalControlError(
                f"insufficient confirmation frames for {stock}"
            )
        specific_error, _, specific_clip = _frame_errors(stock_operators[stock], frames)
        pooled_error, _, pooled_clip = _frame_errors(pooled_operator, frames)
        relative = (pooled_error - specific_error) / np.maximum(pooled_error, 1e-12)
        metrics = {
            "confirmation_frames": len(frames),
            "stock_specific_win_rate_over_pooled_global": float(
                np.mean(specific_error < pooled_error)
            ),
            "median_relative_improvement_over_pooled_global": float(
                np.median(relative)
            ),
            "worst_relative_improvement_over_pooled_global": float(np.min(relative)),
            "mean_stock_specific_delta_e76": float(np.mean(specific_error)),
            "mean_pooled_global_delta_e76": float(np.mean(pooled_error)),
            "maximum_stock_specific_raw_clip_fraction": float(np.max(specific_clip)),
            "maximum_pooled_global_raw_clip_fraction": float(np.max(pooled_clip)),
        }
        checks = {
            "confirmation_support": len(frames) >= minimum,
            "win_rate": metrics["stock_specific_win_rate_over_pooled_global"]
            >= float(gates["minimum_stock_specific_win_rate_over_pooled_global"]),
            "median_incremental_value": metrics[
                "median_relative_improvement_over_pooled_global"
            ]
            >= float(gates["minimum_median_relative_improvement_over_pooled_global"]),
            "worst_tail": metrics["worst_relative_improvement_over_pooled_global"]
            >= float(gates["minimum_worst_relative_improvement_over_pooled_global"]),
            "finite": all(np.isfinite(value) for value in metrics.values()),
        }
        stock_results[stock] = {
            "selected_stock_specific_operator": stock_names[stock],
            "development_roll_heldout_mean_delta_e76": stock_selection_errors[stock],
            "metrics": metrics,
            "gates": checks,
            "automatic_pass": all(checks.values()),
        }

    automatic_pass = all(row["automatic_pass"] for row in stock_results.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _sha256(contract_raw),
        "k1_contract_path": k1_path.relative_to(root).as_posix(),
        "k1_contract_sha256": _sha256_file(k1_path),
        "common_confirmation_scenes": common_scenes,
        "all_operator_selections_frozen_before_confirmation_evaluation": True,
        "confirmation_target_fit_or_selection_reads": 0,
        "selected_pooled_global_operator": pooled_name,
        "pooled_development_roll_heldout_mean_delta_e76": pooled_selection_errors,
        "pooled_operator": pooled_operator.to_dict(),
        "stock_results": stock_results,
        "adaptive_or_retrieval_models_fitted": 0,
        "latent_modes_fitted": 0,
        "automatic_pass": automatic_pass,
        "decision": (
            contract["decision_if_pass"]
            if automatic_pass
            else contract["decision_if_fail"]
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _sha256(_canonical(core))}


__all__ = [
    "CONTRACT_SCHEMA",
    "REPORT_SCHEMA",
    "ThreeStockPooledGlobalControlError",
    "evaluate",
    "load_contract",
]
