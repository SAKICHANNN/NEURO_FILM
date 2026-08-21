"""SF3.A2 three-stock global K=1 explicit-operator baseline core."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from skimage.color import rgb2lab

from src.real_film.gold_transform_consistency import (
    ColorOperator,
    PairedFrameSamples,
    evaluate_operator,
    fit_full_affine,
    fit_per_channel_affine,
    fit_seplut17,
)

CONTRACT_SCHEMA = "neuro-film.sf3-a2-three-stock-k1-baseline-contract.v1"
REPORT_SCHEMA = "neuro-film.sf3-a2-three-stock-k1-baseline-report.v1"
SINGLE_STOCK_REPORT_SCHEMA = "neuro-film.sf3-a2-single-stock-k1-baseline-report.v1"


class ThreeStockK1BaselineError(ValueError):
    """Raised when the frozen three-stock K=1 experiment contract is invalid."""


@dataclass(frozen=True)
class StockFrameSamples:
    """One aligned scene sample from one physical stock observation."""

    scene_id: str
    frame_id: str
    roll_id: str
    source: np.ndarray
    target: np.ndarray

    def __post_init__(self) -> None:
        source = _pixels(self.source)
        target = _pixels(self.target)
        if source.shape != target.shape:
            raise ThreeStockK1BaselineError("source and target sample shapes differ")
        if not self.scene_id or not self.frame_id or not self.roll_id:
            raise ThreeStockK1BaselineError("sample identities must be non-empty")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)


def _pixels(value: np.ndarray) -> np.ndarray:
    pixels = np.asarray(value, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 3 or len(pixels) < 4:
        raise ThreeStockK1BaselineError("RGB samples must have shape (N>=4, 3)")
    if not np.isfinite(pixels).all() or np.min(pixels) < 0.0 or np.max(pixels) > 1.0:
        raise ThreeStockK1BaselineError("RGB samples must be finite and bounded")
    return pixels


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


def load_contract(path: Path, *, root: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    contract = json.loads(raw)
    if not isinstance(contract, dict) or contract.get("schema") != CONTRACT_SCHEMA:
        raise ThreeStockK1BaselineError("unsupported SF3.A2 contract")
    expected = ["fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"]
    if contract.get("required_stocks") != expected:
        raise ThreeStockK1BaselineError("SF3.A2 stock order drift")
    parent = contract.get("parent", {}).get("integrity_contract", {})
    parent_path = root / Path(str(parent.get("path", "")))
    if not parent_path.is_file() or _sha256_file(parent_path) != parent.get("sha256"):
        raise ThreeStockK1BaselineError("SF3.A1 parent contract hash drift")
    candidates = contract.get("operator_selection", {}).get(
        "candidate_order_simplest_first"
    )
    if candidates != [
        "bounded_per_channel_affine",
        "bounded_ridge_3x3_affine",
        "seplut17_monotone_plus_bounded_ridge_3x3",
    ]:
        raise ThreeStockK1BaselineError("SF3.A2 candidate order drift")
    if (
        contract["operator_selection"].get(
            "confirmation_target_colour_fit_or_score_access_before_selection_frozen"
        )
        != 0
    ):
        raise ThreeStockK1BaselineError("confirmation target fit/score gate drift")
    return raw, contract


def _paired(frames: Sequence[StockFrameSamples]) -> list[PairedFrameSamples]:
    return [
        PairedFrameSamples(frame.frame_id, frame.roll_id, frame.source, frame.target)
        for frame in frames
    ]


def _fit(
    name: str,
    frames: Sequence[StockFrameSamples],
    contract: Mapping[str, Any],
) -> ColorOperator:
    paired = _paired(frames)
    bounds = contract["operator_bounds"]
    working_space = str(contract["working_space"])
    if name == "bounded_per_channel_affine":
        return fit_per_channel_affine(paired, bounds, working_space)
    if name == "bounded_ridge_3x3_affine":
        return fit_full_affine(paired, bounds, working_space)
    if name == "seplut17_monotone_plus_bounded_ridge_3x3":
        return fit_seplut17(paired, bounds, working_space)
    raise ThreeStockK1BaselineError(f"unsupported candidate: {name}")


def _mean_error(operator: ColorOperator, frames: Sequence[StockFrameSamples]) -> float:
    return float(
        evaluate_operator(operator, _paired(frames))["mean_delta_e76_to_target"]
    )


def _select_operator(
    development: Sequence[StockFrameSamples], contract: Mapping[str, Any]
) -> tuple[str, dict[str, float]]:
    rolls = sorted({frame.roll_id for frame in development})
    minimum = int(contract["operator_selection"]["minimum_development_rolls_per_stock"])
    if len(rolls) < minimum:
        raise ThreeStockK1BaselineError("insufficient development rolls")
    candidates = contract["operator_selection"]["candidate_order_simplest_first"]
    fold_errors: dict[str, list[float]] = {name: [] for name in candidates}
    for held_roll in rolls:
        train = [frame for frame in development if frame.roll_id != held_roll]
        held = [frame for frame in development if frame.roll_id == held_roll]
        if not train or not held:
            raise ThreeStockK1BaselineError("empty development selection fold")
        for name in candidates:
            fold_errors[name].append(_mean_error(_fit(name, train, contract), held))
    means = {name: float(np.mean(values)) for name, values in fold_errors.items()}
    best = min(means.values())
    margin = float(contract["operator_selection"]["simplest_relative_error_margin"])
    selected = next(name for name in candidates if means[name] <= best * (1.0 + margin))
    return selected, means


def _frame_errors(
    operator: ColorOperator, frames: Sequence[StockFrameSamples]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_errors: list[float] = []
    identity_errors: list[float] = []
    clip_fractions: list[float] = []
    for frame in frames:
        raw = operator.apply(frame.source)
        if not np.isfinite(raw).all():
            raise ThreeStockK1BaselineError("operator produced non-finite output")
        clipped = np.clip(raw, 0.0, 1.0)
        target_lab = rgb2lab(frame.target.reshape(-1, 1, 3)).reshape(-1, 3)
        output_lab = rgb2lab(clipped.reshape(-1, 1, 3)).reshape(-1, 3)
        source_lab = rgb2lab(frame.source.reshape(-1, 1, 3)).reshape(-1, 3)
        target_errors.append(
            float(np.mean(np.linalg.norm(output_lab - target_lab, axis=1)))
        )
        identity_errors.append(
            float(np.mean(np.linalg.norm(source_lab - target_lab, axis=1)))
        )
        clip_fractions.append(float(np.mean((raw < 0.0) | (raw > 1.0))))
    return (
        np.asarray(target_errors),
        np.asarray(identity_errors),
        np.asarray(clip_fractions),
    )


def _validate_population(
    stocks: Sequence[str],
    development: Mapping[str, Sequence[StockFrameSamples]],
    confirmation: Mapping[str, Sequence[StockFrameSamples]],
) -> list[str]:
    if set(development) != set(stocks) or set(confirmation) != set(stocks):
        raise ThreeStockK1BaselineError("stock population drift")
    common_scenes: set[str] | None = None
    common_sources: dict[str, np.ndarray] = {}
    for stock in stocks:
        dev_ids = {frame.frame_id for frame in development[stock]}
        conf_ids = {frame.frame_id for frame in confirmation[stock]}
        if len(dev_ids) != len(development[stock]) or len(conf_ids) != len(
            confirmation[stock]
        ):
            raise ThreeStockK1BaselineError("duplicate frame identity")
        if dev_ids & conf_ids:
            raise ThreeStockK1BaselineError("development/confirmation frame leakage")
        scenes = {frame.scene_id for frame in confirmation[stock]}
        common_scenes = scenes if common_scenes is None else common_scenes & scenes
        for frame in confirmation[stock]:
            prior = common_sources.get(frame.scene_id)
            if prior is None:
                common_sources[frame.scene_id] = frame.source
            elif not np.array_equal(prior, frame.source):
                raise ThreeStockK1BaselineError(
                    "confirmation source samples differ across stocks"
                )
    exact_scene_sets = {
        frozenset(frame.scene_id for frame in confirmation[stock]) for stock in stocks
    }
    if len(exact_scene_sets) != 1 or not common_scenes:
        raise ThreeStockK1BaselineError("confirmation scene sets differ across stocks")
    return sorted(common_scenes)


def _validate_single_stock_population(
    development: Sequence[StockFrameSamples],
    confirmation: Sequence[StockFrameSamples],
) -> None:
    dev_ids = {frame.frame_id for frame in development}
    conf_ids = {frame.frame_id for frame in confirmation}
    if len(dev_ids) != len(development) or len(conf_ids) != len(confirmation):
        raise ThreeStockK1BaselineError("duplicate frame identity")
    if dev_ids & conf_ids:
        raise ThreeStockK1BaselineError("development/confirmation frame leakage")
    conf_scenes = {frame.scene_id for frame in confirmation}
    if len(conf_scenes) != len(confirmation):
        raise ThreeStockK1BaselineError("duplicate confirmation scene identity")


def evaluate_single_stock(
    contract_path: Path,
    *,
    root: Path,
    stock: str,
    development: Sequence[StockFrameSamples],
    confirmation: Sequence[StockFrameSamples],
) -> dict[str, Any]:
    """Evaluate one stock as K=1 without claiming cross-stock evidence.

    This is a lower-claim execution entry for the acquisition order in which one
    stock can become ready before the complete three-stock population.  It uses
    the exact frozen SF3.A2 selection family and within-stock gates; only the
    wrong-stock and pairwise-distinguishability gates remain unavailable.
    """

    contract_raw, contract = load_contract(contract_path, root=root)
    if stock not in contract["required_stocks"]:
        raise ThreeStockK1BaselineError("unsupported stock identity")
    frames = list(confirmation)
    dev_frames = list(development)
    _validate_single_stock_population(dev_frames, frames)

    selected, selection_errors = _select_operator(dev_frames, contract)
    operator = _fit(selected, dev_frames, contract)
    gates_cfg = contract["gates"]
    if len(frames) < int(gates_cfg["minimum_confirmation_frames_per_stock"]):
        raise ThreeStockK1BaselineError("insufficient confirmation frames")
    error, identity, clipping = _frame_errors(operator, frames)
    relative = (identity - error) / np.maximum(identity, 1e-12)
    metrics = {
        "confirmation_frames": len(frames),
        "improvement_rate_over_identity": float(np.mean(error < identity)),
        "median_relative_improvement_over_identity": float(np.median(relative)),
        "worst_relative_improvement_over_identity": float(np.min(relative)),
        "mean_delta_e76_to_target": float(np.mean(error)),
        "mean_identity_delta_e76_to_target": float(np.mean(identity)),
        "maximum_raw_output_clip_fraction": float(np.max(clipping)),
    }
    checks = {
        "confirmation_support": len(frames)
        >= int(gates_cfg["minimum_confirmation_frames_per_stock"]),
        "improvement_rate": metrics["improvement_rate_over_identity"]
        >= float(gates_cfg["minimum_confirmation_improvement_rate_over_identity"]),
        "median_improvement": metrics["median_relative_improvement_over_identity"]
        >= float(gates_cfg["minimum_median_relative_improvement_over_identity"]),
        "worst_tail": metrics["worst_relative_improvement_over_identity"]
        >= float(gates_cfg["minimum_worst_relative_improvement_over_identity"]),
        "raw_output_clip": metrics["maximum_raw_output_clip_fraction"]
        <= float(gates_cfg["maximum_raw_output_clip_fraction"]),
    }
    automatic_pass = all(checks.values())
    core = {
        "schema": SINGLE_STOCK_REPORT_SCHEMA,
        "experiment_id": f"{contract['experiment_id']}.{stock}",
        "contract_sha256": _sha256(contract_raw),
        "stock": stock,
        "selection_used_confirmation_targets": False,
        "confirmation_frames_evaluated_after_selection_frozen": len(frames),
        "selected_operator": selected,
        "development_roll_heldout_mean_delta_e76": selection_errors,
        "operator": operator.to_dict(),
        "metrics": metrics,
        "gates": checks,
        "wrong_stock_control_evaluated": False,
        "cross_stock_distinguishability_evaluated": False,
        "automatic_pass": automatic_pass,
        "decision": (
            "RETAIN_SINGLE_STOCK_K1_CANDIDATE_PENDING_THREE_STOCK_CONTROLS"
            if automatic_pass
            else "RETAIN_SINGLE_STOCK_IDENTITY_BASELINE_WITHOUT_CAPACITY_OR_ROUTER_RESCUE"
        ),
        "adaptive_or_retrieval_models_fitted": 0,
        "latent_modes_fitted": 0,
        "claim_ceiling": (
            "Controlled paired development and sealed-confirmation evidence for "
            "one stock-labelled K=1 explicit operator only. Passing does not "
            "establish wrong-stock rejection, stock distinguishability, calibrated "
            "stock response, population preference, product promotion, adaptive "
            "routing, K>1 or multi-stock completion."
        ),
    }
    return {**core, "stable_evidence_id": _sha256(_canonical(core))}


def evaluate(
    contract_path: Path,
    *,
    root: Path,
    development: Mapping[str, Sequence[StockFrameSamples]],
    confirmation: Mapping[str, Sequence[StockFrameSamples]],
) -> dict[str, Any]:
    """Select three independent global operators, then read confirmation once."""

    contract_raw, contract = load_contract(contract_path, root=root)
    stocks = list(contract["required_stocks"])
    common_scenes = _validate_population(stocks, development, confirmation)
    selected_names: dict[str, str] = {}
    selection_errors: dict[str, dict[str, float]] = {}
    operators: dict[str, ColorOperator] = {}
    for stock in stocks:
        selected, errors = _select_operator(development[stock], contract)
        selected_names[stock] = selected
        selection_errors[stock] = errors
        operators[stock] = _fit(selected, development[stock], contract)

    gates_cfg = contract["gates"]
    stock_results: dict[str, Any] = {}
    for stock in stocks:
        frames = list(confirmation[stock])
        if len(frames) < int(gates_cfg["minimum_confirmation_frames_per_stock"]):
            raise ThreeStockK1BaselineError("insufficient confirmation frames")
        error, identity, clipping = _frame_errors(operators[stock], frames)
        relative = (identity - error) / np.maximum(identity, 1e-12)
        wrong_errors = []
        for frame in frames:
            per_wrong = []
            for other in stocks:
                if other == stock:
                    continue
                raw = np.clip(operators[other].apply(frame.source), 0.0, 1.0)
                target_lab = rgb2lab(frame.target.reshape(-1, 1, 3)).reshape(-1, 3)
                output_lab = rgb2lab(raw.reshape(-1, 1, 3)).reshape(-1, 3)
                per_wrong.append(
                    float(np.mean(np.linalg.norm(output_lab - target_lab, axis=1)))
                )
            wrong_errors.append(float(np.median(per_wrong)))
        wrong = np.asarray(wrong_errors)
        metrics = {
            "confirmation_frames": len(frames),
            "improvement_rate_over_identity": float(np.mean(error < identity)),
            "median_relative_improvement_over_identity": float(np.median(relative)),
            "worst_relative_improvement_over_identity": float(np.min(relative)),
            "correct_operator_win_rate_over_median_wrong_stock": float(
                np.mean(error < wrong)
            ),
            "mean_delta_e76_to_target": float(np.mean(error)),
            "mean_identity_delta_e76_to_target": float(np.mean(identity)),
            "maximum_raw_output_clip_fraction": float(np.max(clipping)),
        }
        checks = {
            "confirmation_support": len(frames)
            >= int(gates_cfg["minimum_confirmation_frames_per_stock"]),
            "improvement_rate": metrics["improvement_rate_over_identity"]
            >= float(gates_cfg["minimum_confirmation_improvement_rate_over_identity"]),
            "median_improvement": metrics["median_relative_improvement_over_identity"]
            >= float(gates_cfg["minimum_median_relative_improvement_over_identity"]),
            "worst_tail": metrics["worst_relative_improvement_over_identity"]
            >= float(gates_cfg["minimum_worst_relative_improvement_over_identity"]),
            "wrong_stock_control": metrics[
                "correct_operator_win_rate_over_median_wrong_stock"
            ]
            >= float(
                gates_cfg["minimum_correct_operator_win_rate_over_median_wrong_stock"]
            ),
            "raw_output_clip": metrics["maximum_raw_output_clip_fraction"]
            <= float(gates_cfg["maximum_raw_output_clip_fraction"]),
        }
        stock_results[stock] = {
            "selected_operator": selected_names[stock],
            "development_roll_heldout_mean_delta_e76": selection_errors[stock],
            "operator": operators[stock].to_dict(),
            "metrics": metrics,
            "gates": checks,
            "automatic_pass": all(checks.values()),
        }

    pairwise: dict[str, Any] = {}
    for left_index, left in enumerate(stocks):
        for right in stocks[left_index + 1 :]:
            values = []
            left_frames = {frame.scene_id: frame for frame in confirmation[left]}
            for scene_id in common_scenes:
                source = left_frames[scene_id].source
                left_output = np.clip(operators[left].apply(source), 0.0, 1.0)
                right_output = np.clip(operators[right].apply(source), 0.0, 1.0)
                left_lab = rgb2lab(left_output.reshape(-1, 1, 3)).reshape(-1, 3)
                right_lab = rgb2lab(right_output.reshape(-1, 1, 3)).reshape(-1, 3)
                values.extend(np.linalg.norm(left_lab - right_lab, axis=1).tolist())
            median = float(np.median(values))
            pairwise[f"{left}__{right}"] = {
                "median_output_delta_e76": median,
                "automatic_pass": median
                >= float(gates_cfg["minimum_pairwise_stock_output_median_delta_e76"]),
            }

    stock_pass = all(value["automatic_pass"] for value in stock_results.values())
    pair_pass = all(value["automatic_pass"] for value in pairwise.values())
    automatic_pass = stock_pass and pair_pass
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _sha256(contract_raw),
        "selection_used_confirmation_targets": False,
        "outer_runner_must_decode_confirmation_targets_after_selection_freeze": True,
        "confirmation_frames_evaluated_after_all_selections_frozen": sum(
            len(confirmation[stock]) for stock in stocks
        ),
        "common_confirmation_scenes": common_scenes,
        "stocks": stock_results,
        "pairwise_stock_output_distinguishability": pairwise,
        "all_three_stocks_pass": stock_pass,
        "all_three_stock_pairs_distinguishable": pair_pass,
        "automatic_pass": automatic_pass,
        "decision": contract[
            "decision_if_pass" if automatic_pass else "decision_if_fail"
        ],
        "adaptive_or_retrieval_models_fitted": 0,
        "latent_modes_fitted": 0,
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _sha256(_canonical(core))}


__all__ = [
    "CONTRACT_SCHEMA",
    "REPORT_SCHEMA",
    "SINGLE_STOCK_REPORT_SCHEMA",
    "StockFrameSamples",
    "ThreeStockK1BaselineError",
    "evaluate",
    "evaluate_single_stock",
    "load_contract",
]
