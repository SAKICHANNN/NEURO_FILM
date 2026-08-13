"""Four-stock source-only robustness after admitting Kodak VISION3 200T."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader

from src.eval.kodak_vision3_200t_raster_signature import STOCK as STOCK_200T
from src.eval.kodak_vision3_200t_raster_signature import (
    _axis_pixel,
    _axis_value,
    _extract_images,
    _trace_dark_curve,
)
from src.eval.kodak_vision3_200t_raster_signature import (
    load_contract as load_200t_contract,
)
from src.eval.vision3_source_signature_robustness import (
    CHANNEL_SUBSETS,
    GRANULARITY_GRID,
    MTF_GRID,
    PATTERNS,
    _distance,
    _load_locked_json,
    _offset,
    _profile_features,
    canonical_json,
    hash_file,
)
from src.film_physics.observed_source_profile import CHANNELS, STOCKS

SCHEMA = "neuro_film.u5-r2bu10-vision3-200t-profile-bank-robustness-contract.v1"
REPORT_SCHEMA = "neuro_film.u5-r2bu10-vision3-200t-profile-bank-robustness-report.v1"
ALL_STOCKS = (*STOCKS, STOCK_200T)


class ProfileBankRobustnessError(RuntimeError):
    """Raised when a frozen BU10 input or protocol drifts."""


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != "U5.R2BU10"
        or tuple(payload.get("comparison", {}).get("stocks", ())) != ALL_STOCKS
        or tuple(payload["comparison"].get("perturbation_patterns", ())) != PATTERNS
        or payload["comparison"].get("digitization_uncertainty_pixels") != 2.0
        or payload.get("gates")
        != {
            "required_case_count": 80,
            "minimum_joint_top1_accuracy": 0.95,
            "minimum_joint_recall_per_stock": 0.9,
            "minimum_joint_accuracy_per_channel_subset": 0.9,
            "minimum_normalized_worst_margin": 0.02,
            "maximum_cyclic_label_control_accuracy": 0.26,
            "legacy_cases_must_remain_correct": True,
            "two_byte_identical_audits": True,
            "no_render_authority": True,
        }
    ):
        raise ProfileBankRobustnessError("BU10 contract drift")
    return payload


def _shape(values: Sequence[float]) -> np.ndarray:
    array = np.log(np.asarray(values, dtype=np.float64))
    return array - array[0]


def _features_200t(
    config: Mapping[str, Any],
    root: Path,
    data_root: Path,
    pattern: str,
    uncertainty: float,
) -> dict[str, dict[str, np.ndarray]]:
    lock = config["source_200t_contract"]
    contract_path = root / lock["path"]
    if not contract_path.is_file() or hash_file(contract_path) != lock["sha256"]:
        raise ProfileBankRobustnessError("BU10 200T contract drift")
    source_config = load_200t_contract(contract_path)
    pdf = data_root / source_config["source"]["path"]
    if not pdf.is_file() or hash_file(pdf) != source_config["source"]["sha256"]:
        raise ProfileBankRobustnessError("BU10 200T PDF drift")
    page = PdfReader(pdf).pages[source_config["raster_source"]["page_zero_based"]]
    images = _extract_images(page, source_config)
    result: dict[str, dict[str, np.ndarray]] = {"mtf": {}, "granularity": {}}
    grids = {"mtf": MTF_GRID, "granularity": GRANULARITY_GRID}
    for domain in ("mtf", "granularity"):
        spec = source_config["raster_source"][domain]
        for channel in CHANNELS:
            nominal_pixels = [
                _axis_pixel(float(value), spec["x_axis"]) for value in grids[domain]
            ]
            trace = _trace_dark_curve(
                np.asarray(images[domain].convert("L")),
                x_start=round(min(nominal_pixels)) - 2,
                x_end=round(max(nominal_pixels)) + 2,
                y_min=int(spec["trace_y_bounds"][0]),
                y_max=int(spec["trace_y_bounds"][1]),
                endpoint_y=int(spec["endpoint_y"][channel]),
                maximum_step=int(spec["maximum_step"]),
                continuity_penalty=float(spec["continuity_penalty"]),
            )
            lookup = {x: y for x, y in trace}
            values: list[float] = []
            for index, coordinate in enumerate(grids[domain]):
                dx, dy = _offset(pattern, index, uncertainty)
                x = round(_axis_pixel(float(coordinate), spec["x_axis"]) + dx)
                values.append(_axis_value(float(lookup[x]) + dy, spec["y_axis"]))
            result[domain][channel] = _shape(values)
    return result


def _predict(
    observation: Mapping[str, Mapping[str, np.ndarray]],
    templates: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    channels: Sequence[str],
) -> tuple[str, dict[str, float], float]:
    distances = {
        stock: _distance(observation, templates[stock], channels, ("mtf", "granularity"))
        for stock in ALL_STOCKS
    }
    ordered = sorted(distances.items(), key=lambda item: (item[1], item[0]))
    margin = (ordered[1][1] - ordered[0][1]) / (ordered[1][1] + ordered[0][1] + 1e-12)
    return ordered[0][0], distances, float(margin)


def audit_profile_bank(
    config: Mapping[str, Any], root: Path, *, data_root: Path
) -> dict[str, Any]:
    bu4 = _load_locked_json(root, config["parent_bu4_decision"])
    bu9 = _load_locked_json(root, config["parent_bu9_evidence"])
    compiler = _load_locked_json(root, config["legacy_compiler_contract"])
    mtf = _load_locked_json(root, compiler["parents"]["mtf_trace"])
    gran = _load_locked_json(root, compiler["parents"]["granularity_trace"])
    uncertainty = float(config["comparison"]["digitization_uncertainty_pixels"])
    templates = {
        stock: _profile_features(mtf, gran, stock, "nominal", uncertainty)
        for stock in STOCKS
    }
    templates[STOCK_200T] = _features_200t(
        config, root, data_root, "nominal", uncertainty
    )
    rows: list[dict[str, Any]] = []
    correct_by_stock = {stock: 0 for stock in ALL_STOCKS}
    total_by_stock = {stock: 0 for stock in ALL_STOCKS}
    correct_by_subset = {"+".join(row): 0 for row in CHANNEL_SUBSETS}
    total_by_subset = {"+".join(row): 0 for row in CHANNEL_SUBSETS}
    margins: list[float] = []
    cyclic_correct = 0
    cyclic = {ALL_STOCKS[i]: ALL_STOCKS[(i + 1) % len(ALL_STOCKS)] for i in range(len(ALL_STOCKS))}
    for truth in ALL_STOCKS:
        for pattern in PATTERNS:
            observation = (
                _features_200t(config, root, data_root, pattern, uncertainty)
                if truth == STOCK_200T
                else _profile_features(mtf, gran, truth, pattern, uncertainty)
            )
            for subset in CHANNEL_SUBSETS:
                predicted, distances, margin = _predict(observation, templates, subset)
                correct = predicted == truth
                correct_by_stock[truth] += int(correct)
                total_by_stock[truth] += 1
                key = "+".join(subset)
                correct_by_subset[key] += int(correct)
                total_by_subset[key] += 1
                margins.append(margin)
                cyclic_correct += int(cyclic[predicted] == truth)
                rows.append({"truth_stock_id": truth, "perturbation": pattern, "channels": list(subset), "prediction": predicted, "distances": distances, "normalized_margin": margin})
    count = len(rows)
    accuracy = sum(correct_by_stock.values()) / count
    recalls = {stock: correct_by_stock[stock] / total_by_stock[stock] for stock in ALL_STOCKS}
    subset_accuracy = {key: correct_by_subset[key] / total_by_subset[key] for key in correct_by_subset}
    legacy_correct = all(row["prediction"] == row["truth_stock_id"] for row in rows if row["truth_stock_id"] != STOCK_200T)
    gates = {
        "parents_exact_and_passed": bu4.get("decision") == "retain_robust_non_renderable_manufacturer_source_signature" and bu9.get("status") == "PASS_NON_RENDERABLE_SOURCE_PROFILE",
        "required_case_count": count == config["gates"]["required_case_count"],
        "joint_top1_accuracy": accuracy >= config["gates"]["minimum_joint_top1_accuracy"],
        "joint_recall_per_stock": min(recalls.values()) >= config["gates"]["minimum_joint_recall_per_stock"],
        "joint_accuracy_per_channel_subset": min(subset_accuracy.values()) >= config["gates"]["minimum_joint_accuracy_per_channel_subset"],
        "normalized_worst_margin": min(margins) >= config["gates"]["minimum_normalized_worst_margin"],
        "cyclic_label_control": cyclic_correct / count <= config["gates"]["maximum_cyclic_label_control_accuracy"],
        "legacy_cases_remain_correct": legacy_correct,
        "no_render_authority": True,
    }
    stable = {"experiment_id": config["experiment_id"], "case_count": count, "accuracy": accuracy, "recall_by_stock": recalls, "accuracy_by_channel_subset": subset_accuracy, "worst_normalized_margin": min(margins), "cyclic_label_control_accuracy": cyclic_correct / count, "gate_results": gates, "rows": rows}
    passed = all(gates.values())
    return {"schema": REPORT_SCHEMA, **stable, "audit_pass": passed, "stable_evidence_id": hashlib.sha256(canonical_json(stable)).hexdigest(), "decision": config["branch_rule"]["pass" if passed else "fail"], "claim_ceiling": config["claim_ceiling"]}
