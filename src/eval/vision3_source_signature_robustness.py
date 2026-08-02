"""U5.R2BU4 robustness audit for non-renderable VISION3 source signatures."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.vision3_observed_source_profile import compile_observed_bundle
from src.eval.vision3_observed_source_profile import (
    load_contract as load_compiler_contract,
)
from src.film_physics.observed_source_profile import (
    CHANNELS,
    EXECUTION_AUTHORITY,
    STOCKS,
)

SCHEMA = "neuro_film.u5_r2bu4_vision3_source_signature_robustness_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2bu4_vision3_source_signature_robustness_report.v1"
PATTERNS = (
    "nominal",
    "all_y_plus",
    "all_y_minus",
    "alternating_xy",
    "inverse_alternating_xy",
)
CHANNEL_SUBSETS = (
    ("blue", "green", "red"),
    ("blue", "green"),
    ("blue", "red"),
    ("green", "red"),
)
MTF_GRID = (28.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 63.0)
GRANULARITY_GRID = (
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.25,
    3.5,
    3.75,
    4.0,
)


class SourceSignatureRobustnessError(RuntimeError):
    """Raised when the frozen BU4 inputs or protocol drift."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise SourceSignatureRobustnessError("BU4 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    comparison = payload.get("comparison", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != "U5.R2BU4"
        or payload.get("parent_decision", {}).get("sha256")
        != "d4c318a857f788b3ea5fdef292ba44f24f2d4386a03cb02928c77a24424c95a9"
        or payload.get("compiler_contract", {}).get("sha256")
        != "8144dfc018839e61ba202c1f4182679624269ad33eb02e22cc4e524e53cdd3af"
        or tuple(comparison.get("stocks", ())) != STOCKS
        or tuple(comparison.get("channels", ())) != CHANNELS
        or tuple(tuple(row) for row in comparison.get("channel_subsets", ()))
        != CHANNEL_SUBSETS
        or tuple(comparison.get("mtf_frequencies_cycles_per_mm", ())) != MTF_GRID
        or tuple(comparison.get("granularity_log_relative_exposures", ()))
        != GRANULARITY_GRID
        or tuple(comparison.get("perturbation_patterns", ())) != PATTERNS
        or comparison.get("digitization_uncertainty_pixels") != 2.0
        or gates
        != {
            "required_case_count": 60,
            "minimum_joint_top1_accuracy": 0.95,
            "minimum_joint_recall_per_stock": 0.9,
            "minimum_joint_accuracy_per_channel_subset": 0.9,
            "minimum_normalized_worst_margin": 0.02,
            "maximum_cyclic_label_control_accuracy": 0.34,
            "joint_accuracy_must_not_be_lower_than_each_single_domain": True,
            "exact_bundle_recompile": True,
            "two_byte_identical_audits": True,
            "no_render_authority": True,
        }
    ):
        raise SourceSignatureRobustnessError("BU4 frozen contract drift")
    _relative_path(str(payload.get("parent_decision", {}).get("path", "")))
    _relative_path(str(payload.get("compiler_contract", {}).get("path", "")))
    return payload


def _load_locked_json(root: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    path = root / _relative_path(str(lock["path"]))
    if not path.is_file() or hash_file(path) != lock["sha256"]:
        raise SourceSignatureRobustnessError(f"BU4 locked input mismatch: {lock['path']}")
    return json.loads(path.read_text(encoding="utf-8"))


def _linear_value_from_pixel(pixel: float, anchors: Sequence[Sequence[float]]) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    return float(value0 + (pixel - pixel0) / (pixel1 - pixel0) * (value1 - value0))


def _log_value_from_pixel(pixel: float, anchors: Sequence[Sequence[float]]) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    fraction = (pixel - pixel0) / (pixel1 - pixel0)
    return float(
        10.0
        ** (
            math.log10(float(value0))
            + fraction * (math.log10(float(value1)) - math.log10(float(value0)))
        )
    )


def _offset(pattern: str, index: int, uncertainty: float) -> tuple[float, float]:
    if pattern == "nominal":
        return 0.0, 0.0
    if pattern == "all_y_plus":
        return 0.0, uncertainty
    if pattern == "all_y_minus":
        return 0.0, -uncertainty
    sign = 1.0 if index % 2 == 0 else -1.0
    if pattern == "alternating_xy":
        return sign * uncertainty, sign * uncertainty
    if pattern == "inverse_alternating_xy":
        return -sign * uncertainty, sign * uncertainty
    raise SourceSignatureRobustnessError(f"unknown perturbation: {pattern}")


def _trace_curve(
    row: Mapping[str, Any],
    channel: str,
    *,
    domain: str,
    pattern: str,
    uncertainty: float,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates: list[float] = []
    values: list[float] = []
    for index, (x_pixel, y_pixel) in enumerate(row["curves"][channel]):
        dx, dy = _offset(pattern, index, uncertainty)
        if domain == "mtf":
            coordinates.append(
                _log_value_from_pixel(
                    float(x_pixel) + dx, row["graph_axes"]["x_value_pixels"]
                )
            )
            values.append(
                _log_value_from_pixel(
                    float(y_pixel) + dy, row["graph_axes"]["y_value_pixels"]
                )
                / 100.0
            )
        else:
            coordinates.append(
                _linear_value_from_pixel(
                    float(x_pixel) + dx, row["graph_axes"]["x_value_pixels"]
                )
            )
            values.append(
                _log_value_from_pixel(
                    float(y_pixel) + dy, row["graph_axes"]["y_value_pixels"]
                )
            )
    x = np.asarray(coordinates, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if np.any(np.diff(x) <= 0.0) or np.any(y <= 0.0) or not np.isfinite(y).all():
        raise SourceSignatureRobustnessError("BU4 perturbation produced invalid curve")
    return x, y


def _shape_feature(
    x: np.ndarray, y: np.ndarray, grid: Sequence[float], *, domain: str
) -> np.ndarray:
    query = np.asarray(grid, dtype=np.float64)
    if query[0] < x[0] or query[-1] > x[-1]:
        raise SourceSignatureRobustnessError("BU4 common grid exceeds curve support")
    if domain == "mtf":
        sampled = np.interp(np.log10(query), np.log10(x), np.log(y))
    else:
        sampled = np.interp(query, x, np.log(y))
    return sampled - sampled[0]


def _profile_features(
    mtf_trace: Mapping[str, Any],
    granularity_trace: Mapping[str, Any],
    stock: str,
    pattern: str,
    uncertainty: float,
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {"mtf": {}, "granularity": {}}
    for channel in CHANNELS:
        x, y = _trace_curve(
            mtf_trace["stocks"][stock],
            channel,
            domain="mtf",
            pattern=pattern,
            uncertainty=uncertainty,
        )
        result["mtf"][channel] = _shape_feature(x, y, MTF_GRID, domain="mtf")
        x, y = _trace_curve(
            granularity_trace["stocks"][stock],
            channel,
            domain="granularity",
            pattern=pattern,
            uncertainty=uncertainty,
        )
        result["granularity"][channel] = _shape_feature(
            x, y, GRANULARITY_GRID, domain="granularity"
        )
    return result


def _distance(
    observation: Mapping[str, Mapping[str, np.ndarray]],
    template: Mapping[str, Mapping[str, np.ndarray]],
    channels: Sequence[str],
    domains: Sequence[str],
) -> float:
    errors = [
        float(np.sqrt(np.mean(np.square(observation[domain][channel] - template[domain][channel]))))
        for domain in domains
        for channel in channels
    ]
    return float(np.mean(errors))


def _predict(
    observation: Mapping[str, Mapping[str, np.ndarray]],
    templates: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    channels: Sequence[str],
    domains: Sequence[str],
) -> tuple[str, dict[str, float], float]:
    distances = {
        stock: _distance(observation, templates[stock], channels, domains)
        for stock in STOCKS
    }
    ordered = sorted(distances.items(), key=lambda item: (item[1], item[0]))
    best, second = ordered[0], ordered[1]
    margin = float((second[1] - best[1]) / (second[1] + best[1] + 1e-12))
    return best[0], distances, margin


def audit_source_signature(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parent = _load_locked_json(root, config["parent_decision"])
    compiler_path = root / _relative_path(config["compiler_contract"]["path"])
    compiler_config = load_compiler_contract(compiler_path)
    first_compile, first_bundle = compile_observed_bundle(compiler_config, root)
    second_compile, second_bundle = compile_observed_bundle(compiler_config, root)
    bundle_sha = hashlib.sha256(canonical_json(first_bundle)).hexdigest()
    parent_gate = bool(
        parent.get("decision") == config["parent_decision"]["required_decision"]
        and parent.get("formal_evidence", {}).get("bundle_sha256")
        == config["parent_decision"]["required_bundle_sha256"]
    )
    recompile_gate = bool(
        canonical_json(first_compile) == canonical_json(second_compile)
        and canonical_json(first_bundle) == canonical_json(second_bundle)
        and bundle_sha == config["parent_decision"]["required_bundle_sha256"]
    )
    parents = compiler_config["parents"]
    mtf_trace = _load_locked_json(root, parents["mtf_trace"])
    granularity_trace = _load_locked_json(root, parents["granularity_trace"])
    uncertainty = float(config["comparison"]["digitization_uncertainty_pixels"])
    templates = {
        stock: _profile_features(
            mtf_trace, granularity_trace, stock, "nominal", uncertainty
        )
        for stock in STOCKS
    }
    rows: list[dict[str, Any]] = []
    domain_correct = {"joint": 0, "mtf": 0, "granularity": 0}
    stock_correct = {stock: 0 for stock in STOCKS}
    stock_total = {stock: 0 for stock in STOCKS}
    subset_correct = {"+".join(subset): 0 for subset in CHANNEL_SUBSETS}
    subset_total = {"+".join(subset): 0 for subset in CHANNEL_SUBSETS}
    joint_margins: list[float] = []
    cyclic_correct = 0
    cyclic = {STOCKS[index]: STOCKS[(index + 1) % len(STOCKS)] for index in range(len(STOCKS))}
    for truth in STOCKS:
        for pattern in PATTERNS:
            observation = _profile_features(
                mtf_trace, granularity_trace, truth, pattern, uncertainty
            )
            for subset in CHANNEL_SUBSETS:
                predictions: dict[str, str] = {}
                distances: dict[str, dict[str, float]] = {}
                margins: dict[str, float] = {}
                for label, domains in (
                    ("joint", ("mtf", "granularity")),
                    ("mtf", ("mtf",)),
                    ("granularity", ("granularity",)),
                ):
                    prediction, scores, margin = _predict(
                        observation, templates, subset, domains
                    )
                    predictions[label] = prediction
                    distances[label] = scores
                    margins[label] = margin
                    domain_correct[label] += int(prediction == truth)
                joint_correct = predictions["joint"] == truth
                stock_correct[truth] += int(joint_correct)
                stock_total[truth] += 1
                subset_key = "+".join(subset)
                subset_correct[subset_key] += int(joint_correct)
                subset_total[subset_key] += 1
                joint_margins.append(margins["joint"])
                cyclic_correct += int(cyclic[predictions["joint"]] == truth)
                rows.append(
                    {
                        "truth_stock_id": truth,
                        "perturbation": pattern,
                        "channels": list(subset),
                        "predictions": predictions,
                        "distances": distances,
                        "normalized_margins": margins,
                    }
                )
    case_count = len(rows)
    accuracies = {
        key: float(value / case_count) for key, value in domain_correct.items()
    }
    recall_by_stock = {
        stock: float(stock_correct[stock] / stock_total[stock]) for stock in STOCKS
    }
    accuracy_by_subset = {
        key: float(subset_correct[key] / subset_total[key]) for key in subset_correct
    }
    cyclic_accuracy = float(cyclic_correct / case_count)
    gate_results = {
        "parent": parent_gate,
        "exact_bundle_recompile": recompile_gate,
        "required_case_count": case_count == int(config["gates"]["required_case_count"]),
        "joint_top1_accuracy": accuracies["joint"]
        >= float(config["gates"]["minimum_joint_top1_accuracy"]),
        "joint_recall_per_stock": min(recall_by_stock.values())
        >= float(config["gates"]["minimum_joint_recall_per_stock"]),
        "joint_accuracy_per_channel_subset": min(accuracy_by_subset.values())
        >= float(config["gates"]["minimum_joint_accuracy_per_channel_subset"]),
        "normalized_worst_margin": min(joint_margins)
        >= float(config["gates"]["minimum_normalized_worst_margin"]),
        "cyclic_label_control": cyclic_accuracy
        <= float(config["gates"]["maximum_cyclic_label_control_accuracy"]),
        "joint_not_lower_than_single_domains": accuracies["joint"]
        >= max(accuracies["mtf"], accuracies["granularity"]),
        "no_render_authority": first_bundle.get("execution_authority")
        == EXECUTION_AUTHORITY,
    }
    audit_pass = all(gate_results.values())
    stable_payload = {
        "experiment_id": config["experiment_id"],
        "bundle_sha256": bundle_sha,
        "case_count": case_count,
        "accuracies": accuracies,
        "recall_by_stock": recall_by_stock,
        "accuracy_by_channel_subset": accuracy_by_subset,
        "worst_normalized_joint_margin": min(joint_margins),
        "cyclic_label_control_accuracy": cyclic_accuracy,
        "gate_results": gate_results,
        "rows": rows,
    }
    report = {
        "schema": REPORT_SCHEMA,
        **stable_payload,
        "stable_evidence_id": hashlib.sha256(canonical_json(stable_payload)).hexdigest(),
        "audit_pass": audit_pass,
        "decision": (
            "retain_robust_non_renderable_manufacturer_source_signature"
            if audit_pass
            else "close_source_signature_robustness_without_rescue"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    return report
