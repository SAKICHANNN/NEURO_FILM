"""BU13 cross-source ordinal test for VISION3 spatial evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader

from src.eval.kodak_vision3_200t_raster_signature import (
    _extract_images,
    _trace_domain,
)
from src.eval.kodak_vision3_200t_raster_signature import (
    load_contract as load_200t_contract,
)
from src.eval.kodak_vision3_mtf_diversity import (
    _log_value_from_pixel,
    hash_file,
    load_trace,
)

SCHEMA = "neuro_film.u5-r2bu13-vision3-mtf-resolution-transfer-contract.v1"
REPORT_SCHEMA = "neuro_film.u5-r2bu13-vision3-mtf-resolution-transfer-report.v1"
STOCK_MAP = {
    "50D": "kodak_vision3_50d_5203_7203",
    "500T": "kodak_vision3_500t_5219_7219",
}


class ResolutionTransferError(RuntimeError):
    """Raised when the frozen BU13 inputs or protocol drift."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    comparison = payload.get("comparison", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != "U5.R2BU13"
        or comparison.get("stocks") != ["50D", "200T", "500T"]
        or comparison.get("channels") != ["blue", "green", "red"]
        or comparison.get("frequencies_cycles_per_mm") != [42.0, 53.0]
        or comparison.get("summaries")
        != ["minimum_channel_response", "geometric_mean_channel_response"]
        or comparison.get("pixel_fit_allowed") is not False
        or comparison.get("render_allowed") is not False
    ):
        raise ResolutionTransferError("BU13 contract drift")
    return payload


def _locked_json(root: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(lock["path"])
    if not path.is_file() or hash_file(path) != lock["sha256"]:
        raise ResolutionTransferError(f"locked input drift: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _legacy_responses(
    trace: Mapping[str, Any], stock: str, frequencies: np.ndarray
) -> dict[str, np.ndarray]:
    row = trace["stocks"][STOCK_MAP[stock]]
    axes = row["graph_axes"]
    result: dict[str, np.ndarray] = {}
    for channel in ("blue", "green", "red"):
        pixels = np.asarray(row["curves"][channel], dtype=np.float64)
        x = np.asarray(
            [_log_value_from_pixel(v, axes["x_value_pixels"]) for v in pixels[:, 0]]
        )
        y = np.asarray(
            [_log_value_from_pixel(v, axes["y_value_pixels"]) for v in pixels[:, 1]]
        )
        if frequencies[0] < x[0] or frequencies[-1] > x[-1]:
            raise ResolutionTransferError(f"BU13 query outside {stock}/{channel} support")
        result[channel] = np.exp(np.interp(np.log(frequencies), np.log(x), np.log(y)))
    return result


def _responses_200t(
    root: Path, source_contract: Mapping[str, Any], frequencies: np.ndarray
) -> dict[str, np.ndarray]:
    config = load_200t_contract(root / str(source_contract["path"]))
    pdf = root / str(config["source"]["path"])
    if not pdf.is_file() or hash_file(pdf) != config["source"]["sha256"]:
        raise ResolutionTransferError("BU13 200T source drift")
    page = PdfReader(pdf).pages[config["raster_source"]["page_zero_based"]]
    images = _extract_images(page, config)
    values, _ = _trace_domain(
        images["mtf"], config["raster_source"]["mtf"], frequencies.tolist()
    )
    return {
        channel: np.asarray(values[channel]["values"], dtype=np.float64)
        for channel in ("blue", "green", "red")
    }


def evaluate(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parents = config["parents"]
    bu0 = _locked_json(root, parents["bu0_decision"])
    bu10 = _locked_json(root, parents["bu10_evidence"])
    bu12 = _locked_json(root, parents["bu12_evidence"])
    trace = load_trace(root / str(parents["legacy_mtf_trace"]["path"]))
    if (
        bu0.get("decision") != parents["bu0_decision"]["required_decision"]
        or bu10.get("decision") != parents["bu10_evidence"]["required_decision"]
        or bu12.get("decision") != parents["bu12_evidence"]["required_decision"]
    ):
        raise ResolutionTransferError("BU13 parent decision drift")
    frequencies = np.asarray(
        config["comparison"]["frequencies_cycles_per_mm"], dtype=np.float64
    )
    responses = {
        "50D": _legacy_responses(trace, "50D", frequencies),
        "200T": _responses_200t(root, parents["source_200t_contract"], frequencies),
        "500T": _legacy_responses(trace, "500T", frequencies),
    }
    scores: dict[str, dict[str, list[float]]] = {}
    for stock, channels in responses.items():
        matrix = np.stack([channels[name] for name in ("blue", "green", "red")])
        scores[stock] = {
            "minimum_channel_response": np.min(matrix, axis=0).tolist(),
            "geometric_mean_channel_response": np.exp(
                np.mean(np.log(matrix), axis=0)
            ).tolist(),
        }
    rows: list[dict[str, Any]] = []
    fast_noninferior = {"50D": 0, "200T": 0}
    fast_strict = {"50D": 0, "200T": 0}
    for summary in config["comparison"]["summaries"]:
        for index, frequency in enumerate(frequencies):
            values = {stock: scores[stock][summary][index] for stock in scores}
            for stock in ("50D", "200T"):
                fast_noninferior[stock] += int(values[stock] >= values["500T"])
                fast_strict[stock] += int(values[stock] > values["500T"])
            rows.append(
                {
                    "summary": summary,
                    "frequency_cycles_per_mm": float(frequency),
                    "responses_percent": values,
                    "predicted_slowest_stock": min(values, key=lambda key: (values[key], key)),
                }
            )
    comparisons = len(rows)
    primary_score = sum(fast_noninferior.values()) + sum(fast_strict.values())
    wrong_scores = {}
    for proposed_slow in ("50D", "200T"):
        proposed_fast = [stock for stock in scores if stock != proposed_slow]
        ni = sum(
            int(row["responses_percent"][stock] >= row["responses_percent"][proposed_slow])
            for row in rows
            for stock in proposed_fast
        )
        strict = sum(
            int(row["responses_percent"][stock] > row["responses_percent"][proposed_slow])
            for row in rows
            for stock in proposed_fast
        )
        wrong_scores[proposed_slow] = ni + strict
    gates = config["gates"]
    rate = min(fast_noninferior.values()) / comparisons
    gate_results = {
        "required_frequency_count": len(frequencies) == gates["required_frequency_count"],
        "required_summary_count": len(config["comparison"]["summaries"])
        == gates["required_summary_count"],
        "fast_noninferiority_rate": rate >= gates["minimum_fast_noninferiority_rate"],
        "each_fast_strict_wins": min(fast_strict.values())
        >= gates["minimum_each_fast_strict_wins"],
        "assignment_margin": primary_score - max(wrong_scores.values())
        >= gates["minimum_assignment_margin"],
        "all_queries_inside_trace_support": True,
        "no_pixel_fit_or_render": True,
    }
    stable = {
        "experiment_id": config["experiment_id"],
        "frequencies_cycles_per_mm": frequencies.tolist(),
        "scores": scores,
        "rows": rows,
        "fast_noninferior_counts": fast_noninferior,
        "fast_strict_win_counts": fast_strict,
        "fast_noninferiority_rate": rate,
        "primary_assignment_score": primary_score,
        "wrong_slow_assignment_scores": wrong_scores,
        "assignment_score_margin": primary_score - max(wrong_scores.values()),
        "gate_results": gate_results,
    }
    passed = all(gate_results.values())
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "automatic_pass": passed,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
        "decision": config["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = ["ResolutionTransferError", "evaluate", "load_contract"]
