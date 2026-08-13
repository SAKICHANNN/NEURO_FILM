"""Ordinal processed-film resolution evidence from the Noever Table 3 source."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "neuro_film.u5-r2bu12-noever-vision3-resolution-order-contract.v1"
REPORT_SCHEMA = "neuro_film.u5-r2bu12-noever-vision3-resolution-order-report.v1"
STOCKS = ("50D", "200T", "500T")
EXPECTED_VALUES = {
    "50D": {"super16": {"2k": 42, "4k": 53, "6k": 67}, "super35": {"2k": 42, "4k": 84, "6k": 84}},
    "200T": {"super16": {"2k": 42, "4k": 53, "6k": 53}, "super35": {"2k": 42, "4k": 84, "6k": 84}},
    "500T": {"super16": {"2k": 42, "4k": 42, "6k": 53}, "super35": {"2k": 42, "4k": 67, "6k": 84}},
}


class ResolutionOrderError(RuntimeError):
    """Raised when a frozen source or contract drifts."""


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != "U5.R2BU12"
        or tuple(payload.get("comparison", {}).get("fast_stocks", ()))
        != ("50D", "200T")
        or payload.get("comparison", {}).get("slow_stock") != "500T"
        or tuple(payload.get("comparison", {}).get("formats", ()))
        != ("super16", "super35")
        or tuple(payload.get("comparison", {}).get("scan_resolutions", ()))
        != ("2k", "4k", "6k")
        or payload.get("source", {}).get("values_lp_per_mm") != EXPECTED_VALUES
        or payload.get("gates")
        != {
            "required_strata": 6,
            "minimum_fast_stock_noninferiority_rate": 1.0,
            "minimum_each_fast_stock_strict_wins": 2,
            "minimum_total_strict_wins": 5,
            "minimum_assignment_score_margin": 1,
            "require_aliasing_flags_reported_not_discarded": True,
            "no_pixel_fit_or_render_authority": True,
        }
    ):
        raise ResolutionOrderError("BU12 contract drift")
    return payload


def _load_locked(root: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(lock["path"])
    if not path.is_file() or hash_file(path) != lock["sha256"]:
        raise ResolutionOrderError(f"locked input drift: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _score_assignment(
    values: Mapping[str, Any], assignment: tuple[str, str, str], strata: list[tuple[str, str]]
) -> dict[str, Any]:
    fast_a, fast_b, slow = assignment
    noninferior_a = 0
    noninferior_b = 0
    strict_a = 0
    strict_b = 0
    for film_format, resolution in strata:
        a = int(values[fast_a][film_format][resolution])
        b = int(values[fast_b][film_format][resolution])
        c = int(values[slow][film_format][resolution])
        noninferior_a += int(a >= c)
        noninferior_b += int(b >= c)
        strict_a += int(a > c)
        strict_b += int(b > c)
    return {
        "fast_stocks": [fast_a, fast_b],
        "slow_stock": slow,
        "noninferior_counts": {fast_a: noninferior_a, fast_b: noninferior_b},
        "strict_win_counts": {fast_a: strict_a, fast_b: strict_b},
        "total_strict_wins": strict_a + strict_b,
        "score": 2 * (noninferior_a + noninferior_b) + strict_a + strict_b,
    }


def evaluate(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    bu10 = _load_locked(root, config["parents"]["bu10_evidence"])
    bu11 = _load_locked(root, config["parents"]["bu11_evidence"])
    source = config["source"]
    table_path = root / str(source["table_image_path"])
    if not table_path.is_file() or hash_file(table_path) != source["table_image_sha256"]:
        raise ResolutionOrderError("Noever Table 3 image drift")
    if (
        bu10.get("decision")
        != "retain_robust_four_stock_non_renderable_manufacturer_source_profile_bank"
        or bu11.get("decision") != "close_noever_resolution_source_without_replacement_or_rescue"
    ):
        raise ResolutionOrderError("parent decision drift")

    values = source["values_lp_per_mm"]
    aliasing = source["aliasing_flags"]
    if set(values) != set(STOCKS) or set(aliasing) != set(STOCKS):
        raise ResolutionOrderError("stock table drift")
    strata = [
        (film_format, resolution)
        for film_format in config["comparison"]["formats"]
        for resolution in config["comparison"]["scan_resolutions"]
    ]
    rows: list[dict[str, Any]] = []
    for film_format, resolution in strata:
        rows.append(
            {
                "format": film_format,
                "scan_resolution": resolution,
                "values_lp_per_mm": {
                    stock: int(values[stock][film_format][resolution])
                    for stock in STOCKS
                },
                "aliasing_flags": {
                    stock: str(aliasing[stock][film_format][resolution])
                    for stock in STOCKS
                },
            }
        )
    assignment_scores = [
        _score_assignment(values, assignment, strata)
        for assignment in itertools.permutations(STOCKS)
    ]
    assignment_scores.sort(
        key=lambda row: (
            -row["score"],
            tuple(row["fast_stocks"]),
            row["slow_stock"],
        )
    )
    primary = next(
        row
        for row in assignment_scores
        if row["fast_stocks"] == ["50D", "200T"] and row["slow_stock"] == "500T"
    )
    best_wrong = max(
        row["score"]
        for row in assignment_scores
        if row["slow_stock"] != "500T"
    )
    gates = config["gates"]
    rate = min(primary["noninferior_counts"].values()) / len(strata)
    gate_results = {
        "required_strata": len(strata) == gates["required_strata"],
        "fast_stock_noninferiority_rate": rate
        >= gates["minimum_fast_stock_noninferiority_rate"],
        "each_fast_stock_strict_wins": min(primary["strict_win_counts"].values())
        >= gates["minimum_each_fast_stock_strict_wins"],
        "total_strict_wins": primary["total_strict_wins"]
        >= gates["minimum_total_strict_wins"],
        "assignment_score_margin": primary["score"] - best_wrong
        >= gates["minimum_assignment_score_margin"],
        "aliasing_flags_reported_not_discarded": all(
            len(row["aliasing_flags"]) == 3 for row in rows
        ),
        "no_pixel_fit_or_render_authority": True,
    }
    passed = all(gate_results.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "source_table_sha256": source["table_image_sha256"],
        "rows": rows,
        "primary_assignment": primary,
        "best_wrong_slow_stock_score": best_wrong,
        "assignment_score_margin": primary["score"] - best_wrong,
        "fast_stock_noninferiority_rate": rate,
        "assignment_scores": assignment_scores,
        "gate_results": gate_results,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "automatic_pass": passed,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
        "decision": config["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = ["ResolutionOrderError", "evaluate", "hash_file", "load_contract"]
