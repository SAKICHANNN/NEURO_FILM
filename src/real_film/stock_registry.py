"""Validation and source cross-checks for the stock-first registry."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


class StockRegistryError(ValueError):
    """Raised when the stock-first evidence contract is inconsistent."""


REQUIRED_STOCK_FIELDS = {
    "film_stock_id", "display_name", "manufacturer", "product_line",
    "nominal_iso", "film_type", "emulsion_generation", "catalog_code",
    "market_status", "source_dataset", "source_label", "label_status",
    "data_evidence_grade", "expert_evidence_grade", "roll_count",
    "frame_count", "aligned_public_frame_count", "selected_first_pilot",
    "claim_ceiling",
}

DATA_GRADES = {"S0", "S1-candidate", "S1", "S2", "S3"}
EXPERT_GRADES = {"none", "S1", "S2", "S3"}
MARKET_STATUSES = {"unknown", "current", "discontinued"}


def validate_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    if registry.get("schema_version") != 1:
        raise StockRegistryError("unsupported registry schema")
    source_datasets = registry.get("source_datasets")
    if not isinstance(source_datasets, Mapping) or not source_datasets:
        raise StockRegistryError("registry must define source datasets")
    stocks = registry.get("stocks")
    if not isinstance(stocks, Sequence) or not stocks:
        raise StockRegistryError("registry must contain stock rows")
    ids: list[str] = []
    labels: list[tuple[str, str]] = []
    for row in stocks:
        missing = REQUIRED_STOCK_FIELDS - set(row)
        if missing:
            raise StockRegistryError(f"stock row missing fields: {sorted(missing)}")
        ids.append(str(row["film_stock_id"]))
        labels.append((str(row["source_dataset"]), str(row["source_label"])))
        if not str(row["film_stock_id"]).strip() or not str(row["source_label"]).strip():
            raise StockRegistryError("stock ids and source labels must be non-empty")
        if row["source_dataset"] not in source_datasets:
            raise StockRegistryError("stock row references an unknown source dataset")
        if not isinstance(row["nominal_iso"], int) or int(row["nominal_iso"]) <= 0:
            raise StockRegistryError("nominal_iso must be a positive integer")
        if int(row["roll_count"]) < 1 or int(row["frame_count"]) < int(row["roll_count"]):
            raise StockRegistryError("invalid roll/frame counts")
        if int(row["aligned_public_frame_count"]) > int(row["frame_count"]):
            raise StockRegistryError("aligned frame count exceeds frames")
        if row["label_status"] == "claimed_stock_hint" and row["data_evidence_grade"] != "S0":
            raise StockRegistryError("claimed hints must remain S0")
        if row["data_evidence_grade"] not in DATA_GRADES:
            raise StockRegistryError("invalid data grade")
        if row["expert_evidence_grade"] not in EXPERT_GRADES:
            raise StockRegistryError("invalid expert grade")
        if row["market_status"] not in MARKET_STATUSES:
            raise StockRegistryError("invalid market status")
        if row["selected_first_pilot"] and row["data_evidence_grade"] == "S0":
            raise StockRegistryError("S0 stock hints cannot be selected as pilots")
    if len(ids) != len(set(ids)) or len(labels) != len(set(labels)):
        raise StockRegistryError("film_stock_id and source labels must be unique")
    selected = [row for row in stocks if bool(row["selected_first_pilot"])]
    if not 2 <= len(selected) <= 4:
        raise StockRegistryError("first stock pilot count must be between two and four")
    coverage = registry.get("coverage")
    if not isinstance(coverage, Mapping):
        raise StockRegistryError("registry must define coverage")
    expected_coverage = {
        "named_stock_data_s1_or_higher": sum(
            row["data_evidence_grade"] in {"S1", "S2", "S3"} for row in stocks
        ),
        "named_stock_experts_s2_or_higher": sum(
            row["expert_evidence_grade"] in {"S2", "S3"} for row in stocks
        ),
        "calibrated_stock_experts_s3": sum(
            row["expert_evidence_grade"] == "S3" for row in stocks
        ),
    }
    for key, expected in expected_coverage.items():
        if coverage.get(key) != expected:
            raise StockRegistryError(f"coverage mismatch for {key}")
    if coverage.get("coverage_accounts_must_not_be_merged") is not True:
        raise StockRegistryError("named and historical coverage must remain separate")
    return {
        "stock_rows": len(stocks),
        "selected_pilots": [str(row["film_stock_id"]) for row in selected],
        "s0_rows": sum(row["data_evidence_grade"] == "S0" for row in stocks),
        "current_s1_rows": sum(row["data_evidence_grade"] == "S1" for row in stocks),
        "s2_experts": expected_coverage["named_stock_experts_s2_or_higher"],
    }


def crosscheck_blueneg(
    registry: Mapping[str, Any],
    frame_rows: Sequence[Mapping[str, Any]],
    roll_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stocks = {
        str(row["source_label"]): row
        for row in registry["stocks"]
        if row["source_dataset"] == "blueneg"
    }
    frames_by_stock = Counter(str(row["film_type"]) for row in frame_rows)
    rolls_by_stock: dict[str, set[str]] = defaultdict(set)
    aligned_by_stock = Counter()
    for row in roll_rows:
        label = str(row["film_type"])
        rolls_by_stock[label].add(str(row["roll_id"]))
        aligned_by_stock[label] += int(row["public_non_test_pseudogt_frames"])
    if set(stocks) != set(frames_by_stock) or set(stocks) != set(rolls_by_stock):
        raise StockRegistryError("registry and BlueNeg film strings differ")
    mismatches: list[dict[str, Any]] = []
    for label, stock in sorted(stocks.items()):
        observed = {
            "roll_count": len(rolls_by_stock[label]),
            "frame_count": frames_by_stock[label],
            "aligned_public_frame_count": aligned_by_stock[label],
        }
        expected = {key: int(stock[key]) for key in observed}
        if expected != observed:
            mismatches.append({"source_label": label, "expected": expected, "observed": observed})
    return {
        "passed": not mismatches,
        "film_strings": len(stocks),
        "rolls": len({str(row["roll_id"]) for row in roll_rows}),
        "frames": len(frame_rows),
        "mismatches": mismatches,
    }
